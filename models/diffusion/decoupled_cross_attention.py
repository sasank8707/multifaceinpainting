"""
Decoupled Identity Cross-Attention Architecture for Stable Diffusion.

Preserves the existing pretrained CLIP text cross-attention pathway 100% intact:
  text prompt -> pretrained CLIP text encoder -> existing UNet text cross-attention

Adds a parallel, dedicated identity cross-attention pathway:
  references -> real ArcFace spatial features -> Phase-4 regional reference fusion
  -> T_fused [B, 256, 512] -> Linear to_k_id -> Linear to_v_id -> softmax(Q K_id^T / sqrt(d)) V_id

Final attention output:
  O = O_text + lambda * O_id

Properties:
- True decoupling: zero modification to text encoder or text conditioning vectors.
- At lambda=0: mathematically and bit-exact equivalent to vanilla UNet.
- At lambda>0: consumes all 256 spatial tokens of T_fused directly without text-sequence padding.
- Zero target identity leakage: keys and values derive strictly from reference features.
- Mask-aware regional fusion preserved: Phase-4 mask weighting feeds directly into T_fused.
- Zero reference pixel copying: purely latent attention guidance.
"""

import math
from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention


class DecoupledIdentityAttnProcessor(nn.Module):
    """
    Decoupled Cross-Attention Processor for Stable Diffusion UNet (attn2).
    
    Computes standard text cross-attention and adds a parallel, dedicated
    identity cross-attention branch operating on Phase 4 spatial tokens (256x512).
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int = 768,
        id_dim: int = 512,
        heads: int = 8,
        default_scale: float = 0.0,
        zero_init_v: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.id_dim = id_dim
        self.heads = heads
        self.scale = default_scale

        # Dedicated Identity Key and Value Linear Projections
        # Projects 512-dim ArcFace spatial tokens to UNet inner dimension
        self.to_k_id = nn.Linear(id_dim, hidden_size, bias=False)
        self.to_v_id = nn.Linear(id_dim, hidden_size, bias=False)

        # Standard initialization for sensitivity, or zero-init for zero-initial residual
        if zero_init_v:
            nn.init.zeros_(self.to_v_id.weight)

        # Stored identity tokens when passed via manager
        self._stored_identity_tokens: Optional[torch.Tensor] = None

        # Diagnostic capture
        self.record_diagnostics: bool = False
        self.latest_diagnostics: Dict[str, Any] = {}

    def set_identity_tokens(self, tokens: Optional[torch.Tensor]):
        """Sets cached identity tokens for UNet forward pass."""
        self._stored_identity_tokens = tokens

    def set_scale(self, scale: float):
        """Sets coupling scale lambda."""
        self.scale = float(scale)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        identity_tokens: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """
        Executes decoupled text and identity cross-attention.
        """
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # Query projection from spatial latents
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        # 1. Existing Pretrained Text Cross-Attention Pathway (100% UNCHANGED)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        text_out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Resolve identity tokens and effective scale lambda
        id_tokens = identity_tokens
        effective_scale = self.scale

        if cross_attention_kwargs is not None:
            if "identity_tokens" in cross_attention_kwargs and id_tokens is None:
                id_tokens = cross_attention_kwargs["identity_tokens"]
            if "scale" in cross_attention_kwargs:
                effective_scale = float(cross_attention_kwargs["scale"])

        if id_tokens is None:
            id_tokens = self._stored_identity_tokens

        # 2. Parallel Dedicated Identity Cross-Attention Pathway
        if effective_scale != 0.0 and id_tokens is not None:
            # Match training scale lambda=0.10 under Classifier-Free Guidance (batch_size=2)
            # By applying id_tokens to both uncond and cond branches, text CFG scales prompt difference
            # while identity cross-attention is preserved at exactly its calibrated training scale lambda.
            if id_tokens.shape[0] != batch_size:
                curr_id_tokens = id_tokens.repeat(batch_size // id_tokens.shape[0], 1, 1)
            else:
                curr_id_tokens = id_tokens

            # Dynamically ensure projection weights match query device and dtype
            if self.to_k_id.weight.device != query.device or self.to_k_id.weight.dtype != query.dtype:
                self.to_k_id.to(device=query.device, dtype=query.dtype)
                self.to_v_id.to(device=query.device, dtype=query.dtype)

            # Ensure curr_id_tokens matches device and dtype of attention query
            curr_id_tokens = curr_id_tokens.to(device=query.device, dtype=query.dtype)

            # curr_id_tokens: [B, num_id_tokens, id_dim] (e.g., [2, 256, 512])
            ip_key = self.to_k_id(curr_id_tokens)
            ip_value = self.to_v_id(curr_id_tokens)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # Ensure key and value strictly match query device and dtype before attention
            if ip_key.device != query.device or ip_key.dtype != query.dtype:
                ip_key = ip_key.to(device=query.device, dtype=query.dtype)
            if ip_value.device != query.device or ip_value.dtype != query.dtype:
                ip_value = ip_value.to(device=query.device, dtype=query.dtype)

            id_out = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            # Record internal attention diagnostics if requested
            if self.record_diagnostics:
                with torch.no_grad():
                    # Attention logits: Q K^T / sqrt(d)
                    scale_factor = 1.0 / math.sqrt(head_dim)
                    id_logits = torch.matmul(query, ip_key.transpose(-1, -2)) * scale_factor
                    id_probs = torch.softmax(id_logits, dim=-1)
                    
                    entropy = float(-(id_probs * torch.log(id_probs + 1e-12)).sum(dim=-1).mean().cpu())
                    max_prob = float(id_probs.max().cpu())
                    mean_patch_mass = id_probs.mean(dim=(0, 1, 2)).cpu().numpy()
                    
                    self.latest_diagnostics = {
                        "query_device": str(query.device),
                        "query_dtype": str(query.dtype),
                        "ip_key_device": str(ip_key.device),
                        "ip_key_dtype": str(ip_key.dtype),
                        "ip_value_device": str(ip_value.device),
                        "ip_value_dtype": str(ip_value.dtype),
                        "id_key_norm": float(ip_key.norm(dim=-1).mean().cpu()),
                        "id_value_norm": float(ip_value.norm(dim=-1).mean().cpu()),
                        "logit_mean": float(id_logits.mean().cpu()),
                        "logit_std": float(id_logits.std().cpu()),
                        "logit_max": float(id_logits.max().cpu()),
                        "logit_min": float(id_logits.min().cpu()),
                        "entropy": entropy,
                        "max_prob": max_prob,
                        "mean_patch_mass": mean_patch_mass,
                        "has_nan": bool(torch.isnan(id_out).any().cpu()),
                        "has_inf": bool(torch.isinf(id_out).any().cpu())
                    }

            # Decoupled Combination: O = O_text + lambda * O_id
            hidden_states = text_out + effective_scale * id_out
        else:
            # At lambda=0: identity branch contributes exactly zero
            hidden_states = text_out

        # Reshape to sequence dimension
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Linear projection and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class DecoupledIdentityCrossAttentionManager:
    """
    Manager for attaching, configuring, and querying decoupled identity cross-attention
    processors across all 16 cross-attention blocks of a Stable Diffusion UNet.
    """

    def __init__(self, unet: nn.Module, id_dim: int = 512, default_scale: float = 0.0):
        self.unet = unet
        self.id_dim = id_dim
        self.default_scale = default_scale
        self.processors: Dict[str, DecoupledIdentityAttnProcessor] = {}
        self.original_processors: Dict[str, Any] = {}
        self._installed = False

    def install(self, zero_init_v: bool = False) -> Dict[str, DecoupledIdentityAttnProcessor]:
        """Installs decoupled identity processors onto all 16 UNet cross-attention modules."""
        if self._installed:
            return self.processors

        self.original_processors = dict(self.unet.attn_processors)
        new_processors = {}

        for name, attn_proc in self.unet.attn_processors.items():
            if name.endswith("attn2.processor"):
                # Extract hidden size and heads from corresponding Attention module
                module_name = name.replace(".processor", "")
                sub_mod = self.unet
                for part in module_name.split("."):
                    sub_mod = getattr(sub_mod, part)
                
                hidden_size = sub_mod.to_q.out_features
                heads = sub_mod.heads
                cross_attention_dim = sub_mod.to_k.in_features

                proc = DecoupledIdentityAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    id_dim=self.id_dim,
                    heads=heads,
                    default_scale=self.default_scale,
                    zero_init_v=zero_init_v
                )
                self.processors[name] = proc
                new_processors[name] = proc
            else:
                new_processors[name] = attn_proc

        self.unet.set_attn_processor(dict(new_processors))
        self._installed = True

        # Synchronize processor layers to UNet device and dtype
        try:
            sample_param = next(self.unet.parameters())
            self.to(device=sample_param.device, dtype=sample_param.dtype)
        except StopIteration:
            pass

        return self.processors

    def restore_vanilla(self):
        """Restores original vanilla attention processors."""
        if self._installed and self.original_processors:
            self.unet.set_attn_processor(dict(self.original_processors))
            self._installed = False

    def set_identity_tokens(self, identity_tokens: Optional[torch.Tensor]):
        """Sets cached identity tokens across all processors."""
        for proc in self.processors.values():
            proc.set_identity_tokens(identity_tokens)

    def set_scale(self, scale: float):
        """Sets lambda coupling factor across all processors."""
        self.default_scale = float(scale)
        for proc in self.processors.values():
            proc.set_scale(scale)

    def set_diagnostics_enabled(self, enabled: bool = True):
        """Enables or disables diagnostic recording."""
        for proc in self.processors.values():
            proc.record_diagnostics = enabled

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Returns all to_k_id and to_v_id parameters across all 16 processors."""
        params = []
        for proc in self.processors.values():
            params.extend(proc.to_k_id.parameters())
            params.extend(proc.to_v_id.parameters())
        return params

    def count_parameters(self) -> Dict[str, int]:
        """Counts total and trainable parameters in the identity cross-attention pathway."""
        total_k_params = sum(p.numel() for proc in self.processors.values() for p in proc.to_k_id.parameters())
        total_v_params = sum(p.numel() for proc in self.processors.values() for p in proc.to_v_id.parameters())
        return {
            "num_processors": len(self.processors),
            "k_id_parameters": total_k_params,
            "v_id_parameters": total_v_params,
            "total_identity_parameters": total_k_params + total_v_params
        }

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state dictionary containing all to_k_id and to_v_id weights."""
        sd = {}
        for name, proc in self.processors.items():
            sd[f"{name}.to_k_id"] = proc.to_k_id.state_dict()
            sd[f"{name}.to_v_id"] = proc.to_v_id.state_dict()
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Loads state dictionary into processors."""
        for name, proc in self.processors.items():
            k_key = f"{name}.to_k_id"
            v_key = f"{name}.to_v_id"
            if k_key in state_dict:
                proc.to_k_id.load_state_dict(state_dict[k_key], strict=strict)
            elif strict:
                raise KeyError(f"Missing key in state_dict: {k_key}")
            if v_key in state_dict:
                proc.to_v_id.load_state_dict(state_dict[v_key], strict=strict)
            elif strict:
                raise KeyError(f"Missing key in state_dict: {v_key}")

    def save_checkpoint(self, path: str, extra_data: Optional[Dict[str, Any]] = None):
        """Saves weights and metadata to file."""
        from pathlib import Path
        import time
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "scale": self.default_scale,
            "id_dim": self.id_dim,
            "state_dict": self.state_dict(),
            "timestamp": time.time()
        }
        if extra_data:
            ckpt.update(extra_data)
        torch.save(ckpt, p)

    def load_checkpoint(self, path: str, strict: bool = True) -> Dict[str, Any]:
        """Loads weights and metadata from file."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "scale" in ckpt:
            self.set_scale(ckpt["scale"])
        if "state_dict" in ckpt:
            self.load_state_dict(ckpt["state_dict"], strict=strict)

        # Synchronize processor layers to UNet device and dtype
        try:
            sample_param = next(self.unet.parameters())
            self.to(device=sample_param.device, dtype=sample_param.dtype)
        except StopIteration:
            pass

        return ckpt

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Moves all processor layers to target device and/or dtype."""
        for proc in self.processors.values():
            if device is not None:
                proc.to(device=device)
            if dtype is not None:
                proc.to(dtype=dtype)
        return self


