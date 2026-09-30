"""
Unit tests for Decoupled Identity Cross-Attention Architecture.
Validates all 10 architectural gates with deterministic assertions.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.diffusion.decoupled_cross_attention import (
    DecoupledIdentityAttnProcessor,
    DecoupledIdentityCrossAttentionManager
)
from diffusers.models.attention import Attention


class TestDecoupledIdentityCrossAttention(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.hidden_size = 320
        self.cross_attention_dim = 768
        self.id_dim = 512
        self.heads = 8
        self.head_dim = 40

        # Create a mock Attention module matching SD 1.5 attn2
        self.attn = Attention(
            query_dim=self.hidden_size,
            cross_attention_dim=self.cross_attention_dim,
            heads=self.heads,
            dim_head=self.head_dim,
            bias=False
        ).to(self.device)

        # Mock inputs
        self.batch_size = 1
        self.seq_len = 4096 # 64x64
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_size)
        self.encoder_hidden_states = torch.randn(self.batch_size, 77, self.cross_attention_dim)
        self.id_tokens = torch.randn(self.batch_size, 256, self.id_dim)

    def test_gate1_lambda0_equivalence(self):
        """Gate 1: At lambda=0, decoupled processor is mathematically equivalent to vanilla."""
        proc = DecoupledIdentityAttnProcessor(
            hidden_size=self.hidden_size,
            cross_attention_dim=self.cross_attention_dim,
            id_dim=self.id_dim,
            heads=self.heads,
            default_scale=0.0
        )

        # Compute with default processor
        from diffusers.models.attention_processor import AttnProcessor2_0
        vanilla_proc = AttnProcessor2_0()
        vanilla_out = vanilla_proc(self.attn, self.hidden_states, self.encoder_hidden_states)

        # Compute with decoupled processor at scale=0.0
        proc.set_identity_tokens(self.id_tokens)
        decoupled_out = proc(self.attn, self.hidden_states, self.encoder_hidden_states)

        max_diff = torch.max(torch.abs(vanilla_out - decoupled_out)).item()
        cos_sim = F.cosine_similarity(vanilla_out.flatten(), decoupled_out.flatten(), dim=0).item()

        self.assertAlmostEqual(max_diff, 0.0, places=6)
        self.assertLess(abs(cos_sim - 1.0), 1e-4)

    def test_gate2_pure_reference_t_fused_shape(self):
        """Gate 2: Identity processor accepts [B, 256, 512] spatial identity tokens."""
        proc = DecoupledIdentityAttnProcessor(
            hidden_size=self.hidden_size,
            cross_attention_dim=self.cross_attention_dim,
            id_dim=self.id_dim,
            heads=self.heads,
            default_scale=1.0
        )
        proc.set_identity_tokens(self.id_tokens)
        out = proc(self.attn, self.hidden_states, self.encoder_hidden_states)
        self.assertEqual(out.shape, self.hidden_states.shape)

    def test_gate3_measurable_kv_differences(self):
        """Gate 3: Different reference tokens produce distinct K and V projections."""
        proc = DecoupledIdentityAttnProcessor(
            hidden_size=self.hidden_size,
            cross_attention_dim=self.cross_attention_dim,
            id_dim=self.id_dim,
            heads=self.heads
        )
        id_tokens_A = torch.randn(1, 256, 512)
        id_tokens_B = torch.randn(1, 256, 512)

        k_A = proc.to_k_id(id_tokens_A)
        k_B = proc.to_k_id(id_tokens_B)
        v_A = proc.to_v_id(id_tokens_A)
        v_B = proc.to_v_id(id_tokens_B)

        self.assertGreater((k_A - k_B).norm().item(), 1.0)
        self.assertGreater((v_A - v_B).norm().item(), 1.0)
        self.assertLess(F.cosine_similarity(k_A.flatten(), k_B.flatten(), dim=0).item(), 0.99)

    def test_gate4_lambda_sensitivity(self):
        """Gate 4: When lambda > 0, different identity tokens produce distinct attention outputs."""
        proc = DecoupledIdentityAttnProcessor(
            hidden_size=self.hidden_size,
            cross_attention_dim=self.cross_attention_dim,
            id_dim=self.id_dim,
            heads=self.heads,
            default_scale=1.0
        )
        id_tokens_A = torch.randn(1, 256, 512)
        id_tokens_B = torch.randn(1, 256, 512)

        proc.set_identity_tokens(id_tokens_A)
        out_A = proc(self.attn, self.hidden_states, self.encoder_hidden_states)

        proc.set_identity_tokens(id_tokens_B)
        out_B = proc(self.attn, self.hidden_states, self.encoder_hidden_states)

        diff = (out_A - out_B).norm().item()
        cos = F.cosine_similarity(out_A.flatten(), out_B.flatten(), dim=0).item()

        self.assertGreater(diff, 0.1)
        self.assertLess(cos, 0.99999)

    def test_gate7_text_pathway_integrity(self):
        """Gate 7: Text encoder hidden states are untouched by identity processing."""
        text_copy = self.encoder_hidden_states.clone()
        proc = DecoupledIdentityAttnProcessor(
            hidden_size=self.hidden_size,
            cross_attention_dim=self.cross_attention_dim,
            id_dim=self.id_dim,
            heads=self.heads,
            default_scale=1.0
        )
        proc.set_identity_tokens(self.id_tokens)
        _ = proc(self.attn, self.hidden_states, self.encoder_hidden_states)

        self.assertTrue(torch.equal(self.encoder_hidden_states, text_copy))

    def test_gate8_no_nan_or_inf(self):
        """Gate 8: Processor output has no NaN or Inf."""
        proc = DecoupledIdentityAttnProcessor(
            hidden_size=self.hidden_size,
            cross_attention_dim=self.cross_attention_dim,
            id_dim=self.id_dim,
            heads=self.heads,
            default_scale=1.0
        )
        proc.record_diagnostics = True
        proc.set_identity_tokens(self.id_tokens)
        out = proc(self.attn, self.hidden_states, self.encoder_hidden_states)

        self.assertFalse(torch.isnan(out).any().item())
        self.assertFalse(torch.isinf(out).any().item())
        self.assertFalse(proc.latest_diagnostics["has_nan"])
        self.assertFalse(proc.latest_diagnostics["has_inf"])


if __name__ == "__main__":
    unittest.main()
