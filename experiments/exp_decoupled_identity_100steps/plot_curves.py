import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

exp_dir = Path("experiments/exp_decoupled_identity_100steps")

with open(exp_dir / "training_metrics.json") as f:
    t_metrics = json.load(f)
with open(exp_dir / "diagnostic_evaluations.json") as f:
    diags = json.load(f)

steps_train = [m["step"] for m in t_metrics]
losses = [m["loss"] for m in t_metrics]
k_grads = [m["mean_k_grad_norm"] for m in t_metrics]
v_grads = [m["mean_v_grad_norm"] for m in t_metrics]
param_deltas = [m["max_param_change"] for m in t_metrics]

diag_steps = [d["step"] for d in diags]
diff_id_l2 = [d["reference_sensitivity"]["ref_A_vs_diff_id"]["l2_diff"] for d in diags]
same_id_b = [d["reference_sensitivity"]["ref_A_vs_ref_B"]["l2_diff"] for d in diags]
same_id_c = [d["reference_sensitivity"]["ref_A_vs_ref_C"]["l2_diff"] for d in diags]
id_vanilla_l2 = [d["unet_epsilon_response"]["identity_vs_vanilla"]["l2_diff"] for d in diags]
mask_ratios = [d["masked_vs_unmasked"]["masked_ratio"] for d in diags]
entropies = [d["identity_attention_stats"]["mean_entropy"] for d in diags]

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, axs = plt.subplots(2, 3, figsize=(18, 10))

# 1. Loss Curve
axs[0, 0].plot(steps_train, losses, color="#1f77b4", alpha=0.6, label="Batch Loss")
# Exponential moving average
ema_loss = [losses[0]]
for l in losses[1:]:
    ema_loss.append(0.85 * ema_loss[-1] + 0.15 * l)
axs[0, 0].plot(steps_train, ema_loss, color="#d62728", lw=2, label="EMA Loss")
axs[0, 0].set_title("A. Diffusion Noise MSE Loss (0 -> 100)", fontsize=12, fontweight="bold")
axs[0, 0].set_xlabel("Optimization Step")
axs[0, 0].set_ylabel("MSE Loss")
axs[0, 0].set_yscale("log")
axs[0, 0].legend()

# 2. Gradient Curves
axs[0, 1].plot(steps_train, v_grads, color="#ff7f0e", lw=1.5, label="V_id Grad Norm")
axs[0, 1].plot(steps_train, k_grads, color="#2ca02c", lw=1.5, label="K_id Grad Norm")
axs[0, 1].set_title("B. Trainable Gradient Norms (K_id vs V_id)", fontsize=12, fontweight="bold")
axs[0, 1].set_xlabel("Optimization Step")
axs[0, 1].set_ylabel("Grad Norm (L2)")
axs[0, 1].set_yscale("log")
axs[0, 1].legend()

# 3. Param Delta
axs[0, 2].plot(steps_train, param_deltas, color="#9467bd", lw=2)
axs[0, 2].set_title("C. Max Trainable Weight Delta (ΔW)", fontsize=12, fontweight="bold")
axs[0, 2].set_xlabel("Optimization Step")
axs[0, 2].set_ylabel("Max Absolute Param Change")

# 4. Reference Sensitivity (Ref A vs Same B/C vs Diff ID)
axs[1, 0].plot(diag_steps, diff_id_l2, marker="o", lw=2, color="#d62728", label="Ref A vs Diff ID (3616)")
axs[1, 0].plot(diag_steps, same_id_b, marker="s", lw=1.5, color="#1f77b4", label="Ref A vs Ref B (Same ID)")
axs[1, 0].plot(diag_steps, same_id_c, marker="^", lw=1.5, color="#2ca02c", label="Ref A vs Ref C (Same ID)")
axs[1, 0].set_title("D. Reference Sensitivity at Checkpoints", fontsize=12, fontweight="bold")
axs[1, 0].set_xlabel("Step")
axs[1, 0].set_ylabel("Epsilon L2 Difference")
axs[1, 0].legend()

# 5. Identity-vs-Vanilla Response vs Mask Ratio
ax5 = axs[1, 1]
ax5_r = ax5.twinx()
p1 = ax5.plot(diag_steps, id_vanilla_l2, marker="o", color="#e377c2", lw=2, label="Identity vs Vanilla L2")
p2 = ax5_r.plot(diag_steps, mask_ratios, marker="D", color="#17becf", lw=2, linestyle="--", label="Masked/Unmasked Ratio")
ax5.set_title("E. Identity Perturbation & Mask Focus", fontsize=12, fontweight="bold")
ax5.set_xlabel("Step")
ax5.set_ylabel("Identity vs Vanilla L2", color="#e377c2")
ax5_r.set_ylabel("Masked / Unmasked Ratio", color="#17becf")
lines = p1 + p2
labels = [l.get_label() for l in lines]
ax5.legend(lines, labels, loc="center right")

# 6. Attention Entropy (Uniformity Check)
axs[1, 2].plot(diag_steps, entropies, marker="o", color="#8c564b", lw=2, label="Mean Identity Attention Entropy")
axs[1, 2].axhline(y=np.log(256), color="black", linestyle=":", label=f"Theoretical Uniform ln(256)={np.log(256):.3f}")
axs[1, 2].set_title("F. Cross-Attention Entropy (Uniformity Audit)", fontsize=12, fontweight="bold")
axs[1, 2].set_xlabel("Step")
axs[1, 2].set_ylabel("Entropy (Nats)")
axs[1, 2].set_ylim(5.45, 5.56)
axs[1, 2].legend()

plt.tight_layout()
out_path = exp_dir / "training_and_diagnostics_curves.png"
plt.savefig(out_path, dpi=200)
print(f"Saved diagnostic curves to {out_path}")
