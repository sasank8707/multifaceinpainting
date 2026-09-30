import json
from pathlib import Path

exp_dir = Path("experiments/exp_decoupled_identity_100steps")

with open(exp_dir / "training_config.json") as f:
    cfg = json.load(f)
with open(exp_dir / "training_metrics.json") as f:
    t_metrics = json.load(f)
with open(exp_dir / "gradient_metrics.json") as f:
    g_metrics = json.load(f)
with open(exp_dir / "diagnostic_evaluations.json") as f:
    diags = json.load(f)
with open(exp_dir / "frozen_parameter_integrity.json") as f:
    frozen = json.load(f)

print("=== DIAGNOSTICS ACROSS CHECKPOINTS ===")
header = f"{'Step':<5} | {'RefA-B L2':<10} | {'RefA-C L2':<10} | {'Diff-ID L2':<10} | {'Id-Van L2':<10} | {'Mask Ratio':<10} | {'Entropy':<8} | {'Max Prob':<8} | {'Mean K':<7} | {'Mean V':<7} | {'Leak L2':<7}"
print(header)
print("-" * len(header))
for d in diags:
    s = d['step']
    ref_b = d['reference_sensitivity']['ref_A_vs_ref_B']['l2_diff']
    ref_c = d['reference_sensitivity']['ref_A_vs_ref_C']['l2_diff']
    diff_id = d['reference_sensitivity']['ref_A_vs_diff_id']['l2_diff']
    id_van = d['unet_epsilon_response']['identity_vs_vanilla']['l2_diff']
    m_ratio = d['masked_vs_unmasked']['masked_ratio']
    entropy = d['identity_attention_stats']['mean_entropy']
    max_p = d['identity_attention_stats']['mean_max_prob']
    k_norm = d['identity_kv_norms']['mean_key_norm']
    v_norm = d['identity_kv_norms']['mean_value_norm']
    leak = d['target_leakage']['l2_diff']
    print(f"{s:<5d} | {ref_b:<10.6f} | {ref_c:<10.6f} | {diff_id:<10.6f} | {id_van:<10.6f} | {m_ratio:<10.2f} | {entropy:<8.4f} | {max_p:<8.4f} | {k_norm:<7.2f} | {v_norm:<7.2f} | {leak:<7.1f}")

print("\n=== TRAINING LOSS & GRADIENTS AT LOGGED STEPS ===")
hdr_train = f"{'Step':<5} | {'Loss':<10} | {'K Grad Norm':<12} | {'V Grad Norm':<12} | {'Max Delta':<10} | {'Time (s)':<8} | {'RAM (MB)':<8}"
print(hdr_train)
print("-" * len(hdr_train))
for tm in t_metrics:
    s = tm['step']
    if s == 1 or s % 10 == 0:
        print(f"{s:<5d} | {tm['loss']:<10.6f} | {tm['mean_k_grad_norm']:<12.6f} | {tm['mean_v_grad_norm']:<12.6f} | {tm['max_param_change']:<10.6f} | {tm['step_time_seconds']:<8.1f} | {tm['memory_rss_mb']:<8.1f}")

print("\n=== LOSS CONVERGENCE SUMMARY ===")
losses = [m['loss'] for m in t_metrics]
print(f"Initial Loss (Step 1): {losses[0]:.6f}")
print(f"Final Loss (Step 100):   {losses[-1]:.6f}")
print(f"Min Loss:               {min(losses):.6f} (at step {losses.index(min(losses))+1})")
print(f"Max Loss:               {max(losses):.6f} (at step {losses.index(max(losses))+1})")
print(f"Mean Loss (1-100):      {sum(losses)/len(losses):.6f}")

print("\n=== FROZEN PARAMETER INTEGRITY ===")
for k, v in frozen.items():
    print(f"  {k}: {v}")

print("\n=== RUNTIME & COMPUTE SUMMARY ===")
step_times = [m['step_time_seconds'] for m in t_metrics]
total_sec = cfg['total_runtime_seconds']
print(f"Total Wall-Clock Time: {total_sec:.1f}s ({total_sec/60:.2f} min / {total_sec/3600:.2f} hours)")
print(f"Average Step Time:     {sum(step_times)/len(step_times):.2f}s")
print(f"Min / Max Step Time:   {min(step_times):.1f}s / {max(step_times):.1f}s")
print(f"Peak RAM:              {max(m['memory_rss_mb'] for m in t_metrics):.1f} MB")
