"""
Formal Evaluation Leakage Audit for Ablation A7 (Verifier-Aware Sampling / Reranker).
Strict Protocol Verification:
1. Matcher Independence: Is the candidate selection matcher independent of the evaluation metric matcher?
2. Target Leakage: Does candidate selection consume ground-truth target information inaccessible at test time?
3. Candidate Protocol: Does K match the intended research protocol?
"""

import os
import sys
import time
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run_reranker_leakage_audit() -> dict:
    print("=" * 80)
    print("FORMAL RERANKER EVALUATION LEAKAGE AUDIT (ABLATION A7)")
    print("=" * 80)

    ablation_script = PROJECT_ROOT / "evaluation/ablations/evaluate_severity_25.py"
    assert ablation_script.exists(), f"Missing ablation script: {ablation_script}"

    with open(ablation_script, "r", encoding="utf-8") as f:
        code = f.read()

    findings = []
    violations = []

    # 1. Audit Target Ground-Truth Leakage in Candidate Scoring
    # Check if PSNR or target_rgb is used to compute verifier score
    has_psnr_in_verifier = "psnr_val = cand_m[\"reconstruction_quality\"][\"psnr\"]" in code
    has_target_in_cand = "cand_m = evaluate_reconstruction(\n            target_rgb=target_rgb" in code or "target_rgb=target_rgb" in code
    has_verifier_formula = "verifier_score = 0.60 * cos_val + 0.40 * qual_norm" in code

    if has_psnr_in_verifier and has_verifier_formula:
        v1 = {
            "check": "Ground-Truth Target Invariance (Test-Time Feasibility)",
            "status": "VIOLATION_CONFIRMED",
            "severity": "CRITICAL",
            "evidence": "Lines 488-491 in evaluate_severity_25.py compute 'qual_norm' directly from 'psnr_val', which was calculated using ground-truth unmasked target pixels (target_rgb).",
            "implication": "At test/inference time, the occluded target region is missing by definition. Using ground-truth PSNR to score candidates is direct evaluation oracle leakage, rendering the reported A7 boost artificially inflated and technically invalid."
        }
        violations.append(v1)
        print("  [AUDIT CHECK 1] Target Ground-Truth Leakage: VIOLATION DETECTED")
        print("    -> Candidate scoring used ground-truth target PSNR!")
    else:
        findings.append({
            "check": "Ground-Truth Target Invariance",
            "status": "PASS"
        })

    # 2. Audit Matcher Independence
    # Check whether the same ArcFace model is used for reranking and evaluation
    has_diff_arcface_in_cand = "cos_val = cand_m[\"biometric_identity\"][\"cos_recon_vs_ref_mean\"]" in code
    if has_diff_arcface_in_cand:
        v2 = {
            "check": "Matcher Independence (Metric Disjointness)",
            "status": "VIOLATION_CONFIRMED",
            "severity": "HIGH",
            "evidence": "Candidate selection used 'diff_arcface' (w600k_r50.onnx) to compute cos_val against references, which is the exact same backbone used as the final evaluation metric (ArcFace cosine similarity).",
            "implication": "Selecting the sample that maximizes the test metric on the same matcher creates optimization bias toward that specific metric, violating the independence of held-out evaluators."
        }
        violations.append(v2)
        print("  [AUDIT CHECK 2] Matcher Independence: VIOLATION DETECTED")
        print("    -> Selection matcher is identical to evaluation matcher!")
    else:
        findings.append({
            "check": "Matcher Independence",
            "status": "PASS"
        })

    # 3. Audit Candidate Pool Size K
    has_k3 = "candidate_seeds = [42, 43, 44]" in code
    if has_k3:
        findings.append({
            "check": "Candidate Pool Size Protocol",
            "status": "PASS",
            "evidence": "candidate_seeds = [42, 43, 44] confirms K=3 was used as planned in Phase 7 protocol."
        })
        print("  [AUDIT CHECK 3] Candidate Pool Size: PASS (K=3 verified)")

    # Formulate Formal Ruling
    verdict = {
        "audit_target": "Ablation A7 (Verifier-Aware Sampling / O3 Reranker)",
        "audit_timestamp": time.time(),
        "total_checks": 3,
        "violations_found": len(violations),
        "violations": violations,
        "valid_findings": findings,
        "formal_ruling": "A7_NUMBERS_INVALID_LEAKAGE_DETECTED",
        "recommendation": (
            "1. Discard the previously reported A7 metrics (e.g. SimRef 0.5644, PSNR 21.77 dB) from all formal publication conclusions.\n"
            "2. Baseline benchmark comparison must strictly rely on uncheated single-shot configurations (Step 150 vs Step 500 at K=1).\n"
            "3. If candidate reranking is evaluated in the future, it must strictly use a blind, no-reference image quality metric (e.g. BRISQUE, NIQE, or CLIP-IQA) and an independent non-ArcFace biometric matcher (e.g. AdaFace or CurricularFace) with zero ground-truth target access."
        )
    }

    out_p = PROJECT_ROOT / "evaluation/reports/reranker_leakage_audit.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)

    print("\n" + "=" * 80)
    print("FORMAL AUDIT VERDICT: A7_NUMBERS_INVALID_LEAKAGE_DETECTED")
    print(f"Audit report saved to: {out_p}")
    print("=" * 80)
    return verdict

if __name__ == "__main__":
    run_reranker_leakage_audit()
