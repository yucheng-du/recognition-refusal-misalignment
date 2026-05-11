"""Aggregate per-model ablation_nullpc_results JSONs into 11-model main-grid summary.

Reads `experiments/ablation_nullpc_results_model-{model}_ds-{dataset}.json` for the
11-model main grid × {math800, code800} = 22 cells, computes:

  - Total configs and per-cell (null_md, pc_md, full_md) values
  - md_null_gt_pc count   → §4.1 main "Null > PC" ordering claim (X/22)
  - md_null_gt_full count → secondary ordering claim
  - Mean and gap distribution

Writes `experiments/ablation_nullpc_results_11model.json` WITHOUT overwriting the
legacy `ablation_nullpc_results.json` (which has the original 8-model 16-cell aggregate
that the prior paper draft cites as 14/16).

Usage:
  python3 scripts/aggregate_ablation_v2.py
"""
import json
import os
import sys

# 11-model main grid (per drafts/qwen3_32b_postrun_assessment.md and
# drafts/qwen2_5_vs_qwen3_32b_comparison.md scope block).
MAIN_GRID_11 = [
    "smollm2", "phi4mini", "gemma3_4b", "mistral",
    "qwen3_8b", "llama", "qwen3_14b", "olmo13b",
    "mistral_small", "qwen3_32b", "llama70b",
]
DATASETS = ["math800", "code800"]

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    os.chdir(REPO)
    per_config = []
    missing = []

    for model in MAIN_GRID_11:
        for ds in DATASETS:
            path = f"experiments/ablation_nullpc_results_model-{model}_ds-{ds}.json"
            if not os.path.exists(path):
                missing.append(path)
                continue
            with open(path) as f:
                d = json.load(f)
            cfg = d["per_config"][0]
            per_config.append({
                "model":     cfg["model"],
                "dataset":   cfg["dataset"],
                "layer":     cfg["layer"],
                "pooling":   cfg["pooling"],
                "D":         cfg["D"],
                "pc_dim":    cfg["pc_dim"],
                "null_dim":  cfg["null_dim"],
                "null_md":   cfg["null_md"],
                "pc_md":     cfg["pc_md"],
                "full_md":   cfg["full_md"],
                "gap_null_pc":   cfg["null_md"] - cfg["pc_md"],
                "gap_null_full": cfg["null_md"] - cfg["full_md"],
            })

    if missing:
        print("ERROR: missing per-model JSONs:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    n = len(per_config)
    md_null_gt_pc   = sum(1 for c in per_config if c["null_md"] > c["pc_md"])
    md_null_gt_full = sum(1 for c in per_config if c["null_md"] > c["full_md"])

    means = {
        "null": sum(c["null_md"] for c in per_config) / n,
        "pc":   sum(c["pc_md"]   for c in per_config) / n,
        "full": sum(c["full_md"] for c in per_config) / n,
    }
    gap_pc_min   = min(c["gap_null_pc"]   for c in per_config)
    gap_pc_max   = max(c["gap_null_pc"]   for c in per_config)
    gap_full_min = min(c["gap_null_full"] for c in per_config)
    gap_full_max = max(c["gap_null_full"] for c in per_config)

    exceptions_pc = [
        f"{c['model']}/{c['dataset']} L{c['layer']} (gap={c['gap_null_pc']:+.4f})"
        for c in per_config if c["null_md"] <= c["pc_md"]
    ]
    exceptions_full = [
        f"{c['model']}/{c['dataset']} L{c['layer']} (gap={c['gap_null_full']:+.4f})"
        for c in per_config if c["null_md"] <= c["full_md"]
    ]

    out = {
        "scope": "11-model main grid (smollm2, phi4mini, gemma3_4b, mistral, qwen3_8b, "
                 "llama, qwen3_14b, olmo13b, mistral_small, qwen3_32b, llama70b) × "
                 "{math800, code800} = 22 configs",
        "per_config": per_config,
        "averages": {"meandiff": means},
        "counts": {
            "n_configs":       n,
            "md_null_gt_pc":   md_null_gt_pc,
            "md_null_gt_full": md_null_gt_full,
        },
        "gap_distribution": {
            "null_minus_pc":   {"min": gap_pc_min,   "max": gap_pc_max},
            "null_minus_full": {"min": gap_full_min, "max": gap_full_max},
        },
        "exceptions": {
            "null_le_pc":   exceptions_pc,
            "null_le_full": exceptions_full,
        },
        "note": "Layer per cell follows the fixed analysis layers used by the "
                "existing paper pipeline (the hardcoded best_layers in scripts/, "
                "matching existing direction_comparison files). LE peak differences "
                "are tracked separately in drafts/ but not re-aggregated here.",
    }

    out_path = "experiments/ablation_nullpc_results_11model.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # ── Print human-readable summary ─────────────────────────────────────
    print(f"=== Aggregate ablation results: 11-model main grid ===")
    print(f"  n_configs:       {n}/22 expected")
    print(f"  md_null_gt_pc:   {md_null_gt_pc}/{n}  (paper §4.1 main ordering)")
    print(f"  md_null_gt_full: {md_null_gt_full}/{n}  (secondary ordering)")
    print(f"")
    print(f"  Means: null={means['null']:.4f}, pc={means['pc']:.4f}, full={means['full']:.4f}")
    print(f"  null-pc gap range:   [{gap_pc_min:+.4f}, {gap_pc_max:+.4f}]")
    print(f"  null-full gap range: [{gap_full_min:+.4f}, {gap_full_max:+.4f}]")
    print(f"")
    if exceptions_pc:
        print(f"  Exceptions (null ≤ pc):")
        for e in exceptions_pc:
            print(f"    {e}")
    if exceptions_full:
        print(f"  Exceptions (null ≤ full):")
        for e in exceptions_full:
            print(f"    {e}")
    print(f"")
    print(f"Saved to: {out_path}")
    print(f"  (legacy aggregate at experiments/ablation_nullpc_results.json preserved)")


if __name__ == "__main__":
    main()
