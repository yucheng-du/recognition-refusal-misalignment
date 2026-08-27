"""Aggregate full 11-model main-grid facts table for paper update.

Unified one-shot script: reads per-model JSONs from experiments/ and produces
a single comprehensive facts JSON containing the headline numbers cited in
the accompanying paper.

Sections aggregated (each section corresponds to a paper claim):

  §4.1   Detection / Null > PC ordering (already in ablation_nullpc_results_11model.json;
         cross-referenced here)
  §4.2   Orthogonality cos(d_imp, d_ref): 22 cells
  §4.3   Pretraining origin / base–instruct paired comparison: 6 pairs
         (5 with behavior-verified base d_ref; 1 Llama-3.1-70B proxy base)
  §5     Steering breadth: 33 cells (11 models × 3 datasets)
  §5.1   Cross-domain transfer: 22 directional drops
  §5.x   Form / category conditionality: 22 cells (characterization subsection;
         exact §5 numbering follows current paper outline)

Output: experiments/main_grid_facts_v2.json
        + human-readable summary printed to stdout

Conventions:
  - "main grid" = updated 11-model expanded set
  - Layer per cell follows the fixed analysis layer used by the existing
    paper pipeline (matches direction_comparison files; layer info is
    embedded in each per-model JSON).

Usage:
  python3 scripts/aggregate_main_grid_v2.py
"""
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAIN_GRID_11 = [
    "smollm2", "phi4mini", "gemma3_4b", "mistral",
    "qwen3_8b", "llama", "qwen3_14b", "olmo13b",
    "mistral_small", "qwen3_32b", "llama70b",
]
DATASETS = ["math800", "code800"]
STEER_DATASETS = ["math800", "code800", "fact800"]

# §4.3 base/instruct paired comparisons (matched-layer).
# 5 pairs are FULLY verified (both instruct AND base have behavior_verified=True);
# the Llama-3.3-vs-3.1-70B pair is "proxy base" — Meta confirmed shared pretraining
# checkpoint, but Llama-3.1-70B base does not refuse harmful prompts often enough
# to fit a behavior-verified d_ref, so its d_ref is computed via proxy. This pair
# remains valuable as the only vendor-confirmed post-training-only ablation in the grid.
BASE_INSTRUCT_PAIRS = [
    ("qwen",      "qwen_base"),       # Qwen2.5-7B    — verified
    ("qwen14b",   "qwen14b_base"),    # Qwen2.5-14B   — verified
    ("qwen32b",   "qwen32b_base"),    # Qwen2.5-32B   — verified
    ("qwen3_8b",  "qwen3_8b_base"),   # Qwen3-8B      — verified
    ("qwen3_14b", "qwen3_14b_base"),  # Qwen3-14B     — verified
    ("llama70b",  "llama70b_base"),   # Llama-3.3 vs Llama-3.1-70B — proxy base
]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── §4.2 Orthogonality ────────────────────────────────────────────────────
def aggregate_orthogonality():
    """Read direction_comparison files for 11 main × 2 datasets = 22 cells."""
    cells, missing = [], []
    for model in MAIN_GRID_11:
        for ds in DATASETS:
            if ds == "math800":
                path = f"experiments/direction_comparison_{model}.json"
            else:
                # code800: file has _code800_L{N}.json suffix
                pattern = f"experiments/direction_comparison_{model}_code800_L*.json"
                matches = sorted(glob.glob(pattern))
                if not matches:
                    missing.append(f"{model}/{ds}")
                    continue
                path = matches[0]  # Take first match (should be only one canonical)
            d = load_json(path)
            if d is None:
                missing.append(f"{model}/{ds}")
                continue
            cells.append({
                "model": model,
                "dataset": ds,
                "layer": d["layer"],
                "cos_matched_full": d["cos_matched_full"],
                "ci95_lo": d.get("bootstrap_cos_ci95_lo"),
                "ci95_hi": d.get("bootstrap_cos_ci95_hi"),
                "over_random": d.get("cos_matched_over_random"),
                "behavior_verified": d.get("behavior_verified"),
                "n_harmful_verified": d.get("n_harmful_verified_raw"),
                "heldout_refusal_auc": d.get("heldout_refusal_auc"),
                "auc_imp": d.get("auc_impossibility_on_dataset"),
            })

    n = len(cells)
    cos_values = [c["cos_matched_full"] for c in cells]
    n_verified = sum(1 for c in cells if c["behavior_verified"])
    return {
        "n_cells": n,
        "missing_cells": missing,
        "n_behavior_verified": n_verified,
        "cos_min": min(cos_values) if cos_values else None,
        "cos_max": max(cos_values) if cos_values else None,
        "cos_mean": sum(cos_values) / n if n else None,
        "in_band_005_017_count": sum(1 for v in cos_values if 0.05 <= v <= 0.17),
        "in_band_004_014_count": sum(1 for v in cos_values if 0.04 <= v <= 0.14),
        "cells": cells,
    }


# ── §5.x Form / category conditionality (characterization subsection) ────
def aggregate_form():
    cells, missing = [], []
    for model in MAIN_GRID_11:
        for ds in DATASETS:
            d = load_json(f"experiments/form_conditionality_results_model-{model}_ds-{ds}.json")
            if d is None:
                missing.append(f"{model}/{ds}")
                continue
            key = "part1_per_config" if "part1_per_config" in d else "part1_per_category"
            c = d[key][0]
            aucs = [x["auc"] for x in c.get("per_category", [])]
            cells.append({
                "model": model,
                "dataset": ds,
                "layer": c["layer"],
                "global_cosnsrt_auc": c["global_cosnsrt_auc"],
                "n_categories": len(aucs),
                "auc_min": min(aucs) if aucs else None,
                "auc_max": max(aucs) if aucs else None,
                "ns_snr_rho": c["correlations"]["ns_snr"]["spearman_rho"],
                "ns_snr_p":   c["correlations"]["ns_snr"]["spearman_p"],
            })

    n = len(cells)
    aucs = [c["global_cosnsrt_auc"] for c in cells]
    rhos = [c["ns_snr_rho"] for c in cells]
    sig_count = sum(1 for c in cells if c["ns_snr_p"] < 0.05)
    return {
        "n_cells": n,
        "missing_cells": missing,
        "global_auc_mean": sum(aucs) / n if n else None,
        "global_auc_min": min(aucs) if aucs else None,
        "global_auc_max": max(aucs) if aucs else None,
        "rho_mean": sum(rhos) / n if n else None,
        "rho_significant_count": sig_count,
        "cells": cells,
    }


# ── §5.1 Cross-domain transfer ────────────────────────────────────────────
def aggregate_cross_domain():
    drops, missing = [], []
    for model in MAIN_GRID_11:
        d = load_json(f"experiments/cross_domain_transfer_model-{model}.json")
        if d is None:
            missing.append(model)
            continue
        for r in d:
            drops.append({
                "model": model,
                "train": r["train"],
                "test": r["test"],
                "within_mean": r["within_mean"],
                "cross_mean":  r["cross_mean"],
                "drop":        r["drop"],
            })

    n = len(drops)
    drop_values = [d["drop"] for d in drops]
    return {
        "n_directional_drops": n,
        "missing_models": missing,
        "drop_mean": sum(drop_values) / n if n else None,
        "drop_min": min(drop_values) if drops else None,
        "drop_max": max(drop_values) if drops else None,
        "smallest_drop_cell": min(drops, key=lambda x: x["drop"]) if drops else None,
        "largest_drop_cell":  max(drops, key=lambda x: x["drop"]) if drops else None,
        "drops": drops,
    }


# ── §5 Steering breadth ───────────────────────────────────────────────────
def aggregate_steering():
    cells, missing = [], []
    for model in MAIN_GRID_11:
        for ds in STEER_DATASETS:
            pattern = f"experiments/steering/steering_{model}_{ds}_L*.json"
            matches = sorted(glob.glob(pattern))
            if not matches:
                missing.append(f"{model}/{ds}")
                continue
            d = load_json(matches[0])
            cells.append({
                "model": model,
                "dataset": ds,
                "layer": d["layer"],
                "best_alpha": d.get("best_alpha"),
                "halluc_reduction_abs": d.get("hallucination_reduction"),
                "halluc_reduction_pct": d.get("hallucination_reduction_pct"),
                "non_refusal_cost": d.get("non_refusal_cost"),
            })

    n = len(cells)
    by_ds = {ds: [c for c in cells if c["dataset"] == ds] for ds in STEER_DATASETS}
    summary_by_ds = {}
    for ds, lst in by_ds.items():
        n_pos = sum(1 for c in lst if c["halluc_reduction_abs"] and c["halluc_reduction_abs"] > 0)
        # Skip cells with best_α=0 (no useful net steering) when computing means
        active = [c for c in lst if c["best_alpha"] and c["best_alpha"] > 0]
        if active:
            mean_red = sum(c["halluc_reduction_abs"] or 0 for c in active) / len(active)
            mean_cost = sum(c["non_refusal_cost"] or 0 for c in active) / len(active)
        else:
            mean_red = mean_cost = None
        summary_by_ds[ds] = {
            "n_cells": len(lst),
            "n_positive_reduction": n_pos,           # cells with halluc_reduction > 0
            "n_best_alpha_zero": sum(1 for c in lst if c["best_alpha"] == 0),  # cells where grid search picked alpha=0 (no useful net steering)
            "mean_halluc_reduction_active": mean_red,    # mean over best_alpha > 0 only
            "mean_preservation_cost_active": mean_cost,  # mean over best_alpha > 0 only
        }
    return {
        "n_cells_total": n,
        "missing_cells": missing,
        "by_dataset": summary_by_ds,
        "cells": cells,
    }


# ── §4.3 Base/Instruct paired comparisons ─────────────────────────────────
def aggregate_base_pairs():
    pairs = []
    for instruct, base in BASE_INSTRUCT_PAIRS:
        # Read instruct cos (math, default-named file)
        inst_d = load_json(f"experiments/direction_comparison_{instruct}.json")
        if inst_d is None:
            pairs.append({"instruct": instruct, "base": base, "status": "INSTRUCT_MISSING"})
            continue

        # Read base cos at matched layer (look for matched layer file)
        # base files are direction_comparison_{base}_math800_L{N}.json (note _math800_)
        base_pattern = f"experiments/direction_comparison_{base}_math800_L*.json"
        base_matches = sorted(glob.glob(base_pattern))
        if not base_matches:
            # fallback: try without _math800 suffix
            base_alt = load_json(f"experiments/direction_comparison_{base}.json")
            if base_alt is None:
                pairs.append({"instruct": instruct, "base": base, "status": "BASE_MISSING"})
                continue
            base_d = base_alt
        else:
            # Pick the one at instruct's layer if available, else the middle
            inst_layer = inst_d.get("layer")
            best_match = None
            for m in base_matches:
                bd = load_json(m)
                if bd and bd.get("layer") == inst_layer:
                    best_match = bd
                    break
            base_d = best_match if best_match else load_json(base_matches[len(base_matches) // 2])

        pairs.append({
            "instruct": instruct,
            "base": base,
            "instruct_layer": inst_d.get("layer"),
            "base_layer": base_d.get("layer"),
            "instruct_cos": inst_d.get("cos_matched_full"),
            "base_cos": base_d.get("cos_matched_full"),
            "delta_cos": (inst_d.get("cos_matched_full") or 0) - (base_d.get("cos_matched_full") or 0),
            "instruct_behavior_verified": inst_d.get("behavior_verified"),
            "base_behavior_verified": base_d.get("behavior_verified"),
            "instruct_n_harmful_verified": inst_d.get("n_harmful_verified_raw"),
            "base_n_harmful_verified": base_d.get("n_harmful_verified_raw"),
            "status": "OK",
        })

    n = len([p for p in pairs if p.get("status") == "OK"])
    n_fully_verified = sum(
        1 for p in pairs
        if p.get("status") == "OK"
        and p.get("instruct_behavior_verified") and p.get("base_behavior_verified")
    )
    n_proxy_base = sum(
        1 for p in pairs
        if p.get("status") == "OK"
        and p.get("instruct_behavior_verified") and not p.get("base_behavior_verified")
    )
    deltas = [p["delta_cos"] for p in pairs if p.get("status") == "OK" and p.get("delta_cos") is not None]
    return {
        "n_pairs": n,
        "n_fully_verified": n_fully_verified,
        "n_proxy_base":     n_proxy_base,
        "delta_cos_mean": sum(deltas) / len(deltas) if deltas else None,
        "delta_cos_min":  min(deltas) if deltas else None,
        "delta_cos_max":  max(deltas) if deltas else None,
        "pairs": pairs,
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    os.chdir(REPO)

    facts = {
        "scope": "11-model main grid (updated 2026-05); see drafts/qwen3_32b_postrun_assessment.md",
        "main_grid_11": MAIN_GRID_11,
        "appendix_5": ["gemma2", "phi3", "qwen", "qwen14b", "qwen32b"],
        "ablation_4_1_ref": "experiments/ablation_nullpc_results_11model.json",
        "orthogonality":  aggregate_orthogonality(),
        "form":           aggregate_form(),
        "cross_domain":   aggregate_cross_domain(),
        "steering":       aggregate_steering(),
        "base_pairs":     aggregate_base_pairs(),
    }

    out_path = "experiments/main_grid_facts_v2.json"
    with open(out_path, "w") as f:
        json.dump(facts, f, indent=2)

    # ── Print human summary ─────────────────────────────────────────────
    o = facts["orthogonality"]
    print(f"\n=== §4.2 Orthogonality (cos d_imp ⊥ d_ref) ===")
    print(f"  cells: {o['n_cells']}/22, behavior_verified: {o['n_behavior_verified']}/{o['n_cells']}")
    print(f"  cos range: [{o['cos_min']:.4f}, {o['cos_max']:.4f}], mean={o['cos_mean']:.4f}")
    print(f"  in [0.04, 0.14]: {o['in_band_004_014_count']}/{o['n_cells']}")
    print(f"  in [0.05, 0.17]: {o['in_band_005_017_count']}/{o['n_cells']} (prior paper band)")

    f_ = facts["form"]
    print(f"\n=== §5.x Form / Category Conditionality (characterization subsection) ===")
    print(f"  cells: {f_['n_cells']}/22")
    print(f"  global AUC range: [{f_['global_auc_min']:.4f}, {f_['global_auc_max']:.4f}], mean={f_['global_auc_mean']:.4f}")
    print(f"  NS_SNR↔AUC ρ mean: {f_['rho_mean']:.3f}, significant (p<0.05): {f_['rho_significant_count']}/{f_['n_cells']}")

    cd = facts["cross_domain"]
    print(f"\n=== §5.1 Cross-Domain Transfer ===")
    print(f"  directional drops: {cd['n_directional_drops']}/22")
    print(f"  drop range: [{cd['drop_min']:.4f}, {cd['drop_max']:.4f}], mean={cd['drop_mean']:.4f}")
    sd = cd['smallest_drop_cell']; ld = cd['largest_drop_cell']
    print(f"  smallest: {sd['model']} {sd['train']}→{sd['test']} drop={sd['drop']:.4f}")
    print(f"  largest:  {ld['model']} {ld['train']}→{ld['test']} drop={ld['drop']:.4f}")

    s = facts["steering"]
    print(f"\n=== §5 Steering Breadth (11 × 3 = 33 cells) ===")
    print(f"  total cells: {s['n_cells_total']}/33")
    for ds, sd in s['by_dataset'].items():
        mean_red = sd['mean_halluc_reduction_active']
        mean_cost = sd['mean_preservation_cost_active']
        red_str = f"{mean_red:.3f}" if mean_red is not None else "n/a"
        cost_str = f"{mean_cost:.3f}" if mean_cost is not None else "n/a"
        print(f"  {ds}: {sd['n_positive_reduction']}/{sd['n_cells']} positive Δhalluc, "
              f"{sd['n_best_alpha_zero']} best_α=0 (no useful net steering), "
              f"active-mean Δhalluc={red_str}, cost={cost_str}")

    bp = facts["base_pairs"]
    print(f"\n=== §4.3 Base/Instruct Paired Comparisons ===")
    print(f"  pairs: {bp['n_pairs']}/{len(BASE_INSTRUCT_PAIRS)}  "
          f"(fully behavior-verified: {bp['n_fully_verified']}; proxy base: {bp['n_proxy_base']})")
    if bp['delta_cos_mean'] is not None:
        print(f"  Δcos (instruct − base): range [{bp['delta_cos_min']:+.4f}, {bp['delta_cos_max']:+.4f}], mean={bp['delta_cos_mean']:+.4f}")
    for p in bp['pairs']:
        if p.get('status') == 'OK':
            i_ok = p['instruct_behavior_verified']; b_ok = p['base_behavior_verified']
            tag = "fully verified" if (i_ok and b_ok) else ("proxy base" if i_ok and not b_ok else "PARTIAL")
            print(f"  {p['instruct']:11s} L{p['instruct_layer']}: instruct cos={p['instruct_cos']:.4f}, "
                  f"base cos={p['base_cos']:.4f}, Δ={p['delta_cos']:+.4f}  [{tag}]")
        else:
            print(f"  {p['instruct']}: {p['status']}")

    print(f"\nFull JSON: {out_path}")

    # ── Hard-fail check: facts file is the input to factsheet edits, so any
    # missing cells or unexpected verified/proxy split must block downstream
    # paper work, not just print to stderr.
    expectations = [
        ("orthogonality.n_cells",            facts["orthogonality"]["n_cells"],       22),
        ("form.n_cells",                     facts["form"]["n_cells"],                22),
        ("cross_domain.n_directional_drops", facts["cross_domain"]["n_directional_drops"], 22),
        ("steering.n_cells_total",           facts["steering"]["n_cells_total"],      33),
        ("base_pairs.n_pairs",               facts["base_pairs"]["n_pairs"],          len(BASE_INSTRUCT_PAIRS)),
        # Expected base-pair verification split: 5 fully verified + 1 proxy (Llama70B).
        # If this changes (e.g. someone adds a new pair), the script blocks until
        # the constants and footnote framing are reviewed.
        ("base_pairs.n_fully_verified",      facts["base_pairs"]["n_fully_verified"], 5),
        ("base_pairs.n_proxy_base",          facts["base_pairs"]["n_proxy_base"],     1),
    ]
    errors = [(k, got, want) for k, got, want in expectations if got != want]
    # Surface per-section missing lists too (debugging convenience).
    for section in ("orthogonality", "form", "cross_domain", "steering"):
        miss = facts[section].get("missing_cells") or facts[section].get("missing_models") or []
        if miss:
            errors.append((f"{section}.missing", miss, "[]"))
    if errors:
        print(f"\nFAIL — facts file integrity check (block factsheet edits until resolved):", file=sys.stderr)
        for k, got, want in errors:
            print(f"  {k}: got={got}  want={want}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
