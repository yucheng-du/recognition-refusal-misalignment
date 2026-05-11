#!/usr/bin/env python3
"""
Extract Qwen2.5-32B summary numbers from all paper-load-bearing JSONs
and print a markdown table ready to paste into paper_outline_and_factsheet.md
Part B2.2 (Qwen2.5-32B Supplementary Numbers).

Usage:
    python3 analysis/extract_qwen32b_summary.py

Run from repo root.
"""

from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"


def load(p: Path) -> dict | list:
    with open(p) as f:
        return json.load(f)


def fmt_num(x, ndigits: int = 3) -> str:
    if x is None:
        return "—"
    try:
        return f"{x:.{ndigits}f}"
    except Exception:
        return str(x)


def section(title: str) -> None:
    print()
    print(f"## {title}")
    print()


# ---------------------------------------------------------------------------
# 1. Orthogonality (direction_comparison_*)
# ---------------------------------------------------------------------------
def emit_orthogonality():
    section("Orthogonality (`cos_matched_full`)")
    print("| Cell | layer | cos_matched_full | 95% CI | cos_matched_null | held-out AUC | AUC d_imp on ds | AUC d_ref on ds |")
    print("|------|------:|-----------------:|--------|-----------------:|-------------:|----------------:|----------------:|")

    cells = []
    for p in sorted(EXP.glob("direction_comparison_qwen32b*.json")):
        d = load(p)
        name = p.stem.replace("direction_comparison_", "")
        cells.append((name, d, p))

    # Order: instruct first, then base
    cells.sort(key=lambda x: ("base" in x[0], x[0]))

    for name, d, p in cells:
        layer = d.get("layer", "—")
        cos_full = d.get("cos_matched_full")
        cos_null = d.get("cos_matched_null")
        ci_lo = d.get("bootstrap_cos_ci95_lo")
        ci_hi = d.get("bootstrap_cos_ci95_hi")
        ho_auc = d.get("heldout_refusal_auc")
        auc_imp = d.get("auc_impossibility_on_dataset")
        auc_ref = d.get("auc_refusal_on_dataset")
        ci = f"[{fmt_num(ci_lo)}, {fmt_num(ci_hi)}]" if ci_lo is not None else "—"
        print(f"| `{name}` | {layer} | {fmt_num(cos_full, 4)} | {ci} | {fmt_num(cos_null, 4)} | {fmt_num(ho_auc, 3)} | {fmt_num(auc_imp, 3)} | {fmt_num(auc_ref, 3)} |")


# ---------------------------------------------------------------------------
# 2. Detection / GSRS ablation (ablation_nullpc_*)
# ---------------------------------------------------------------------------
def emit_ablation():
    section("Detection / GSRS ablation (`ablation_nullpc_results_model-qwen32b_ds-*`)")
    print("| Dataset | Layer | Null-MD | PC-MD | Full-MD | Null > PC | Null > Full |")
    print("|---------|------:|--------:|------:|--------:|----------:|------------:|")

    for ds in ("math800", "code800"):
        p = EXP / f"ablation_nullpc_results_model-qwen32b_ds-{ds}.json"
        if not p.exists():
            continue
        d = load(p)
        # averages.meandiff has {null, pc, full}
        avg = d.get("averages", {}).get("meandiff", {})
        null_md = avg.get("null")
        pc_md = avg.get("pc")
        full_md = avg.get("full")
        # counts has md_null_gt_pc, md_null_gt_full, n_configs
        counts = d.get("counts", {})
        n_cfg = counts.get("n_configs", "—")
        gt_pc = counts.get("md_null_gt_pc", "—")
        gt_full = counts.get("md_null_gt_full", "—")
        # Layer from per_config[0]
        layer = "—"
        per_config = d.get("per_config", [])
        if per_config:
            layer = per_config[0].get("layer", "—")
        print(
            f"| {ds} | {layer} "
            f"| {fmt_num(null_md, 4)} | {fmt_num(pc_md, 4)} | {fmt_num(full_md, 4)} "
            f"| {gt_pc}/{n_cfg} | {gt_full}/{n_cfg} |"
        )


# ---------------------------------------------------------------------------
# 3. Layer emergence (peak layer + peak AUC + curve summary)
# ---------------------------------------------------------------------------
def emit_layer_emergence():
    section("Layer emergence (peak layer + NullMD curve summary)")
    print("| Dataset | n_layers | Peak layer | Peak Null-MD | Layer 0 Null-MD | Last layer Null-MD | mean Null−PC gap | max gap (layer) |")
    print("|---------|---------:|----------:|-------------:|---------------:|-------------------:|-----------------:|-----------------|")

    for ds in ("math800", "code800"):
        p = EXP / f"layer_emergence_results_model-qwen32b_ds-{ds}.json"
        if not p.exists():
            continue
        d = load(p)
        per_config = d.get("per_config", [])
        if not per_config:
            print(f"| {ds} | — | — | — | — | — | — | — |")
            continue
        ent = per_config[0]
        n_layers = ent.get("n_layers", "—")
        peak_layer = ent.get("peak_layer")
        peak_md = ent.get("peak_null_md")
        mean_gap = ent.get("mean_null_pc_gap")
        max_gap = ent.get("max_null_pc_gap")
        max_gap_layer = ent.get("max_null_pc_gap_layer")
        layers = ent.get("layers", [])
        first_md = layers[0]["null_md"] if layers else None
        last_md = layers[-1]["null_md"] if layers else None
        print(
            f"| {ds} | {n_layers} | {peak_layer} "
            f"| {fmt_num(peak_md, 4)} "
            f"| {fmt_num(first_md, 4)} | {fmt_num(last_md, 4)} "
            f"| {fmt_num(mean_gap, 3)} "
            f"| {fmt_num(max_gap, 3)} (L{max_gap_layer}) |"
        )


# ---------------------------------------------------------------------------
# 4. Form / category conditionality
# ---------------------------------------------------------------------------
def emit_form():
    section("Form / category conditionality")
    print("Global CosNSRT AUC (sanity check) and NS_SNR ↔ Cohen-d Spearman ρ from `part1_per_category`.")
    print()
    print("| Dataset | Layer | Global AUC | n categories | min/median/max per-cat AUC | Spearman ρ (NS_SNR ↔ AUC) |")
    print("|---------|------:|-----------:|-------------:|-----------------:|--------------------------:|")

    for ds in ("math800", "code800"):
        p = EXP / f"form_conditionality_results_model-qwen32b_ds-{ds}.json"
        if not p.exists():
            continue
        d = load(p)
        part1 = d.get("part1_per_category", [])
        if not part1:
            print(f"| {ds} | — | — | — | — | — |")
            continue
        ent = part1[0]
        layer = ent.get("layer", "—")
        global_auc = ent.get("global_cosnsrt_auc")
        per_cat = ent.get("per_category", [])
        n_cat = len(per_cat)
        aucs = sorted([c.get("auc") for c in per_cat if c.get("auc") is not None])
        if aucs:
            mn, mx = aucs[0], aucs[-1]
            md = aucs[len(aucs) // 2]
            auc_summary = f"{fmt_num(mn, 3)} / {fmt_num(md, 3)} / {fmt_num(mx, 3)}"
        else:
            auc_summary = "—"
        # Correlation: ns_snr ↔ auc
        corr = ent.get("correlations", {})
        snr_corr = corr.get("ns_snr", {}) if isinstance(corr, dict) else {}
        spearman = snr_corr.get("spearman_rho")
        spearman_p = snr_corr.get("spearman_p")
        spearman_str = f"{fmt_num(spearman, 3)} (p={fmt_num(spearman_p, 3)})" if spearman is not None else "—"
        print(
            f"| {ds} | {layer} | {fmt_num(global_auc, 4)} | {n_cat} "
            f"| {auc_summary} "
            f"| {spearman_str} |"
        )

    # Also print per-category breakdown (compact)
    print()
    print("**Per-category breakdown (math800):**")
    print()
    print("| Category | AUC | Cohen d | NS_SNR | n_A | n_U |")
    print("|----------|----:|--------:|-------:|----:|----:|")
    p = EXP / "form_conditionality_results_model-qwen32b_ds-math800.json"
    if p.exists():
        d = load(p)
        per_cat = (d.get("part1_per_category") or [{}])[0].get("per_category", [])
        for c in per_cat:
            print(
                f"| `{c.get('category')}` "
                f"| {fmt_num(c.get('auc'), 3)} "
                f"| {fmt_num(c.get('cohen_d'), 2)} "
                f"| {fmt_num(c.get('ns_snr'), 1)} "
                f"| {int(c.get('n_A', 0))} | {int(c.get('n_U', 0))} |"
            )

    print()
    print("**Per-category breakdown (code800):**")
    print()
    print("| Category | AUC | Cohen d | NS_SNR | n_A | n_U |")
    print("|----------|----:|--------:|-------:|----:|----:|")
    p = EXP / "form_conditionality_results_model-qwen32b_ds-code800.json"
    if p.exists():
        d = load(p)
        per_cat = (d.get("part1_per_category") or [{}])[0].get("per_category", [])
        for c in per_cat:
            print(
                f"| `{c.get('category')}` "
                f"| {fmt_num(c.get('auc'), 3)} "
                f"| {fmt_num(c.get('cohen_d'), 2)} "
                f"| {fmt_num(c.get('ns_snr'), 1)} "
                f"| {int(c.get('n_A', 0))} | {int(c.get('n_U', 0))} |"
            )


# ---------------------------------------------------------------------------
# 5. Cross-domain transfer
# ---------------------------------------------------------------------------
def emit_cross_domain():
    section("Cross-domain transfer (math ↔ code)")
    print("| Train | Test | within_mean | cross_mean | drop |")
    print("|-------|------|------------:|-----------:|-----:|")
    p = EXP / "cross_domain_transfer_model-qwen32b.json"
    if not p.exists():
        print("| — | — | — | — | — |")
        return
    d = load(p)
    for entry in d:
        print(
            f"| {entry.get('train', '?')} | {entry.get('test', '?')} "
            f"| {fmt_num(entry.get('within_mean'), 4)} "
            f"| {fmt_num(entry.get('cross_mean'), 4)} "
            f"| {fmt_num(entry.get('drop'), 4)} |"
        )


# ---------------------------------------------------------------------------
# 6. Steering breadth
# ---------------------------------------------------------------------------
def emit_steering():
    section("Steering breadth (Phase 9)")
    print("| Dataset | Layer | Best α (×σ) | Halluc reduction | Halluc red. % | Non-refusal cost (A-side) |")
    print("|---------|------:|------------:|-----------------:|--------------:|--------------------------:|")
    for fname in sorted((EXP / "steering").glob("steering_qwen32b_*.json")):
        d = load(fname)
        ds = d.get("dataset", "?")
        layer = d.get("layer", "—")
        best_a = d.get("best_alpha")
        red = d.get("hallucination_reduction")
        red_pct = d.get("hallucination_reduction_pct")
        cost = d.get("non_refusal_cost")
        print(
            f"| {ds} | {layer} | {best_a} "
            f"| {fmt_num(red, 3)} "
            f"| {fmt_num(red_pct, 1)}% "
            f"| {fmt_num(cost, 3)} |"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("# Qwen2.5-32B Supplementary Summary")
    print()
    print(f"Source: `{EXP.relative_to(REPO)}/` JSON outputs (RunPod A100 run 2026-04-27)")

    emit_orthogonality()
    emit_ablation()
    emit_layer_emergence()
    emit_form()
    emit_cross_domain()
    emit_steering()

    print()
    print("---")
    print("_Generated by `analysis/extract_qwen32b_summary.py`_")


if __name__ == "__main__":
    main()
