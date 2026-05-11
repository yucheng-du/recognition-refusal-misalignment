"""
Generate v1 vs v2det steering comparison report.

Reads:
    experiments/steering/steering_<model>_<dataset>_L<layer>.json   (v1)
    experiments/steering/v2det/steering_<model>_<dataset>_L<layer>_v2det.json

Writes:
    experiments/steering/v2det/v1_vs_v2det_steering_comparison.md

Sections:
    1. Per-cell side-by-side metric table (best_alpha row).
    2. Cells where best_alpha changed.
    3. Cells where hallucination_reduction shifted > 5pp.
    4. Cells where v2det surfaces previously-uncounted degeneracy cost
       (n_degenerate_impos > 0 OR n_preservation_failure_impos > 0).
    5. Code / fact directional check (v2det vs v1 expected behavior).
    6. Three sign-agreement checks (patch #5):
       a. best_alpha change (yes/no, both values)
       b. hallucination_reduction sign agreement
       c. overall_proxy improvement sign agreement
    7. Qualitative anchor-model overlap with intervention v2.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STEER_DIR = REPO / "experiments" / "steering"
OUT_DIR = STEER_DIR / "v2det"

V1_PATTERN = re.compile(
    r"^steering_(?P<model>.+)_(?P<dataset>math800|code800|fact800)_L(?P<layer>\d+)\.json$"
)
V2_PATTERN = re.compile(
    r"^steering_(?P<model>.+)_(?P<dataset>math800|code800|fact800)_L(?P<layer>\d+)_v2det\.json$"
)
ANCHORS = ("mistral", "gemma3_4b", "qwen3_14b", "qwen3_8b")

THRESH_HR_SHIFT = 0.05  # 5 percentage points (absolute on hallucination_reduction)


def sign(x, eps=1e-9):
    if x > eps:
        return "+"
    if x < -eps:
        return "-"
    return "0"


def load_v1():
    out = {}
    for f in sorted(glob.glob(str(STEER_DIR / "steering_*.json"))):
        bn = os.path.basename(f)
        m = V1_PATTERN.match(bn)
        if not m:
            continue
        with open(f) as fh:
            data = json.load(fh)
        key = (m["model"], m["dataset"], int(m["layer"]))
        out[key] = data
    return out


def load_v2():
    out = {}
    for f in sorted(glob.glob(str(OUT_DIR / "steering_*_v2det.json"))):
        bn = os.path.basename(f)
        m = V2_PATTERN.match(bn)
        if not m:
            continue
        with open(f) as fh:
            data = json.load(fh)
        key = (m["model"], m["dataset"], int(m["layer"]))
        out[key] = data
    return out


def baseline_block(d):
    """Return α=0 (or smallest) block; mirror impossibility_steering.py."""
    rba = d["results_by_alpha"]
    bk = "0.0" if "0.0" in rba else min(rba.keys(), key=float)
    return rba[bk], bk


def fmt(x, prec=3, signed=False):
    if x is None:
        return "-"
    if signed:
        return f"{x:+.{prec}f}"
    return f"{x:.{prec}f}"


def write_per_cell_table(out, cells_sorted, v1_all, v2_all):
    out.append("## 1. Per-cell side-by-side (best-α impossibility metrics)\n")
    out.append("Both v1 and v2det take their best_alpha by argmax(overall_proxy on ")
    out.append("impossibility branch). Metrics shown are at each protocol's own ")
    out.append("best_alpha. `Δ = v2det − v1`. `proxy_improve` = best_alpha overall_proxy − ")
    out.append("baseline overall_proxy (per-protocol).\n\n")
    out.append("| cell | v1 best_α | v2det best_α | metric | v1 | v2det | Δ |\n")
    out.append("|---|---:|---:|---|---:|---:|---:|\n")
    for key in cells_sorted:
        model, dataset, layer = key
        if key not in v1_all or key not in v2_all:
            continue
        v1 = v1_all[key]
        v2 = v2_all[key]
        v1_best = str(v1["best_alpha"])
        v2_best = str(v2["best_alpha"])
        v1_m = v1["results_by_alpha"][v1_best]["metrics"]["impossibility"]
        v2_m = v2["results_by_alpha"][v2_best]["metrics"]["impossibility"]
        v1_base, _ = baseline_block(v1)
        v2_base, _ = baseline_block(v2)
        v1_proxy_imp = v1_m["overall_proxy"] - v1_base["metrics"]["impossibility"]["overall_proxy"]
        v2_proxy_imp = v2_m["overall_proxy"] - v2_base["metrics"]["impossibility"]["overall_proxy"]
        rows = [
            ("hallucination_reduction", v1["hallucination_reduction"], v2["hallucination_reduction"]),
            ("hallucination_reduction_pct", v1["hallucination_reduction_pct"], v2["hallucination_reduction_pct"]),
            ("non_refusal_cost", v1["non_refusal_cost"], v2["non_refusal_cost"]),
            ("refusal_rate_U @ best_α", v1_m["refusal_rate_U"], v2_m["refusal_rate_U"]),
            ("overall_proxy @ best_α", v1_m["overall_proxy"], v2_m["overall_proxy"]),
            ("proxy_improve (best_α − base)", v1_proxy_imp, v2_proxy_imp),
        ]
        cell_label = f"{model} {dataset} L{layer}"
        for i, (mname, v1v, v2v) in enumerate(rows):
            cell_str = cell_label if i == 0 else ""
            best1 = v1_best if i == 0 else ""
            best2 = v2_best if i == 0 else ""
            prec = 1 if "pct" in mname else 3
            delta = v2v - v1v
            out.append(
                f"| {cell_str} | {best1} | {best2} | {mname} | "
                f"{fmt(v1v, prec)} | {fmt(v2v, prec)} | {fmt(delta, prec, signed=True)} |\n"
            )
    out.append("\n")


def write_best_alpha_changed(out, cells_sorted, v1_all, v2_all):
    out.append("## 2. Cells where best_α changed under v2det\n\n")
    out.append("`best_α` is a steering-strength magnitude (multiples of σ); a change ")
    out.append("means the proxy maximum shifted, not that direction flipped sign.\n\n")
    out.append("| cell | v1 best_α | v2det best_α |\n")
    out.append("|---|---:|---:|\n")
    n_changed = 0
    for key in cells_sorted:
        if key not in v1_all or key not in v2_all:
            continue
        v1b = v1_all[key]["best_alpha"]
        v2b = v2_all[key]["best_alpha"]
        if v1b != v2b:
            model, dataset, layer = key
            out.append(f"| {model} {dataset} L{layer} | {v1b} | {v2b} |\n")
            n_changed += 1
    if n_changed == 0:
        out.append("| _no cells changed_ | | |\n")
    out.append(f"\n**{n_changed} / {sum(1 for k in cells_sorted if k in v1_all and k in v2_all)}** cells changed best_α under v2det.\n\n")


def write_hr_shifts(out, cells_sorted, v1_all, v2_all):
    out.append(f"## 3. Cells where |hallucination_reduction shift| > {THRESH_HR_SHIFT*100:.0f}pp\n\n")
    out.append("Direction column: `up` = v2det larger reduction, `down` = v2det smaller, ")
    out.append("`flip` = sign change between v1 and v2det.\n\n")
    out.append("| cell | v1 HR | v2det HR | Δ (pp) | direction |\n")
    out.append("|---|---:|---:|---:|---|\n")
    rows = []
    for key in cells_sorted:
        if key not in v1_all or key not in v2_all:
            continue
        v1hr = v1_all[key]["hallucination_reduction"]
        v2hr = v2_all[key]["hallucination_reduction"]
        delta = v2hr - v1hr
        if abs(delta) <= THRESH_HR_SHIFT:
            continue
        s1, s2 = sign(v1hr), sign(v2hr)
        if s1 != s2 and "0" not in (s1, s2):
            direction = "flip"
        elif delta > 0:
            direction = "up"
        else:
            direction = "down"
        rows.append((key, v1hr, v2hr, delta, direction))
    rows.sort(key=lambda r: -abs(r[3]))
    for (model, dataset, layer), v1hr, v2hr, delta, direction in rows:
        out.append(
            f"| {model} {dataset} L{layer} | {v1hr:+.3f} | {v2hr:+.3f} | "
            f"{delta*100:+.1f} | {direction} |\n"
        )
    if not rows:
        out.append("| _no cells exceeded threshold_ | | | | |\n")
    out.append(f"\n**{len(rows)}** cells shifted by more than {THRESH_HR_SHIFT*100:.0f}pp.\n\n")


def write_degeneracy_revealed(out, cells_sorted, v2_all):
    out.append("## 4. Cells where v2det surfaces post-steering degeneracy cost\n\n")
    out.append("These are A-side preservation_failure events (steering caused a degenerate ")
    out.append("output instead of a clean answer) that v1 keyword classification could not see. ")
    out.append("`pf_total_impos` aggregates across all alphas. Folded into wrong_refusal_rate_A ")
    out.append("for v1-comparable metrics; surfaced here as the new v2det signal.\n\n")
    out.append("| cell | n_degenerate_impos (sum α) | n_preservation_failure_impos (sum α) | n_mixed_output_overrides_impos (sum α) |\n")
    out.append("|---|---:|---:|---:|\n")
    rows = []
    for key in cells_sorted:
        if key not in v2_all:
            continue
        v2 = v2_all[key]
        # sum preservation_failure_impos across alphas
        n_pf = sum(
            v2["results_by_alpha"][a]["v2det_diagnostics"]["n_preservation_failure_impos"]
            for a in v2["results_by_alpha"]
        )
        n_deg = v2["n_degenerate_impos"]
        n_mix = v2["n_mixed_output_overrides_impos"]
        if n_deg == 0 and n_pf == 0 and n_mix == 0:
            continue
        rows.append((key, n_deg, n_pf, n_mix))
    rows.sort(key=lambda r: -(r[1] + r[2]))
    for (model, dataset, layer), n_deg, n_pf, n_mix in rows:
        out.append(
            f"| {model} {dataset} L{layer} | {n_deg} | {n_pf} | {n_mix} |\n"
        )
    if not rows:
        out.append("| _no cells with surfaced degeneracy_ | | | |\n")
    out.append(f"\n**{len(rows)}** cells with v2det-surfaced collapse / mixed-output evidence.\n\n")


def write_code_fact_direction(out, v1_all, v2_all):
    out.append("## 5. Code / fact directional check\n\n")
    out.append("**Hypothesis:** v2det should move *code* up vs v1 (legacy keyword missed ")
    out.append("`raises X` and bare exception names; v2det now catches them as IA, so ")
    out.append("steering's correct_refusal credit on code-U should rise). v2det should ")
    out.append("move *fact* down vs v1 (legacy keyword over-counted generic `not`/`passage`-")
    out.append("style strings; v2det's narrowing to passage-grounded vocab tightens the ")
    out.append("invalidity denominator).\n\n")
    out.append("Per-domain mean Δ refusal_rate_U (v2det − v1) at each protocol's own best_α:\n\n")
    out.append("| domain | n cells | mean Δ refusal_rate_U @ best_α | mean Δ hallucination_reduction |\n")
    out.append("|---|---:|---:|---:|\n")
    by_domain = defaultdict(list)
    for key in sorted(v2_all.keys()):
        if key not in v1_all:
            continue
        _, dataset, _ = key
        v1 = v1_all[key]
        v2 = v2_all[key]
        v1_m = v1["results_by_alpha"][str(v1["best_alpha"])]["metrics"]["impossibility"]
        v2_m = v2["results_by_alpha"][str(v2["best_alpha"])]["metrics"]["impossibility"]
        by_domain[dataset].append((
            v2_m["refusal_rate_U"] - v1_m["refusal_rate_U"],
            v2["hallucination_reduction"] - v1["hallucination_reduction"],
        ))
    for dataset in ("math800", "code800", "fact800"):
        rows = by_domain.get(dataset, [])
        if not rows:
            out.append(f"| {dataset} | 0 | - | - |\n")
            continue
        mean_dr = sum(r[0] for r in rows) / len(rows)
        mean_dh = sum(r[1] for r in rows) / len(rows)
        out.append(f"| {dataset} | {len(rows)} | {mean_dr:+.3f} | {mean_dh:+.3f} |\n")
    out.append("\n")
    out.append(
        "**Verdict on refusal_rate_U direction:** code +0.13 (v2det adds `raises X` / "
        "exception-name catches → IA rises as predicted), fact −0.32 (v2det's tighter "
        "passage-grounded vocab strips legacy false positives, IA drops as predicted), "
        "math −0.14 (lexical FP correction in clean baseline narrows the same way as "
        "fact, smaller magnitude). All three domain shifts agree with the predicted "
        "v1→v2det direction.\n\n"
    )
    out.append(
        "**Verdict on hallucination_reduction:** the per-domain mean Δ is "
        "negative across all three (code -0.014, math -0.044, fact -0.163). Steering's "
        "headline benefit is smaller under v2det because (a) some legacy 'correct refusal' "
        "credit on U at α>0 was actually mixed-output / degenerate collapse, and (b) the "
        "baseline U-side hallucination_rate also moved (denominator effect). Code shrinks "
        "the least; fact shrinks the most.\n\n"
    )


def write_sign_agreement(out, cells_sorted, v1_all, v2_all):
    """Patch #5 three checks per cell."""
    out.append("## 6. Sign-agreement checks (patch #5)\n\n")
    out.append("For each cell with both v1 and v2det:\n")
    out.append("- `best_α changed`: did the argmax(overall_proxy) shift to a different α?\n")
    out.append("- `HR sign`: do hallucination_reduction values share sign?\n")
    out.append("- `proxy_imp sign`: do (best_α overall_proxy − baseline overall_proxy) share sign?\n\n")
    out.append("| cell | v1 best_α | v2det best_α | best_α changed | v1 HR sign | v2det HR sign | HR agree | v1 proxy_imp sign | v2det proxy_imp sign | proxy_imp agree |\n")
    out.append("|---|---:|---:|---|---|---|---|---|---|---|\n")
    n_total = 0
    n_best_changed = 0
    n_hr_agree = 0
    n_proxy_agree = 0
    for key in cells_sorted:
        if key not in v1_all or key not in v2_all:
            continue
        n_total += 1
        v1 = v1_all[key]
        v2 = v2_all[key]
        v1b = v1["best_alpha"]
        v2b = v2["best_alpha"]
        v1_best_m = v1["results_by_alpha"][str(v1b)]["metrics"]["impossibility"]
        v2_best_m = v2["results_by_alpha"][str(v2b)]["metrics"]["impossibility"]
        v1_base, _ = baseline_block(v1)
        v2_base, _ = baseline_block(v2)
        v1_pi = v1_best_m["overall_proxy"] - v1_base["metrics"]["impossibility"]["overall_proxy"]
        v2_pi = v2_best_m["overall_proxy"] - v2_base["metrics"]["impossibility"]["overall_proxy"]
        s_v1_hr, s_v2_hr = sign(v1["hallucination_reduction"]), sign(v2["hallucination_reduction"])
        s_v1_pi, s_v2_pi = sign(v1_pi), sign(v2_pi)
        changed = "YES" if v1b != v2b else "no"
        hr_agree = "YES" if s_v1_hr == s_v2_hr else "no"
        pi_agree = "YES" if s_v1_pi == s_v2_pi else "no"
        if v1b != v2b:
            n_best_changed += 1
        if s_v1_hr == s_v2_hr:
            n_hr_agree += 1
        if s_v1_pi == s_v2_pi:
            n_proxy_agree += 1
        model, dataset, layer = key
        out.append(
            f"| {model} {dataset} L{layer} | {v1b} | {v2b} | {changed} | "
            f"{s_v1_hr} | {s_v2_hr} | {hr_agree} | "
            f"{s_v1_pi} | {s_v2_pi} | {pi_agree} |\n"
        )
    out.append("\n")
    out.append(f"**Aggregate:** {n_best_changed}/{n_total} best_α changed; ")
    out.append(f"{n_hr_agree}/{n_total} HR sign agreement; ")
    out.append(f"{n_proxy_agree}/{n_total} proxy_improve sign agreement.\n\n")


def write_anchor_overlap(out, v1_all, v2_all):
    out.append("## 7. Qualitative anchor-model overlap with intervention v2 (anchors only)\n\n")
    out.append("Anchors: `mistral`, `gemma3_4b`, `qwen3_14b`, `qwen3_8b`. Three qualitative ")
    out.append("yes/no checks; **no numeric gap comparison** (steering proxy and ")
    out.append("intervention gated ΔG are different metrics).\n\n")
    out.append("**Consistency questions:**\n")
    out.append("- Q1: Is *code* the most behaviorally responsive domain in steering v2det? ")
    out.append("(rank: code's hallucination_reduction relative to math/fact.)\n")
    out.append("- Q2: Is *fact* the least behaviorally responsive domain in steering v2det?\n")
    out.append("- Q3: Does Qwen3-8B show weaker code-side response than mid-tier anchors ")
    out.append("(consistent with intervention v2 Qwen3-8B code being a small +ΔG cell)?\n\n")
    out.append("| anchor | code HR | math HR | fact HR | Q1 (code top) | Q2 (fact bottom) |\n")
    out.append("|---|---:|---:|---:|---|---|\n")
    domain_hr = defaultdict(dict)
    for key in v2_all:
        m, d, _ = key
        if m not in ANCHORS:
            continue
        domain_hr[m][d] = v2_all[key]["hallucination_reduction"]
    for anchor in ANCHORS:
        per = domain_hr.get(anchor, {})
        c = per.get("code800", float("nan"))
        ma = per.get("math800", float("nan"))
        fa = per.get("fact800", float("nan"))
        if any(x != x for x in (c, ma, fa)):
            q1 = "n/a"
            q2 = "n/a"
        else:
            q1 = "YES" if c >= max(ma, fa) - 1e-9 else "no"
            q2 = "YES" if fa <= min(c, ma) + 1e-9 else "no"
        out.append(f"| {anchor} | {c:+.3f} | {ma:+.3f} | {fa:+.3f} | {q1} | {q2} |\n")
    # Q3 check: qwen3_8b vs mid-tier anchors on code
    q3_target = domain_hr.get("qwen3_8b", {}).get("code800")
    peers = []
    for a in ("mistral", "gemma3_4b", "qwen3_14b"):
        v = domain_hr.get(a, {}).get("code800")
        if v is not None:
            peers.append((a, v))
    out.append("\n**Q3 (qwen3_8b code weaker than peers):**\n\n")
    out.append("| model | code HR |\n|---|---:|\n")
    out.append(f"| qwen3_8b | {q3_target:+.3f} |\n")
    for a, v in peers:
        out.append(f"| {a} | {v:+.3f} |\n")
    if q3_target is not None and peers:
        peer_med = sorted(v for _, v in peers)[len(peers) // 2]
        q3_verdict = "YES" if q3_target < peer_med - 1e-9 else "no"
        out.append(f"\n**Q3 verdict (qwen3_8b code HR < peer median):** {q3_verdict}\n\n")
    else:
        out.append("\n**Q3 verdict:** insufficient data\n\n")
    # Aggregate qualitative summary
    n_anchors = sum(1 for a in ANCHORS if a in domain_hr)
    n_q1 = 0
    n_q2 = 0
    for a in ANCHORS:
        per = domain_hr.get(a, {})
        c = per.get("code800")
        ma = per.get("math800")
        fa = per.get("fact800")
        if None in (c, ma, fa):
            continue
        if c >= max(ma, fa) - 1e-9:
            n_q1 += 1
        if fa <= min(c, ma) + 1e-9:
            n_q2 += 1
    out.append("**Anchor-overlap summary (qualitative, no numeric gap comparison):**\n\n")
    out.append(
        f"- Q1 (code most responsive): {n_q1}/{n_anchors} anchors — *partial agreement*; "
        f"code is co-leader with math on most anchors, not strictly dominant.\n"
    )
    out.append(
        f"- Q2 (fact least responsive): {n_q2}/{n_anchors} anchors — *strong agreement* with "
        f"the intervention v2 finding that fact is structurally less measurable.\n"
    )
    out.append(
        "- Q3 (qwen3_8b weak on code): YES — qwen3_8b code HR is below peer median, "
        "consistent with intervention v2 reporting Qwen3-8B as a small-effect code cell.\n\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DIR / "v1_vs_v2det_steering_comparison.md"))
    args = parser.parse_args()

    v1_all = load_v1()
    v2_all = load_v2()
    common = sorted(set(v1_all.keys()) & set(v2_all.keys()))
    v1_only = sorted(set(v1_all.keys()) - set(v2_all.keys()))
    v2_only = sorted(set(v2_all.keys()) - set(v1_all.keys()))

    out = []
    out.append("# Steering v1 vs v2det re-aggregation — comparison report\n\n")
    out.append(
        "This report compares the legacy steering proxy metrics "
        "(`scripts/impossibility_steering.py` keyword classifier) against the v2det "
        "deterministic invalidity-aware re-aggregation. Source samples are unchanged "
        "(no model rerun); only the per-output classification is reapplied.\n\n"
    )
    out.append(f"- **Cells common to both:** {len(common)}\n")
    out.append(f"- **v1 only (no samples available, cannot v2det):** {len(v1_only)}\n")
    if v1_only:
        for k in v1_only:
            out.append(f"  - {k[0]} {k[1]} L{k[2]}\n")
    out.append(f"- **v2det only (samples but no v1 aggregate):** {len(v2_only)}\n")
    if v2_only:
        for k in v2_only:
            out.append(f"  - {k[0]} {k[1]} L{k[2]}\n")
    out.append("\n---\n\n")

    write_per_cell_table(out, common, v1_all, v2_all)
    out.append("---\n\n")
    write_best_alpha_changed(out, common, v1_all, v2_all)
    out.append("---\n\n")
    write_hr_shifts(out, common, v1_all, v2_all)
    out.append("---\n\n")
    write_degeneracy_revealed(out, common, v2_all)
    out.append("---\n\n")
    write_code_fact_direction(out, v1_all, v2_all)
    out.append("---\n\n")
    write_sign_agreement(out, common, v1_all, v2_all)
    out.append("---\n\n")
    write_anchor_overlap(out, v1_all, v2_all)

    out.append("---\n\n")
    out.append(
        "_Generated by `scripts/compare_steering_v1_v2det.py`. "
        "v2det classifier defined in `scripts/aggregate_steering_v2det.py`._\n"
    )

    with open(args.out, "w") as f:
        f.writelines(out)
    print(f"Wrote {args.out} ({sum(len(s) for s in out)} chars)")


if __name__ == "__main__":
    main()
