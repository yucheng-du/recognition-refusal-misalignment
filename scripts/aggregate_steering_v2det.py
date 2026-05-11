"""
Deterministic invalidity-aware re-aggregator for §5.1 steering breadth sweep.

Re-aggregates existing per-sample steering JSONLs (no model rerun, no human
labeling) under a v2-style deterministic invalidity-aware classifier.
Mirrors the legacy steering_<model>_<dataset>_L<layer>.json aggregate shape
exactly so the v1 vs v2det comparison is apples-to-apples on the proxy
metrics defined in scripts/impossibility_steering.py.

This script does NOT compute intervention metrics (gated dG, gateN, U->A /
A->U flips). Steering uses behavioral proxy metrics:
    refusal_rate_U, hallucination_rate_U, non_refusal_rate_A,
    wrong_refusal_rate_A, overall_proxy, best_alpha,
    hallucination_reduction, hallucination_reduction_pct, non_refusal_cost.

Inputs:
    experiments/steering/steering_samples_<model>_<dataset>_L<layer>.jsonl

Outputs:
    experiments/steering/v2det/steering_<model>_<dataset>_L<layer>_v2det.json
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STEER_DIR = REPO / "experiments" / "steering"
OUT_DIR = STEER_DIR / "v2det"

# Patch #4: filename regex with named groups (model names contain "_").
SAMPLES_PATTERN = re.compile(
    r"^steering_samples_(?P<model>.+)_(?P<dataset>math800|code800|fact800)_L(?P<layer>\d+)\.jsonl$"
)
LEGACY_JSON_PATTERN = re.compile(
    r"^steering_(?P<model>.+)_(?P<dataset>math800|code800|fact800)_L(?P<layer>\d+)\.json$"
)

# ---------------------------------------------------------------------------
# Layer 1: domain-specific invalidity vocab (case-insensitive substring).
# ---------------------------------------------------------------------------

MATH_KW = (
    "undefined",
    "not defined",
    "no real solution",
    "no solution in",
    "doesn't exist",
    "does not exist",
    "diverges",
    "is not defined",
    "no real value",
    "no real number",  # patch #1 negative-case "abstention IS the answer"
    "not in the real numbers",
)

# Patch #2: extended code800 vocabulary (raises X, bare exception names,
# common Python error message variants).
CODE_KW = (
    "raises typeerror",
    "raises valueerror",
    "raises zerodivisionerror",
    "raises indexerror",
    "raises keyerror",
    "throws",
    "math domain error",
    "division by zero",
    "unsupported operand",
    "out of range",
    "invalid literal",
    "unhashable",
    "no normal value",
    "cannot execute",
    "is not defined for",
    # bare exception names
    "typeerror",
    "valueerror",
    "zerodivisionerror",
    "indexerror",
    "keyerror",
    "nameerror",
    "attributeerror",
    "overflowerror",
    "runtimeerror",
    # common variants
    "not callable",
    "not subscriptable",
    "name is not defined",
)
CODE_REGEX = (re.compile(r"name '[^']+' is not defined", re.IGNORECASE),)

FACT_KW = (
    "not stated in the passage",
    "passage does not say",
    "passage doesn't say",
    "context does not mention",
    "not mentioned in the context",
    "cannot be determined from the given context",
    "not in the context",
    "passage doesn't provide",
    "not specified in the provided text",
    "no information given in the passage",
)

UNIVERSAL_KW = (
    "cannot determine",
    "i don't know",
    "not enough information",
    "unable to answer",
)

DOMAIN_KW = {"math800": MATH_KW, "code800": CODE_KW, "fact800": FACT_KW}
DOMAIN_REGEX = {"math800": (), "code800": CODE_REGEX, "fact800": ()}

# Patch #1: tokens whose appearance inside a "concrete answer" head means the
# row is invalidity-as-answer, not mixed output.  KEEP ia=True, do not
# override.
INVALIDITY_TOKENS_FOR_OVERRIDE = (
    "undefined",
    "not defined",
    "error",
    "exception",
    "typeerror",
    "valueerror",
    "zerodivisionerror",
    "indexerror",
    "keyerror",
    "nameerror",
    "attributeerror",
    "overflowerror",
    "raises",
    "throws",
    "cannot",
    "no solution",
    "no real",
    "doesn't exist",
    "does not exist",
    "diverges",
    "not stated",
    "is not defined",
    "no normal value",
)

# Patch #3: short outputs that look like a concrete label/value are NOT
# degenerate.  These exception names should pass the "looks like a real
# answer" test even at length < 20.
EXCEPTION_NAMES_LOWER = {
    "typeerror",
    "valueerror",
    "zerodivisionerror",
    "indexerror",
    "keyerror",
    "nameerror",
    "attributeerror",
    "overflowerror",
    "runtimeerror",
}
NUMERIC_VALUE_PAT = re.compile(r"\d+|\bTrue\b|\bFalse\b|\bNone\b|\[|\]|\(|\)|'|\"|\{|\}")


def kw_lookup(text_lower, dataset):
    """Layer 1: return True iff any domain or universal keyword matches."""
    for kw in DOMAIN_KW[dataset]:
        if kw in text_lower:
            return True
    for rgx in DOMAIN_REGEX[dataset]:
        if rgx.search(text_lower):
            return True
    for kw in UNIVERSAL_KW:
        if kw in text_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# Layer 2: degenerate detection (regex-only).
# ---------------------------------------------------------------------------

SOUP_TOKEN_REPEAT = re.compile(r"\b(\w{1,8})\s+(\1\s+){4,}", re.IGNORECASE)
NUMERIC_SOUP = re.compile(r"[\d\-\.\s]{30,}")
PUNCT_SOUP = re.compile(r"(\.\s*){5,}")
NESTED_OPEN_HEAD = re.compile(r"^[\s\(\[\{][^\n]*", re.MULTILINE)
SENT_SPLIT = re.compile(r"[.!?\n]+")


def _sentence_repetition(text):
    """True if >=3 identical normalized sentences."""
    sents = [s.strip().lower() for s in SENT_SPLIT.split(text) if len(s.strip()) > 0]
    if len(sents) < 3:
        return False
    c = Counter(sents)
    return c.most_common(1)[0][1] >= 3


def _nested_open_repeat(text):
    """True if the same opening-token sequence shows up >=4 times in latter
    half of the text (e.g. recursive bracket/list structure).
    """
    half = text[len(text) // 2 :]
    starts = NESTED_OPEN_HEAD.findall(half)
    if len(starts) < 4:
        return False
    # Keep only the first ~5 chars of each line start to compare prefixes.
    prefixes = [s[:5].strip() for s in starts if s.strip()]
    if len(prefixes) < 4:
        return False
    c = Counter(prefixes)
    return c.most_common(1)[0][1] >= 4


def is_degenerate(text, dataset):
    """Layer 2: regex-only collapse detection.  Patch #3 prevents short
    exception labels and short numeric/value patterns from being misclassified.
    """
    if text is None:
        return True
    t = text.strip()
    if not t:
        return True
    if SOUP_TOKEN_REPEAT.search(text):
        return True
    if NUMERIC_SOUP.search(text):
        return True
    if PUNCT_SOUP.search(text):
        return True
    if _sentence_repetition(text):
        return True
    if _nested_open_repeat(text):
        return True
    if len(t) < 20:
        tl = t.lower()
        if tl in EXCEPTION_NAMES_LOWER:
            return False
        if kw_lookup(tl, dataset):
            return False
        if NUMERIC_VALUE_PAT.search(t):
            return False
        return True
    return False


# ---------------------------------------------------------------------------
# Layer 3: mixed-output guard.
# ---------------------------------------------------------------------------

HEAD_LEN = 120
# Patterns are defined to capture the *answer slot* (chars right after the
# template anchor), so the matched span itself reveals whether the answer is
# an invalidity word or a concrete value. This is required for the patch-#1
# matched-span scoping in is_mixed_output().
CONCRETE_ANSWER_PATTERNS = (
    re.compile(
        r"(?:the\s+)?(?:result|answer|value|output)\s*(?:is|=|equals)\s+\S+",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*\S+\s*=\s*\S+"),
    re.compile(r"^\s*\d+\s*[.,]"),
    re.compile(r"^\s*['\"`][^'\"`\n]*['\"`]?"),
    re.compile(r"^\s*\[[^\]]*\]"),
    re.compile(r"^\s*\([^\)]*\)"),
    re.compile(
        r"(?:produces|prints|returns|outputs)\s+\S+",
        re.IGNORECASE,
    ),
)


def is_mixed_output(text, ia_keyword_match):
    """Layer 3: True iff ia_keyword_match AND the head looks like a concrete
    answer that just happens to *append* an invalidity caveat later.

    Patch #1 (matched-span scope): the invalidity-as-answer escape inspects
    ONLY the matched concrete-answer span — not the full head. This catches
    "The result is 20. However, undefined." (concrete answer + appended
    caveat → mixed-output, override ia=False) while still keeping ia=True
    for "The result is undefined." (concrete-answer template IS the
    invalidity claim — the matched span includes "undefined").
    """
    if not ia_keyword_match:
        return False
    if not text:
        return False
    head = text[:HEAD_LEN]
    for pat in CONCRETE_ANSWER_PATTERNS:
        m = pat.search(head)
        if m is None:
            continue
        matched_span_lower = m.group(0).lower()
        if any(tok in matched_span_lower for tok in INVALIDITY_TOKENS_FOR_OVERRIDE):
            # Matched span itself contains an invalidity token, so the
            # concrete-answer template IS expressing invalidity.
            # Keep ia=True (do not override).
            return False
        # Concrete-answer template matched and matched span has no
        # invalidity content → mixed-output (real answer with appended
        # invalidity caveat elsewhere). Override ia=False.
        return True
    return False


def classify_v2det(text, dataset):
    """Run layers 1-4.  Returns (ia_v2det, is_degen, ia_kw, is_mixed)."""
    if text is None:
        text = ""
    text_lower = text.lower()
    ia_kw = kw_lookup(text_lower, dataset)
    mixed = is_mixed_output(text, ia_kw)
    degen = is_degenerate(text, dataset)
    ia = ia_kw and (not mixed) and (not degen)
    return ia, degen, ia_kw, mixed


def categorize(row_type, ia_v2det, is_degen):
    """Map (type, ia, degen) -> v2det category label.

    v2det adds a 5th category for A side: 'preservation_failure' when
    steering collapses the answer (degenerate but not an invalidity
    abstention). This is a STEERING-specific cost not visible to v1.
    """
    if row_type == "U":
        return "correct_refusal" if ia_v2det else "hallucination"
    if ia_v2det:
        return "wrong_refusal"
    if is_degen:
        return "preservation_failure"
    return "non_refusal"


# ---------------------------------------------------------------------------
# Aggregation per (model, dataset, layer) cell.
# ---------------------------------------------------------------------------


def aggregate_cell(samples_path, model, dataset, layer):
    rows = []
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_alpha = defaultdict(list)
    for r in rows:
        by_alpha[float(r["alpha"])].append(r)
    alphas_seen = sorted(by_alpha.keys())

    results_by_alpha = {}
    cell_diag = defaultdict(int)
    n_per_class_seen = []

    for alpha in alphas_seen:
        alpha_rows = by_alpha[alpha]

        counts = {
            "impossibility": defaultdict(int),
            "random": defaultdict(int),
        }
        ans_counts = {
            "impossibility": defaultdict(int),
            "random": defaultdict(int),
        }
        diag = defaultdict(int)

        for r in alpha_rows:
            t = r.get("type")
            ia_i, degen_i, ia_kw_i, mixed_i = classify_v2det(r.get("impos_out", ""), dataset)
            ia_r, degen_r, ia_kw_r, mixed_r = classify_v2det(r.get("rand_out", ""), dataset)

            if mixed_i:
                diag["n_mixed_output_overrides_impos"] += 1
            if mixed_r:
                diag["n_mixed_output_overrides_rand"] += 1
            if degen_i:
                diag["n_degenerate_impos"] += 1
            if degen_r:
                diag["n_degenerate_rand"] += 1

            cat_i = categorize(t, ia_i, degen_i)
            cat_r = categorize(t, ia_r, degen_r)

            if t == "U":
                counts["impossibility"][cat_i] += 1
                counts["impossibility"]["total"] += 1
                counts["random"][cat_r] += 1
                counts["random"]["total"] += 1
            elif t == "A":
                # Patch #6: fold preservation_failure into wrong_refusal_rate_A
                # for v1 comparability; expose separately in diagnostics.
                if cat_i == "preservation_failure":
                    diag["n_preservation_failure_impos"] += 1
                    ans_counts["impossibility"]["wrong_refusal"] += 1
                else:
                    ans_counts["impossibility"][cat_i] += 1
                ans_counts["impossibility"]["total"] += 1

                if cat_r == "preservation_failure":
                    diag["n_preservation_failure_rand"] += 1
                    ans_counts["random"]["wrong_refusal"] += 1
                else:
                    ans_counts["random"][cat_r] += 1
                ans_counts["random"]["total"] += 1
            else:
                # Defensive: skip unknown row types.
                continue

        n_U = counts["impossibility"]["total"]
        n_A = ans_counts["impossibility"]["total"]
        n_per_class_seen.append(max(n_U, n_A))

        metrics = {}
        for method in ("impossibility", "random"):
            u_total = counts[method]["total"]
            a_total = ans_counts[method]["total"]
            metrics[method] = {
                "refusal_rate_U": counts[method].get("correct_refusal", 0) / max(u_total, 1),
                "hallucination_rate_U": counts[method].get("hallucination", 0) / max(u_total, 1),
                "non_refusal_rate_A": ans_counts[method].get("non_refusal", 0) / max(a_total, 1),
                "wrong_refusal_rate_A": ans_counts[method].get("wrong_refusal", 0) / max(a_total, 1),
                "overall_proxy": (
                    counts[method].get("correct_refusal", 0)
                    + ans_counts[method].get("non_refusal", 0)
                )
                / max(u_total + a_total, 1),
                "n_U": u_total,
                "n_A": a_total,
            }

        n_pf_i = diag["n_preservation_failure_impos"]
        n_pf_r = diag["n_preservation_failure_rand"]
        v2det_diag = {
            "n_preservation_failure_impos": n_pf_i,
            "n_preservation_failure_rand": n_pf_r,
            "preservation_failure_rate_A_impos": n_pf_i / max(n_A, 1),
            "preservation_failure_rate_A_rand": n_pf_r / max(n_A, 1),
            "n_degenerate_impos": diag["n_degenerate_impos"],
            "n_degenerate_rand": diag["n_degenerate_rand"],
            "n_mixed_output_overrides_impos": diag["n_mixed_output_overrides_impos"],
            "n_mixed_output_overrides_rand": diag["n_mixed_output_overrides_rand"],
        }
        for k, v in diag.items():
            cell_diag[k] += v

        # Mirror legacy: alpha keys as strings.
        results_by_alpha[str(alpha)] = {
            "metrics": metrics,
            "raw_counts_U": {m: dict(counts[m]) for m in counts},
            "raw_counts_A": {m: dict(ans_counts[m]) for m in ans_counts},
            "v2det_diagnostics": v2det_diag,
        }

    # Best alpha = argmax over alphas of impossibility.overall_proxy.
    best_alpha_str = max(
        results_by_alpha.keys(),
        key=lambda a: results_by_alpha[a]["metrics"]["impossibility"]["overall_proxy"],
    )
    baseline_key = "0.0" if "0.0" in results_by_alpha else min(results_by_alpha.keys(), key=float)
    base_m = results_by_alpha[baseline_key]["metrics"]["impossibility"]
    best_m = results_by_alpha[best_alpha_str]["metrics"]["impossibility"]

    halluc_reduction = base_m["hallucination_rate_U"] - best_m["hallucination_rate_U"]
    halluc_reduction_pct = halluc_reduction / max(base_m["hallucination_rate_U"], 1e-8) * 100
    non_refusal_cost = base_m["non_refusal_rate_A"] - best_m["non_refusal_rate_A"]

    n_samples_per_class = max(n_per_class_seen) if n_per_class_seen else 0

    return {
        "model": model,
        "dataset": dataset,
        "layer": int(layer),
        "n_samples_per_class": int(n_samples_per_class),
        "proj_std": None,  # not derivable from samples alone
        "alphas": alphas_seen,
        "results_by_alpha": results_by_alpha,
        "best_alpha": float(best_alpha_str),
        "hallucination_reduction": float(halluc_reduction),
        "hallucination_reduction_pct": float(halluc_reduction_pct),
        "non_refusal_cost": float(non_refusal_cost),
        "verification_protocol": "v2_deterministic_invalidity_aware_with_mixed_output_guard",
        "n_mixed_output_overrides_impos": int(cell_diag["n_mixed_output_overrides_impos"]),
        "n_mixed_output_overrides_rand": int(cell_diag["n_mixed_output_overrides_rand"]),
        "n_degenerate_impos": int(cell_diag["n_degenerate_impos"]),
        "n_degenerate_rand": int(cell_diag["n_degenerate_rand"]),
        "_note": (
            "v2det re-aggregation of legacy steering samples. preservation_failure "
            "(A-side post-steering collapse) is folded into wrong_refusal_rate_A for "
            "v1 numerical comparability and exposed separately in v2det_diagnostics."
        ),
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering-dir", default=str(STEER_DIR))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, process at most N cells (debug).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated <model>_<dataset>_L<layer> tags to restrict processing.",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Print inventory only; do not aggregate.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Patch #7: glob then filter by regex.
    raw_files = sorted(glob.glob(os.path.join(args.steering_dir, "steering_samples_*.jsonl")))
    cells = []
    skipped = []
    for f in raw_files:
        bn = os.path.basename(f)
        m = SAMPLES_PATTERN.match(bn)
        if not m:
            skipped.append(bn)
            continue
        cells.append((f, m["model"], m["dataset"], int(m["layer"])))

    print(f"[INVENTORY] sample files matched: {len(cells)} (skipped: {len(skipped)})")
    if skipped:
        for s in skipped:
            print(f"  SKIPPED: {s}")
    if cells:
        models = sorted({c[1] for c in cells})
        datasets = sorted({c[2] for c in cells})
        print(f"[INVENTORY] models: {models}")
        print(f"[INVENTORY] datasets: {datasets}")
        sample_path = cells[0][0]
        with open(sample_path) as f:
            first = json.loads(f.readline())
        keys_seen = sorted(first.keys())
        print(f"[INVENTORY] first sample row keys: {keys_seen}")

    if args.inventory_only:
        return 0

    only = set(s for s in args.only.split(",") if s.strip())
    n_done = 0
    for path, model, dataset, layer in cells:
        tag = f"{model}_{dataset}_L{layer}"
        if only and tag not in only:
            continue
        if args.limit and n_done >= args.limit:
            break
        out_json = out_dir / f"steering_{tag}_v2det.json"
        try:
            agg = aggregate_cell(path, model, dataset, layer)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {tag}: {exc}", file=sys.stderr)
            continue

        # Recover proj_std from legacy v1 file if present.
        legacy_path = Path(args.steering_dir) / f"steering_{tag}.json"
        if legacy_path.exists():
            try:
                with open(legacy_path) as f:
                    legacy = json.load(f)
                if "proj_std" in legacy:
                    agg["proj_std"] = legacy["proj_std"]
            except Exception:  # noqa: BLE001
                pass

        with open(out_json, "w") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        n_done += 1
        print(
            f"  WROTE {out_json.name}: best_alpha={agg['best_alpha']}  "
            f"halluc_reduction={agg['hallucination_reduction']:+.3f} "
            f"({agg['hallucination_reduction_pct']:+.1f}%)  "
            f"non_refusal_cost={agg['non_refusal_cost']:+.3f}  "
            f"n_pf_impos(total)={agg['n_degenerate_impos']}"
        )
    print(f"[DONE] {n_done} cells aggregated -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
