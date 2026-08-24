r"""
recompute_gated_dG_from_labels.py — Stage 3 of §4.3 intervention verification upgrade.

Purpose
-------
Consume LLM-assisted candidate labels plus an optional audit-subset label-
override TSV and the original samples TSV, then recompute the gated flip rate
ΔG = signal_gated − random_gated under TWO criteria. The override file may
contain provisional fills or independently completed human review; this script
does not infer either provenance from the filename or populated `final_*` cells.

  1. invalidity_aware  (broad — accepts "raises X" / "undefined" / "diverges")
  2. refusal_only      (strict — only explicit refusal/uncertainty phrasing)

Joins per-sample by (sample_id, condition, branch, alpha) — already encoded
in the batch keys produced by `prepare_intervention_review_batches.py`.

Condition-aware flip definition (CRITICAL)
------------------------------------------
A→U (inject signal): target = induce abstention on clean-non-abstaining baseline.
    flip(s) = (clean[C]==no) AND (branch[C]==yes)
    gate(s) = (clean[C]==no)

U→A (remove signal): target = remove abstention on clean-abstaining baseline.
    flip(s) = (clean[C]==yes) AND (branch[C]==no)
    gate(s) = (clean[C]==yes)

The gate is per-criterion (different for invalidity_aware vs refusal_only on
the same sample), per-condition (opposite directions), and per-branch in the
sense that uncertain labels on the relevant branch exclude the sample.

`uncertain` rule
----------------
Any sample whose clean label OR the relevant branch label is `uncertain` for
that criterion is excluded from BOTH numerator and denominator for that
(criterion, branch) computation. The excluded count is reported per cell.

`degenerate` rule (option C — punish branch collapse, exclude clean collapse)
-----------------------------------------------------------------------------
- If the clean generation is degenerate, the sample is excluded from the
  cell entirely (both numerator and denominator) for both criteria. The
  per-cell count is `n_clean_degenerate_excluded`.
- If the signal or random generation is degenerate, the sample stays in
  the gate denominator (assuming clean admits it) but the flip is forced
  to False regardless of the invalidity / refusal label values. The
  per-(criterion, branch) count is `n_branch_degenerate_in_gate`.

`degenerate_rate_signal` / `degenerate_rate_random` continue to report the
fraction of branch generations that were token-collapsed, computed over the
post-clean-exclusion sample set so it isolates *intervention*-induced
collapse from clean-side collapse.

CLI
---
    python scripts/recompute_gated_dG_from_labels.py \
        --candidate-labels experiments/intervention/labels/<cell>_labels.tsv \
        [--label-overrides experiments/intervention/labels/<cell>_overrides.tsv] \
        --tsv experiments/intervention/samples_<m>_<ds>_L<L>.tsv \
        --out experiments/intervention/intervention_<m>_<ds>_L<L>_v2.json \
        [--original-json experiments/intervention/intervention_<m>_<ds>_L<L>.json]

Why two label flags
-------------------
The validator emits an *audit subset* TSV containing only selected rows. If
the recompute script accepted only that subset, the remaining generations
would be silently dropped from the gate denominators and ΔG would collapse
without error.

Therefore:
  --candidate-labels  the FULL pre-labeling output, one row per generation
                      key (~900 rows for the canonical 50 × 5 × 2 grid).
  --label-overrides   OPTIONAL audit-subset TSV with `final_*` columns.
                      Rows here override candidate labels per field (only
                      fields whose `final_*` is non-blank). `--adjudicated`
                      is retained as a backwards-compatible alias only.

Hard-fail conditions
--------------------
1. duplicate `key` in --candidate-labels
2. duplicate `key` in --label-overrides
3. a `key` in --label-overrides does not exist in --candidate-labels
4. merge sanity: aggregated row count != candidate-labels row count

A warning (not a fail) is emitted when --candidate-labels does not cover
the full grid universe derivable from --tsv (fewer rows than expected).

If --original-json is omitted, the script auto-detects the sibling legacy
JSON by stripping `_v2` off the --out path. Top-level metadata (proj_std,
mu_norm, alpha_val per cell) is inherited from there if available; otherwise
those fields are emitted as null.

Output
------
JSON with the same outer shape as the legacy intervention JSONs, plus extra
per-cell fields under each result entry, plus a top-level
`verification_protocol` field.

Markdown sidecar (same path, .md): per-(condition, α) side-by-side table for
both criteria, plus degenerate rates and uncertain-excluded counts.

Notes
-----
- The audit-set TSV emitted by `validate_intervention_labels.py` already
  carries `candidate_*` columns alongside `final_*`. When passed as
  `--label-overrides`, only the `final_*` columns are read; the canonical
  candidate labels still come from `--candidate-labels`.
- This script does NOT call any external API and writes no .tex files.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

CONDITION_FROM_CODE = {
    "A→U": "A→U (inject signal)",
    "U→A": "U→A (remove signal)",
}
CONDITION_TO_CODE = {v: k for k, v in CONDITION_FROM_CODE.items()}

CRITERIA = ("invalidity_aware", "refusal_only")


# ── label-loading ─────────────────────────────────────────────────────


CRITERION_FIELDS = ("invalidity_aware", "refusal_only", "degenerate")


def _read_value(row: dict, *col_names: str) -> str:
    """Return the first non-empty value among `col_names`, lowercased.

    Useful for accepting either plain `invalidity_aware` or prefixed
    `candidate_invalidity_aware` column schemas in the same loader.
    """
    for c in col_names:
        v = (row.get(c) or "").strip()
        if v:
            return v.lower()
    return ""


# Column aliases accepted in --candidate-labels. The first match wins.
IA_CANDIDATE_ALIASES = (
    "invalidity_aware",
    "invalidity_aware_abstention",
    "candidate_invalidity_aware",
)
RO_CANDIDATE_ALIASES = (
    "refusal_only",
    "refusal_only_abstention",
    "candidate_refusal_only",
)
DG_CANDIDATE_ALIASES = ("degenerate", "candidate_degenerate")


def load_candidate_labels(path: Path) -> dict[str, dict[str, str]]:
    """Load the FULL candidate-labels TSV.

    Accepts column-name aliases for each criterion:
      invalidity_aware: `invalidity_aware`, `invalidity_aware_abstention`,
                        `candidate_invalidity_aware`
      refusal_only:     `refusal_only`,     `refusal_only_abstention`,
                        `candidate_refusal_only`
      degenerate:       `degenerate`,       `candidate_degenerate`

    Hard-fails on duplicate `key` rows and on missing required columns.
    """
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = set(reader.fieldnames or [])
        if "key" not in cols:
            raise ValueError(f"candidate-labels TSV missing `key` column: {reader.fieldnames}")
        has_ia = any(c in cols for c in IA_CANDIDATE_ALIASES)
        has_ro = any(c in cols for c in RO_CANDIDATE_ALIASES)
        has_dg = any(c in cols for c in DG_CANDIDATE_ALIASES)
        if not (has_ia and has_ro and has_dg):
            missing = []
            if not has_ia: missing.append(f"invalidity_aware (any of {IA_CANDIDATE_ALIASES})")
            if not has_ro: missing.append(f"refusal_only (any of {RO_CANDIDATE_ALIASES})")
            if not has_dg: missing.append(f"degenerate (any of {DG_CANDIDATE_ALIASES})")
            raise ValueError(
                f"candidate-labels TSV missing required columns: {missing}; got {sorted(cols)}"
            )
        for row in reader:
            k = row["key"]
            if k in out:
                raise ValueError(f"duplicate key in --candidate-labels: {k}")
            out[k] = {
                "invalidity_aware": _read_value(row, *IA_CANDIDATE_ALIASES),
                "refusal_only":     _read_value(row, *RO_CANDIDATE_ALIASES),
                "degenerate":       _read_value(row, *DG_CANDIDATE_ALIASES),
            }
    return out


def load_label_overrides(path: Path) -> dict[str, dict[str, str]]:
    """Load label overrides. Returns {key: {ia, ro, dg}} where each
    field value is the non-empty `final_*` cell, or "" if blank.

    Hard-fails on duplicate `key` rows and on missing `final_*` columns.
    """
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = set(reader.fieldnames or [])
        if "key" not in cols:
            raise ValueError(f"label-override TSV missing `key` column: {reader.fieldnames}")
        for c in CRITERION_FIELDS:
            if f"final_{c}" not in cols:
                raise ValueError(f"label-override TSV missing column: final_{c}")
        for row in reader:
            k = row["key"]
            if k in out:
                raise ValueError(f"duplicate key in --label-overrides: {k}")
            out[k] = {
                "invalidity_aware": (row.get("final_invalidity_aware") or "").strip().lower(),
                "refusal_only":     (row.get("final_refusal_only")     or "").strip().lower(),
                "degenerate":       (row.get("final_degenerate")       or "").strip().lower(),
            }
    return out


def merge_labels(
    candidates: dict[str, dict[str, str]],
    overrides: dict[str, dict[str, str]] | None,
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Per-field merge: override fields whose `final_*` is non-blank.

    Hard-fails if any override key is missing from `candidates`.

    Returns (merged_labels, merge_stats). `merge_stats` includes:
        audit_overrides_applied      # rows whose any final_* override was applied
        non_audit_passthrough        # rows untouched by the override file
        total_generations_aggregated
        candidate_labels_row_count
        per_field_overrides[field]   # per-criterion override counts
    """
    if overrides is None:
        overrides = {}

    extra = sorted(set(overrides) - set(candidates))
    if extra:
        head = ", ".join(extra[:10])
        more = f" (+{len(extra) - 10} more)" if len(extra) > 10 else ""
        raise ValueError(
            f"label-override TSV contains {len(extra)} key(s) not in candidate-labels: {head}{more}"
        )

    merged: dict[str, dict[str, str]] = {}
    audit_overrides_applied = 0
    non_audit_passthrough = 0
    per_field: dict[str, int] = {c: 0 for c in CRITERION_FIELDS}

    for k, cand in candidates.items():
        ov = overrides.get(k)
        if ov is None:
            merged[k] = dict(cand)
            non_audit_passthrough += 1
            continue
        row = dict(cand)
        any_override = False
        for c in CRITERION_FIELDS:
            if ov[c]:
                row[c] = ov[c]
                per_field[c] += 1
                any_override = True
        merged[k] = row
        if any_override:
            audit_overrides_applied += 1
        else:
            non_audit_passthrough += 1

    stats = {
        "audit_overrides_applied": audit_overrides_applied,
        "non_audit_passthrough": non_audit_passthrough,
        "total_generations_aggregated": len(merged),
        "candidate_labels_row_count": len(candidates),
        "per_field_overrides": per_field,
    }
    if stats["total_generations_aggregated"] != stats["candidate_labels_row_count"]:
        # This is a tautology by construction; assert defensively to catch bugs.
        raise ValueError(
            f"merge sanity violated: aggregated={stats['total_generations_aggregated']} "
            f"!= candidate_labels_row_count={stats['candidate_labels_row_count']}"
        )
    return merged, stats


def grid_universe(tsv: Path) -> set[str]:
    """All generation keys derivable from the source TSV (the full universe).

    Mirrors the dedup logic in `prepare_intervention_review_batches.py` so we
    can warn when the candidate-labels file doesn't cover the full grid.
    """
    keys: set[str] = set()
    with tsv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cond = CONDITION_TO_CODE.get(row["condition"], row["condition"])
            try:
                alpha = float(row["alpha_mult"])
            except (KeyError, ValueError):
                continue
            sid = row["sample_id"]
            keys.add(f"{sid}__{cond}__clean__0")
            if alpha != 0.0:
                keys.add(f"{sid}__{cond}__signal__{alpha}")
                keys.add(f"{sid}__{cond}__random__{alpha}")
    return keys


# ── TSV loading (gives us full sample × condition × alpha grid) ───────


def load_samples_grid(tsv: Path) -> tuple[dict, list[float], int]:
    """Return (grid, alphas, n_samples_per_class).

    grid[(condition_code, alpha)][sample_id] = alpha_val (signed) for legacy meta.
    """
    grid: dict[tuple[str, float], dict[str, float]] = defaultdict(dict)
    alphas_seen: set[float] = set()
    samples_per_cond: dict[str, set[str]] = defaultdict(set)
    with tsv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cond = CONDITION_TO_CODE.get(row["condition"], row["condition"])
            try:
                alpha = float(row["alpha_mult"])
            except (KeyError, ValueError):
                continue
            alphas_seen.add(alpha)
            samples_per_cond[cond].add(row["sample_id"])
            grid[(cond, alpha)][row["sample_id"]] = float(row.get("alpha_mult", 0.0))
    n_per_class = max((len(v) for v in samples_per_cond.values()), default=0)
    return grid, sorted(alphas_seen), n_per_class


# ── core computation ──────────────────────────────────────────────────


def gate_class_for(cond_code: str) -> str:
    """The clean-label class that admits a sample into the gate.

    A→U gate = clean is NOT abstaining (we want to inject one).
    U→A gate = clean IS abstaining (we want to remove it).
    """
    return "no" if cond_code == "A→U" else "yes"


def flip_predicate(cond_code: str, clean_v: str, branch_v: str) -> bool:
    """Per-sample flip given clean and branch labels (both yes/no, no uncertain)."""
    if cond_code == "A→U":
        return clean_v == "no" and branch_v == "yes"
    if cond_code == "U→A":
        return clean_v == "yes" and branch_v == "no"
    raise ValueError(f"unknown condition code: {cond_code}")


def key_for(sample_id: str, cond_code: str, branch: str, alpha: float) -> str:
    if branch == "clean":
        return f"{sample_id}__{cond_code}__clean__0"
    return f"{sample_id}__{cond_code}__{branch}__{alpha}"


def compute_cell(labels: dict, sample_ids: list[str], cond_code: str, alpha: float,
                 criterion: str, branch: str) -> dict:
    """Compute gated flip rate for one (condition, alpha, criterion, branch) cell.

    Degenerate handling (option C):
      • Caller is expected to have already removed clean-degenerate samples
        from `sample_ids` (they are excluded from the cell entirely, both
        numerator and denominator, for both criteria).
      • Branch-degenerate samples ARE kept in `sample_ids` and counted in
        the gate denominator if their clean label admits them, but the flip
        is forced to False regardless of the invalidity / refusal label.
        This punishes branch token collapse instead of treating it as a
        successful flip.
    """
    gate_v = gate_class_for(cond_code)
    n_clean_gate = 0
    n_flip = 0
    n_excluded = 0
    n_branch_degenerate_in_gate = 0
    for sid in sample_ids:
        clean_lbl = labels.get(key_for(sid, cond_code, "clean", alpha))
        branch_lbl = labels.get(key_for(sid, cond_code, branch, alpha))
        if clean_lbl is None or branch_lbl is None:
            n_excluded += 1
            continue
        clean_v = clean_lbl.get(criterion, "")
        branch_v = branch_lbl.get(criterion, "")
        if clean_v not in ("yes", "no") or branch_v not in ("yes", "no"):
            n_excluded += 1
            continue
        if clean_v != gate_v:
            continue  # outside the gate (legitimately) — not "excluded", just N/A
        n_clean_gate += 1
        # Branch token collapse: keep sample in the gate, force flip=False.
        if (branch_lbl.get("degenerate") or "").strip().lower() == "yes":
            n_branch_degenerate_in_gate += 1
            continue
        if flip_predicate(cond_code, clean_v, branch_v):
            n_flip += 1
    # If gate is empty, the rate is methodologically NOT MEASURABLE — emit
    # null instead of 0.0 so downstream tables/aggregators don't conflate
    # "no clean abstentions to flip" with "intervention had zero effect".
    rate = (n_flip / n_clean_gate) if n_clean_gate > 0 else None
    return {
        "n_clean_gate": n_clean_gate,
        "flip": n_flip,
        "n_excluded_uncertain": n_excluded,
        "n_branch_degenerate_in_gate": n_branch_degenerate_in_gate,
        "rate_gated": rate,
    }


def degenerate_rate(labels: dict, sample_ids: list[str], cond_code: str,
                    alpha: float, branch: str) -> tuple[int, int, float]:
    n = 0
    n_dg = 0
    for sid in sample_ids:
        lbl = labels.get(key_for(sid, cond_code, branch, alpha))
        if lbl is None:
            continue
        n += 1
        if lbl.get("degenerate") == "yes":
            n_dg += 1
    rate = (n_dg / n) if n > 0 else 0.0
    return n, n_dg, rate


# ── per-condition sample-id resolution ────────────────────────────────


def sample_ids_for_condition(grid: dict, cond_code: str) -> list[str]:
    sids: set[str] = set()
    for (c, a), per_sample in grid.items():
        if c == cond_code:
            sids.update(per_sample.keys())
    return sorted(sids)


# ── filename parsing (auto-detect legacy JSON) ────────────────────────


def auto_legacy_json(out_path: Path) -> Path:
    """`intervention_<m>_<ds>_L<L>_v2.json` → `intervention_<m>_<ds>_L<L>.json`."""
    name = out_path.name
    if name.endswith("_v2.json"):
        return out_path.with_name(name[:-len("_v2.json")] + ".json")
    return out_path  # not a v2 name; caller should pass --original-json explicitly


# ── main ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate-labels", required=True,
                    help="FULL pre-labeling output TSV (one row per generation key)")
    ap.add_argument("--label-overrides", "--adjudicated", dest="label_overrides",
                    default=None, help="Optional audit-subset TSV with `final_*` "
                    "overrides; --adjudicated is a legacy alias and does not "
                    "assert human provenance")
    ap.add_argument("--tsv", required=True, help="Source samples TSV")
    ap.add_argument("--out", required=True,
                    help="Output JSON (intervention_<m>_<ds>_L<L>_v2.json)")
    ap.add_argument("--original-json", default=None,
                    help="Legacy intervention JSON to inherit metadata from "
                         "(auto-detected from --out if name ends in _v2.json)")
    args = ap.parse_args()

    candidate_path = Path(args.candidate_labels)
    label_overrides_path = Path(args.label_overrides) if args.label_overrides else None
    tsv_path = Path(args.tsv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not candidate_path.exists():
        print(f"[error] candidate-labels TSV not found: {candidate_path}", file=sys.stderr)
        return 2
    if label_overrides_path is not None and not label_overrides_path.exists():
        print(f"[error] label-override TSV not found: {label_overrides_path}", file=sys.stderr)
        return 2
    if not tsv_path.exists():
        print(f"[error] samples TSV not found: {tsv_path}", file=sys.stderr)
        return 2

    legacy_path = Path(args.original_json) if args.original_json else auto_legacy_json(out_path)
    legacy: dict = {}
    if legacy_path.exists():
        try:
            with legacy_path.open(encoding="utf-8") as f:
                legacy = json.load(f)
        except json.JSONDecodeError:
            print(f"[warn] legacy JSON malformed; ignoring: {legacy_path}", file=sys.stderr)

    candidates = load_candidate_labels(candidate_path)
    overrides = load_label_overrides(label_overrides_path) if label_overrides_path else None
    labels, merge_stats = merge_labels(candidates, overrides)

    grid, alphas, n_per_class = load_samples_grid(tsv_path)
    universe = grid_universe(tsv_path)
    candidate_keys = set(candidates.keys())
    missing_in_candidates = sorted(universe - candidate_keys)
    extra_in_candidates = sorted(candidate_keys - universe)
    if missing_in_candidates:
        print(
            f"[warn] candidate-labels covers {len(candidate_keys)} keys but the "
            f"TSV-derived grid universe is {len(universe)}; "
            f"{len(missing_in_candidates)} key(s) not labeled — gateN will reflect "
            f"missing data",
            file=sys.stderr,
        )
    if extra_in_candidates:
        print(
            f"[warn] candidate-labels has {len(extra_in_candidates)} key(s) not "
            f"present in the TSV-derived grid universe (orphan labels)",
            file=sys.stderr,
        )

    # Build legacy alpha_val lookup (per condition, alpha_mult)
    legacy_cell: dict[tuple[str, float], dict] = {}
    for r in legacy.get("results", []):
        key = (CONDITION_TO_CODE.get(r.get("condition", ""), r.get("condition", "")),
               float(r.get("alpha_mult", 0.0)))
        legacy_cell[key] = r

    # Compute per-cell
    results: list[dict] = []
    total_uncertain_excluded = 0
    total_clean_degenerate_excluded = 0
    for cond_code in ("U→A", "A→U"):
        sample_ids = sample_ids_for_condition(grid, cond_code)
        for alpha in alphas:
            # Hoist clean-degenerate exclusion out of compute_cell: clean is
            # α-invariant, so the exclusion set is shared across both
            # criteria and both branches. Samples whose clean is degenerate
            # are removed from BOTH numerator and denominator for the cell.
            clean_dg_excluded = {
                sid for sid in sample_ids
                if (
                    (lbl := labels.get(key_for(sid, cond_code, "clean", alpha))) is not None
                    and (lbl.get("degenerate") or "").strip().lower() == "yes"
                )
            }
            sample_ids_eff = [sid for sid in sample_ids if sid not in clean_dg_excluded]

            cell: dict = {
                "condition": CONDITION_FROM_CODE[cond_code],
                "alpha_mult": alpha,
                "alpha_val": legacy_cell.get((cond_code, alpha), {}).get("alpha_val"),
                "n_samples": len(sample_ids),
                "n_clean_degenerate_excluded": len(clean_dg_excluded),
                "n_samples_effective": len(sample_ids_eff),
            }
            cell_uncertain = 0
            stub = {
                "n_clean_gate": 0, "flip": 0, "n_excluded_uncertain": 0,
                "n_branch_degenerate_in_gate": 0, "rate_gated": None,
            }
            # criterion blocks
            for crit in CRITERIA:
                sig = compute_cell(labels, sample_ids_eff, cond_code, alpha, crit, "signal") if alpha != 0.0 \
                    else dict(stub)
                rnd = compute_cell(labels, sample_ids_eff, cond_code, alpha, crit, "random") if alpha != 0.0 \
                    else dict(stub)
                # delta is None when either branch's gate is empty (not measurable)
                if sig["rate_gated"] is None or rnd["rate_gated"] is None:
                    delta = None
                else:
                    delta = sig["rate_gated"] - rnd["rate_gated"]
                cell[f"criterion_{crit}"] = {
                    "signal": sig,
                    "random": rnd,
                    "delta_gated": delta,
                }
                cell_uncertain += sig["n_excluded_uncertain"] + rnd["n_excluded_uncertain"]
            # convenience top-level fields
            cell["gated_flip_rate_invalidity_aware"] = {
                "signal": cell["criterion_invalidity_aware"]["signal"]["rate_gated"],
                "random": cell["criterion_invalidity_aware"]["random"]["rate_gated"],
                "delta": cell["criterion_invalidity_aware"]["delta_gated"],
            }
            cell["gated_flip_rate_refusal_only"] = {
                "signal": cell["criterion_refusal_only"]["signal"]["rate_gated"],
                "random": cell["criterion_refusal_only"]["random"]["rate_gated"],
                "delta": cell["criterion_refusal_only"]["delta_gated"],
            }
            # degenerate rates (over sample_ids_eff so they isolate
            # *intervention*-induced collapse, not clean-side collapse)
            if alpha != 0.0:
                _, _, dg_sig = degenerate_rate(labels, sample_ids_eff, cond_code, alpha, "signal")
                _, _, dg_rnd = degenerate_rate(labels, sample_ids_eff, cond_code, alpha, "random")
            else:
                dg_sig = dg_rnd = 0.0
            cell["degenerate_rate_signal"] = dg_sig
            cell["degenerate_rate_random"] = dg_rnd
            cell["n_uncertain_excluded"] = cell_uncertain
            total_uncertain_excluded += cell_uncertain
            total_clean_degenerate_excluded += len(clean_dg_excluded)
            results.append(cell)

    out_obj = {
        "model": legacy.get("model"),
        "dataset": legacy.get("dataset"),
        "layer": legacy.get("layer"),
        "reps_type": legacy.get("reps_type"),
        "n_samples_per_class": legacy.get("n_samples_per_class", n_per_class),
        "proj_std": legacy.get("proj_std"),
        "mu_norm": legacy.get("mu_norm"),
        "alphas": alphas,
        "verification_protocol": (
            "llm_assisted_candidate_plus_label_override_file_invalidity_aware_v2"
            if label_overrides_path else
            "llm_assisted_candidate_passthrough_invalidity_aware_v2"
        ),
        "n_uncertain_excluded_total": total_uncertain_excluded,
        "n_clean_degenerate_excluded_total": total_clean_degenerate_excluded,
        "candidate_labels_source": str(candidate_path),
        "label_override_source": (str(label_overrides_path) if label_overrides_path else None),
        "label_override_provenance": "not_inferred_by_recompute_script",
        "samples_tsv_source": str(tsv_path),
        "merge_stats": merge_stats,
        "grid_coverage": {
            "universe_size": len(universe),
            "candidate_keys": len(candidate_keys),
            "missing_in_candidates": len(missing_in_candidates),
            "extra_in_candidates": len(extra_in_candidates),
        },
        "results": results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    # Markdown sidecar
    md_path = out_path.with_suffix(".md")
    md_lines: list[str] = []
    md_lines.append(f"# v2 gated ΔG — {out_obj.get('model')} / {out_obj.get('dataset')} / L{out_obj.get('layer')}")
    md_lines.append("")
    md_lines.append(f"Protocol: `{out_obj['verification_protocol']}`  ")
    md_lines.append(f"Total `uncertain` excluded (counting signal + random): {total_uncertain_excluded}  ")
    md_lines.append(f"Total clean-degenerate samples excluded (cell-level): {total_clean_degenerate_excluded}  ")
    md_lines.append(f"Candidate labels: `{candidate_path}`  ")
    md_lines.append(f"Label-override file: `{label_overrides_path or '(none)'}`  ")
    md_lines.append("Override provenance: not inferred by this script.  ")
    md_lines.append(f"Source TSV: `{tsv_path}`  ")
    md_lines.append(
        f"Merge stats: overrides={merge_stats['audit_overrides_applied']}, "
        f"passthrough={merge_stats['non_audit_passthrough']}, "
        f"aggregated={merge_stats['total_generations_aggregated']}, "
        f"candidate_rows={merge_stats['candidate_labels_row_count']}  "
    )
    md_lines.append("")
    md_lines.append(
        "**Degenerate handling (option C):** clean-degenerate samples are "
        "excluded from the cell entirely (both numerator and denominator). "
        "Branch-degenerate samples remain in the gate denominator but are "
        "forced to flip=False (`n_branch_degenerate_in_gate` counts them)."
    )
    md_lines.append("")
    for cond_code in ("U→A", "A→U"):
        cond_full = CONDITION_FROM_CODE[cond_code]
        md_lines.append(f"## {cond_full}")
        md_lines.append("")
        md_lines.append("| α | criterion | n_gate_sig | flip_sig | rate_sig_g | n_gate_rnd | flip_rnd | rate_rnd_g | ΔG_gated |")
        md_lines.append("|---|-----------|-----------:|---------:|-----------:|-----------:|---------:|-----------:|---------:|")
        for cell in results:
            if cell["condition"] != cond_full:
                continue
            for crit in CRITERIA:
                blk = cell[f"criterion_{crit}"]
                # Render N/A for empty-gate cells (rate_gated/delta None)
                def _r(v):
                    return "N/A" if v is None else f"{v:.3f}"
                def _d(v):
                    return "N/A" if v is None else f"{v:+.3f}"
                md_lines.append(
                    f"| {cell['alpha_mult']:g} | {crit} | "
                    f"{blk['signal']['n_clean_gate']} | {blk['signal']['flip']} | {_r(blk['signal']['rate_gated'])} | "
                    f"{blk['random']['n_clean_gate']} | {blk['random']['flip']} | {_r(blk['random']['rate_gated'])} | "
                    f"{_d(blk['delta_gated'])} |"
                )
        md_lines.append("")
        md_lines.append("Degenerate accounting by α:")
        md_lines.append("")
        md_lines.append(
            "| α | clean_dg_excluded | n_branch_dg_in_gate (sig / rnd, ia) | "
            "degen_rate_sig | degen_rate_rnd |"
        )
        md_lines.append(
            "|---|------------------:|-----------------------------------:|"
            "---------------:|---------------:|"
        )
        for cell in results:
            if cell["condition"] != cond_full:
                continue
            ia_blk = cell["criterion_invalidity_aware"]
            md_lines.append(
                f"| {cell['alpha_mult']:g} | {cell['n_clean_degenerate_excluded']} | "
                f"{ia_blk['signal']['n_branch_degenerate_in_gate']} / "
                f"{ia_blk['random']['n_branch_degenerate_in_gate']} | "
                f"{cell['degenerate_rate_signal']:.3f} | "
                f"{cell['degenerate_rate_random']:.3f} |"
            )
        md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[recompute] candidate labels      : {merge_stats['candidate_labels_row_count']}  ({candidate_path})")
    print(f"[recompute] label-override rows   : "
          f"{(0 if overrides is None else len(overrides))}  "
          f"({label_overrides_path or '(none)'})")
    print(f"[recompute] audit overrides       : {merge_stats['audit_overrides_applied']}")
    print(f"[recompute] non-audit passthrough : {merge_stats['non_audit_passthrough']}")
    print(f"[recompute] total aggregated      : {merge_stats['total_generations_aggregated']}")
    print(f"[recompute] per-field overrides   : {merge_stats['per_field_overrides']}")
    print(f"[recompute] grid universe         : {len(universe)}  "
          f"(missing in candidates: {len(missing_in_candidates)}, "
          f"extra in candidates: {len(extra_in_candidates)})")
    print(f"[recompute] alphas                : {alphas}")
    print(f"[recompute] n_samples_per_cond    : {n_per_class}")
    print(f"[recompute] uncertain excluded    : {total_uncertain_excluded}")
    print(f"[recompute] clean-deg excluded    : {total_clean_degenerate_excluded}")
    print(f"[recompute] JSON                  : {out_path}")
    print(f"[recompute] Markdown              : {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
