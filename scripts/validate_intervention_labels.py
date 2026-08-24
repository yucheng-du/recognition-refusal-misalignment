r"""
validate_intervention_labels.py — Stage 2 of §4.3 intervention verification upgrade.

Purpose
-------
Given (a) a directory of batch JSONLs produced by
`prepare_intervention_review_batches.py` and (b) a candidate-label TSV produced
by an LLM-assisted pre-labeling pass, this script:

  1. Cross-checks coverage: every batch key has exactly one label row.
  2. Validates that label field values lie in their allowed sets.
  3. Flags the only logical schema error: `refusal_only=yes` with
     `invalidity_aware != yes`. (The reverse — `invalidity_aware=yes` with
     `refusal_only=no` — is the *expected* case for invalidity-recognition
     answers like "raises TypeError" or "undefined", and is NOT an error.)
  4. Reports per-criterion `uncertain` counts (rubric ambiguity signal).
  5. Builds an audit set of rows selected for additional review. Human
     adjudication is a separate optional step and must not be inferred merely
     from the existence of a populated audit-subset file.

CLI
---
    python scripts/validate_intervention_labels.py \
        --labels experiments/intervention/labels/<cell>_labels.tsv \
        --batches-dir experiments/intervention/review_batches/<cell>/ \
        --audit-out experiments/intervention/audit_sets/<cell>_audit.tsv

Label TSV schema (minimum required columns)
-------------------------------------------
    key,
    invalidity_aware (or invalidity_aware_abstention),
    refusal_only     (or refusal_only_abstention),
    degenerate

  invalidity_aware ∈ {yes, no, uncertain}
  refusal_only     ∈ {yes, no, uncertain}
  degenerate       ∈ {yes, no}

The rubric's natural column names use the `*_abstention` suffix; the
short forms are aliases for the same fields. Either set is accepted.
Optional columns are passed through unchanged. `notes` (free text) is
recommended for tricky rows.

Audit-set selection rules
-------------------------
A row enters the audit set if ANY of the following hold:
  • branch == clean                      → reason: "clean_baseline"
  • candidate_invalidity_aware == "yes"  → reason: "invalidity_aware_yes"
  • candidate_degenerate == "yes"        → reason: "degenerate_yes"
  • candidate_invalidity_aware == "no", chosen via stratified 10% sample
                                         (seed=42) → reason: "stratified_no_sample"
  • condition-aware candidate flip under EITHER criterion (signal/random
    branches only). For each criterion C in {invalidity_aware, refusal_only}:
        candidate_flip_A→U = (clean[C]=="no")  and (branch[C]=="yes")
        candidate_flip_U→A = (clean[C]=="yes") and (branch[C]=="no")
    → reason: "candidate_flip_invalidity_aware" / "candidate_flip_refusal_only"
    Skipped when the branch generation is degenerate — recompute already
    forces flip=False there, and the row is already audited via
    `degenerate_yes`.
  • either side of that comparison is uncertain (clean OR branch label, on
    EITHER criterion) → reason: "uncertain_for_candidate_flip"
  • candidate `invalidity_aware` labels imply a new-flip verdict that
    disagrees with the source-TSV legacy `keyword_flip_*` boolean
    → reason: "legacy_disagreement"
  • the legacy comparison above can't be computed because clean or branch
    `invalidity_aware` is uncertain
    → reason: "uncertain_for_legacy_comparison"

Why the candidate-flip rule is condition-aware
----------------------------------------------
Selecting only `candidate_invalidity_aware == "yes"` rows audits A→U
positive flips (clean=no → branch=yes) but systematically misses U→A
positive flips (clean=yes → branch=no), which look like
`branch_invalidity_aware="no"` and would otherwise pass to recompute on the
LLM candidate label without review. The rule above mirrors the flip
definition in `recompute_gated_dG_from_labels` so every condition-aware
positive flip — under either criterion — is selected for the audit set.

Multiple reasons can stack on one row; reasons are comma-joined and
deduplicated.

Output audit TSV columns
------------------------
    key, prompt, generation, branch, condition, alpha,
    candidate_invalidity_aware, candidate_refusal_only, candidate_degenerate,
    audit_reason,
    final_invalidity_aware, final_refusal_only, final_degenerate

The `final_*` columns are intentionally left blank. A reviewer may fill them,
but a populated file may also contain provisional LLM-assisted fills; its
provenance must be recorded separately. The downstream
`recompute_gated_dG_from_labels.py` consumes the optional override TSV,
preferring populated `final_*` fields and falling back to `candidate_*`
otherwise.

Notes
-----
- This script does not call any external API and writes no .tex files.
- It deliberately does NOT mutate the input label TSV; output is a separate
  audit set that can be reviewed independently.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize)

VALID_TRINARY = {"yes", "no", "uncertain"}
VALID_BINARY = {"yes", "no"}

CONDITION_CODE = {
    "A→U (inject signal)": "A→U",
    "U→A (remove signal)": "U→A",
}


def load_batches(batches_dir: Path) -> tuple[dict[str, dict], dict]:
    """Load all batch_*.jsonl entries plus the manifest.

    Returns (entries_by_key, manifest_dict).
    """
    manifest_path = batches_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json missing in {batches_dir}")
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    entries: dict[str, dict] = {}
    for fname in manifest["batch_files"]:
        fpath = batches_dir / fname
        with fpath.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                e = json.loads(line)
                if e["key"] in entries:
                    raise ValueError(f"duplicate batch key: {e['key']}")
                entries[e["key"]] = e
    return entries, manifest


IA_COLUMN_ALIASES = ("invalidity_aware", "invalidity_aware_abstention")
RO_COLUMN_ALIASES = ("refusal_only", "refusal_only_abstention")


def load_labels(labels_path: Path) -> dict[str, dict]:
    """Load candidate-label TSV keyed by `key`. Detects duplicate keys.

    Accepts column aliases for the two trinary criteria so the rubric's
    natural names (`invalidity_aware_abstention`, `refusal_only_abstention`)
    are loadable without a TSV rename. Internally everything is normalized
    to the short forms (`invalidity_aware`, `refusal_only`).
    """
    out: dict[str, dict] = {}
    with labels_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = set(reader.fieldnames or [])
        if "key" not in cols:
            raise ValueError(f"label TSV missing `key` column: {reader.fieldnames}")
        ia_col = next((c for c in IA_COLUMN_ALIASES if c in cols), None)
        ro_col = next((c for c in RO_COLUMN_ALIASES if c in cols), None)
        dg_col = "degenerate" if "degenerate" in cols else None
        missing: list[str] = []
        if ia_col is None:
            missing.append("invalidity_aware (or invalidity_aware_abstention)")
        if ro_col is None:
            missing.append("refusal_only (or refusal_only_abstention)")
        if dg_col is None:
            missing.append("degenerate")
        if missing:
            raise ValueError(
                f"label TSV missing required columns: {missing}; "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            k = row["key"]
            if k in out:
                raise ValueError(f"duplicate label row for key: {k}")
            normalized = dict(row)
            normalized["invalidity_aware"] = row.get(ia_col, "")
            normalized["refusal_only"] = row.get(ro_col, "")
            normalized["degenerate"] = row.get(dg_col, "")
            out[k] = normalized
    return out


def load_legacy_keyword_flags(source_tsv: Path) -> dict[tuple[str, str, str], dict[str, bool]]:
    """Map (sample_id, condition_code, alpha_str) → {signal: bool, random: bool}.

    `alpha_str` is the str(float(...)) form to match the format used in keys.
    """
    flags: dict[tuple[str, str, str], dict[str, bool]] = {}
    with source_tsv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cond = CONDITION_CODE.get(row["condition"], row["condition"])
            try:
                alpha_str = str(float(row["alpha_mult"]))
            except (KeyError, ValueError):
                continue
            key = (row["sample_id"], cond, alpha_str)
            flags[key] = {
                "signal": row.get("keyword_flip_signal", "").strip().lower() == "true",
                "random": row.get("keyword_flip_random", "").strip().lower() == "true",
            }
    return flags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="Candidate-label TSV")
    ap.add_argument("--batches-dir", required=True,
                    help="Directory containing batch_*.jsonl + manifest.json")
    ap.add_argument("--audit-out", required=True,
                    help="Output audit TSV (rows selected for additional review)")
    ap.add_argument("--stratified-no-frac", type=float, default=0.10,
                    help="Fraction of invalidity_aware=no rows to sample (default 0.10)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for stratified sample (default 42)")
    args = ap.parse_args()

    labels_path = Path(args.labels)
    batches_dir = Path(args.batches_dir)
    audit_out = Path(args.audit_out)
    audit_out.parent.mkdir(parents=True, exist_ok=True)

    if not labels_path.exists():
        print(f"[error] labels TSV not found: {labels_path}", file=sys.stderr)
        return 2
    if not batches_dir.exists():
        print(f"[error] batches dir not found: {batches_dir}", file=sys.stderr)
        return 2

    entries_by_key, manifest = load_batches(batches_dir)
    labels_by_key = load_labels(labels_path)
    source_tsv = Path(manifest["source_tsv"])
    legacy = load_legacy_keyword_flags(source_tsv) if source_tsv.exists() else {}
    if not legacy:
        print(f"[warn] could not load legacy keyword flags from {source_tsv} "
              "— legacy_keyword_disagreement audit reason will be empty",
              file=sys.stderr)

    # ── 1. Coverage ────────────────────────────────────────────────────
    universe = set(entries_by_key.keys())
    labeled = set(labels_by_key.keys())
    missing = sorted(universe - labeled)
    extra = sorted(labeled - universe)

    # ── 2/3. Field validity + schema check ─────────────────────────────
    field_violations: list[tuple[str, str, str]] = []  # (key, field, value)
    schema_errors: list[tuple[str, str]] = []  # (key, detail)
    uncertain_counts = Counter()

    for k, lab in labels_by_key.items():
        ia = lab["invalidity_aware"].strip().lower()
        ro = lab["refusal_only"].strip().lower()
        dg = lab["degenerate"].strip().lower()
        if ia not in VALID_TRINARY:
            field_violations.append((k, "invalidity_aware", ia))
        if ro not in VALID_TRINARY:
            field_violations.append((k, "refusal_only", ro))
        if dg not in VALID_BINARY:
            field_violations.append((k, "degenerate", dg))
        if ia == "uncertain":
            uncertain_counts["invalidity_aware"] += 1
        if ro == "uncertain":
            uncertain_counts["refusal_only"] += 1
        # Schema error: refusal_only=yes ⇒ invalidity_aware=yes.
        if ro == "yes" and ia != "yes":
            schema_errors.append(
                (k, f"refusal_only=yes but invalidity_aware={ia!r}")
            )

    # ── 4. Build audit set ─────────────────────────────────────────────
    audit_rows: list[dict] = []
    audit_reasons: dict[str, list[str]] = {}

    def add_reason(key: str, reason: str) -> None:
        bucket = audit_reasons.setdefault(key, [])
        if reason not in bucket:
            bucket.append(reason)

    # 4a. all clean baselines
    for k, e in entries_by_key.items():
        if e.get("branch") == "clean":
            add_reason(k, "clean_baseline")

    # 4b. all candidate_invalidity_aware == "yes"
    # 4c. all candidate_degenerate == "yes"
    invalidity_no_keys: list[str] = []
    for k, lab in labels_by_key.items():
        if k not in entries_by_key:
            continue
        ia = lab["invalidity_aware"].strip().lower()
        dg = lab["degenerate"].strip().lower()
        if ia == "yes":
            add_reason(k, "invalidity_aware_yes")
        elif ia == "no":
            invalidity_no_keys.append(k)
        if dg == "yes":
            add_reason(k, "degenerate_yes")

    # 4d. stratified 10% of invalidity_aware=no
    if invalidity_no_keys:
        rng = random.Random(args.seed)
        invalidity_no_keys.sort()  # determinism
        n_sample = max(1, int(round(len(invalidity_no_keys) * args.stratified_no_frac)))
        for k in rng.sample(invalidity_no_keys, min(n_sample, len(invalidity_no_keys))):
            add_reason(k, "stratified_no_sample")

    # 4e. condition-aware legacy disagreement (signal/random branches only).
    # Mirror the flip definition used by recompute_gated_dG_from_labels:
    #   A→U: new_flip = (clean_ia == "no")  and (branch_ia == "yes")
    #   U→A: new_flip = (clean_ia == "yes") and (branch_ia == "no")
    # If clean or branch ia is "uncertain", we can't compute new_flip — flag
    # for human review under "uncertain_for_legacy_comparison".
    for k, e in entries_by_key.items():
        if e.get("branch") not in ("signal", "random"):
            continue
        if k not in labels_by_key:
            continue
        sample_id = k.split("__", 1)[0]
        cond = e["condition"]
        alpha_str = str(float(e["alpha"]))
        legacy_row = legacy.get((sample_id, cond, alpha_str))
        if legacy_row is None:
            continue
        legacy_flag = legacy_row[e["branch"]]  # bool
        branch_ia = labels_by_key[k]["invalidity_aware"].strip().lower()
        clean_key = f"{sample_id}__{cond}__clean__0"
        clean_lab = labels_by_key.get(clean_key)
        if clean_lab is None:
            # No clean label for this unit → can't compute new_flip.
            add_reason(k, "uncertain_for_legacy_comparison")
            continue
        clean_ia = (clean_lab.get("invalidity_aware") or "").strip().lower()
        if clean_ia == "uncertain" or branch_ia == "uncertain":
            add_reason(k, "uncertain_for_legacy_comparison")
            continue
        if clean_ia not in ("yes", "no") or branch_ia not in ("yes", "no"):
            # Field-validity violation already reported elsewhere; skip.
            continue
        if cond == "A→U":
            new_flip = (clean_ia == "no") and (branch_ia == "yes")
        elif cond == "U→A":
            new_flip = (clean_ia == "yes") and (branch_ia == "no")
        else:
            continue
        if legacy_flag != new_flip:
            add_reason(k, "legacy_disagreement")

    # 4f. condition-aware candidate positive flip — under EITHER criterion.
    # Without this rule, U→A positive flips (clean=yes → branch=no) silently
    # pass to recompute on the LLM candidate label without human review,
    # because they look like `branch_invalidity_aware="no"` and slip past
    # the "invalidity_aware=yes" filter (which only catches A→U flips).
    # Mirrors the flip definition in `recompute_gated_dG_from_labels`.
    #
    # If the branch generation is degenerate, the row is already audited via
    # `degenerate_yes` AND recompute will force flip=False for it regardless
    # of the invalidity/refusal labels. Adding `candidate_flip_*` here would
    # imply a successful flip that recompute then negates — confusing the
    # taxonomy. So skip the candidate-flip rule entirely for those rows.
    CRITERIA = ("invalidity_aware", "refusal_only")
    for k, e in entries_by_key.items():
        if e.get("branch") not in ("signal", "random"):
            continue
        if k not in labels_by_key:
            continue
        sample_id = k.split("__", 1)[0]
        cond = e["condition"]
        if cond not in ("A→U", "U→A"):
            continue
        branch_lab = labels_by_key[k]
        if (branch_lab.get("degenerate") or "").strip().lower() == "yes":
            # Already in audit via degenerate_yes; recompute negates the flip.
            continue
        clean_key = f"{sample_id}__{cond}__clean__0"
        clean_lab = labels_by_key.get(clean_key)
        for crit in CRITERIA:
            branch_v = (branch_lab.get(crit) or "").strip().lower()
            clean_v = (clean_lab.get(crit) or "").strip().lower() if clean_lab else ""
            # Treat missing clean as "uncertain" for the purposes of this audit
            # rule — we cannot compute the flip without it.
            if clean_v == "" or clean_v == "uncertain" or branch_v == "uncertain":
                add_reason(k, "uncertain_for_candidate_flip")
                continue
            if clean_v not in ("yes", "no") or branch_v not in ("yes", "no"):
                # Field-validity violation already reported elsewhere; skip.
                continue
            if cond == "A→U":
                cand_flip = (clean_v == "no") and (branch_v == "yes")
            else:  # U→A
                cand_flip = (clean_v == "yes") and (branch_v == "no")
            if cand_flip:
                add_reason(k, f"candidate_flip_{crit}")

    # 4g. assemble audit rows
    audit_keys = sorted(audit_reasons.keys())
    for k in audit_keys:
        e = entries_by_key.get(k)
        if e is None:
            continue
        lab = labels_by_key.get(k, {})
        audit_rows.append({
            "key": k,
            "prompt": e.get("prompt", ""),
            "generation": e.get("generation", ""),
            "branch": e.get("branch", ""),
            "condition": e.get("condition", ""),
            "alpha": e.get("alpha", ""),
            "candidate_invalidity_aware": lab.get("invalidity_aware", ""),
            "candidate_refusal_only": lab.get("refusal_only", ""),
            "candidate_degenerate": lab.get("degenerate", ""),
            "audit_reason": ",".join(audit_reasons[k]),
            "final_invalidity_aware": "",
            "final_refusal_only": "",
            "final_degenerate": "",
        })

    # 4h. write audit TSV
    columns = [
        "key", "prompt", "generation", "branch", "condition", "alpha",
        "candidate_invalidity_aware", "candidate_refusal_only", "candidate_degenerate",
        "audit_reason",
        "final_invalidity_aware", "final_refusal_only", "final_degenerate",
    ]
    with audit_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t",
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in audit_rows:
            writer.writerow(row)

    # ── 5. Stdout report ───────────────────────────────────────────────
    print(f"[validate] universe size       : {len(universe)} (from batches)")
    print(f"[validate] labeled rows        : {len(labeled)}")
    print(f"[validate] missing labels      : {len(missing)}")
    if missing[:5]:
        for k in missing[:5]:
            print(f"           missing key       : {k}")
        if len(missing) > 5:
            print(f"           ... +{len(missing) - 5} more")
    print(f"[validate] extra label rows    : {len(extra)}")
    if extra[:5]:
        for k in extra[:5]:
            print(f"           extra key         : {k}")
        if len(extra) > 5:
            print(f"           ... +{len(extra) - 5} more")
    print(f"[validate] field violations    : {len(field_violations)}")
    for k, f, v in field_violations[:10]:
        print(f"           bad value         : {k}  {f}={v!r}")
    if len(field_violations) > 10:
        print(f"           ... +{len(field_violations) - 10} more")
    print(f"[validate] schema errors (refusal_only=yes & invalidity_aware!=yes): {len(schema_errors)}")
    for k, why in schema_errors[:10]:
        print(f"           schema offender   : {k}  {why}")
    if len(schema_errors) > 10:
        print(f"           ... +{len(schema_errors) - 10} more")
    print(f"[validate] uncertain counts    : invalidity_aware={uncertain_counts['invalidity_aware']}  "
          f"refusal_only={uncertain_counts['refusal_only']}")
    print(f"[validate] audit rows          : {len(audit_rows)} → {audit_out}")
    reason_counts: Counter = Counter()
    for reasons in audit_reasons.values():
        for r in reasons:
            reason_counts[r] += 1
    if reason_counts:
        print("[validate] audit reason breakdown (rows hit by each reason):")
        for reason in sorted(reason_counts.keys()):
            print(f"             {reason:36s} : {reason_counts[reason]}")

    # Non-zero exit if anything is wrong with required structure.
    if missing or extra or field_violations or schema_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
