#!/usr/bin/env python3
"""
fetch_falseqa.py — Download FalseQA from upstream and convert to the JSONL
schema used by this repo.

WHY THIS SCRIPT EXISTS
----------------------
At the time of this anonymous release, the upstream FalseQA repository
(github.com/thunlp/FalseQA) does not contain an explicit LICENSE file.
Per default copyright law, this means no automatic redistribution
permission is granted. To stay clearly within fair-use academic
reproducibility, this repo does NOT ship `data/falseqa.jsonl`; instead,
this script downloads the data directly from the upstream source so the
reviewer obtains it under whatever terms the upstream authors apply.

By running this script you are accepting that you download FalseQA
yourself from the upstream repository.

USAGE
-----
    python src/data/fetch_falseqa.py    # writes data/falseqa.jsonl (1374 rows)
    python src/data/clean_falseqa.py    # applies the local cleanup pass

EXPECTED OUTPUT
---------------
data/falseqa.jsonl with 1374 rows (687 answerable + 687 false-premise),
schema per row:
    {
      "id":             "fqa_NNNNa" | "fqa_NNNNu",   # zero-padded 4-digit;
                                                     # U: 0000u..0686u,
                                                     # A: 0687a..1373a
      "form":           "QA",
      "answerable":     "A" | "U",
      "category":       "true_premise" | "false_premise",   # A→true, U→false
      "prompt":         "Answer concisely: <question>",
      "source_dataset": "falseqa"
    }

REQUIREMENTS
------------
- `git` available on PATH
- Python 3.9+
- Network access to github.com

If upstream layout has changed since this script was written and
auto-detection fails, the script prints the upstream `dataset/` listing
and exits with a clear error so you can adjust the input-file selection
manually.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
UPSTREAM_REPO = "https://github.com/thunlp/FalseQA.git"
OUT_PATH = Path("data/falseqa.jsonl")
EXPECTED_TOTAL_ROWS = 1374  # paper App. A.5: 687 A + 687 U
EXPECTED_PER_LABEL = 687
PROMPT_PREFIX = "Answer concisely: "


# ---------------------------------------------------------------------------
def clone_upstream(target_dir: Path) -> None:
    """Shallow-clone the upstream FalseQA repo into target_dir."""
    print(f"[fetch_falseqa] cloning {UPSTREAM_REPO} into temp dir ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", UPSTREAM_REPO, str(target_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[fetch_falseqa] git clone failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)


def find_dataset_files(clone_dir: Path) -> list[Path]:
    """Locate FalseQA data file(s). Returns candidates in priority order."""
    # Upstream layout at time of writing: dataset/{train,valid,test}.csv
    # and/or dataset/false_qa.json. We try the test split first since the
    # paper's 1374 rows corresponds to FalseQA's test split as of 2023.
    dataset_dir = clone_dir / "dataset"
    if not dataset_dir.exists():
        return []

    print(f"[fetch_falseqa] upstream dataset/ listing:")
    for p in sorted(dataset_dir.iterdir()):
        print(f"  {p.name}")

    # Priority: test → valid → train → any csv/json
    name_priority = [
        "test.csv", "test.json", "test.jsonl",
        "valid.csv", "valid.json", "valid.jsonl",
        "train.csv", "train.json", "train.jsonl",
        "false_qa.json", "falseqa.json", "FalseQA.json",
    ]
    found = []
    for name in name_priority:
        p = dataset_dir / name
        if p.exists():
            found.append(p)
    # Fallback: any csv / json in dataset/
    if not found:
        for p in sorted(dataset_dir.iterdir()):
            if p.suffix.lower() in {".csv", ".json", ".jsonl"} and p.is_file():
                found.append(p)
    return found


def load_upstream_rows(path: Path) -> list[dict]:
    """Read an upstream file as a list of dicts. Handles csv / json / jsonl."""
    print(f"[fetch_falseqa] reading {path.name} ...")
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    if path.suffix.lower() == ".json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            # Try common wrapping keys
            for key in ("data", "examples", "rows", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            raise ValueError(
                f"{path.name} is a dict; expected list or known wrapper key."
            )
        if isinstance(data, list):
            return data
        raise ValueError(f"{path.name} top-level type {type(data)} not handled.")
    if path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    raise ValueError(f"unrecognised extension for {path}")


def normalize_question(text: str) -> str:
    """Normalize whitespace; do NOT apply the dedup/punctuation cleanups —
    that's clean_falseqa.py's job. Just collapse internal whitespace."""
    return re.sub(r"\s+", " ", text.strip())


def detect_question_field(rows: list[dict]) -> str:
    """Find which field holds the question text. Try common names."""
    candidates = ["question", "Question", "prompt", "text", "Q"]
    if not rows:
        raise ValueError("upstream returned zero rows")
    keys = list(rows[0].keys())
    for c in candidates:
        if c in keys:
            return c
    raise ValueError(
        f"could not auto-detect question field; upstream row 0 keys: {keys}"
    )


def detect_label_field(rows: list[dict]) -> str:
    """Find which field holds the FalseQA label. Hardcoded for FalseQA.

    Per upstream README at github.com/thunlp/FalseQA: label=1 is a
    false-premise question, label=0 is a true-premise question. We do
    NOT use a generic auto-detection heuristic here because numeric
    labels (0/1) are convention-dependent and previously generated
    silently-inverted output when an auto-detection guess matched the
    wrong convention.
    """
    if not rows:
        raise ValueError("upstream returned zero rows")
    keys = list(rows[0].keys())
    for fname in ("label", "Label"):
        if fname in keys:
            return fname
    raise ValueError(
        f"could not find 'label' field; upstream row 0 keys: {keys}\n"
        "FalseQA upstream is expected to have a 'label' column with values "
        "0 (true premise) / 1 (false premise). Adjust if upstream changed."
    )


# FalseQA's documented label semantics — DO NOT change without verifying
# upstream README. Per github.com/thunlp/FalseQA: 1=false_premise, 0=true_premise.
FALSEQA_LABEL_MAP = {
    "1": ("U", "false_premise"),
    "0": ("A", "true_premise"),
}


def convert(upstream_rows: list[dict]) -> list[dict]:
    """Convert upstream rows → our schema.

    Critical FalseQA-specific invariants (do NOT change without verifying
    the canonical data/falseqa.jsonl produced by the paper):
      - label=1 → answerable="U", category="false_premise"
      - label=0 → answerable="A", category="true_premise"
      - ID numbering is CONTINUOUS across U then A:
          U rows: fqa_0000u .. fqa_0686u
          A rows: fqa_0687a .. fqa_1373a
        (clean_falseqa.py has hardcoded rewrites referencing fqa_0981a,
        fqa_0697a, fqa_0294u, fqa_0010u — these IDs only exist under
        continuous numbering. Resetting A's counter to fqa_0000a would
        silently break the cleanup pass.)
      - Upstream row order within each label class is PRESERVED.
    """
    qfield = detect_question_field(upstream_rows)
    lfield = detect_label_field(upstream_rows)
    print(f"[fetch_falseqa] question field='{qfield}', label field='{lfield}'")
    print(f"[fetch_falseqa] hardcoded FalseQA label map: 1→U/false_premise, 0→A/true_premise")

    u_rows, a_rows = [], []
    for r in upstream_rows:
        raw_label = str(r.get(lfield, "")).strip()
        if raw_label not in FALSEQA_LABEL_MAP:
            continue  # skip unrecognised labels
        au, category = FALSEQA_LABEL_MAP[raw_label]
        q = normalize_question(str(r.get(qfield, "")))
        if not q:
            continue
        out = {
            "form": "QA",
            "answerable": au,
            "category": category,
            "prompt": PROMPT_PREFIX + q,
            "source_dataset": "falseqa",
        }
        if au == "U":
            u_rows.append(out)
        else:
            a_rows.append(out)

    # Continuous ID numbering: U first (0000u..NNNNu), then A continues
    # (NNNN+1 .. NNNN+M). This matches the canonical falseqa.jsonl ID layout
    # that clean_falseqa.py's hardcoded edits depend on.
    n_u = len(u_rows)
    for i, r in enumerate(u_rows):
        r["id"] = f"fqa_{i:04d}u"
    for i, r in enumerate(a_rows):
        r["id"] = f"fqa_{n_u + i:04d}a"

    return u_rows + a_rows


def write_output(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[fetch_falseqa] wrote {len(rows)} rows to {out_path}")


def validate(rows: list[dict]) -> None:
    """Validate the converted rows match the canonical falseqa.jsonl invariants.

    Beyond the count check, we verify the structural invariants that
    clean_falseqa.py implicitly assumes:
      - A rows all have category="true_premise"; U rows all have
        category="false_premise" (label-mapping integrity)
      - The IDs clean_falseqa.py hardcodes (fqa_0981a, fqa_0697a,
        fqa_0294u, fqa_0010u) all exist after fetch
    If any of these fail, the cleanup pass would silently produce
    wrong data; we exit non-zero so the user sees the problem.
    """
    a_count = sum(1 for r in rows if r["answerable"] == "A")
    u_count = sum(1 for r in rows if r["answerable"] == "U")
    print(f"[fetch_falseqa] counts: A={a_count}, U={u_count}, total={len(rows)}")

    fatal = []

    if len(rows) != EXPECTED_TOTAL_ROWS:
        fatal.append(
            f"expected {EXPECTED_TOTAL_ROWS} total rows; got {len(rows)}"
        )
    if a_count != EXPECTED_PER_LABEL or u_count != EXPECTED_PER_LABEL:
        fatal.append(
            f"expected A=U={EXPECTED_PER_LABEL}; got A={a_count}, U={u_count}"
        )

    # Category-vs-label integrity
    bad_a = [r["id"] for r in rows if r["answerable"] == "A" and r["category"] != "true_premise"]
    bad_u = [r["id"] for r in rows if r["answerable"] == "U" and r["category"] != "false_premise"]
    if bad_a:
        fatal.append(
            f"{len(bad_a)} A row(s) have category != 'true_premise' "
            f"(first 3: {bad_a[:3]})"
        )
    if bad_u:
        fatal.append(
            f"{len(bad_u)} U row(s) have category != 'false_premise' "
            f"(first 3: {bad_u[:3]})"
        )

    # Canonical ID presence — clean_falseqa.py hardcodes these
    by_id = {r["id"]: r for r in rows}
    required_ids = {
        "fqa_0981a": "A",  # cross-label duplicate fix target (clean script rewrites this row)
        "fqa_0294u": "U",  # the matching U-side duplicate (clean script leaves this untouched)
        "fqa_0697a": "A",  # 'If i' → 'If I' capitalization fix target
        "fqa_0010u": "U",  # same fix, U side
    }
    for rid, expected_au in required_ids.items():
        if rid not in by_id:
            fatal.append(
                f"canonical ID '{rid}' missing from output — clean_falseqa.py's "
                f"hardcoded edit for this row would silently no-op"
            )
            continue
        actual_au = by_id[rid]["answerable"]
        if actual_au != expected_au:
            fatal.append(
                f"canonical ID '{rid}' has answerable={actual_au!r}, expected {expected_au!r}"
            )

    if fatal:
        print("[fetch_falseqa] VALIDATION FAILED:", file=sys.stderr)
        for msg in fatal:
            print(f"  - {msg}", file=sys.stderr)
        print(
            "\nThe upstream may have changed since this script was written, "
            "or auto-detection picked the wrong upstream file. Inspect the "
            "candidate file used (printed above) and adjust convert() / "
            "find_dataset_files() as needed. Do NOT run clean_falseqa.py on "
            "the output — its hardcoded edits would target the wrong rows.",
            file=sys.stderr,
        )
        sys.exit(2)

    print("[fetch_falseqa] validation OK")


# ---------------------------------------------------------------------------
def main() -> None:
    if shutil.which("git") is None:
        print("[fetch_falseqa] error: `git` not on PATH", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        clone_dir = Path(tmp) / "FalseQA"
        clone_upstream(clone_dir)

        candidates = find_dataset_files(clone_dir)
        if not candidates:
            print(
                "[fetch_falseqa] error: could not find any dataset file in "
                "upstream `dataset/` directory. Inspect the upstream repo "
                "and adjust find_dataset_files() in this script.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Try each candidate until one yields the expected row count.
        last_rows = None
        for cand in candidates:
            try:
                upstream_rows = load_upstream_rows(cand)
                converted = convert(upstream_rows)
                if len(converted) == EXPECTED_TOTAL_ROWS:
                    print(f"[fetch_falseqa] {cand.name} matched expected row count")
                    last_rows = converted
                    break
                else:
                    print(
                        f"[fetch_falseqa] {cand.name} → {len(converted)} rows "
                        f"(expected {EXPECTED_TOTAL_ROWS}); trying next candidate"
                    )
                    last_rows = converted  # keep latest non-empty for fallback
            except Exception as exc:
                print(f"[fetch_falseqa] skipping {cand.name}: {exc}")

        if last_rows is None:
            print(
                "[fetch_falseqa] error: no candidate upstream file converted "
                "successfully. Adjust the auto-detection logic.",
                file=sys.stderr,
            )
            sys.exit(1)

        validate(last_rows)
        write_output(last_rows, OUT_PATH)
        print(
            "[fetch_falseqa] done. Next step: run "
            "`python src/data/clean_falseqa.py` to apply the documented "
            "cleanup pass (see paper Appendix A.5)."
        )


if __name__ == "__main__":
    main()
