"""
prepare_squad2.py — Build matched-pair prompts.jsonl from SQuAD 2.0.

SQuAD 2.0 adds unanswerable questions to SQuAD 1.1. Every paragraph contains
both answerable and unanswerable questions on the same context passage. We use
this structure to construct matched pairs: one A prompt and one U prompt that
share identical context, differing only in whether the question can be answered.

This gives natural matched pairs in the FACT domain — a real-world counterpart
to the hand-crafted FACT pairs in the workshop paper, allowing us to test
whether the geometric null result holds on naturally-occurring unanswerable QA.

Supports two input modes:
  (1) HuggingFace datasets library (auto-download)
  (2) Raw SQuAD 2.0 JSON file (if already downloaded)
      Download from: https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

Usage:
    # Mode 1: HuggingFace auto-download
    python src/data/prepare_squad2.py \\
        --out-dir experiments/squad2/runs/llama \\
        --n-pairs 50

    # Mode 2: raw JSON file
    python src/data/prepare_squad2.py \\
        --squad-json /path/to/dev-v2.0.json \\
        --out-dir experiments/squad2/runs/llama \\
        --n-pairs 50

After generating prompts.jsonl, run the existing extraction + analysis pipeline:

    # NOTE: run_extract.py and analyze_controlled.py have been retired; this
    # docstring is kept for historical context. Current pipeline uses
    # scripts/run_extract_minimal.py.
    # Step 1: extract representations from the repo root using the current pipeline.
    python run_extract.py --model llama   --run-dir experiments/squad2/runs/llama
    python run_extract.py --model qwen    --run-dir experiments/squad2/runs/qwen
    python run_extract.py --model mistral --run-dir experiments/squad2/runs/mistral

    # Step 2: analyze
    python analyze_controlled.py \\
        --run-dirs experiments/squad2/runs/llama \\
        --label squad2_llama --forms FACT

Output (in --out-dir):
    prompts.jsonl    2 × n_pairs rows, form="FACT", answerable="A"/"U"

Design notes:
    - Context truncated to MAX_CONTEXT_WORDS to keep prompt length comparable
      to workshop FACT prompts and avoid atypical tokenization.
    - At most MAX_PAIRS_PER_ARTICLE pairs are drawn from any single article,
      ensuring topical diversity across the dataset.
    - form="FACT" for pipeline compatibility and form-conditionality comparison.
    - A questions are only admitted when at least one official answer span is
      fully preserved after truncation (strict span check, not substring match).
    - The expected result is a null signal (p > 0.3, Cohen's d < 0.5),
      consistent with the workshop FACT null result. A null here validates that
      the FACT null is not an artifact of synthetic prompt construction.
"""

import argparse
import json
import os
import random

MAX_CONTEXT_WORDS    = 150   # truncate context to this many whitespace-split words
MAX_PAIRS_PER_ARTICLE = 3    # cap pairs per article for topical diversity

PROMPT_TEMPLATE = (
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer:"
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _word_boundary(text, max_words=MAX_CONTEXT_WORDS):
    """Return the char offset just past the last character of the max_words-th
    whitespace-delimited word in *text*, scanning the original string so that
    all original whitespace (tabs, multiple spaces, newlines) is preserved.

    Returns len(text) if the text has <= max_words words.
    """
    n_words = 0
    pos = 0
    length = len(text)
    while pos < length and n_words < max_words:
        # skip whitespace
        while pos < length and text[pos].isspace():
            pos += 1
        if pos >= length:
            break
        # skip word
        while pos < length and not text[pos].isspace():
            pos += 1
        n_words += 1
    return pos


def truncate(text, max_words=MAX_CONTEXT_WORDS):
    """Truncate *text* to at most *max_words* whitespace-delimited words.

    Unlike a naive ``" ".join(text.split()[:max_words])``, this preserves the
    original text verbatim up to the truncation point — tabs, multiple spaces,
    and other whitespace are kept intact.  This is critical for keeping SQuAD
    ``answer_start`` character offsets valid in the truncated string.
    """
    if len(text.split()) <= max_words:
        return text
    boundary = _word_boundary(text, max_words)
    return text[:boundary] + " ..."


def answer_span_survives_truncation(context, answer_start, answer_text,
                                    max_words=MAX_CONTEXT_WORDS):
    """Return True iff the official answer span is fully preserved in the
    truncated context AND the span literally matches at the original offset.

    Two checks:
    1. The span [answer_start, answer_start+len) falls within the kept prefix.
    2. context[answer_start:answer_start+len] == answer_text (literal alignment).
    """
    boundary = _word_boundary(context, max_words)
    span_end = answer_start + len(answer_text)
    if span_end > boundary:
        return False
    # Literal alignment check — catches whitespace-normalization mismatches
    return context[answer_start:span_end] == answer_text


# ── data loading ─────────────────────────────────────────────────────────────

def load_from_hf(split="validation"):
    """Load SQuAD 2.0 via HuggingFace datasets."""
    from datasets import load_dataset
    print(f"Downloading SQuAD 2.0 ({split}) from HuggingFace...")
    ds = load_dataset("rajpurkar/squad_v2", split=split)
    print(f"  {len(ds)} QA items loaded.")
    return [
        {
            "title":        row["title"],
            "context":      row["context"],
            "question":     row["question"],
            "is_impossible": len(row["answers"]["text"]) == 0,
            "answers": [
                {"text": t, "answer_start": s}
                for t, s in zip(row["answers"]["text"],
                                row["answers"]["answer_start"])
            ],
        }
        for row in ds
    ]


def load_from_json(path):
    """Load SQuAD 2.0 from the official raw JSON file (dev-v2.0.json etc.)."""
    print(f"Loading SQuAD 2.0 from {path} ...")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for article in raw["data"]:
        title = article["title"]
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                rows.append({
                    "title":         title,
                    "context":       context,
                    "question":      qa["question"],
                    "is_impossible": qa.get("is_impossible", False),
                    "answers": [
                        {"text": a["text"], "answer_start": a["answer_start"]}
                        for a in qa.get("answers", [])
                    ],
                })
    print(f"  {len(rows)} QA items loaded.")
    return rows


# ── pair construction ─────────────────────────────────────────────────────────

def build_pairs(rows, n_pairs, seed=42):
    """
    Group QAs by paragraph context. For each paragraph with at least one
    answerable (is_impossible=False) and one unanswerable (is_impossible=True)
    question, create one matched pair. Return up to n_pairs pairs.
    """
    rng = random.Random(seed)

    # Group by (title, context) — context uniquely identifies a paragraph
    para_map = {}
    for row in rows:
        key = (row["title"], row["context"])
        if key not in para_map:
            para_map[key] = {"title": row["title"], "context": row["context"],
                             "A": [], "U": []}
        if row["is_impossible"]:
            para_map[key]["U"].append(row["question"])
        else:
            para_map[key]["A"].append({
                "question": row["question"],
                "answers":  row["answers"],
            })

    # Keep only paragraphs with both answerable and unanswerable questions
    eligible = [v for v in para_map.values() if v["A"] and v["U"]]
    rng.shuffle(eligible)
    print(f"  {len(eligible)} eligible paragraphs (have both A and U questions).")

    pairs = []
    article_counts = {}
    for para in eligible:
        if len(pairs) >= n_pairs:
            break
        title = para["title"]
        if article_counts.get(title, 0) >= MAX_PAIRS_PER_ARTICLE:
            continue

        context = truncate(para["context"])

        # Strict span check: only keep A questions where at least one
        # official answer span is fully preserved after truncation.
        valid_a = [
            a for a in para["A"]
            if any(
                answer_span_survives_truncation(
                    para["context"],
                    ans["answer_start"],
                    ans["text"],
                )
                for ans in a["answers"]
            )
        ]
        if not valid_a:
            continue

        a_item = rng.choice(valid_a)
        q_u = rng.choice(para["U"])
        pairs.append({"context": context, "q_a": a_item["question"], "q_u": q_u})
        article_counts[title] = article_counts.get(title, 0) + 1

    return pairs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare SQuAD 2.0 matched-pair prompts for the EMNLP pipeline."
    )
    parser.add_argument(
        "--squad-json", default=None,
        help="Path to raw SQuAD 2.0 JSON (e.g. dev-v2.0.json). "
             "If omitted, downloads via HuggingFace datasets."
    )
    parser.add_argument(
        "--split", default="validation", choices=["train", "validation"],
        help="HuggingFace split to use (ignored if --squad-json is given). Default: validation."
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to write output file (created if absent). Default: dirname of --out-file."
    )
    parser.add_argument(
        "--out-file", default=None,
        help="Full output path, e.g. data/fact800.jsonl. Overrides --out-dir filename."
    )
    parser.add_argument(
        "--n-pairs", type=int, default=50,
        help="Number of matched pairs to sample (default: 50)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)."
    )
    args = parser.parse_args()

    # Load data
    if args.squad_json:
        rows = load_from_json(args.squad_json)
    else:
        try:
            rows = load_from_hf(split=args.split)
        except ImportError:
            raise ImportError(
                "Install HuggingFace datasets with:\n"
                "    pip install datasets\n"
                "Or download dev-v2.0.json manually and pass --squad-json <path>"
            )

    # Build pairs
    pairs = build_pairs(rows, args.n_pairs, seed=args.seed)
    if len(pairs) < args.n_pairs:
        print(f"  Warning: only {len(pairs)} pairs found "
              f"(requested {args.n_pairs}). "
              f"Try --split train for more data.")
    print(f"  {len(pairs)} matched pairs ready.")

    # Resolve output path
    if args.out_file:
        out_path = args.out_file
    elif args.out_dir:
        out_path = os.path.join(args.out_dir, "prompts.jsonl")
    else:
        raise ValueError("Specify --out-file or --out-dir")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    rows_out = []
    for i, pair in enumerate(pairs, start=1):
        pid = f"sq{i:03d}"
        rows_out.append({
            "id":         f"{pid}a",
            "form":       "FACT",
            "answerable": "A",
            "prompt":     PROMPT_TEMPLATE.format(
                              context=pair["context"], question=pair["q_a"]),
        })
        rows_out.append({
            "id":         f"{pid}u",
            "form":       "FACT",
            "answerable": "U",
            "prompt":     PROMPT_TEMPLATE.format(
                              context=pair["context"], question=pair["q_u"]),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows_out)} prompts ({len(pairs)} A/U pairs) → {out_path}")
    print(f"\n{'='*60}")
    print(f"Next steps (run from EMNLP_2026 repo root):")
    print(f"")
    stem = os.path.splitext(os.path.basename(out_path))[0]  # e.g. fact800
    print(f"  # Extract representations for the three models with the current minimal extractor")
    print(f"  python scripts/run_extract_minimal.py --model llama   --prompts {out_path} --run-dir experiments/signals/{stem}_llama_allL   --forms FACT --all-layers --no-gradients")
    print(f"  python scripts/run_extract_minimal.py --model qwen    --prompts {out_path} --run-dir experiments/signals/{stem}_qwen_allL    --forms FACT --all-layers --no-gradients")
    print(f"  python scripts/run_extract_minimal.py --model mistral --prompts {out_path} --run-dir experiments/signals/{stem}_mistral_allL --forms FACT --all-layers --no-gradients")
    print(f"")
    print(f"  # Analyze (analyze_controlled.py has been retired; current pipeline runs)")
    print(f"  python scripts/analyze_form_conditionality.py --model llama --dataset {stem}")
    print(f"  python scripts/compare_impossibility_vs_refusal_direction.py --model llama --dataset {stem}")
    print(f"{'='*60}")
    print(f"\nExpected result: p > 0.3, Cohen's d < 0.5 (null signal),")
    print(f"consistent with workshop FACT null result.")


if __name__ == "__main__":
    main()
