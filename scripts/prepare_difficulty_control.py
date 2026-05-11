"""
prepare_difficulty_control.py — Build a difficulty control dataset from GSM8K.

Purpose: Negative control for the "difficulty vs unanswerability" confound.
All prompts are ANSWERABLE math word problems from GSM8K test split.
Split into easy (1-2 solution steps) vs hard (≥5 steps) via `difficulty_label`.

Schema note:
  - `answerable` is ALWAYS "A" — every prompt is genuinely answerable.
  - `difficulty_label` ("easy" / "hard") is the experimental variable.
  - This dataset does NOT use the A/U convention of the main datasets.
  - A dedicated evaluation script (eval_difficulty_control.py) reads
    `difficulty_label` for binary classification, NOT `answerable`.

If the impossibility direction does NOT separate easy from hard (AUC ≈ 0.5),
we can conclude it encodes unanswerability, not difficulty.

Note on overlap:
  - This dataset draws from the same GSM8K test split as abstentionbench_gsm8k.
  - ~94% of prompts overlap with abstentionbench A-class prompts.
  - This is a same-source difficulty probe, NOT an independent external dataset.

Design decisions:
  - Step count = number of "<<...>>" calculation markers in the answer field
  - easy: 1 ≤ n_steps ≤ 2  (0-step anomalies excluded — missing markers)
  - hard: n_steps ≥ 5
  - Balanced: min(N_easy, N_hard) per group, capped at 200
  - Prompt format: "Answer concisely: {question}" (matches math800/abstentionbench)

Output: data/difficulty_control_gsm8k.jsonl
"""

import json
import os
import re
import random

from datasets import load_dataset

# ── config ──────────────────────────────────────────────────
EASY_MIN_STEPS = 1   # exclude 0-step anomalies (missing <<>> markers)
EASY_MAX_STEPS = 2
HARD_MIN_STEPS = 5
MAX_PER_GROUP = 200
SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT_FILE = os.path.join(OUT_DIR, "difficulty_control_gsm8k.jsonl")

# ── helpers ─────────────────────────────────────────────────

def count_steps(answer_text: str) -> int:
    """Count solution steps by <<...>> calculation markers in GSM8K answers."""
    return len(re.findall(r"<<[^>]+>>", answer_text))


def extract_final_answer(answer_text: str) -> str:
    """Extract the numeric answer after #### in GSM8K format."""
    match = re.search(r"####\s*(.+)", answer_text)
    return match.group(1).strip() if match else answer_text.strip()


# ── main ────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Difficulty Control Dataset — GSM8K easy vs hard")
    print("All prompts are ANSWERABLE. difficulty_label = easy/hard.")
    print("=" * 60)

    # Load GSM8K test split
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    print(f"Loaded GSM8K test split: {len(gsm8k)} problems")

    # Count steps for each problem
    easy_pool = []
    hard_pool = []
    step_counts = []

    for i, q in enumerate(gsm8k):
        n_steps = count_steps(q["answer"])
        step_counts.append(n_steps)

        entry = {
            "gsm8k_idx": i,
            "question": q["question"],
            "answer_text": q["answer"],
            "final_answer": extract_final_answer(q["answer"]),
            "n_steps": n_steps,
        }

        if EASY_MIN_STEPS <= n_steps <= EASY_MAX_STEPS:
            easy_pool.append(entry)
        elif n_steps >= HARD_MIN_STEPS:
            hard_pool.append(entry)

    print(f"\nStep count distribution (all {len(gsm8k)} problems):")
    from collections import Counter
    dist = Counter(step_counts)
    for k in sorted(dist.keys()):
        print(f"  {k} steps: {dist[k]} problems")

    print(f"\nEasy pool ({EASY_MIN_STEPS}≤steps≤{EASY_MAX_STEPS}): {len(easy_pool)}")
    print(f"Hard pool (steps≥{HARD_MIN_STEPS}): {len(hard_pool)}")

    # Balance and sample
    n_per_group = min(len(easy_pool), len(hard_pool), MAX_PER_GROUP)
    print(f"Sampling {n_per_group} per group (seed={SEED})")

    rng = random.Random(SEED)
    easy_sample = rng.sample(easy_pool, n_per_group)
    hard_sample = rng.sample(hard_pool, n_per_group)

    # Build JSONL records — ALL answerable, difficulty_label distinguishes groups
    os.makedirs(OUT_DIR, exist_ok=True)
    records = []

    for i, e in enumerate(easy_sample):
        records.append({
            "id": f"diffctrl_{i:04d}_easy",
            "form": "MATH",
            "answerable": "A",
            "difficulty_label": "easy",
            "category": "gsm8k_easy",
            "prompt": f"Answer concisely: {e['question']}",
            "source_dataset": "difficulty_control_gsm8k",
            "original_answer": e["final_answer"],
            "n_steps": e["n_steps"],
        })

    for i, e in enumerate(hard_sample):
        records.append({
            "id": f"diffctrl_{i:04d}_hard",
            "form": "MATH",
            "answerable": "A",
            "difficulty_label": "hard",
            "category": "gsm8k_hard",
            "prompt": f"Answer concisely: {e['question']}",
            "source_dataset": "difficulty_control_gsm8k",
            "original_answer": e["final_answer"],
            "n_steps": e["n_steps"],
        })

    # Verify no overlap with abstentionbench prompts (documentation purpose)
    abs_path = os.path.join(OUT_DIR, "abstentionbench_gsm8k.jsonl")
    if os.path.exists(abs_path):
        abs_prompts = set()
        with open(abs_path) as f:
            for line in f:
                abs_prompts.add(json.loads(line)["prompt"])
        overlap = sum(1 for r in records if r["prompt"] in abs_prompts)
        print(f"\nOverlap with abstentionbench_gsm8k: {overlap}/{len(records)} prompts")
        print("  (Expected: same GSM8K source. This is a same-source difficulty probe.)")

    with open(OUT_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n_easy = sum(1 for r in records if r["difficulty_label"] == "easy")
    n_hard = sum(1 for r in records if r["difficulty_label"] == "hard")
    print(f"\nWritten: {len(records)} records ({n_easy} easy, {n_hard} hard)")
    print(f"All answerable={set(r['answerable'] for r in records)}")
    print(f"Path: {OUT_FILE}")
    print(f"\nExample easy: {records[0]['prompt'][:100]}... ({records[0]['n_steps']} steps)")
    print(f"Example hard: {records[n_per_group]['prompt'][:100]}... ({records[n_per_group]['n_steps']} steps)")

    # ── Length confound analysis (character-level) ──
    print("\n" + "=" * 60)
    print("LENGTH CONFOUND ANALYSIS (character-level)")
    print("=" * 60)

    easy_recs = [r for r in records if r["difficulty_label"] == "easy"]
    hard_recs = [r for r in records if r["difficulty_label"] == "hard"]

    easy_cl = [len(r["prompt"]) for r in easy_recs]
    hard_cl = [len(r["prompt"]) for r in hard_recs]
    ratio = sum(hard_cl) / sum(easy_cl)
    print(f"Char length: easy mean={sum(easy_cl)/len(easy_cl):.0f}, "
          f"hard mean={sum(hard_cl)/len(hard_cl):.0f}, ratio={ratio:.2f}x")

    # Compute char-level length AUC
    n_correct = sum(1 for h in hard_cl for e in easy_cl if h > e)
    n_tie = sum(1 for h in hard_cl for e in easy_cl if h == e)
    char_length_auc = (n_correct + 0.5 * n_tie) / (len(hard_cl) * len(easy_cl))
    print(f"Char-level length AUC: {char_length_auc:.3f}")

    # Token-level length AUC (project standard: Mistral tokenizer)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.3", trust_remote_code=True
        )
        easy_tok = [len(tok.encode(r["prompt"])) for r in easy_recs]
        hard_tok = [len(tok.encode(r["prompt"])) for r in hard_recs]
        n_c = sum(1 for h in hard_tok for e in easy_tok if h > e)
        n_t = sum(1 for h in hard_tok for e in easy_tok if h == e)
        token_length_auc = (n_c + 0.5 * n_t) / (len(hard_tok) * len(easy_tok))
        print(f"Token-level length AUC (Mistral): {token_length_auc:.3f}")
    except Exception as e:
        print(f"Token-level length AUC: could not compute ({e})")
        print("  (Install transformers + download Mistral tokenizer to enable)")

    print("\n  Hard problems are longer — this is expected, as difficulty")
    print("  correlates with problem length. Key question: does the")
    print("  IMPOSSIBILITY direction also separate easy/hard?")
    print("  If CosNSRT AUC ≈ 0.5 despite length AUC ~0.75, the impossibility")
    print("  direction is specific to unanswerability, not difficulty/length.")

    # ── Usage ──
    print("\n" + "=" * 60)
    print("USAGE")
    print("=" * 60)
    print(f"""
Step 1 — Extract signals (same pipeline as core datasets, run from repo root):
  python scripts/run_extract_signals.py \\
    --model <MODEL> \\
    --prompts data/difficulty_control_gsm8k.jsonl \\
    --run-dir experiments/signals/difficulty_control_gsm8k_<MODEL>_allL \\
    --forms MATH --all-layers --no-gradients

Step 2 — Evaluate with dedicated script:
  python scripts/eval_difficulty_control.py --model <MODEL>

  This script:
    1. Loads math800 impossibility direction (d_imp)
    2. Projects difficulty_control reps onto d_imp
    3. Computes AUC(easy vs hard) using difficulty_label, NOT answerable
    4. Expected AUC ≈ 0.5 (impossibility direction ⊥ difficulty axis)
""")


if __name__ == "__main__":
    main()
