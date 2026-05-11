"""
fix_fact800.py — Fix 22 truncated A samples in fact800.jsonl.

The original prepare_squad2.py truncated context to 150 words but didn't check
if the A answer span was still in the truncated context. This script replaces
the 22 affected pairs with new ones from SQuAD 2.0 train split.

Usage (from the repo root):
    source .venv/bin/activate
    python src/data/fix_fact800.py
"""

import json
import random
import re
import sys

MAX_CONTEXT_WORDS = 150
MAX_PAIRS_PER_ARTICLE = 3

PROMPT_TEMPLATE = (
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer:"
)

AFFECTED_IDS = [
    "sq003", "sq075", "sq167", "sq187", "sq189", "sq205", "sq255", "sq267",
    "sq294", "sq338", "sq348", "sq372", "sq467", "sq475", "sq514", "sq516",
    "sq583", "sq669", "sq706", "sq761", "sq769", "sq791",
]

INPUT = OUTPUT = "data/fact800.jsonl"


def truncate(text, max_words=MAX_CONTEXT_WORDS):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def extract_context(prompt):
    """Extract context from a fact800 prompt."""
    m = re.match(r"Context: (.*?)\nQuestion:", prompt, re.DOTALL)
    return m.group(1) if m else None


def main():
    print("Loading fact800.jsonl...")
    with open(INPUT) as f:
        entries = [json.loads(l) for l in f]

    by_id = {e["id"]: e for e in entries}

    # Collect existing contexts to avoid duplicates
    existing_contexts = set()
    for e in entries:
        ctx = extract_context(e["prompt"])
        if ctx:
            existing_contexts.add(ctx)

    # Count existing article usage (approximation: first 50 chars of context)
    # We need to track which articles are used to respect MAX_PAIRS_PER_ARTICLE
    # But since we're replacing existing pairs, the count stays the same.

    # Load SQuAD 2.0 train split
    print("Loading SQuAD 2.0 train split from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="train")
    print(f"  {len(ds)} QA items loaded.")

    # Group by (title, context)
    para_map = {}
    for row in ds:
        key = (row["title"], row["context"])
        if key not in para_map:
            para_map[key] = {
                "title": row["title"],
                "context": row["context"],
                "A": [],  # (question, answer_text)
                "U": [],  # (question,)
            }
        if len(row["answers"]["text"]) == 0:
            para_map[key]["U"].append(row["question"])
        else:
            para_map[key]["A"].append({
                "question": row["question"],
                "answer_text": row["answers"]["text"][0],
            })

    # Filter eligible paragraphs:
    # 1. Has both A and U questions
    # 2. Context ≤ 150 words OR after truncation, at least one A answer is still in context
    # 3. Context (truncated) not already in fact800
    eligible = []
    for para in para_map.values():
        if not para["A"] or not para["U"]:
            continue

        ctx_truncated = truncate(para["context"])

        # Skip if this context is already used
        if ctx_truncated in existing_contexts:
            continue

        # Find A questions whose answer is still in the truncated context
        valid_a = [
            a for a in para["A"]
            if a["answer_text"] and a["answer_text"] in ctx_truncated
        ]
        if not valid_a:
            continue

        eligible.append({
            "title": para["title"],
            "context_truncated": ctx_truncated,
            "valid_a": valid_a,
            "u_questions": para["U"],
        })

    print(f"  {len(eligible)} eligible paragraphs for replacement.")

    # Sample replacements
    rng = random.Random(2026)
    rng.shuffle(eligible)

    # Track article usage for diversity
    # First, count how many pairs each article already has (excluding affected ones)
    article_counts = {}
    affected_set = set()
    for pid in AFFECTED_IDS:
        affected_set.add(f"{pid}a")
        affected_set.add(f"{pid}u")

    # We can't easily recover article titles from the current data, so we'll
    # just enforce MAX_PAIRS_PER_ARTICLE on the new replacements
    new_article_counts = {}

    replacements = []
    for para in eligible:
        if len(replacements) >= len(AFFECTED_IDS):
            break

        title = para["title"]
        if new_article_counts.get(title, 0) >= MAX_PAIRS_PER_ARTICLE:
            continue

        a_item = rng.choice(para["valid_a"])
        u_question = rng.choice(para["u_questions"])

        replacements.append({
            "context": para["context_truncated"],
            "q_a": a_item["question"],
            "q_u": u_question,
            "answer_text": a_item["answer_text"],
            "title": title,
        })
        new_article_counts[title] = new_article_counts.get(title, 0) + 1

    print(f"  Sampled {len(replacements)} replacement pairs.")

    if len(replacements) < len(AFFECTED_IDS):
        print(f"  ERROR: not enough replacements! Need {len(AFFECTED_IDS)}, got {len(replacements)}")
        sys.exit(1)

    # Apply replacements
    print("\n=== Applying replacements ===")
    for i, pid in enumerate(AFFECTED_IDS):
        repl = replacements[i]
        aid = f"{pid}a"
        uid = f"{pid}u"

        old_a = by_id[aid]
        old_u = by_id[uid]

        new_a_prompt = PROMPT_TEMPLATE.format(
            context=repl["context"], question=repl["q_a"]
        )
        new_u_prompt = PROMPT_TEMPLATE.format(
            context=repl["context"], question=repl["q_u"]
        )

        old_a["prompt"] = new_a_prompt
        old_u["prompt"] = new_u_prompt

        # Verify answer is in context
        assert repl["answer_text"] in repl["context"], \
            f"{aid}: answer '{repl['answer_text'][:30]}' not in context!"

        print(f"  {pid}: [{repl['title'][:40]}] "
              f"Q_A: {repl['q_a'][:50]}... "
              f"Ans: {repl['answer_text'][:30]}")

    # ── Full verification ────────────────────────────────────────────────
    print("\n=== Full verification: checking all 800 A entries ===")
    # Re-load SQuAD to verify answers
    # Build a lookup: context+question -> answer_text
    squad_lookup = {}
    for row in ds:
        if len(row["answers"]["text"]) > 0:
            ctx_trunc = truncate(row["context"])
            key = (ctx_trunc, row["question"])
            squad_lookup[key] = row["answers"]["text"][0]

    # For the original entries that weren't replaced, we need to check too
    # But we don't have their original answer_text stored in fact800.jsonl
    # So let's just verify the new replacements have answers in context
    # and do a broader check using the squad lookup

    verified_ok = 0
    verified_fail = 0
    verified_no_lookup = 0

    for e in entries:
        if e["answerable"] != "A":
            continue
        ctx = extract_context(e["prompt"])
        q_match = re.search(r"Question: (.*?)\nAnswer:", e["prompt"], re.DOTALL)
        question = q_match.group(1) if q_match else ""

        key = (ctx, question)
        if key in squad_lookup:
            answer = squad_lookup[key]
            if answer in ctx:
                verified_ok += 1
            else:
                verified_fail += 1
                print(f"  FAIL: {e['id']}: answer '{answer[:40]}' not in context")
        else:
            # Can't verify (original entry not in train split - might be from validation)
            verified_no_lookup += 1

    print(f"  Verified: {verified_ok} OK, {verified_fail} FAIL, {verified_no_lookup} no lookup")

    # For entries without lookup, let's at least check no more truncated contexts
    # have the issue (the 22 we fixed should now all be OK)
    for pid in AFFECTED_IDS:
        aid = f"{pid}a"
        e = by_id[aid]
        ctx = extract_context(e["prompt"])
        q_match = re.search(r"Question: (.*?)\nAnswer:", e["prompt"], re.DOTALL)
        question = q_match.group(1) if q_match else ""
        key = (ctx, question)
        if key in squad_lookup:
            answer = squad_lookup[key]
            if answer not in ctx:
                print(f"  CRITICAL: {aid} still has truncated answer!")
                sys.exit(1)

    # ── Write output ─────────────────────────────────────────────────────
    print(f"\n=== Writing {OUTPUT} (in-place) ===")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  Written {len(entries)} entries")

    # Summary
    a_count = sum(1 for e in entries if e["answerable"] == "A")
    u_count = sum(1 for e in entries if e["answerable"] == "U")
    print(f"\n=== Summary ===")
    print(f"  Total: {len(entries)} ({a_count} A, {u_count} U)")
    print(f"  Replaced: {len(AFFECTED_IDS)} pairs (A+U)")
    print(f"\n== Done! ==")


if __name__ == "__main__":
    main()
