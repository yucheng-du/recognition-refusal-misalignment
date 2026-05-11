"""
clean_abstentionbench_gsm8k.py — Lightweight cleanup for abstentionbench_gsm8k.jsonl.

Fixes:
  A. 3 groups of exact-duplicate U prompts (8 entries total, 5 need rewrite)
     — minimal semantic-preserving reword to disambiguate
  B. Typo in absgsm_0007a / absgsm_0007u: "How load" → "How long"
  C. Smart quotes (U+2019 right single quote) → ASCII apostrophe
  D. Collapse repeated spaces to single space (except after "Answer concisely:")
  E. original_answer thousand-separator commas stripped (14 entries)
  F. En dash (U+2013) / em dash (U+2014) → ASCII hyphen (1 entry: absgsm_1120a)
  G. Non-breaking space (U+00A0) → regular ASCII space (absgsm_0098a/u)
     Note: retains U+20AC (€) and U+00BE (¾) as legitimate currency/math symbols.

Does NOT change row count (2426), A/U balance (1213/1213), or pair structure.

Usage:
    source .venv/bin/activate
    python src/data/clean_abstentionbench_gsm8k.py
"""

import json
import re

INPUT = OUTPUT = "data/abstentionbench_gsm8k.jsonl"


# ── A: Duplicate U prompt rewrites ──────────────────────────────────────────
# Each group: keep the first occurrence unchanged, rewrite the rest minimally.

PROMPT_REWRITES = {
    # Group 1: "How much did he pay?" — keep absgsm_0109u, rewrite the other 3
    "absgsm_0602u": "Answer concisely: How much did he pay in total?",
    "absgsm_0740u": "Answer concisely: How much money did he pay?",
    "absgsm_1210u": "Answer concisely: How much did he pay altogether?",
    # Group 2: "How much did he spend in total?" — keep absgsm_0383u, rewrite absgsm_0955u
    "absgsm_0955u": "Answer concisely: How much did he spend altogether?",
    # Group 3: "How much did everything cost?" — keep absgsm_0594u, rewrite absgsm_0611u
    "absgsm_0611u": "Answer concisely: What was the total cost of everything?",
}


def main():
    with open(INPUT) as f:
        entries = [json.loads(l) for l in f]
    by_id = {e["id"]: e for e in entries}

    changes = {"rewrite": 0, "typo": 0, "smart_quote": 0, "double_space": 0,
               "comma": 0, "dash": 0, "nbsp": 0}

    for e in entries:
        # ── A: Duplicate U prompt rewrites ─────────────────────────────
        if e["id"] in PROMPT_REWRITES:
            old = e["prompt"]
            e["prompt"] = PROMPT_REWRITES[e["id"]]
            print(f"  REWRITE {e['id']}: {old!r} → {e['prompt']!r}")
            changes["rewrite"] += 1

        # ── B: Typo "How load" → "How long" ───────────────────────────
        if "How load does" in e["prompt"]:
            e["prompt"] = e["prompt"].replace("How load does", "How long does")
            print(f"  TYPO {e['id']}: How load → How long")
            changes["typo"] += 1

        # ── C: Smart quotes → ASCII ───────────────────────────────────
        old_p = e["prompt"]
        e["prompt"] = (e["prompt"]
                       .replace("\u2018", "'").replace("\u2019", "'")
                       .replace("\u201c", '"').replace("\u201d", '"'))
        if e["prompt"] != old_p:
            changes["smart_quote"] += 1

        # ── G: Non-breaking space → ASCII space ──────────────────────
        # (Must run before double-space collapse, in case NBSP sits next to a
        # regular space — we don't want a lingering double space afterwards.)
        old_p = e["prompt"]
        e["prompt"] = e["prompt"].replace("\u00a0", " ")
        if e["prompt"] != old_p:
            print(f"  NBSP {e['id']}: U+00A0 → space")
            changes["nbsp"] += 1

        # ── D: Collapse repeated spaces ───────────────────────────────
        old_p = e["prompt"]
        e["prompt"] = re.sub(r"  +", " ", e["prompt"])
        if e["prompt"] != old_p:
            changes["double_space"] += 1

        # ── E: Strip commas from original_answer ──────────────────────
        if e.get("original_answer") and "," in str(e["original_answer"]):
            old_ans = e["original_answer"]
            e["original_answer"] = e["original_answer"].replace(",", "")
            print(f"  COMMA {e['id']}: {old_ans} → {e['original_answer']}")
            changes["comma"] += 1

        # ── F: En/em dash → ASCII hyphen ──────────────────────────────
        old_p = e["prompt"]
        e["prompt"] = e["prompt"].replace("\u2013", "-").replace("\u2014", "-")
        if e["prompt"] != old_p:
            print(f"  DASH {e['id']}: en/em dash → ASCII")
            changes["dash"] += 1

    # ── Verification ─────────────────────────────────────────────────────
    print(f"\n=== Verification ===")
    total = len(entries)
    a = sum(1 for e in entries if e["answerable"] == "A")
    u = sum(1 for e in entries if e["answerable"] == "U")
    print(f"  Rows: {total}, A: {a}, U: {u}")
    assert total == 2426 and a == 1213 and u == 1213

    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "Duplicate IDs!"
    print(f"  Unique IDs: {len(set(ids))} ✓")

    prompts = [e["prompt"] for e in entries]
    dup_prompts = {}
    for i, p in enumerate(prompts):
        dup_prompts.setdefault(p, []).append(entries[i]["id"])
    dups = {p: ids for p, ids in dup_prompts.items() if len(ids) > 1}
    if dups:
        print(f"  WARNING: {len(dups)} duplicate prompt groups remain:")
        for p, ids in dups.items():
            print(f"    {ids}: {p[:60]}...")
    else:
        print(f"  0 duplicate prompts ✓")

    # Smart quotes check
    sq = sum(1 for e in entries if any(c in e["prompt"] for c in "\u2018\u2019\u201c\u201d"))
    print(f"  Remaining smart quotes: {sq}")

    # NBSP check
    nbsp = sum(1 for e in entries if "\u00a0" in e["prompt"])
    print(f"  Remaining non-breaking spaces: {nbsp}")

    # Double spaces check
    ds = sum(1 for e in entries if "  " in e["prompt"])
    print(f"  Remaining double spaces: {ds}")

    # Comma in original_answer
    ca = sum(1 for e in entries if e.get("original_answer") and "," in str(e["original_answer"]))
    print(f"  Remaining comma answers: {ca}")

    # Pair completeness
    id_set = set(ids)
    for e in entries:
        base = e["id"][:-1]
        partner = base + ("u" if e["id"][-1] == "a" else "a")
        assert partner in id_set, f"Missing partner for {e['id']}"
    print(f"  All pairs complete ✓")

    # ── Write ────────────────────────────────────────────────────────────
    print(f"\n=== Writing {OUTPUT} ===")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  Written {total} entries")

    print(f"\n=== Summary ===")
    for k, v in changes.items():
        print(f"  {k}: {v}")
    print("== Done! ==")


if __name__ == "__main__":
    main()
