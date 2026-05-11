"""
clean_falseqa.py — Lightweight cleanup for falseqa.jsonl.

Fixes:
  A. 1 cross-label exact-duplicate prompt (fqa_0294u / fqa_0981a)
     — LOCAL DEDUP REWRITE: rewrites fqa_0981a to break the exact duplicate.
       This is NOT a source-faithful restoration of original FalseQA wording;
       it is a minimal edit to eliminate a cross-label collision in the shipped
       JSONL so that no prompt text appears with both A and U labels.
  B. Smart quotes (U+2019) → ASCII apostrophe (2 entries)
  C. Collapse repeated spaces (9 entries)
  D. Add terminal '?' to question-like prompts that lack any punctuation
     — only for prompts that contain question words and don't already end with '?' or '.'
  E. Fix question-like prompts ending with '.' → '?'
     — only for prompts starting with question words (How/Why/What/Is/Does/etc.)
       that DON'T have a mid-sentence '?' (which indicates a "question? options." pattern)
     — imperative forms ("List...", "Name...", "Give...") keep '.'
  F. Add terminal '.' to imperative prompts lacking any punctuation
  G. Fix lowercase "If i" → "If I" (fqa_0010u, fqa_0697a)

Does NOT change row count (1374), A/U balance (687/687), or ID structure.

Usage:
    source .venv/bin/activate
    python src/data/clean_falseqa.py
"""

import json
import re

INPUT = OUTPUT = "data/falseqa.jsonl"

# ── A: Cross-label duplicate fix ────────────────────────────────────────────
# fqa_0294u (U/false_premise): "How to light and put out the fire at the same time?"
# fqa_0981a (A/true_premise):  same text — rewrite A side minimally
PROMPT_REWRITES = {
    "fqa_0981a": "Answer concisely: How to light a fire and then put it out?",
}

# Question words that signal a prompt should end with '?'
_QUESTION_WORDS_RE = re.compile(
    r"\b(how|why|what|when|where|which|who|whom|whose|does|do|did|is|are|was|were|can|could|will|would|should)\b",
    re.IGNORECASE,
)


def main():
    with open(INPUT) as f:
        entries = [json.loads(l) for l in f]

    changes = {"rewrite": 0, "smart_quote": 0, "double_space": 0,
               "add_question_mark": 0, "dot_to_question": 0, "add_dot": 0,
               "lowercase_i": 0}

    for e in entries:
        # ── A: Duplicate rewrite ───────────────────────────────────────
        if e["id"] in PROMPT_REWRITES:
            old = e["prompt"]
            e["prompt"] = PROMPT_REWRITES[e["id"]]
            print(f"  REWRITE {e['id']}: {old!r} → {e['prompt']!r}")
            changes["rewrite"] += 1

        # ── G: Lowercase "If i" → "If I" ─────────────────────────────
        if "If i " in e["prompt"]:
            e["prompt"] = e["prompt"].replace("If i ", "If I ")
            changes["lowercase_i"] += 1

        # ── B: Smart quotes → ASCII ───────────────────────────────────
        old_p = e["prompt"]
        e["prompt"] = (e["prompt"]
                       .replace("\u2018", "'").replace("\u2019", "'")
                       .replace("\u201c", '"').replace("\u201d", '"'))
        if e["prompt"] != old_p:
            changes["smart_quote"] += 1

        # ── C: Collapse repeated spaces ───────────────────────────────
        old_p = e["prompt"]
        e["prompt"] = re.sub(r"  +", " ", e["prompt"])
        if e["prompt"] != old_p:
            changes["double_space"] += 1

        # ── D: Add terminal '?' to questions missing all punctuation ──
        p = e["prompt"].rstrip()
        body = p[len("Answer concisely: "):] if p.startswith("Answer concisely: ") else p
        if not p.endswith("?") and not p.endswith("."):
            if _QUESTION_WORDS_RE.search(body):
                e["prompt"] = p + "?"
                changes["add_question_mark"] += 1
            elif re.match(r"^(Name|List|Give|list)\b", body):
                # ── F: Imperative with no punctuation → add '.' ───────
                e["prompt"] = p + "."
                changes["add_dot"] += 1

        # ── E: Simple question ending with '.' → '?' ─────────────────
        # Only if body starts with a question word AND there's no mid-sentence '?'
        # (mid-sentence '?' means "question? options." pattern — leave '.' alone)
        p = e["prompt"].rstrip()
        body = p[len("Answer concisely: "):] if p.startswith("Answer concisely: ") else p
        if p.endswith("."):
            is_q_start = bool(re.match(
                r"^(How|Why|What|When|Where|Which|Who|Whom|Whose|"
                r"Does|Do|Did|Is|Are|Was|Were|Can|Could|Will|Would|Should|If)\b",
                body, re.IGNORECASE))
            has_mid_q = "?" in body[:-1]
            if is_q_start and not has_mid_q:
                e["prompt"] = p[:-1] + "?"
                changes["dot_to_question"] += 1

    # ── Verification ─────────────────────────────────────────────────────
    print(f"\n=== Verification ===")
    total = len(entries)
    a = sum(1 for e in entries if e["answerable"] == "A")
    u = sum(1 for e in entries if e["answerable"] == "U")
    print(f"  Rows: {total}, A: {a}, U: {u}")
    assert total == 1374 and a == 687 and u == 687

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
            labels = [(eid, [e for e in entries if e["id"] == eid][0]["answerable"]) for eid in ids]
            print(f"    {labels}: {p[:60]}...")
    else:
        print(f"  0 duplicate prompts ✓")

    # Cross-label check
    prompt_labels = {}
    for e in entries:
        prompt_labels.setdefault(e["prompt"], set()).add(e["answerable"])
    cross = {p: lbls for p, lbls in prompt_labels.items() if len(lbls) > 1}
    print(f"  Cross-label duplicates: {len(cross)}")

    # Smart quotes check
    sq = sum(1 for e in entries if any(c in e["prompt"] for c in "\u2018\u2019\u201c\u201d"))
    print(f"  Remaining smart quotes: {sq}")

    # Double spaces check
    ds = sum(1 for e in entries if "  " in e["prompt"])
    print(f"  Remaining double spaces: {ds}")

    # Punctuation stats
    ends_q = sum(1 for e in entries if e["prompt"].rstrip().endswith("?"))
    ends_dot = sum(1 for e in entries if e["prompt"].rstrip().endswith("."))
    ends_other = total - ends_q - ends_dot
    print(f"  Ends with '?': {ends_q}, '.': {ends_dot}, other: {ends_other}")

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
