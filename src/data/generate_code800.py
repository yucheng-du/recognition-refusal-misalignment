"""
generate_code800.py — Generate CODE-800 dataset via Anthropic API.

Generates 8 categories × 100 pairs each = 800 matched A/U pairs.
Each pair: one answerable (A) + one structurally unanswerable (U) Python expression.

U prompts must raise a runtime exception in standard CPython (TypeError, ValueError,
ZeroDivisionError, IndexError, KeyError, AttributeError, NameError, ValueError from
math domain, or hang forever). NOT: syntax errors, expressions returning None,
platform-dependent behavior.

Prompt format: "Answer concisely: <template>`X`<template>" where <template> is one of
seven canonical templates (see PROMPT_TEMPLATES below).
(NOT "What does X return?" — U prompts raise exceptions, not return values)

Usage (from the repository root):
    # Full run — all 8 categories, 100 pairs each (= 800 pairs total)
    python src/data/generate_code800.py --out data/code800_raw.jsonl

    # Single category test (5 pairs)
    python src/data/generate_code800.py --category type_error --n-pairs 5 --out /tmp/test.jsonl

    # Resume interrupted run
    python src/data/generate_code800.py --out data/code800_raw.jsonl --resume

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=...
"""

import argparse
import json
import os
import re
import time

import anthropic

# ── Category definitions ──────────────────────────────────────────────────────

CATEGORIES = {
    "type_error": {
        "description": "Operations between incompatible types that raise TypeError in CPython",
        "a_examples": [
            'abs(-5)',
            '"hello" * 3',
            'len([1, 2, 3])',
            'round(3.14159, 2)',
        ],
        "u_examples": [
            'abs("hello")',
            '"hello" * "world"',
            'len(42)',
            'round("pi", 2)',
        ],
        "u_rule": "The expression must raise TypeError in standard CPython. Do NOT use: syntax errors, expressions returning None, NameError (different category).",
    },
    "value_error": {
        "description": "Operations with structurally invalid values that raise ValueError in CPython (not math domain — that is a separate category)",
        "a_examples": [
            'max([1, 2, 3])',
            'int("42")',
            'float("3.14")',
            'sorted([3, 1, 2])',
        ],
        "u_examples": [
            'max([])',
            'int("forty-two")',
            'float("hello")',
            'min([])',
        ],
        "u_rule": "Must raise ValueError. Do NOT use: math.sqrt(-1) (math_domain_error category), TypeError (different category). Good examples: max([]), min([]), int('abc'), bytes.fromhex('xyz'), int('', 16).",
    },
    "zero_division": {
        "description": "Division or modulo by zero in Python (raises ZeroDivisionError)",
        "a_examples": [
            '10 // 3',
            'divmod(17, 5)',
            '15 % 4',
            '100 / 7',
        ],
        "u_examples": [
            '10 // 0',
            'divmod(17, 0)',
            '15 % 0',
            '100 / 0',
        ],
        "u_rule": "The divisor or modulus must be exactly 0. Include integer division (//), true division (/), modulo (%), and divmod(). Vary the dividend.",
    },
    "index_key_error": {
        "description": "Accessing an index or key that does not exist (raises IndexError or KeyError)",
        "a_examples": [
            '[1, 2, 3][0]',
            '{"a": 1, "b": 2}["a"]',
            '(10, 20, 30)[2]',
            '"hello"[1]',
        ],
        "u_examples": [
            '[][0]',
            '{}["key"]',
            '[1, 2][5]',
            '{"a": 1}["b"]',
        ],
        "u_rule": "Must raise IndexError (list/tuple/string out of bounds) or KeyError (dict missing key). For IndexError: use out-of-range positive or negative indices. For KeyError: use a key that is not in the dict. Vary collection types and indices.",
    },
    "attribute_error": {
        "description": "Calling a method or accessing an attribute that does not exist on the object (raises AttributeError)",
        "a_examples": [
            '"hello".upper()',
            '[1, 2, 3].index(2)',
            '(3.14).is_integer()',
            '"world".replace("o", "0")',
        ],
        "u_examples": [
            '(42).upper()',
            'None.replace("l", "r")',
            '(3).append(4)',
            '(1.5).split(",")',
        ],
        "u_rule": "The method/attribute must not exist on that type in standard CPython. Must raise AttributeError. Good targets: calling string methods on int/float/None, calling list methods on int/tuple, etc.",
    },
    "infinite_iter": {
        "description": "Expressions that hang forever (non-terminating iteration) — never produce a result",
        "a_examples": [
            'sum(range(100))',
            'len(list(range(1000)))',
            'list(range(10))',
            'max(range(50))',
        ],
        "u_examples": [
            'sum(itertools.count())',
            'list(range(float("inf")))',
            'len(list(itertools.count()))',
            'max(itertools.count())',
        ],
        "u_rule": "The expression must hang forever (non-terminating). Good patterns: sum/max/min/list/len applied to itertools.count(), range(float('inf')), or other infinite iterators. The A counterpart uses a finite range/list of the same operation. Always use itertools.count() or range(float('inf')) for U.",
    },
    "name_error": {
        "description": "Using a variable name that has not been defined in the current scope (raises NameError)",
        "a_examples": [
            'len([1, 2, 3])',
            'str(42)',
            'type(True)',
            'bool(0)',
        ],
        "u_examples": [
            'len(items)',
            'str(my_value)',
            'type(result)',
            'sum(scores)',
        ],
        "u_rule": "The expression must use a variable name that is NOT a Python builtin and NOT imported. Use realistic-looking but undefined names: items, scores, result, data, values, numbers, my_list, count, total, etc. The A counterpart performs the SAME operation but on a literal value.",
    },
    "math_domain_error": {
        "description": "math module functions called outside their mathematical domain (raises ValueError in Python)",
        "a_examples": [
            'math.sqrt(16)',
            'math.log(100)',
            'math.asin(0.5)',
            'math.log2(8)',
        ],
        "u_examples": [
            'math.sqrt(-1)',
            'math.log(-5)',
            'math.asin(2)',
            'math.log(0)',
        ],
        "u_rule": "Must raise ValueError (Python's math domain error). Good functions: math.sqrt(negative), math.log(0 or negative), math.log2(0 or negative), math.log10(0 or negative), math.asin/acos(x where |x|>1), math.acosh(x<1), math.atanh(|x|>=1). Always prefix with 'math.'.",
    },
}

N_PAIRS_DEFAULT = 100

# ── Prompt templates ─────────────────────────────────────────────────────────
# The dataset uses 7 canonical prompt templates for surface-form diversity.
# All begin with "Answer concisely:" and contain the expression in backticks.

PROMPT_TEMPLATES = [
    "Answer concisely: What is the result of evaluating `{expr}` in Python?",
    "Answer concisely: What does `{expr}` produce in Python?",
    "Answer concisely: What is the value of `{expr}` in Python?",
    "Answer concisely: What output does `{expr}` give in Python?",
    "Answer concisely: Compute `{expr}` in Python.",
    "Answer concisely: Evaluate the Python expression `{expr}`.",
    "Answer concisely: Run `{expr}` in Python and give the result.",
]

SYSTEM_PROMPT = """You are generating a Python expression dataset for NLP research.
You generate matched pairs: one answerable (A) and one structurally unanswerable (U)
Python expression. When evaluated in standard CPython, A expressions produce a value;
U expressions either raise a runtime exception or hang forever.

CRITICAL RULES for U expressions:
- Must fail due to a structural/type/domain error in standard CPython
- Must NOT be a syntax error (too trivial)
- Must NOT return None (None is a valid value)
- Must NOT be platform-dependent
- Must NOT use undefined variables UNLESS the category is name_error

Prompt format: ALL prompts (A and U) must use one of these 7 templates:
  "Answer concisely: What is the result of evaluating `X` in Python?"
  "Answer concisely: What does `X` produce in Python?"
  "Answer concisely: What is the value of `X` in Python?"
  "Answer concisely: What output does `X` give in Python?"
  "Answer concisely: Compute `X` in Python."
  "Answer concisely: Evaluate the Python expression `X`."
  "Answer concisely: Run `X` in Python and give the result."
(Vary templates across prompts for surface-form diversity.)

Format: return ONLY a JSON array, no markdown, no explanation:
[
  {"answerable": "A", "prompt": "Answer concisely: What is the result of evaluating `X` in Python?"},
  {"answerable": "U", "prompt": "Answer concisely: What does `Y` produce in Python?"},
  ...
]

Rules:
- A and U in each pair address the same operation/function, differing only in the argument
- Vary expressions across pairs — no two pairs should use the exact same numbers/values
- Do NOT number pairs or add commentary
"""

def make_user_prompt(category_key, category, n_pairs, already_generated=None):
    a_ex = "\n".join(
        f'  A: "What is the result of evaluating `{e}` in Python?"'
        for e in category["a_examples"]
    )
    u_ex = "\n".join(
        f'  U: "What is the result of evaluating `{e}` in Python?"'
        for e in category["u_examples"]
    )

    avoid = ""
    if already_generated:
        samples = already_generated[-8:]
        avoid = "\nAlready generated (avoid repeating similar expressions):\n"
        for item in samples:
            avoid += f'  {item["answerable"]}: {item["prompt"]}\n'

    return f"""Generate {n_pairs} matched A/U pairs for category: {category["description"]}

Category rule for U prompts: {category["u_rule"]}

Examples of the style to follow:
{a_ex}
{u_ex}
{avoid}
Generate exactly {n_pairs} pairs ({n_pairs} A prompts and {n_pairs} U prompts, interleaved).
Return ONLY the JSON array."""


# ── Quality checks ────────────────────────────────────────────────────────────

_TEMPLATE_PREFIXES = [
    "Answer concisely: What is the result of evaluating",
    "Answer concisely: What does",
    "Answer concisely: What is the value of",
    "Answer concisely: What output does",
    "Answer concisely: Compute",
    "Answer concisely: Evaluate the Python expression",
    "Answer concisely: Run",
]


def _ast_key(expr):
    """Return an AST-normalized key for dedup, or None if unparseable."""
    import ast as _ast
    try:
        tree = _ast.parse(expr.strip(), mode="eval")
        return _ast.dump(tree, include_attributes=False)
    except Exception:
        return None


def _extract_expr(prompt):
    import re as _re
    m = _re.search(r'`([^`]+)`', prompt)
    return m.group(1) if m else None


# Accumulated AST keys across the entire generation run, used by quality_check
# to reject AST-equivalent duplicates within and across categories.
_seen_ast_keys = set()


def seed_seen_ast(items):
    """Pre-populate _seen_ast_keys from already-generated items.

    Call this before quality_check when resuming a run, so the guardrail
    also covers expressions already written to disk.
    """
    for item in items:
        expr = _extract_expr(item.get("prompt", ""))
        if expr:
            key = _ast_key(expr)
            if key:
                _seen_ast_keys.add(key)


def quality_check(item, category_key):
    """Returns (ok: bool, reason: str)."""
    p = item.get("prompt", "")
    ans = item.get("answerable", "")

    if ans not in ("A", "U"):
        return False, f"invalid answerable={ans}"

    if not any(p.startswith(prefix) for prefix in _TEMPLATE_PREFIXES):
        return False, "wrong prompt format (must match one of 7 canonical templates)"

    # Must contain a backtick-enclosed expression
    if "`" not in p:
        return False, "no backtick-enclosed expression found"

    # AST-normalized dedup: reject if an AST-equivalent expression was already seen
    expr = _extract_expr(p)
    if expr:
        key = _ast_key(expr)
        if key:
            if key in _seen_ast_keys:
                return False, f"AST-duplicate expression: `{expr}`"
            _seen_ast_keys.add(key)

    p_lower = p.lower()

    # U-specific checks
    if ans == "U":
        bad_patterns = ["syntaxerror", "syntax error"]
        for bp in bad_patterns:
            if bp in p_lower:
                return False, f"syntax error pattern: '{bp}'"

    return True, "ok"


def parse_response(text, category_key):
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return [], "no JSON array found"
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError as e:
        return [], f"JSON parse error: {e}"

    normalized = []
    for item in items:
        if isinstance(item.get("answerable"), bool):
            item["answerable"] = "A" if item["answerable"] else "U"
        normalized.append(item)
    return normalized, "ok"


# ── Main generation loop ──────────────────────────────────────────────────────

def generate_category(client, category_key, category, n_pairs, existing=None):
    results = list(existing or [])
    if len(results) // 2 >= n_pairs:
        print(f"  [{category_key}] already complete, skipping")
        return results

    print(f"  [{category_key}] generating {n_pairs - len(results)//2} more pairs...")

    max_attempts = 6
    batch_size = min(n_pairs, 20)
    attempt = 0

    while len(results) // 2 < n_pairs and attempt < max_attempts:
        still_needed = n_pairs - len(results) // 2
        this_batch = min(still_needed, batch_size)

        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=6000,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": make_user_prompt(
                        category_key, category, this_batch,
                        already_generated=results[-16:] if results else None
                    )
                }]
            )
            text = response.content[0].text
        except Exception as e:
            print(f"    API error: {e}. Retrying in 10s...")
            time.sleep(10)
            attempt += 1
            continue

        items, parse_msg = parse_response(text, category_key)
        if not items:
            print(f"    Parse failed: {parse_msg}. Retrying...")
            attempt += 1
            continue

        accepted = rejected = 0
        for item in items:
            ok, reason = quality_check(item, category_key)
            if ok:
                results.append(item)
                accepted += 1
            else:
                rejected += 1
                print(f"    REJECTED ({reason}): {item.get('prompt','')[:60]}")

        print(f"    batch: +{accepted} accepted, {rejected} rejected "
              f"→ {len(results)//2}/{n_pairs} pairs")
        attempt += 1
        time.sleep(1)

    return results


def assign_ids(items, category_key, start_idx=1):
    abbr = category_key[:4]
    a_items = [x for x in items if x["answerable"] == "A"]
    u_items = [x for x in items if x["answerable"] == "U"]
    n = min(len(a_items), len(u_items))
    pairs = []
    for i in range(n):
        idx = start_idx + i
        pairs.append({
            "id": f"c{abbr}{idx:03d}a",
            "form": "CODE",
            "answerable": "A",
            "category": category_key,
            "prompt": a_items[i]["prompt"],
        })
        pairs.append({
            "id": f"c{abbr}{idx:03d}u",
            "form": "CODE",
            "answerable": "U",
            "category": category_key,
            "prompt": u_items[i]["prompt"],
        })
    return pairs


def load_existing(out_path):
    existing = {}
    if not os.path.isfile(out_path):
        return existing
    with open(out_path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                cat = item.get("category", "unknown")
                existing.setdefault(cat, [])
                existing[cat].append({
                    "answerable": item["answerable"],
                    "prompt": item["prompt"],
                })
    return existing


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate CODE-800 dataset via Anthropic API")
    parser.add_argument("--out", default="data/code800_raw.jsonl")
    parser.add_argument("--category", default=None)
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS_DEFAULT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default="claude-sonnet-4-5")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")

    client = anthropic.Anthropic(api_key=api_key)
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    if args.category:
        if args.category not in CATEGORIES:
            raise ValueError(f"Unknown category: {args.category}. Options: {list(CATEGORIES)}")
        cats_to_run = {args.category: CATEGORIES[args.category]}
    else:
        cats_to_run = CATEGORIES

    existing_by_cat = load_existing(args.out) if args.resume else {}

    # Pre-populate dedup guardrail with already-written expressions so that
    # resume mode rejects AST-duplicates against the on-disk file, not just
    # within the current process's new candidates.
    if existing_by_cat:
        all_existing = [item for items in existing_by_cat.values() for item in items]
        seed_seen_ast(all_existing)
        print(f"  Seeded AST-dedup guardrail with {len(all_existing)} existing items "
              f"({len(_seen_ast_keys)} unique AST keys).")

    mode = "a" if args.resume else "w"
    total_pairs = 0

    with open(args.out, mode, encoding="utf-8") as out_f:
        for cat_key, cat_def in cats_to_run.items():
            print(f"\n{'='*60}")
            print(f"Category: {cat_key}")
            print(f"{'='*60}")

            existing = existing_by_cat.get(cat_key, [])
            if args.resume and len(existing) // 2 >= args.n_pairs:
                print(f"  Already complete ({len(existing)//2} pairs). Skipping.")
                total_pairs += len(existing) // 2
                continue

            items = generate_category(client, cat_key, cat_def, args.n_pairs, existing=existing)
            pairs = assign_ids(items, cat_key)

            for row in pairs:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

            n = len(pairs) // 2
            total_pairs += n
            print(f"  Written {n} pairs for {cat_key}")

    print(f"\n{'='*60}")
    print(f"Done. Total pairs: {total_pairs}")
    print(f"Output: {args.out}")
    print(f"\nNext: python src/data/check_code800.py --in {args.out}")


if __name__ == "__main__":
    main()
