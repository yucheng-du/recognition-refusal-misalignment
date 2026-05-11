"""
generate_math800.py — Generate MATH-800 dataset via Anthropic API.

Generates 16 categories × 50 pairs each = 800 matched A/U pairs.
Each pair: one answerable (A) + one structurally unanswerable (U) prompt
on the same mathematical concept.

U prompts must be STRUCTURALLY unanswerable — formally undefined, having no
result, or having no solution in the specified domain under standard mathematics.
This includes: undefined operations (e.g., division by zero), non-existent results
(e.g., derivative at a cusp), no solution in domain (e.g., x^2=-4 in reals), and
divergent quantities (e.g., sum of all natural numbers). U must NOT be epistemic
(unknown real-world values, lookup-dependent facts).

Usage (from the repo root):
    # Full run — all 16 categories, 50 pairs each (= 800 pairs total)
    python src/data/generate_math800.py --out data/math800_raw.jsonl

    # Single category test (5 pairs)
    python src/data/generate_math800.py --category div_zero --n-pairs 5 --out /tmp/test.jsonl

    # Resume interrupted run (appends, skips already-done categories)
    python src/data/generate_math800.py --out data/math800_raw.jsonl --resume

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
    "div_zero": {
        "description": "Division by zero (n ÷ 0 for integer or decimal n, including 0÷0)",
        "a_examples": ["What is 84 divided by 7?", "What is 0.5 divided by 2?"],
        "u_examples": ["What is 84 divided by 0?", "What is 0 divided by 0?"],
        "u_rule": "The denominator must be exactly 0. Never use near-zero values.",
    },
    "mod_zero": {
        "description": "Modulo by zero (n mod 0 for any integer n)",
        "a_examples": ["What is 23 modulo 5?", "What is 100 mod 7?"],
        "u_examples": ["What is 23 modulo 0?", "What is 17 mod 0?"],
        "u_rule": "The modulus must be exactly 0.",
    },
    "log_base_zero_or_neg": {
        "description": "Logarithm with base 0 or a negative number (undefined in standard real-valued mathematics)",
        "a_examples": ["What is log base 2 of 8?", "What is log base 10 of 1000?"],
        "u_examples": ["What is log base -2 of 8?", "What is log base 0 of 125?"],
        "u_rule": "Base must be 0 or negative. Include log base -n and log base 0 variants. Do NOT use base 1 (that is a separate edge case).",
    },
    "invalid_combination": {
        "description": "Combination C(n, k) where k > n or k < 0 (undefined under the standard combinatorial definition)",
        "a_examples": ["What is C(10, 3)?", "What is 8 choose 2?"],
        "u_examples": ["What is C(5, 8) using the standard combinatorial definition?", "What is 7 choose negative 2 using the standard combinatorial definition?"],
        "u_rule": "k must be strictly greater than n OR strictly negative. Always include 'using the standard combinatorial definition'. Use 'C(n, k)' or 'n choose k' interchangeably.",
    },
    "sqrt_negative_real": {
        "description": "Square root of a negative number in the real numbers",
        "a_examples": ["What is the square root of 144?", "What is the square root of 0.25?"],
        "u_examples": ["What is the square root of negative 144 in the real numbers?", "What is the square root of negative 25 in the real numbers?"],
        "u_rule": "Always include 'in the real numbers'. Use only square root (not 4th root etc — that is a different category).",
    },
    "even_root_negative": {
        "description": "Even-indexed root (4th, 6th, 8th...) of a negative number in the real numbers",
        "a_examples": ["What is the fourth root of 16?", "What is the sixth root of 64?"],
        "u_examples": ["What is the fourth root of negative 16 in the real numbers?", "What is the sixth root of negative 64 in the real numbers?"],
        "u_rule": "Use only even indices (4, 6, 8...). Always include 'in the real numbers'.",
    },
    "factorial_invalid": {
        "description": "Factorial of a negative integer or non-integer (using the standard integer factorial definition)",
        "a_examples": ["What is 6 factorial?", "What is 0 factorial?"],
        "u_examples": ["What is negative 3 factorial using the standard integer factorial definition?", "What is 2.5 factorial using the standard integer factorial definition?"],
        "u_rule": "Always include 'using the standard integer factorial definition'. Use negative integers OR non-integers (not both in same prompt).",
    },
    "divergent_series": {
        "description": "Sum or product of an infinite series that diverges in standard mathematics",
        "a_examples": ["What is the sum of the first 10 natural numbers?", "What is the sum of 1/2 + 1/4 + 1/8 + ... (geometric series)?"],
        "u_examples": ["What is the exact sum of all natural numbers from 1 to infinity in standard mathematics?", "What is the exact product of all positive integers?"],
        "u_rule": "The series must genuinely diverge. Always include 'exact' and 'in standard mathematics'. Do NOT use Grandi's series or conditionally convergent series. Good choices: sum of all naturals, sum of all odds, product of all primes.",
    },
    "trig_undefined": {
        "description": "Trigonometric functions at undefined points (tan 90°, cot 0°, sec 90°, csc 0°)",
        "a_examples": ["What is the sine of 30 degrees?", "What is the tangent of 45 degrees?"],
        "u_examples": ["What is the tangent of 90 degrees?", "What is the cotangent of 0 degrees?"],
        "u_rule": "Use tan(90° + 180°k), cot(0° + 180°k), sec(90° + 180°k), or csc(0° + 180°k). Do not use sin or cos (always defined).",
    },
    "inv_trig_out_of_domain": {
        "description": "Inverse trig functions applied outside their domain (arcsin/arccos outside [−1, 1])",
        "a_examples": ["What is arcsin of 0.5 in degrees?", "What is arccos of 1 in degrees?"],
        "u_examples": ["What is arcsin of 2 in the real numbers?", "What is arccos of negative 3 in the real numbers?"],
        "u_rule": "The argument must have absolute value strictly greater than 1. Always include 'in the real numbers'.",
    },
    "gcd_lcm_irrational": {
        "description": "GCD or LCM involving an irrational number (undefined outside integers)",
        "a_examples": ["What is the GCD of 24 and 36?", "What is the LCM of 6 and 8?"],
        "u_examples": ["What is the GCD of 6 and pi?", "What is the LCM of 4 and the square root of 2?"],
        "u_rule": "One operand must be an irrational number (π, √2, √3, e, etc.). The other can be any integer.",
    },
    "undefined_derivative": {
        "description": "Derivative at a point where the function is not differentiable",
        "a_examples": ["What is the derivative of x^2 at x = 1?", "What is the derivative of x^3+1 at x = 2?"],
        "u_examples": ["What is the derivative of |x| at x = 0?", "What is the derivative of |x-1| at x = 1?",
                        "What is the derivative of floor(x) at x = 3?", "What is the derivative of sign(x) at x = 0?"],
        "u_rule": "The function must genuinely fail to be differentiable at the given point. "
                  "Accepted families: (1) absolute value cusps — |f(x)| at a zero of f where f changes sign "
                  "(e.g. |x-a| at x=a, |x^2-1| at x=±1, |2x+1| at x=-0.5); "
                  "(2) step discontinuities — floor(x) and ceil(x) at integer points, "
                  "round(x) at half-integer tie points (x = n + 0.5 for integer n); "
                  "(3) sign(x) at x=0 (jump discontinuity); "
                  "(4) x^(1/3) at x=0 (vertical tangent, infinite derivative). "
                  "For absolute value, ensure the inner expression equals zero at the evaluation point "
                  "(e.g. |x^21| at x=1 is WRONG because x^21>0 near x=1). "
                  "The A counterpart uses a smooth polynomial at the same or similar point.",
    },
    "no_real_solution": {
        "description": "Equations with no solution in the real numbers",
        "a_examples": ["Solve for x: 2x plus 3 equals 11.", "Solve for x: x squared equals 25."],
        "u_examples": ["Solve for x: x squared equals negative 9 in the real numbers.", "Solve for x: 5x plus 3 equals 5x plus 7."],
        "u_rule": "Two subtypes allowed: (1) x²=−n in ℝ (no real solution), (2) contradictory equations (0=k, k≠0). Always include 'in the real numbers' for subtype 1.",
    },
    "singular_matrix_inverse": {
        "description": "Inverse of a singular (non-invertible) matrix",
        "a_examples": ["What is the inverse of the 2×2 matrix [[2, 1], [1, 1]]?", "What is the inverse of the matrix [[3, 0], [0, 2]]?"],
        "u_examples": ["What is the inverse of the 2×2 matrix [[1, 2], [2, 4]]?", "What is the inverse of the matrix [[0, 0], [1, 2]]?"],
        "u_rule": "The matrix must have determinant = 0 (singular). Verify this before including. Use only 2×2 matrices for consistency.",
    },
    "nonexistent_limit": {
        "description": "Limits that do not exist (oscillation, jump discontinuity)",
        "a_examples": ["What is the limit of (x squared minus 1) divided by (x minus 1) as x approaches 1?", "What is the limit of 3x plus 2 as x approaches 5?"],
        "u_examples": ["What is the limit of sin(1/x) as x approaches 0?", "What is the limit of the absolute value of x divided by x as x approaches 0?"],
        "u_rule": "The limit must genuinely not exist (oscillation or left/right limits differ). Do NOT use limits that are merely infinite (lim 1/x² as x→0 = +∞ exists in extended reals).",
    },
    "neg_base_frac_exp": {
        "description": "Negative base raised to a fractional power with even denominator, in the real numbers (yields non-real result)",
        "a_examples": ["What is 8 to the power of 1/3?", "What is 27 to the power of 2/3?"],
        "u_examples": ["What is negative 8 to the power of 1/2 in the real numbers?", "What is negative 16 to the power of 1/4 in the real numbers?"],
        "u_rule": "The base must be negative and the exponent must be a fraction p/q where q is even (so the result is non-real). Always include 'in the real numbers'. Do NOT use odd denominators ((-8)^(1/3) = -2 is real and answerable).",
    },
}

N_PAIRS_DEFAULT = 50

# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are generating a mathematics dataset for an NLP research paper.
You generate matched pairs of math prompts: one answerable (A) and one structurally
unanswerable (U) on the same concept.

CRITICAL RULES for U prompts:
- U must be STRUCTURALLY unanswerable under standard mathematics: the operation is
  formally undefined, has no result, has no solution in the specified domain, or diverges.
  Examples: division by zero, derivative at a non-differentiable point, x^2=-4 in reals,
  sum of all natural numbers.
- U must NOT be epistemic (i.e., do not use unknown physical quantities, future values,
  very large but computable numbers, or lookup-dependent values)
- U must NOT be a valid math question with a surprising answer (e.g., "largest prime" is
  answerable: "there is no largest prime")

Format: return ONLY a JSON array of objects, no markdown, no explanation:
[
  {"answerable": "A", "prompt": "Answer concisely: <question>"},
  {"answerable": "U", "prompt": "Answer concisely: <question>"},
  ...
]

Rules for prompts:
- All prompts start with "Answer concisely: "
- Keep questions concise (1–2 sentences)
- A and U in each pair should address the same mathematical operation/concept
- Vary the numbers and surface form across pairs (no duplicates)
- Do NOT number the pairs or add any commentary
"""

def make_user_prompt(category_key, category, n_pairs, already_generated=None):
    a_ex = "\n".join(f'  A: "Answer concisely: {e}"' for e in category["a_examples"])
    u_ex = "\n".join(f'  U: "Answer concisely: {e}"' for e in category["u_examples"])

    avoid = ""
    if already_generated:
        samples = already_generated[-6:]  # last few to avoid repetition
        avoid = "\nAlready generated (avoid repeating similar surface forms):\n"
        for item in samples:
            avoid += f'  {item["answerable"]}: {item["prompt"]}\n'

    return f"""Generate {n_pairs} matched A/U pairs for category: {category["description"]}

Category rule for U prompts: {category["u_rule"]}

Examples of the style to follow:
{a_ex}
{u_ex}
{avoid}
Generate exactly {n_pairs} pairs ({n_pairs} A prompts and {n_pairs} U prompts, interleaved A then U).
Return ONLY the JSON array."""


# ── Quality checks ────────────────────────────────────────────────────────────

EPISTEMIC_KEYWORDS = [
    "current", "right now", "today", "2050", "2100", "2099", "future",
    "will ever", "ever live", "stars in", "atoms in", "grains of sand",
    "active volcanoes", "internet users", "countries in", "population of",
    "speed of light",  # c is known but signals physics lookup
    "number of people", "total wealth",
]

# Accumulated whitespace-stripped prompts, used by quality_check to reject
# whitespace-insensitive duplicates (e.g. "|2x + 1|" vs "|2x+1|").
# Key = all whitespace removed, so *any* spacing difference is caught.
_seen_ws_stripped = set()


def _ws_strip_key(prompt):
    """Normalize a prompt to a whitespace-free key for dedup.

    Removes ALL whitespace so that "|2x + 1|" and "|2x+1|" produce
    the same key.  This is intentionally aggressive — false positives
    (two genuinely different prompts mapping to the same key) are
    extremely unlikely for well-formed math prompts.
    """
    return re.sub(r"\s+", "", prompt)


def seed_seen_ws(items):
    """Pre-populate _seen_ws_stripped from already-generated items.

    Call this before quality_check when resuming a run, so the guardrail
    also covers prompts already written to disk.
    """
    for item in items:
        _seen_ws_stripped.add(_ws_strip_key(item["prompt"]))


def quality_check(item, category_key):
    """Returns (ok: bool, reason: str)."""
    p = item.get("prompt", "").lower()
    ans = item.get("answerable", "")

    if ans not in ("A", "U"):
        return False, f"invalid answerable={ans}"

    if not item["prompt"].startswith("Answer concisely:"):
        return False, "missing 'Answer concisely:' prefix"

    # Whitespace-insensitive dedup (strips ALL whitespace)
    ws_key = _ws_strip_key(item["prompt"])
    if ws_key in _seen_ws_stripped:
        return False, "whitespace-insensitive duplicate prompt"
    _seen_ws_stripped.add(ws_key)

    if ans == "U":
        for kw in EPISTEMIC_KEYWORDS:
            if kw in p:
                return False, f"epistemic keyword: '{kw}'"

    return True, "ok"


def parse_response(text, category_key):
    """Extract and validate JSON array from API response."""
    # Strip markdown fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Find the JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if not match:
        return [], "no JSON array found"

    try:
        items = json.loads(match.group())
    except json.JSONDecodeError as e:
        return [], f"JSON parse error: {e}"

    # Normalize: some models return "answerable": true/false
    normalized = []
    for item in items:
        if isinstance(item.get("answerable"), bool):
            item["answerable"] = "A" if item["answerable"] else "U"
        normalized.append(item)

    return normalized, "ok"


# ── Main generation loop ──────────────────────────────────────────────────────

def generate_category(client, category_key, category, n_pairs, existing=None):
    """Generate n_pairs pairs for one category. Returns list of valid items."""
    results = list(existing or [])
    needed = n_pairs - len(results) // 2  # pairs still needed
    if needed <= 0:
        print(f"  [{category_key}] already have {len(results)} items, skipping")
        return results

    print(f"  [{category_key}] generating {needed} more pairs "
          f"(have {len(results)//2}/{n_pairs})...")

    max_attempts = 5
    batch_size = min(needed, 15)  # generate up to 15 pairs per API call

    attempt = 0
    while len(results) // 2 < n_pairs and attempt < max_attempts:
        still_needed = n_pairs - len(results) // 2
        this_batch = min(still_needed, batch_size)

        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": make_user_prompt(
                        category_key, category, this_batch,
                        already_generated=results[-12:] if results else None
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
              f"→ {len(results)//2}/{n_pairs} pairs total")
        attempt += 1
        time.sleep(1)  # rate limit courtesy

    return results


def assign_ids(items, category_key, start_idx=1):
    """Assign IDs: m{category_abbr}{idx:03d}{a/u}"""
    abbr = category_key[:4]
    pairs = []
    a_items = [x for x in items if x["answerable"] == "A"]
    u_items = [x for x in items if x["answerable"] == "U"]
    n = min(len(a_items), len(u_items))
    for i in range(n):
        idx = start_idx + i
        pairs.append({
            "id": f"m{abbr}{idx:03d}a",
            "form": "MATH",
            "answerable": "A",
            "category": category_key,
            "prompt": a_items[i]["prompt"],
        })
        pairs.append({
            "id": f"m{abbr}{idx:03d}u",
            "form": "MATH",
            "answerable": "U",
            "category": category_key,
            "prompt": u_items[i]["prompt"],
        })
    return pairs


def load_existing(out_path):
    """Load already-generated items grouped by category."""
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
    parser = argparse.ArgumentParser(description="Generate MATH-800 dataset via Anthropic API")
    parser.add_argument("--out", default="data/math800_raw.jsonl",
                        help="Output JSONL file (default: data/math800_raw.jsonl)")
    parser.add_argument("--category", default=None,
                        help="Run only this category (default: all)")
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS_DEFAULT,
                        help=f"Pairs per category (default: {N_PAIRS_DEFAULT})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume by loading existing output and skipping done categories")
    parser.add_argument("--model", default="claude-sonnet-4-5",
                        help="Anthropic model to use")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")

    client = anthropic.Anthropic(api_key=api_key)

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)

    # Determine which categories to run
    if args.category:
        if args.category not in CATEGORIES:
            raise ValueError(f"Unknown category: {args.category}. Choose from: {list(CATEGORIES)}")
        cats_to_run = {args.category: CATEGORIES[args.category]}
    else:
        cats_to_run = CATEGORIES

    # Load existing if resuming
    existing_by_cat = load_existing(args.out) if args.resume else {}

    # Pre-populate dedup guardrail with already-written prompts so that
    # resume mode rejects duplicates against the on-disk file, not just
    # within the current process's new candidates.
    if existing_by_cat:
        all_existing = [item for items in existing_by_cat.values() for item in items]
        seed_seen_ws(all_existing)
        print(f"  Seeded WS-dedup guardrail with {len(all_existing)} existing items.")

    # Write header comment
    mode = "a" if args.resume else "w"
    total_pairs = 0

    with open(args.out, mode, encoding="utf-8") as out_f:
        for cat_key, cat_def in cats_to_run.items():
            print(f"\n{'='*60}")
            print(f"Category: {cat_key}")
            print(f"{'='*60}")

            existing = existing_by_cat.get(cat_key, [])

            # Skip if already complete
            if args.resume and len(existing) // 2 >= args.n_pairs:
                print(f"  Already complete ({len(existing)//2} pairs). Skipping.")
                total_pairs += len(existing) // 2
                continue

            items = generate_category(
                client, cat_key, cat_def, args.n_pairs, existing=existing
            )
            pairs = assign_ids(items, cat_key)

            # Write to file
            for row in pairs:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

            n = len(pairs) // 2
            total_pairs += n
            print(f"  Written {n} pairs for {cat_key}")

    print(f"\n{'='*60}")
    print(f"Done. Total pairs written: {total_pairs}")
    print(f"Output: {args.out}")
    print(f"\nNext steps:")
    print(f"  1. Review sample: head -40 {args.out} | python -m json.tool")
    print(f"  2. Quality check: python src/data/check_math800.py --in {args.out}")


if __name__ == "__main__":
    main()
