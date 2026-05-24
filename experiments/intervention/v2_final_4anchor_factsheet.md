# V2 Final 4-Anchor Factsheet

**Source of truth:** the 12 `intervention_{model}_{dataset}_L{layer}_full_v2.json` files
under `experiments/intervention/`. Sidecar `.md` files were not consulted; all
numbers below are recomputed from JSON.

**4 anchors:** Mistral-7B, Gemma3-4B, Qwen3-14B, Qwen3-8B
**3 datasets:** math800, code800, fact800
**Layers:** Mistral L15/L15/L17 · Gemma3-4B L16/L15/L16 · Qwen3-14B L25/L24/L25 · Qwen3-8B L21/L19/L21

Conventions: ΔG is gated rate gap (signal − random) under the noted
criterion. **IA** = invalidity-aware (concrete answer suppressed,
mixed-output flagged as not-abstention). **RO** = refusal-only (only "I
cannot / I don't know" lexical refusals count as abstention). gateN is
the clean-baseline denominator. ★ marks anchor-quality cells (IA ΔG
≥ +30pp). "Anecdotal" marks gateN ≤ 4 cells (small-N, headline-unsafe).
"Collapse-driven" marks cells where degenerate-rate-signal ≥ 50%.

---

## 1. Headline summary

- **Code is the most consistently steerable domain** (3/4 anchors ≥ +30pp on
  both directions; Qwen3-8B remains positive but weaker).
- **Bidirectional math control is anchor-dependent**: only Mistral-7B
  exceeds +30pp on both math directions; Gemma3-4B and Qwen3-8B show
  strong U→A math control only; Qwen3-14B math fails both directions
  (U→A is +21.4pp under heavy α=20 collapse; A→U is +7.0pp).
- **Fact U→A direction is not a reliable headline metric**: gates are
  empty or extremely small (gateN 0–4) across all 4 anchors, so apparent
  large percentages are anecdotal small-N effects; report as N/A.
- **Refusal-only mechanism is largely absent.** Of 96 non-baseline
  cell × direction × α rows, only **48 are measurable** under the
  refusal-only criterion — the other **48 have empty RO gates**, because
  modern instruct LLMs almost never use explicit "I cannot / I don't know"
  phrasing on math/code/fact U-class clean baselines. Among the 48
  measurable rows, **47 have RO ΔG ≤ +5pp**; the lone exception is
  Mistral math A→U α=20 at +6.0pp (still an order of magnitude below
  the matched IA ΔG of +32.7pp). The combination (48 unmeasurable
  + 47/48 ≤+5pp) provides no evidence that the steered behavior is
  generic refusal-vocabulary expression.
- Within the Qwen3 family, 8B and 14B show different causal profiles
  (8B math U→A anchor / 14B code anchor); we treat this as observation,
  not a size-mediated mechanism.

---

## 2. Full 12-cell ΔG table (anchor-quality marked)

Each row reports the best-IA-ΔG α per (cell, direction). Anecdotal cells
(gateN ≤ 4) are marked N/A here and exposed in §6 only.

### 2.1 U→A direction (remove signal)

| Cell | α | IA ΔG | RO ΔG | gateN | sig% | rnd% | deg_sig | Mark |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mistral / math800 / L15 | 10 | +37.9pp | +0.0pp | 29 | 72.4 | 34.5 | 4.0 | ★ anchor |
| mistral / code800 / L15 | 20 | +44.4pp | +0.0pp | 27 | 77.8 | 33.3 | 0.0 | ★ anchor |
| mistral / fact800 / L17 | — | N/A | N/A | 0 | — | — | — | anecdotal (§6) |
| gemma3_4b / math800 / L16 | 10 | +47.6pp | N/A | 21 | 66.7 | 19.0 | 16.0 | ★ anchor |
| gemma3_4b / code800 / L15 | 10 | +40.0pp | N/A | 15 | 73.3 | 33.3 | 2.0 | ★ anchor |
| gemma3_4b / fact800 / L16 | — | N/A | N/A | 1 | — | — | — | anecdotal (§6) |
| qwen3_14b / math800 / L25 | 20 | +21.4pp | N/A | 28 | 39.3 | 17.9 | 60.4 | + collapse-driven |
| qwen3_14b / code800 / L24 | 5 | +52.2pp | N/A | 23 | 52.2 | 0.0 | 12.5 | ★ anchor |
| qwen3_14b / fact800 / L25 | — | N/A | N/A | 4 | — | — | — | anecdotal (§6) |
| qwen3_8b / math800 / L21 | 20 | +41.7pp | N/A | 24 | 91.7 | 50.0 | 12.5 | ★ anchor |
| qwen3_8b / code800 / L19 | 5 | +23.5pp | N/A | 17 | 35.3 | 11.8 | 18.6 | + |
| qwen3_8b / fact800 / L21 | — | N/A | N/A | 2 | — | — | — | anecdotal (§6) |

### 2.2 A→U direction (inject signal)

| Cell | α | IA ΔG | RO ΔG | gateN | sig% | rnd% | deg_sig | Mark |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| mistral / math800 / L15 | 20 | +32.7pp | +6.0pp | 49 | 38.8 | 6.1 | 6.0 | ★ anchor |
| mistral / code800 / L15 | 40 | +35.4pp | +0.0pp | 48 | 35.4 | 0.0 | 0.0 | ★ anchor |
| mistral / fact800 / L17 | 40 | +4.0pp | +0.0pp | 50 | 4.0 | 0.0 | 66.0 | + collapse-driven |
| gemma3_4b / math800 / L16 | 10 | +16.7pp | +4.1pp | 48 | 16.7 | 0.0 | 8.2 | + |
| gemma3_4b / code800 / L15 | 20 | +35.4pp | +0.0pp | 48 | 37.5 | 2.1 | 0.0 | ★ anchor |
| gemma3_4b / fact800 / L16 | 40 | +24.0pp | +0.0pp | 50 | 24.0 | 0.0 | 8.0 | + |
| qwen3_14b / math800 / L25 | 10 | +7.0pp | +0.0pp | 43 | 7.0 | 0.0 | 6.7 | + |
| qwen3_14b / code800 / L24 | 20 | +37.0pp | +2.1pp | 46 | 37.0 | 0.0 | 8.5 | ★ anchor |
| qwen3_14b / fact800 / L25 | 20 | +20.4pp | +0.0pp | 49 | 20.4 | 0.0 | 12.0 | + |
| qwen3_8b / math800 / L21 | 20 | +20.0pp | +0.0pp | 45 | 20.0 | 0.0 | 48.9 | + |
| qwen3_8b / code800 / L19 | 20 | +22.4pp | +4.1pp | 49 | 24.5 | 2.0 | 20.4 | + |
| qwen3_8b / fact800 / L21 | 20 | +10.0pp | +0.0pp | 50 | 10.0 | 0.0 | 0.0 | + |

**Counts (across both directions, 24 cell-direction slots, anecdotal counted as N/A):**

- ★ anchor-quality (IA ΔG ≥ +30pp, gateN > 4, not collapse-driven): **10 / 24**
  - U→A (6): mistral/math, mistral/code, gemma3_4b/math, gemma3_4b/code, qwen3_14b/code, qwen3_8b/math
  - A→U (4): mistral/math, mistral/code, gemma3_4b/code, qwen3_14b/code
- Positive but sub-anchor (gateN > 4, IA ΔG > 0, below +30pp or collapse-driven): **10 / 24**
  - U→A (2): qwen3_14b/math (collapse-driven), qwen3_8b/code
  - A→U (8): mistral/fact (collapse-driven), gemma3_4b/math, gemma3_4b/fact, qwen3_14b/math, qwen3_14b/fact, qwen3_8b/math, qwen3_8b/code, qwen3_8b/fact
- Anecdotal-only (gateN ≤ 4): **4 / 24** cells (all U→A fact)
- Total: 10 + 10 + 4 = 24 ✓

---

## 3. Cross-anchor patterns

### 3.1 Code domain — robustness across anchors

Code is the most consistently steerable domain in the v2 grid:
- **U→A**: Mistral +44.4pp ★, Gemma3-4B +40.0pp ★, Qwen3-14B +52.2pp ★,
  Qwen3-8B +23.5pp (positive, sub-anchor).
- **A→U**: Mistral +35.4pp ★, Gemma3-4B +35.4pp ★, Qwen3-14B +37.0pp ★,
  Qwen3-8B +22.4pp (positive, sub-anchor).

Three of four anchors hit ≥ +30pp on **both** directions; the fourth
(Qwen3-8B) is positive but weaker on both. Code anchors also show low
degenerate-rate-signal at the best α (Mistral 0.0%, Gemma3-4B 0.0–2.0%,
Qwen3-14B 8.5–12.5%, Qwen3-8B 18.6–20.4%), so the effect is not driven
by collapse.

### 3.2 Math domain — anchor-dependent direction asymmetry

Bidirectional math control is the exception, not the rule:
- **Mistral** is the only anchor that exceeds +30pp on both directions
  (U→A +37.9pp ★ at α=10; A→U +32.7pp ★ at α=20).
- **Gemma3-4B** is anchor-quality U→A only (+47.6pp ★ at α=10) and weak
  A→U (+16.7pp at α=10). Higher α flattens A→U: degenerate-signal goes
  from 6.1% (α=5) to 38.8% (α=10) to 83.7% (α=20) to 71.4% (α=40).
- **Qwen3-8B** matches that pattern: anchor-quality U→A (+41.7pp ★ at
  α=20) and sub-anchor A→U (+20.0pp at α=20, with deg_sig=48.9%).
- **Qwen3-14B math fails both directions** under v2: U→A +21.4pp
  collapse-driven (deg_sig=60.4% at α=20), A→U +7.0pp at α=10.
  Versus v1 (U→A +75.0pp legacy keyword, A→U +31.9pp), this is the
  largest drop in the grid (see §4) and is dominated by the degenerate-output guard
  pruning of degenerate signal flips.

### 3.3 Fact domain — small-N gate problem + structural baseline absence

Fact U→A is structurally hard to measure under v2 because the
invalidity-aware rubric finds very few clean U baselines that pass the
"abstains, no concrete answer in first sentence" filter:
- Mistral: gateN = 0 (cannot measure).
- Gemma3-4B: gateN = 1.
- Qwen3-14B: gateN = 4.
- Qwen3-8B: gateN = 2.

All four U→A fact cells are anecdotal (§6). The "+100pp" at gateN=1
(Gemma3-4B) or "+100pp" at gateN=4 (Qwen3-14B) are not headline-safe
numbers. We report them as N/A in §2 and confine them to §6.

A→U fact, in contrast, is well-measured (gateN ≈ 49–50 across all
anchors) but uniformly sub-anchor: Mistral +4.0pp (collapse, deg_sig
66%), Gemma3-4B +24.0pp, Qwen3-14B +20.4pp, Qwen3-8B +10.0pp. None
hit +30pp.

The right framing is: passage-grounded epistemic abstention is much
less behaviorally coupled than structural impossibility — not "the
effect is zero", but "the U side has too few clean exemplars to anchor
a per-cell ΔG measurement, and the A side, when measurable, sits below
the bidirectional anchor threshold."

### 3.4 Refusal-only mechanism is largely absent

Counted directly from the 12 v2 JSONs: there are **96** non-baseline
condition-α rows (12 cells × 2 directions × 4 non-zero alphas). The
RO criterion has its own per-cell gate (clean baselines where the
model uses explicit "I cannot / I don't know" phrasing); this gate is
much smaller than the IA gate.

- **48 of 96 rows have empty RO gates** (gateN = 0 under the RO
  criterion). These are not "≤ +5pp" — they are unmeasurable and
  provide no evidence either way; report as N/A.
- **Of the 48 measurable RO rows, 47 have ΔG ≤ +5pp**; the lone
  exception is Mistral math A→U α=20 at +6.0pp (sig = 6.0%,
  rnd = 0.0%, RO gateN = 50). Even there, RO sits an order of
  magnitude below the matched IA ΔG (+32.7pp).
- Maximum measurable RO ΔG across the entire grid is +6.0pp;
  second-highest is +4.1pp (Gemma3-4B math A→U α=10; Qwen3-8B code
  A→U α=20). All other measurable RO rows are at +0.0pp or +2.0–2.1pp.

The split (48 unmeasurable + 47/48 measurable ≤ +5pp) is the
methodological evidence for the headline claim: the behavior d_imp
steers is invalidity recognition, not generic refusal phrasing.
Modern instruct LLMs almost never produce explicit "I cannot" / "I
don't know" tokens in math/code/fact U-class baselines, so the RO
criterion has nothing to gate on across most of the grid.

Read: signal removal/injection moves invalidity-aware abstention but
does not move generic refusal-vocabulary abstention. RO is a tight
specificity control, not a competing mechanism that v2 pulled out.

### 3.5 Degenerate handling — what the degenerate-output guard catches

V2 forces flip=False for branches whose generation collapses to a
degenerate (looping / EOS-only / pure-punctuation) trajectory and
excludes degenerate clean baselines from gate denominators. The
contrast against v1 is most visible where degenerate-rate-signal is
high at the best v1 α. Per-cell maxima of degenerate-rate-signal:

| Cell | U→A max deg | A→U max deg |
|---|---:|---:|
| mistral / math / L15 | 86.0% | 92.0% |
| mistral / code / L15 | 4.0% | 2.1% |
| mistral / fact / L17 | 30.0% | 66.0% |
| gemma3_4b / math / L16 | 74.0% | 83.7% |
| gemma3_4b / code / L15 | 61.2% | 4.2% |
| gemma3_4b / fact / L16 | 6.0% | 8.0% |
| qwen3_14b / math / L25 | 97.9% | 82.2% |
| qwen3_14b / code / L24 | 81.2% | 85.1% |
| qwen3_14b / fact / L25 | 67.3% | 100.0% |
| qwen3_8b / math / L21 | 93.8% | 100.0% |
| qwen3_8b / code / L19 | 65.1% | 20.4% |
| qwen3_8b / fact / L21 | 32.0% | 8.0% |

Cells where the v1→v2 drop is concentrated at high α and the v2 best
α is lower than v1's best α generally trace to the degenerate-output guard catching
high-α collapse: Mistral fact A→U (v1 α=40 best, v2 α=40 still best
but ΔG drops 30→4pp under deg_sig=66%), Qwen3-14B math U→A
(v1 +75pp at α=20 → v2 +21pp at α=20 under deg_sig=60.4%), and
Qwen3-8B math A→U (v1 +51pp at α=20 → v2 +20pp at α=20 under
deg_sig=48.9%).

Cells with low max deg (e.g. Mistral code) show small or zero v1→v2
shifts, consistent with degeneracy being non-load-bearing there.

### 3.6 Qwen3 family heterogeneity

Within the Qwen3 family, 8B and 14B show **different** causal profiles
under v2:
- **Qwen3-8B**: math U→A is ★ anchor (+41.7pp); code U→A is sub-anchor
  (+23.5pp).
- **Qwen3-14B**: code U→A is ★ anchor (+52.2pp); math U→A fails
  (+21.4pp under collapse).

A→U directions follow the same split: 8B math A→U +20pp / code A→U
+22pp; 14B math A→U +7pp / code A→U +37pp ★. We treat this as
observation only — there is no size-mediated mechanism we can read off
this 4-anchor grid, and the absence of a Qwen3 mid-size third point
(e.g. 11B) means we cannot decompose size from any other Qwen3 design
choice (post-training mix, rope, layer count) that differs between
8B and 14B.

---

## 4. v1 vs v2 numerical comparison

For each (cell, direction), the best ΔG under v1 (legacy keyword gate)
versus v2 (invalidity-aware + degenerate-aware). v1 chooses α to
maximize `rate_signal_gated − rate_random_gated`; v2 chooses α to
maximize IA `delta_gated`. Cause set: GB = gate broadening (legacy
keyword missed invalidity vocab); MO = mixed-output FP catch (concrete
answer + appended caveat now correctly flagged not-abstention); DEG =
degenerate punishment (the degenerate-output guard); SN = small-N reveal (legacy
gate too small/large to be reliable); LX = lexical FP catch in clean
baseline (legacy keyword over-counted invalidity in clean baseline).

| Cell | Dir | v1 best | v2 best | Δ shift | Primary cause |
|---|---|---:|---:|---:|---|
| mistral / math800 / L15 | U→A | +80.0pp | +37.9pp | −42.1pp | LX + MO + DEG |
| mistral / math800 / L15 | A→U | +40.8pp | +32.7pp | −8.2pp | MO |
| mistral / code800 / L15 | U→A | +71.4pp | +44.4pp | −27.0pp | GB (clean baseline grew, ΔG diluted) |
| mistral / code800 / L15 | A→U | +35.4pp | +35.4pp | +0.0pp | (stable; minimal MO/DEG) |
| mistral / fact800 / L17 | U→A | +100.0pp | N/A | — | SN reveal (v1 gateN=2 → v2 gateN=0; LX in v1 baseline) |
| mistral / fact800 / L17 | A→U | +34.0pp | +4.0pp | −30.0pp | DEG (deg_sig=66% at α=40) + MO |
| gemma3_4b / math800 / L16 | U→A | +43.5pp | +47.6pp | +4.1pp | (slight; v2 LX in clean baseline gives smaller v2 gate, ΔG up) |
| gemma3_4b / math800 / L16 | A→U | +55.1pp | +16.7pp | −38.4pp | MO + DEG |
| gemma3_4b / code800 / L15 | U→A | +25.0pp | +40.0pp | +15.0pp | GB (v1 gateN=4 → v2 gateN=15; v1 was small-N inflation, v2 is a real measurement on a wider gate) |
| gemma3_4b / code800 / L15 | A→U | +68.0pp | +35.4pp | −32.6pp | MO |
| gemma3_4b / fact800 / L16 | U→A | +40.0pp | +100.0pp | +60.0pp | SN (anecdotal; gateN=5 → gateN=1, do not headline) |
| gemma3_4b / fact800 / L16 | A→U | +32.0pp | +24.0pp | −8.0pp | MO |
| qwen3_14b / math800 / L25 | U→A | +75.0pp | +21.4pp | −53.6pp | DEG (deg_sig=60.4% at α=20) + MO |
| qwen3_14b / math800 / L25 | A→U | +31.9pp | +7.0pp | −24.9pp | MO + DEG |
| qwen3_14b / code800 / L24 | U→A | +71.4pp | +52.2pp | −19.3pp | MO + best α dropped 10→5 (degeneracy at α=10 caught) |
| qwen3_14b / code800 / L24 | A→U | +58.0pp | +37.0pp | −21.0pp | MO |
| qwen3_14b / fact800 / L25 | U→A | +40.0pp | +100.0pp | +60.0pp | SN (anecdotal; gateN=5 → gateN=4, do not headline) |
| qwen3_14b / fact800 / L25 | A→U | +51.0pp | +20.4pp | −30.6pp | MO + DEG |
| qwen3_8b / math800 / L21 | U→A | +57.9pp | +41.7pp | −16.2pp | MO + best α moved 10→20 |
| qwen3_8b / math800 / L21 | A→U | +51.0pp | +20.0pp | −31.0pp | MO + DEG (deg_sig=48.9%) |
| qwen3_8b / code800 / L19 | U→A | +50.0pp | +23.5pp | −26.5pp | MO + GB (deg_sig=18.6% at α=5) |
| qwen3_8b / code800 / L19 | A→U | +16.0pp | +22.4pp | +6.4pp | SN reveal at α=20 (v1 best was α=10) |
| qwen3_8b / fact800 / L21 | U→A | +66.7pp | +50.0pp | −16.7pp | SN (anecdotal both v1 and v2; gateN=3 → gateN=2) |
| qwen3_8b / fact800 / L21 | A→U | +40.0pp | +10.0pp | −30.0pp | MO + DEG |

**Aggregate read:** 18 of 24 (cell, direction) slots show v2 < v1; 4
show v2 > v1 (3 are SN-reveal artifacts in fact U→A, only 1 is a
substantive widening — Gemma3-4B code U→A); 2 are essentially flat
(±≤6pp). The downward shifts cluster on math A→U and Qwen3 cells with
high deg_sig.

---

## 5. Protocol upgrade audit (for Appendix)

The v2 protocol is a **refinement** of v1's labeling pipeline, not a
method change. Both protocols are LLM-assisted human-adjudicated; both
operate on the same 900-generation TSVs per cell; both compute gated
rate gaps. Differences are scoped to the rubric, the degenerate
handling, and the audit pass.

### 5.1 What v2 adds over v1

| Aspect | v1 (legacy keyword) | v2 (invalidity-aware + degenerate-aware) |
|---|---|---|
| Rubric | Single keyword/lexical match for "abstention" (e.g. "undefined", "I cannot", "no answer"). | Invalidity-aware rubric with domain-specific structural categories (math: div-zero / sqrt-neg / log-neg etc.; code: raises X; fact: epistemic unanswerability). Two criteria reported in parallel: IA (full structural rubric) and RO (refusal-only sub-criterion using "I cannot / I don't know"). |
| Mixed output | Counted as abstention if any abstention keyword present. | Stage B Rule 3: concrete answer with appended invalidity caveat → IA = no. Caught the systematic FP class (≈27% of provisionally-kept candidate flips on Mistral code). |
| Degenerate generations | Silently included in flip and gate counts. | The degenerate-output guard: degenerate branches (looping / EOS-only / pure-punctuation) forced flip=False; clean degenerate baselines excluded from gate denominator. |
| Audit | LLM-assisted candidate + human override. | LLM-assisted candidate + Stage-B in-memory rubric fill + 10% self-check + pass-2 second-read on the flagged subset (catches both under-apply and over-apply of Rule 3). |
| Empty-gate handling | gateN=0 silently rendered as ΔG = 0. | gateN=0 rendered as N/A (and surfaced as such in §2 / §6). |
| Reported metrics | Single ΔG per (cell, direction, α). | IA ΔG and RO ΔG reported separately; degenerate-signal and degenerate-random rates exposed per row. |

### 5.2 Framing

v2 is described as a "protocol refinement" because:
- Both v1 and v2 operate on the same generations and the same
  signal-projection arithmetic; what changed is what counts as
  "abstention" and what counts as "valid baseline."
- v1's positive direction (large ΔG) is preserved by v2 in code
  cells, where the signal is real and degeneracy is low.
- v1's claim that fact U→A is anchor-quality does not survive v2 —
  not because the underlying behavior changed, but because v2's
  rubric exposes that v1's gate denominator on fact U→A was 2–5
  rows of lexical noise rather than a real impossibility-recognizing
  baseline population.

---

## 6. Anecdotal cell registry

Cells with IA-criterion gateN ≤ 4. **None of these may appear in
headline tables (§2 marks them N/A).** They are listed here for
appendix robustness only.

All anecdotal rows are in the U→A fact direction:

| Cell | Direction | α | gateN | IA ΔG | sig% | rnd% | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| mistral / fact800 / L17 | U→A | 5 | 0 | N/A | N/A | N/A | empty gate, not measurable |
| mistral / fact800 / L17 | U→A | 10 | 0 | N/A | N/A | N/A | empty gate |
| mistral / fact800 / L17 | U→A | 20 | 0 | N/A | N/A | N/A | empty gate |
| mistral / fact800 / L17 | U→A | 40 | 0 | N/A | N/A | N/A | empty gate (deg_sig=30%, deg_rnd=44%) |
| gemma3_4b / fact800 / L16 | U→A | 5 | 1 | +0.0pp | 0.0 | 0.0 | single-row gate |
| gemma3_4b / fact800 / L16 | U→A | 10 | 1 | +100.0pp | 100.0 | 0.0 | single-row gate; misleading |
| gemma3_4b / fact800 / L16 | U→A | 20 | 1 | +0.0pp | 100.0 | 100.0 | single-row gate |
| gemma3_4b / fact800 / L16 | U→A | 40 | 1 | +100.0pp | 100.0 | 0.0 | single-row gate; misleading |
| qwen3_14b / fact800 / L25 | U→A | 5 | 4 | +50.0pp | 50.0 | 0.0 | 4-row gate |
| qwen3_14b / fact800 / L25 | U→A | 10 | 4 | +100.0pp | 100.0 | 0.0 | 4-row gate; misleading |
| qwen3_14b / fact800 / L25 | U→A | 20 | 4 | +50.0pp | 75.0 | 25.0 | 4-row gate |
| qwen3_14b / fact800 / L25 | U→A | 40 | 4 | +75.0pp | 75.0 | 0.0 | 4-row gate; deg_rnd=95.9% |
| qwen3_8b / fact800 / L21 | U→A | 5 | 2 | +50.0pp | 50.0 | 0.0 | 2-row gate |
| qwen3_8b / fact800 / L21 | U→A | 10 | 2 | +0.0pp | 50.0 | 50.0 | 2-row gate |
| qwen3_8b / fact800 / L21 | U→A | 20 | 2 | −50.0pp | 0.0 | 50.0 | 2-row gate; sign flip is noise |
| qwen3_8b / fact800 / L21 | U→A | 40 | 2 | −50.0pp | 50.0 | 100.0 | 2-row gate; deg_rnd=0% but rnd=100% |

**Total anecdotal rows:** 16 (all U→A fact, all four anchors).

The pattern is uniform: SQuAD-style passage-grounded epistemic
unanswerability does not produce many "clean abstention without
appended concrete answer" baselines under the IA rubric. v1 reported
positive U→A fact ΔG numbers because its keyword gate accepted lexical
noise (e.g. "no" / "not" / "unknown" appearing in long-form passages);
v2 surfaces that the IA-clean baseline population is structurally too
small to support a per-cell ΔG measurement on this direction.

---

## Summary stats (printed to stdout for handoff)

- Cells: 12 (4 anchors × 3 datasets)
- Total (cell, direction) slots: 24
- ★ anchor-quality slots (IA ΔG ≥ +30pp, gateN > 4, not collapse-driven): **10 / 24**
  - U→A: 6 (mistral/math, mistral/code, gemma3_4b/math, gemma3_4b/code, qwen3_14b/code, qwen3_8b/math)
  - A→U: 4 (mistral/math, mistral/code, gemma3_4b/code, qwen3_14b/code)
- Sub-anchor positive slots: **10 / 24** (includes 2 collapse-driven: qwen3_14b/math U→A, mistral/fact A→U)
- Anecdotal-only slots (gateN ≤ 4): **4 / 24** (all U→A fact)
- Anecdotal individual rows (gateN ≤ 4 across all α): **16**
- Refusal-only measurable rows: **48 / 96** (other 48 have empty RO gates → N/A, not "≤ +5pp")
- Refusal-only measurable rows ≤ +5pp: **47 / 48** (lone exception: mistral/math A→U α=20, +6.0pp)
- v1→v2 (cell, direction) shift signs: **18** down, **4** up (3 of which are SN artifacts in fact U→A), **2** flat
- v2 cells unmeasurable on U→A direction (gateN=0): **1** (mistral/fact)
