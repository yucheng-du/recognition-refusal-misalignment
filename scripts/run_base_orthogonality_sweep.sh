#!/bin/bash
# Base-model pilot: probe whether the impossibility direction is orthogonal to
# the harmful-vs-harmless axis in a NON-RLHF model.
#
# Supports multi-dataset sweeps (default: math800 + code800).
# Set DATASETS env var to override: DATASETS="math800" for math-only.
#
# ⚠ IMPORTANT NAMING CAVEAT — read before interpreting results:
#
#   For base models that do not reliably refuse (Llama/Mistral base typically 0-2/50),
#   behavior verification falls back to "all 50 harmful + 50 harmless prompts".
#   The resulting vector is NOT a behavior-verified refusal direction; it is a
#   harmful-vs-harmless proxy axis. Check the per-run JSON
#   `behavior_verified` + `direction_interpretation` fields.
#
#   Qwen-family base models are an exception: qwen_base may verify with
#   ~30-40/50 refusals, plausibly reflecting instruction-like or conversational
#   supervision already present in the pretraining regime. This yields a clean
#   verified-vs-verified comparison against qwen instruct.
#
# Pilot default: Llama-3.1-8B base. Override with MODEL env var.
#
# What this runs (for each dataset in DATASETS):
#   1. Extraction on {dataset}_{model}_allL (minimal 3-file output).
#   2. Orthogonality comparison at LAYERS around the instruct peak (±2).
#      Each comparison derives d_imp from the SAME dataset's extraction.
#
# Time on M1/M2/M3 Mac (64GB), per model:
#   extraction/dataset : 1.7B ~10min | 7B ~25min | 8B ~30min | 14B ~60-90min
#   3-layer compare    : 1.7B ~10min | 7B ~15min | 14B ~25min
#   Rough totals (math + code, 6 layer runs): 1.7B ~40min, 7B ~1.3h, 14B ~3-3.5h
#
# Prereq:
#   - `.venv` activated (pilot uses bare `python`).
#   - HF access for any gated models (e.g., meta-llama/Llama-3.1-8B).
#
# Usage (run from repo root):
#   bash scripts/run_base_orthogonality_sweep.sh 2>&1 | tee logs/base_pilot_log.txt
#   DATASETS="math800" bash scripts/run_base_orthogonality_sweep.sh                # math-only fast mode
#   DATASETS="math800 code800" MODEL=qwen_base bash scripts/run_base_orthogonality_sweep.sh
#   SWEEP_LAYERS="18" MODEL=qwen14b_base bash scripts/run_base_orthogonality_sweep.sh   # single-layer mode
#
# Supported base keys:
#   llama_base, mistral_base, qwen_base, qwen14b_base, gemma2_base, smollm2_base,
#   mistral_small_base, olmo13b_base

cd "$(dirname "$0")/.."

MODEL="${MODEL:-llama_base}"
DATASETS="${DATASETS:-math800 code800}"
PYTHON="${PYTHON:-python3}"

# Corresponding instruct baseline + its peak layer from BEST_LAYERS.
# Default sweep is peak ± 2. Override with SWEEP_LAYERS env var.
case "$MODEL" in
    llama_base)    INSTRUCT_BASELINE="llama";   PEAK_LAYER=15 ;;
    mistral_base)  INSTRUCT_BASELINE="mistral"; PEAK_LAYER=15 ;;
    qwen_base)     INSTRUCT_BASELINE="qwen";    PEAK_LAYER=18 ;;
    qwen14b_base)  INSTRUCT_BASELINE="qwen14b"; PEAK_LAYER=34 ;;
    gemma2_base)   INSTRUCT_BASELINE="gemma2";  PEAK_LAYER=16 ;;
    smollm2_base)  INSTRUCT_BASELINE="smollm2"; PEAK_LAYER=11 ;;
    mistral_small_base) INSTRUCT_BASELINE="mistral_small"; PEAK_LAYER=28 ;;
    olmo13b_base)  INSTRUCT_BASELINE="olmo13b"; PEAK_LAYER=20 ;;
    *)             INSTRUCT_BASELINE="";        PEAK_LAYER=15 ;;
esac

SWEEP_LAYERS="${SWEEP_LAYERS:-$((PEAK_LAYER - 2)) $PEAK_LAYER $((PEAK_LAYER + 2))}"

# Dataset → form mapping
dataset_form() {
    case "$1" in
        math800)   echo "MATH" ;;
        code800)   echo "CODE" ;;
        fact800)   echo "FACT" ;;
        falseqa)   echo "QA" ;;
        abstentionbench_gsm8k) echo "MATH" ;;
        difficulty_control_gsm8k) echo "MATH" ;;
        *) echo "UNKNOWN" ;;
    esac
}

echo "============================================================"
echo " BASE-MODEL ORTHOGONALITY PILOT"
echo " Model         : $MODEL"
echo " Datasets      : $DATASETS"
echo " Sweep layers  : $SWEEP_LAYERS"
echo " Instruct comp : ${INSTRUCT_BASELINE:-<none>}"
echo " Start         : $(date)"
echo "============================================================"
echo ""
echo " NOTE: For most base models, the second direction is a HARMFUL-vs-"
echo "       HARMLESS PROXY AXIS (not behavior-verified refusal). See the"
echo "       behavior_verified field in each result JSON."
echo ""

TOTAL_FAIL=0

for DATASET in $DATASETS; do
    FORM=$(dataset_form "$DATASET")
    if [ "$FORM" = "UNKNOWN" ]; then
        echo "  ⚠ Unknown dataset form for $DATASET — skipping"
        continue
    fi

    echo ""
    echo "############################################################"
    echo "# DATASET: $DATASET (form=$FORM)"
    echo "############################################################"

    # ── Extraction ────────────────────────────────────────────────
    OUTDIR="experiments/signals/${DATASET}_${MODEL}_allL/signals"
    MARKER1="$OUTDIR/reps_last_token_all_layers.npy"
    MARKER2="$OUTDIR/meta.jsonl"
    SKIP_EXTRACT=0
    if [ -f "$MARKER1" ] && [ -s "$MARKER1" ] && [ -f "$MARKER2" ] && [ -s "$MARKER2" ]; then
        META_ROWS=$(wc -l < "$MARKER2" | tr -d ' ')
        DATA_ROWS=$(wc -l < "data/${DATASET}.jsonl" | tr -d ' ')
        if [ "$META_ROWS" = "$DATA_ROWS" ]; then
            echo "  SKIP extraction: $DATASET/${MODEL} already done ($META_ROWS rows)"
            SKIP_EXTRACT=1
        fi
    fi

    if [ "$SKIP_EXTRACT" -eq 0 ]; then
        echo "============================================================"
        echo " EXTRACT: $DATASET/${MODEL} — $(date)"
        echo "============================================================"
        if "$PYTHON" scripts/run_extract_minimal.py \
            --model "$MODEL" \
            --prompts "data/${DATASET}.jsonl" \
            --run-dir "experiments/signals/${DATASET}_${MODEL}_allL" \
            --forms "$FORM" \
            --all-layers --no-gradients; then
            echo "  Done extraction $DATASET/$MODEL — $(date)"
        else
            echo "  FAILED extraction $DATASET/$MODEL — $(date)"
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
            echo "  Skipping compare step for $DATASET (extraction missing)."
            continue
        fi
    fi

    # ── Layer sweep comparisons ──────────────────────────────────
    for LAYER in $SWEEP_LAYERS; do
        ORTH_OUT="experiments/direction_comparison_${MODEL}_${DATASET}_L${LAYER}.json"
        if [ -f "$ORTH_OUT" ] && [ -s "$ORTH_OUT" ]; then
            echo ""
            echo "  SKIP compare $DATASET L${LAYER}: $ORTH_OUT exists"
            continue
        fi
        echo ""
        echo "============================================================"
        echo " COMPARE: $MODEL @ $DATASET L$LAYER — $(date)"
        echo "============================================================"
        if "$PYTHON" scripts/compare_impossibility_vs_refusal_direction.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --layer "$LAYER" \
            --out-suffix "_${DATASET}_L${LAYER}"; then
            echo "  Done compare $DATASET L${LAYER} — $(date)"
        else
            echo "  FAILED compare $DATASET L${LAYER} — $(date)"
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
        fi
    done
done

# ── Final summary ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " PIPELINE SUMMARY — $MODEL"
echo "============================================================"
echo " Finished : $(date)"
echo " Failures : $TOTAL_FAIL"
echo ""

"$PYTHON" - <<PY
import json, os

model = "$MODEL"
datasets = "$DATASETS".split()
layers = "$SWEEP_LAYERS".split()
instruct = "$INSTRUCT_BASELINE"

print(f"  Results for {model}:")
print(f"  {'Dataset':<10} {'Layer':<7} {'cos':>8} {'95% CI':<22} {'verified':>10} {'n_harm':>7}")
print(f"  {'-'*70}")

all_rows = []
for ds in datasets:
    for L in layers:
        p = f"experiments/direction_comparison_{model}_{ds}_L{L}.json"
        if not os.path.exists(p):
            print(f"  {ds:<10} L{L:<5}  <missing>")
            continue
        r = json.load(open(p))
        ci = f"[{r['bootstrap_cos_ci95_lo']:+.3f},{r['bootstrap_cos_ci95_hi']:+.3f}]"
        ver = str(r.get('behavior_verified', '?'))
        nh = r.get('n_harmful_verified_raw', '?')
        print(f"  {ds:<10} L{L:<5}  {r['cos_matched_full']:>+8.4f}  "
              f"{ci:<22} {ver:>10} {nh:>7}")
        all_rows.append(r)

if not all_rows:
    raise SystemExit

# Instruct comparison at matched layer
if instruct:
    print()
    for ds in datasets:
        # Instruct was run without --dataset prefix historically; check both paths
        for inst_path in [f"experiments/direction_comparison_{instruct}.json",
                          f"experiments/direction_comparison_{instruct}_{ds}.json",
                          f"experiments/direction_comparison_{instruct}_v2.json"]:
            if os.path.exists(inst_path):
                i = json.load(open(inst_path))
                if i.get("dataset", "math800") != ds:
                    continue   # only compare matching dataset
                print(f"  Instruct {instruct} @ {ds} L{i['layer']}:")
                print(f"    cos = {i['cos_matched_full']:+.4f}  CI = "
                      f"[{i['bootstrap_cos_ci95_lo']:+.3f},"
                      f"{i['bootstrap_cos_ci95_hi']:+.3f}]  "
                      f"verified={i.get('behavior_verified','?')} "
                      f"({i.get('n_harmful_verified_raw','?')}/50)")
                break
PY

echo ""
echo "============================================================"
