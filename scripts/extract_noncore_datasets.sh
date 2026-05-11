#!/bin/bash
# Non-core dataset extraction: abstentionbench_gsm8k, falseqa, difficulty_control_gsm8k
# For each dataset × 7 models. mathtrap is intentionally excluded.
#
# Uses run_extract_minimal.py (3-file output) rather than run_extract_signals.py
# (9-file legacy output). If you later want to run attention/gradient/own_dist
# analyses on these datasets, re-extract them with run_extract_signals.py.
#
# SUPPORTS RESUME: each task checks if output already exists (reps + meta) and skips.
# Run after core (math800/code800/fact800) re-run is complete.
#
# After extraction finishes, run:
#   python scripts/eval_natural_transfer.py             # abstentionbench + falseqa (zero-shot)
#   python scripts/eval_abstentionbench_lengthcontrol.py # length-controlled subset
#   python scripts/eval_difficulty_control.py            # easy-vs-hard projection
#
# Usage:
#   bash scripts/extract_noncore_datasets.sh 2>&1 | tee logs/noncore_extract_log.txt
#
# Estimated total: ~8-12 hours on MPS
#   difficulty_control (400 prompts):    ~5-10 min per model × 7 models ≈ 1 h
#   abstentionbench_gsm8k (2426):        ~30-60 min per model × 7 models ≈ 5-7 h
#   falseqa (1374):                      ~20-35 min per model × 7 models ≈ 2-4 h

cd "$(dirname "$0")/.."

MODELS="smollm2 gemma2 phi3 mistral qwen llama qwen14b"

# Datasets with (name, form) pairs. All use run_extract_minimal.py.
# difficulty_control first because it's smallest (fast sanity check).
# abstentionbench last because it's largest.
NON_CORE_DATASETS=(
    "difficulty_control_gsm8k MATH"
    "falseqa QA"
    "abstentionbench_gsm8k MATH"
)

echo "============================================================"
echo " NON-CORE RE-RUN: abstentionbench + falseqa + difficulty_control"
echo " Models: $MODELS"
echo " Datasets: difficulty_control_gsm8k, falseqa, abstentionbench_gsm8k"
echo " (mathtrap intentionally excluded)"
echo " Started: $(date)"
echo " Resume mode: completed tasks will be skipped"
echo "============================================================"

FAIL_COUNT=0
SKIP_COUNT=0
RUN_COUNT=0

for ENTRY in "${NON_CORE_DATASETS[@]}"; do
    read -r DATASET FORM <<< "$ENTRY"

    # Sanity check the data file exists
    if [ ! -f "data/${DATASET}.jsonl" ]; then
        echo ""
        echo "  MISSING DATA: data/${DATASET}.jsonl not found, skipping entire dataset"
        continue
    fi

    echo ""
    echo "============================================================"
    echo " EXTRACT: $DATASET (form=$FORM)"
    echo "============================================================"

    for MODEL in $MODELS; do
        OUTDIR="experiments/signals/${DATASET}_${MODEL}_allL/signals"
        MARKER1="$OUTDIR/reps_last_token_all_layers.npy"
        MARKER2="$OUTDIR/meta.jsonl"

        if [ -f "$MARKER1" ] && [ -s "$MARKER1" ] && [ -f "$MARKER2" ] && [ -s "$MARKER2" ]; then
            # Additional check: meta.jsonl row count should match dataset row count
            META_ROWS=$(wc -l < "$MARKER2" | tr -d ' ')
            DATA_ROWS=$(wc -l < "data/${DATASET}.jsonl" | tr -d ' ')
            if [ "$META_ROWS" = "$DATA_ROWS" ]; then
                echo "  SKIP: $DATASET / $MODEL (already done: $META_ROWS rows)"
                SKIP_COUNT=$((SKIP_COUNT + 1))
                continue
            else
                echo "  RESUME: $DATASET / $MODEL (partial: meta=$META_ROWS / data=$DATA_ROWS)"
            fi
        fi

        echo ""
        echo "--- $DATASET / $MODEL --- $(date)"
        if python scripts/run_extract_minimal.py \
            --model "$MODEL" \
            --prompts "data/${DATASET}.jsonl" \
            --run-dir "experiments/signals/${DATASET}_${MODEL}_allL" \
            --forms "$FORM" \
            --all-layers \
            --no-gradients; then
            echo "  Done: ${DATASET}_${MODEL}_allL — $(date)"
            RUN_COUNT=$((RUN_COUNT + 1))
        else
            echo "  FAILED: ${DATASET}_${MODEL} — $(date)"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
done

echo ""
echo "============================================================"
echo " EXTRACTION SUMMARY"
echo "============================================================"
echo " Completed this run: $RUN_COUNT"
echo " Skipped (already done): $SKIP_COUNT"
echo " Failed: $FAIL_COUNT"
echo " Finished: $(date)"
echo "============================================================"

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo ""
    echo " ALL DONE. Next steps:"
    echo "   python scripts/eval_natural_transfer.py"
    echo "   python scripts/eval_abstentionbench_lengthcontrol.py"
    echo "   python scripts/eval_difficulty_control.py"
fi
