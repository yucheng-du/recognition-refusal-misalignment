#!/bin/bash
# Model suite — full extension run for adding a new instruct/base model pair to
# the current paper pipeline. Defaults to OLMo-2-13B (the original anchor model
# this suite was built for); override via env vars (see Usage below).
#
# This is the "full suite" for adding a new model to the current paper
# pipeline, not the historical full archive of exploratory scripts.
# It runs:
#   1. minimal extraction for instruct math/code/fact
#   2. minimal extraction for base math/code
#   3. layer emergence on instruct math/code
#   4. detection/GSRS, form/category, cross-domain transfer
#   5. instruct orthogonality on math/code
#   6. base orthogonality local sweeps
#   7. instruct steering breadth on math/code/fact
#
# Heavy intervention (intervention_nullspace.py) is intentionally excluded.
# Run it separately only for a selected causal anchor.
#
# Usage (run from repo root):
#   bash scripts/run_model_suite.sh                                    # OLMo-13B defaults
#
# Override defaults via env vars:
#   MODEL=qwen32b BASE_MODEL=qwen32b_base bash scripts/run_model_suite.sh

cd "$(dirname "$0")/.."

MODEL="${MODEL:-olmo13b}"
# `${BASE_MODEL-default}` (no colon): only fall back when UNSET; explicit
# empty `BASE_MODEL=` stays empty so the Phase 2/8 guards trigger. Use this
# form for families with no base release (e.g. Phi-4-mini).
BASE_MODEL="${BASE_MODEL-olmo13b_base}"
PYTHON="${PYTHON:-python3}"
ALPHAS="${ALPHAS:-0,5,10,20,30,40}"
N_STEER="${N_STEER:-100}"

FAIL_COUNT=0

dataset_form() {
    case "$1" in
        math800) echo "MATH" ;;
        code800) echo "CODE" ;;
        fact800) echo "FACT" ;;
        *) echo "UNKNOWN" ;;
    esac
}

row_count() {
    wc -l < "$1" | tr -d ' '
}

extract_one() {
    local model="$1"
    local dataset="$2"
    local form
    form="$(dataset_form "$dataset")"
    if [ "$form" = "UNKNOWN" ]; then
        echo "UNKNOWN dataset form: $dataset"
        return 1
    fi

    local data_file="data/${dataset}.jsonl"
    local out_dir="experiments/signals/${dataset}_${model}_allL/signals"
    local marker_reps="$out_dir/reps_last_token_all_layers.npy"
    local marker_meta="$out_dir/meta.jsonl"

    if [ ! -f "$data_file" ]; then
        echo "MISSING: $data_file"
        return 1
    fi

    if [ -s "$marker_reps" ] && [ -s "$marker_meta" ]; then
        local meta_rows data_rows
        meta_rows="$(row_count "$marker_meta")"
        data_rows="$(row_count "$data_file")"
        if [ "$meta_rows" = "$data_rows" ]; then
            echo "SKIP extraction: $dataset/$model ($meta_rows rows)"
            return 0
        fi
    fi

    echo ""
    echo "============================================================"
    echo "EXTRACT: $dataset/$model (form=$form) -- $(date)"
    echo "============================================================"
    "$PYTHON" scripts/run_extract_minimal.py \
        --model "$model" \
        --prompts "$data_file" \
        --run-dir "experiments/signals/${dataset}_${model}_allL" \
        --forms "$form" \
        --all-layers --no-gradients
}

run_if_missing() {
    local marker="$1"
    shift
    if [ -s "$marker" ]; then
        echo "SKIP: $marker exists"
        return 0
    fi
    "$@"
}

run_layer_emergence() {
    local dataset="$1"
    local out="experiments/layer_emergence_results_model-${MODEL}_ds-${dataset}.json"
    run_if_missing "$out" "$PYTHON" scripts/analyze_layer_emergence.py --model "$MODEL" --dataset "$dataset"
}

get_peak_layer() {
    local dataset="$1"
    "$PYTHON" - "$MODEL" "$dataset" <<'PY'
import json, sys
model, dataset = sys.argv[1], sys.argv[2]
path = f"experiments/layer_emergence_results_model-{model}_ds-{dataset}.json"
data = json.load(open(path))
rows = [r for r in data["per_config"] if r["model"] == model and r["dataset"] == dataset]
if not rows:
    raise SystemExit(f"No layer emergence row for {model}/{dataset}")
print(int(rows[0]["peak_layer"]))
PY
}

sweep_layers() {
    local peak="$1"
    local lo=$((peak - 2))
    local hi=$((peak + 2))
    if [ "$lo" -lt 0 ]; then
        lo=0
    fi
    echo "$lo $peak $hi"
}

echo "============================================================"
echo " MODEL SUITE — Recognition ⊥ Refusal pipeline"
echo " Instruct : $MODEL"
echo " Base     : $BASE_MODEL"
echo " Start    : $(date)"
echo "============================================================"
echo "This run is resumable. Existing complete outputs are skipped."
echo ""

echo ">>> Phase 1: Instruct extraction"
for ds in math800 code800 fact800; do
    if ! extract_one "$MODEL" "$ds"; then
        echo "FAILED extraction: $ds/$MODEL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        if [ "$ds" = "math800" ]; then
            echo "Core math800 extraction failed; stopping to avoid cascade failures."
            exit 1
        fi
    fi
done

echo ""
echo ">>> Phase 2: Base extraction"
if [ -z "$BASE_MODEL" ]; then
    echo "SKIP base extraction: BASE_MODEL is empty (no base release for this family)"
else
    for ds in math800 code800; do
        if ! extract_one "$BASE_MODEL" "$ds"; then
            echo "FAILED extraction: $ds/$BASE_MODEL"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    done
fi

echo ""
echo ">>> Phase 3: Layer emergence"
for ds in math800 code800; do
    if ! run_layer_emergence "$ds"; then
        echo "FAILED layer emergence: $ds/$MODEL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done

MATH_LAYER="$(get_peak_layer math800)" || exit 1
CODE_LAYER="$(get_peak_layer code800)" || exit 1
FACT_LAYER="$MATH_LAYER"
echo ""
echo "Peak layers:"
echo "  math800: L$MATH_LAYER"
echo "  code800: L$CODE_LAYER"
echo "  fact800: L$FACT_LAYER (using math peak; no layer-emergence sweep for fact)"

echo ""
echo ">>> Phase 4: Detection / GSRS ablation"
run_if_missing "experiments/ablation_nullpc_results_model-${MODEL}_ds-math800.json" \
    "$PYTHON" scripts/ablation_nullspace_vs_pcspace.py --model "$MODEL" --dataset math800 --layer "$MATH_LAYER" \
    || FAIL_COUNT=$((FAIL_COUNT + 1))
run_if_missing "experiments/ablation_nullpc_results_model-${MODEL}_ds-code800.json" \
    "$PYTHON" scripts/ablation_nullspace_vs_pcspace.py --model "$MODEL" --dataset code800 --layer "$CODE_LAYER" \
    || FAIL_COUNT=$((FAIL_COUNT + 1))

echo ""
echo ">>> Phase 5: Form / category analysis"
run_if_missing "experiments/form_conditionality_results_model-${MODEL}_ds-math800.json" \
    "$PYTHON" scripts/analyze_form_conditionality.py --model "$MODEL" --dataset math800 --layer "$MATH_LAYER" \
    || FAIL_COUNT=$((FAIL_COUNT + 1))
run_if_missing "experiments/form_conditionality_results_model-${MODEL}_ds-code800.json" \
    "$PYTHON" scripts/analyze_form_conditionality.py --model "$MODEL" --dataset code800 --layer "$CODE_LAYER" \
    || FAIL_COUNT=$((FAIL_COUNT + 1))

echo ""
echo ">>> Phase 6: Cross-domain transfer"
run_if_missing "experiments/cross_domain_transfer_model-${MODEL}.json" \
    "$PYTHON" scripts/eval_cross_domain_transfer.py --model "$MODEL" --layer "$MATH_LAYER" \
    || FAIL_COUNT=$((FAIL_COUNT + 1))

echo ""
echo ">>> Phase 7: Instruct orthogonality"
run_if_missing "experiments/direction_comparison_${MODEL}.json" \
    "$PYTHON" scripts/compare_impossibility_vs_refusal_direction.py --model "$MODEL" --layer "$MATH_LAYER" \
    || FAIL_COUNT=$((FAIL_COUNT + 1))
run_if_missing "experiments/direction_comparison_${MODEL}_code800_L${CODE_LAYER}.json" \
    "$PYTHON" scripts/compare_impossibility_vs_refusal_direction.py \
        --model "$MODEL" \
        --dataset code800 \
        --layer "$CODE_LAYER" \
        --out-suffix "_code800_L${CODE_LAYER}" \
    || FAIL_COUNT=$((FAIL_COUNT + 1))

echo ""
echo ">>> Phase 8: Base orthogonality sweeps"
if [ -z "$BASE_MODEL" ]; then
    echo "SKIP base orthogonality sweeps: BASE_MODEL is empty (no base release for this family)"
else
    MATH_SWEEP="$(sweep_layers "$MATH_LAYER")"
    CODE_SWEEP="$(sweep_layers "$CODE_LAYER")"
    MODEL="$BASE_MODEL" DATASETS="math800" SWEEP_LAYERS="$MATH_SWEEP" PYTHON="$PYTHON" bash scripts/run_base_orthogonality_sweep.sh \
        || FAIL_COUNT=$((FAIL_COUNT + 1))
    MODEL="$BASE_MODEL" DATASETS="code800" SWEEP_LAYERS="$CODE_SWEEP" PYTHON="$PYTHON" bash scripts/run_base_orthogonality_sweep.sh \
        || FAIL_COUNT=$((FAIL_COUNT + 1))
fi
# Note: env-var prefix on the bash invocations above does NOT mutate this
# shell's MODEL — it stays at the user's chosen instruct model (default olmo13b).

echo ""
echo ">>> Phase 9: Steering breadth"
run_if_missing "experiments/steering/steering_${MODEL}_math800_L${MATH_LAYER}.json" \
    "$PYTHON" scripts/impossibility_steering.py --model "$MODEL" --dataset math800 --layer "$MATH_LAYER" --n-samples "$N_STEER" --alphas "$ALPHAS" \
    || FAIL_COUNT=$((FAIL_COUNT + 1))
run_if_missing "experiments/steering/steering_${MODEL}_code800_L${CODE_LAYER}.json" \
    "$PYTHON" scripts/impossibility_steering.py --model "$MODEL" --dataset code800 --layer "$CODE_LAYER" --n-samples "$N_STEER" --alphas "$ALPHAS" \
    || FAIL_COUNT=$((FAIL_COUNT + 1))
run_if_missing "experiments/steering/steering_${MODEL}_fact800_L${FACT_LAYER}.json" \
    "$PYTHON" scripts/impossibility_steering.py --model "$MODEL" --dataset fact800 --layer "$FACT_LAYER" --n-samples "$N_STEER" --alphas "$ALPHAS" \
    || FAIL_COUNT=$((FAIL_COUNT + 1))

echo ""
echo "============================================================"
echo " Model suite finished"
echo " End      : $(date)"
echo " Failures : $FAIL_COUNT"
echo "============================================================"
exit "$FAIL_COUNT"
