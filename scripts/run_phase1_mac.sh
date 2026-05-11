#!/bin/bash
# Phase 1 Mac kickoff — runs the main suite for the 4 dense-successor upgrades:
#   phi4mini, gemma3_4b, qwen3_8b, qwen3_14b
#
# Usage (from anywhere):
#   bash /path/to/repo/scripts/run_phase1_mac.sh
#
# Or chmod +x once and call as ./scripts/run_phase1_mac.sh from repo root.
#
# Behavior:
#   * Auto-cd into the repo root (resolved from this script's location).
#   * Verifies HF_HOME points at the external drive (warns if not).
#   * Starts caffeinate to keep the Mac awake; cleans it up on exit/Ctrl-C.
#   * Runs the 4 suites sequentially, smallest → largest. Each one is
#     internally resumable — if a model already finished, its phases skip.
#   * tee's each model's stdout+stderr to logs/<model>_<timestamp>.log.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || { echo "ERROR: cannot cd to $REPO"; exit 1; }

mkdir -p logs

# ── 1. Sanity checks ──────────────────────────────────────────────────────────
echo "============================================================"
echo " Phase 1 Mac suite — starting at $(date)"
echo "============================================================"
echo "Repo     : $REPO"
echo "HF_HOME  : ${HF_HOME:-(unset — will fall back to ~/.cache/huggingface)}"
echo "Python   : $(which python)"
echo ""

if [ -z "${HF_HOME:-}" ]; then
    echo "WARNING: HF_HOME is unset. Weights will land in ~/.cache/huggingface"
    echo "         (the default location)."
    echo "         Ctrl-C now and 'export HF_HOME=/path/to/your/hf-cache' if you"
    echo "         want them on a different volume."
    echo "         Continuing in 10s..."
    sleep 10
fi

# ── 2. Caffeinate to keep Mac awake ───────────────────────────────────────────
caffeinate -dimsu &
CAFFEINATE_PID=$!
echo "caffeinate PID: $CAFFEINATE_PID"

cleanup() {
    if kill -0 "$CAFFEINATE_PID" 2>/dev/null; then
        kill "$CAFFEINATE_PID" 2>/dev/null
        echo "caffeinate stopped."
    fi
}
trap cleanup EXIT INT TERM

# ── 3. Run the 4 suites sequentially ──────────────────────────────────────────
run_suite() {
    local model="$1"
    local base_model="$2"
    local ts
    ts="$(date +%Y%m%d_%H%M)"
    local log="logs/${model}_${ts}.log"

    echo ""
    echo "============================================================"
    echo " [$model] starting at $(date)"
    echo " base   : ${base_model:-<empty — base phases will skip>}"
    echo " log    : $log"
    echo "============================================================"

    MODEL="$model" BASE_MODEL="$base_model" \
        bash scripts/run_model_suite.sh 2>&1 | tee "$log"

    local rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
        echo "[$model] suite returned exit $rc (FAIL_COUNT inside the suite)"
        echo "[$model] continuing to next model — inspect $log for failed phases"
    else
        echo "[$model] finished cleanly at $(date)"
    fi
}

run_suite "phi4mini"   ""                  # no base — Phase 2/8 skip
run_suite "gemma3_4b"  "gemma3_4b_base"
run_suite "qwen3_8b"   "qwen3_8b_base"
run_suite "qwen3_14b"  "qwen3_14b_base"

echo ""
echo "============================================================"
echo " Phase 1 Mac suite — finished at $(date)"
echo "============================================================"
echo "Log files:"
ls -lt logs/ | grep -E "phi4mini|gemma3_4b|qwen3_" | head -8
