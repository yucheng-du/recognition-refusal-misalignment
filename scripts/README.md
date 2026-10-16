# Scripts

CLI entry points for the extraction, detection, intervention, steering, and evaluation pipeline. Run each from the repo root (`python scripts/<name>.py` or `bash scripts/<name>.sh`); all scripts resolve paths via env or `Path(__file__)` and do not assume a hardcoded user prefix.

---

## Raw reruns and release boundaries

The lightweight release includes frozen aggregate JSONs for checking results
and regenerating figures. A raw rerun requires model weights downloaded
separately from Hugging Face, approximately 151 GB for the omitted
`experiments/signals/` activation tree across the released runs, and a GPU for
intervention/steering. For example, extraction on one model populates the
layout read by the analysis scripts:

```bash
python scripts/run_extract_minimal.py --model <hf_id> --prompts data/math800.jsonl \
    --run-dir experiments/signals/math800_<key>_allL --all-layers --no-gradients
```

Some label-dependent aggregates additionally require effective intervention-label
TSVs that are not included in this release. The frozen aggregates are the
released reproducibility targets for those comparisons. Labels follow a fixed
invalidity-aware rubric with this provenance:

- In the 4-anchor intervention grid, nine cells use LLM-assisted candidate
  labels plus provisional audit-subset fills; the three Qwen3-8B cells use
  candidate-label passthrough. The supplementary Gemma-3-12B/code cell also
  uses provisional audit-subset fills.
- For `experiments/d_struct_behav_matrix.json`, the six non-Qwen3-8B math/code
  cells use candidate labels plus provisional audit-subset fills; the two
  Qwen3-8B cells use candidate-label passthrough. Its local builder also needs
  the omitted raw activations and is not a lightweight-release entry point.

These describe effective aggregation inputs, not completed human adjudication.
See the [intervention factsheet](../experiments/intervention/v2_final_4anchor_factsheet.md)
and [figure guide](../paper/README.md) for the released artifacts and their use.

## Model checkpoints

The detection/orthogonality main grid uses 11 instruction-tuned checkpoints:

- `HuggingFaceTB/SmolLM2-1.7B-Instruct`, `microsoft/Phi-4-mini-instruct`, `google/gemma-3-4b-it`
- `mistralai/Mistral-7B-Instruct-v0.3`, `Qwen/Qwen3-8B`, `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen3-14B`, `allenai/OLMo-2-1124-13B-Instruct`, `mistralai/Mistral-Small-24B-Instruct-2501`
- `Qwen/Qwen3-32B`, `meta-llama/Llama-3.3-70B-Instruct`

The six base/instruct pairs add Qwen2.5-7B/14B/32B base, Qwen3-8B/14B base,
and the Llama-3.1-70B base proxy. The deterministic 16-model steering-breadth
sweep uses `gemma2`, `gemma3_4b`, `llama`, `mistral`, `mistral_small`,
`mistral_small_3_2`, `olmo13b`, `phi3`, `phi4mini`, `qwen`, `qwen14b`,
`qwen32b`, `qwen3_8b`, `qwen3_14b`, `qwen3_32b`, and `smollm2`.

The frozen JSON artifacts are authoritative for experiment identity; model
registries are in `run_extract_minimal.py` and `impossibility_steering.py`.
Set `HF_HOME` to choose a cache directory. Peak layers are registered in the
relevant scripts; `find_best_layers_smollm2_gemma2_phi3.py` shows how to
re-resolve the small-model subset. Model weights retain their upstream terms.

## Extraction (hidden-state dumps for downstream analysis)

| Script | Role |
|---|---|
| `run_extract_minimal.py` | Forward-pass extraction of last-token + all-layers representations from a HuggingFace model on a prompt file. Writes to `experiments/signals/<run-dir>/signals/reps_*.npy`. |
| `run_extract_signals.py` | Companion that adds optional gradient / attention extraction. |

## Detection / probing / per-cell aggregation

| Script | Role |
|---|---|
| `analyze_layer_emergence.py` | Per-model AUC sweep across layers (used to pick the peak detection layer). |
| `analyze_form_conditionality.py` | Within-form vs cross-form AUC drop. |
| `compare_impossibility_vs_refusal_direction.py` | Computes $\cos(d_{\mathrm{imp}}, d_{\mathrm{ref}})$ + bootstrap CI; writes `experiments/direction_comparison_<model>*.json`. |
| `ablation_nullspace_vs_pcspace.py` | Null-space vs PC-space ablation; writes `experiments/ablation_nullpc_results_*.json`. |
| `find_best_layers_smollm2_gemma2_phi3.py` | Helper for layer selection on the small-model subset. |
| `aggregate_main_grid_v2.py` | Per-cell → 22-cell main-grid aggregate; writes `experiments/main_grid_facts_v2.json`. |
| `aggregate_ablation_v2.py` | Aggregates per-cell ablation JSONs into `experiments/ablation_nullpc_results_11model.json`. |
| `aggregate_steering_v2det.py` | Aggregates the 48-cell × 16-model steering breadth sweep. |
| `recompute_gated_dG_from_labels.py` | Recomputes invalidity-aware gated $\Delta$G from LLM-assisted candidate labels plus an optional label-override TSV. |

## Intervention + steering

| Script | Role |
|---|---|
| `intervention_nullspace.py` | The 4-anchor intervention-generation pipeline that produces per-sample outputs. Requires a GPU; label aggregation is handled separately. |
| `impossibility_steering.py` | Generation-time activation steering along $d_{\mathrm{imp}}$ vs a random-direction control. |
| `validate_intervention_labels.py` | Sanity-check pass on per-cell labels before invalidity-aware aggregation. |
| `compare_steering_v1_v2det.py` | Compares legacy v1 vs invalidity-aware v2 steering rates. |

## Evaluation / transfer

| Script | Role |
|---|---|
| `eval_cross_domain_transfer.py` | Within-domain vs cross-domain $d_{\mathrm{imp}}$ transfer. |
| `eval_natural_transfer.py` | Math800-trained probe → natural-distribution transfer (AbstentionBench-GSM8K, FalseQA). |
| `eval_abstentionbench_lengthcontrol.py` | Length-controlled subset of AbstentionBench-GSM8K. |
| `eval_difficulty_control.py` | Difficulty-controlled split null check. |
| `prepare_difficulty_control.py` | Builds `data/difficulty_control_gsm8k.jsonl` from GSM8K. |
| `verify_core_conclusions.py` | Lightweight consistency check: reads shipped aggregate JSONs, reports headline summaries, and validates released label-provenance metadata. |

## Orchestrators (shell)

| Script | Role |
|---|---|
| `run_model_suite.sh` | Master orchestrator: extracts → finds peak layer → runs detection + form-conditionality + orthogonality + base sweep + steering breadth, for one registered model. Resumable. |
| `run_base_orthogonality_sweep.sh` | Sweep the 6 base/instruct pairs on math800 for the §4.3 base panel. |
| `extract_noncore_datasets.sh` | Convenience wrapper to extract signals for the auxiliary transfer datasets. |
| `run_phase1_mac.sh` | Mac-tier orchestrator for phi-4-mini / gemma-3-4b / qwen3-8b / qwen3-14b. |

## Typical lightweight workflow

```bash
# Sanity check (reads aggregate JSONs, prints headline numbers)
python3 scripts/verify_core_conclusions.py

# Regenerate the 5 matplotlib figures from shipped aggregates
python3 paper/generate_fig5_v2.py
python3 paper/generate_figures.py --figs fig2 fig3 fig4 fig6

# Full pipeline on one model (heavy — requires GPU + ~30 GB activations)
MODEL=mistral BASE_MODEL=mistral_base bash scripts/run_model_suite.sh
```
