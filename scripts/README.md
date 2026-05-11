# Scripts

CLI entry points for the extraction, detection, intervention, steering, and evaluation pipeline. Run each from the repo root (`python scripts/<name>.py` or `bash scripts/<name>.sh`); all scripts resolve paths via env or `Path(__file__)` and do not assume a hardcoded user prefix.

---

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
| `recompute_gated_dG_from_labels.py` | Recomputes invalidity-aware gated $\Delta$G from human verification labels. |

## Intervention + steering

| Script | Role |
|---|---|
| `intervention_nullspace.py` | The 4-anchor intervention pipeline that produces `experiments/intervention/intervention_*_full_v2.json`. Requires GPU + human verification. |
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
| `verify_core_conclusions.py` | One-shot replication check: reads aggregate JSONs, prints the headline numbers that appear in the paper (see OpenReview submission). |

## Orchestrators (shell)

| Script | Role |
|---|---|
| `run_model_suite.sh` | Master orchestrator: extracts → finds peak layer → runs detection + form-conditionality + orthogonality + base sweep + steering breadth, for one registered model. Resumable. |
| `run_base_orthogonality_sweep.sh` | Sweep the 6 base/instruct pairs on math800 for the §4.3 base panel. |
| `extract_noncore_datasets.sh` | Convenience wrapper to extract signals for the auxiliary transfer datasets. |
| `run_phase1_mac.sh` | Mac-tier orchestrator for phi-4-mini / gemma-3-4b / qwen3-8b / qwen3-14b. |

## Typical reviewer workflow

```bash
# Sanity check (reads aggregate JSONs, prints headline numbers)
python scripts/verify_core_conclusions.py

# Regenerate the 5 fully-reproducible figures
python paper/generate_fig5_v2.py
python paper/generate_figures.py --figs fig2 fig3 fig4 fig6

# Full pipeline on one model (heavy — requires GPU + ~30 GB activations)
MODEL=mistral BASE_MODEL=mistral_base bash scripts/run_model_suite.sh
```
