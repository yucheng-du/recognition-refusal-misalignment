# Multidimensional subspace robustness

The five `subspace_overlap_v2_*` files reproduce the frozen 5--10-dimensional robustness results reported in the paper: one energy JSON, two principal-angle JSONs, and the two scripts that produced them.

`subspace_overlap_v2_energies.py` requires the omitted raw `experiments/signals/` tensors and cached directions. `subspace_overlap_v2_angles.py` additionally loads the relevant model weights and reruns the paper's behavior verification. The frozen JSON files are therefore the lightweight reproducibility targets; run `python3 scripts/verify_core_conclusions.py` from the repository root to validate their cell coverage and reported ranges without model inference.
