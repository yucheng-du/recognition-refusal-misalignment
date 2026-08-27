"""Rebuttal check v2 (verified provenance, numpy-only): energy of the paper's
behavior-verified d_ref_safety inside the k-dim impossibility subspace.

v1 (subspace_overlap.py) recomputed d_ref from a fresh forward pass WITHOUT the
behavior-verification filter. v2 instead loads the cached verified direction
(experiments/cached_directions/{model}_L{layer}_d_ref_full.npy), which reproduces
the paper's cos_full_full to machine precision (checked below), and the cached
A-PC basis V_A for the exact paper A-null subspace. Only the energy /
random-null / d_imp-capture numbers are recomputed here; the subspace-subspace
principal angles need per-prompt verified harmful states and are re-run
separately (subspace_overlap_v2_angles.py, model forward pass required).

Usage: python subspace_overlap_v2_energies.py
"""
import json
from pathlib import Path

import numpy as np
from numpy.linalg import norm, svd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

CELLS = [
    ("mistral", "math800", 15),
    ("mistral", "code800", 15),
    ("qwen3_14b", "math800", 25),
    ("qwen3_14b", "code800", 24),
]
KS = (5, 10)
N_RAND_VEC = 2000

# paper values (experiments/direction_comparison_{model}*.json, cos_full_full)
PAPER_COS_FULL_FULL = {
    ("mistral", "math800"): 0.2720494270324707,
    ("mistral", "code800"): 0.2446238249540329,
    ("qwen3_14b", "math800"): 0.2510501742362976,
    ("qwen3_14b", "code800"): 0.23420341312885284,
}


def load_cell(model, dataset, layer):
    sig = ROOT / "experiments" / "signals" / f"{dataset}_{model}_allL" / "signals"
    reps = np.load(sig / "reps_last_token_all_layers.npy", mmap_mode="r")
    with (sig / "meta.jsonl").open(encoding="utf-8") as handle:
        meta = [json.loads(line) for line in handle]
    labels = np.array([m["answerable"] for m in meta])
    X = np.array(reps[:, layer, :], dtype=np.float64)
    cd = ROOT / "experiments" / "cached_directions"
    V_A = np.load(cd / f"{model}_{dataset}_L{layer}_d_imp_V_A.npy").astype(np.float64)
    d_ref = np.load(cd / f"{model}_L{layer}_d_ref_full.npy").astype(np.float64)
    d_imp_full = np.load(cd / f"{model}_{dataset}_L{layer}_d_imp_full.npy").astype(np.float64)
    return X, labels, V_A, d_ref / norm(d_ref), d_imp_full / norm(d_imp_full)


def energy_stats(Vk, d, rng, n=N_RAND_VEC):
    D = Vk.shape[1]
    e = float(norm(Vk @ d) ** 2)
    R = rng.standard_normal((n, D))
    R /= norm(R, axis=1, keepdims=True)
    er = norm(R @ Vk.T, axis=1) ** 2
    return dict(energy=e, rand_mean=float(er.mean()),
                rand_p95=float(np.percentile(er, 95)),
                rand_p99=float(np.percentile(er, 99)),
                percentile_of_d=float((er < e).mean() * 100),
                analytic_kD=Vk.shape[0] / D)


def main():
    out = []
    rng = np.random.default_rng(0)
    for model, dataset, layer in CELLS:
        X, labels, V_A, d_ref, d_imp_full = load_cell(model, dataset, layer)

        # provenance check: cached verified d_ref reproduces paper cos_full_full
        cos_ff = float(d_imp_full @ d_ref)
        ref = PAPER_COS_FULL_FULL[(model, dataset)]
        assert abs(cos_ff - ref) < 1e-4, (model, dataset, cos_ff, ref)

        # exact paper A-null (cached V_A) + seed-42 split as in the paper pipeline
        A_idx = np.where(labels == "A")[0]
        U_idx = np.where(labels == "U")[0]
        r = np.random.RandomState(42)
        pA = r.permutation(len(A_idx)); pU = r.permutation(len(U_idx))
        trA = A_idx[pA[:len(A_idx) // 2]]
        trU = U_idx[pU[:len(U_idx) // 2]]
        teA = A_idx[pA[len(A_idx) // 2:]]
        teU = U_idx[pU[len(U_idx) // 2:]]

        R_null = X - X @ V_A.T @ V_A
        d_imp = R_null[trU].mean(0) - R_null[trA].mean(0)
        d_imp /= norm(d_imp)
        Xdev = R_null[trU] - R_null[trA].mean(0)
        _, _, Vt = svd(Xdev, full_matrices=False)

        d_ref_null = d_ref - V_A.T @ (V_A @ d_ref)
        d_ref_null /= norm(d_ref_null)

        from sklearn.metrics import roc_auc_score
        y = np.r_[np.zeros(len(teA)), np.ones(len(teU))]

        cell = dict(model=model, dataset=dataset, layer=layer,
                    hidden_dim=X.shape[1],
                    provenance="cached behavior-verified d_ref (paper vector); cached V_A",
                    cos_full_full_check=cos_ff,
                    cos_dimp_dref_verified=float(d_imp @ d_ref))
        for k in KS:
            Vk = Vt[:k]
            aucs = [float(max(a, 1 - a)) for a in
                    (roc_auc_score(y, np.r_[R_null[teA] @ Vt[j], R_null[teU] @ Vt[j]])
                     for j in range(k))]
            cell[f"k{k}"] = dict(
                heldout_component_aucs=aucs,
                energy_dimp_in_Vk=float(norm(Vk @ d_imp) ** 2),
                dref_full=energy_stats(Vk, d_ref, rng),
                dref_Anull=energy_stats(Vk, d_ref_null, rng),
            )
        out.append(cell)
        print(json.dumps(cell, indent=2), flush=True)

    with (HERE / "subspace_overlap_v2_energies.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[done]")


if __name__ == "__main__":
    main()
