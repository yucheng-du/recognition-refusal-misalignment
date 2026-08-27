"""Rebuttal check v2 (verified provenance, model forward pass): principal angles
between the k-dim impossibility subspace and a k-dim refusal-deviation subspace
built from the BEHAVIOR-VERIFIED harmful/harmless prompt subsets (mirroring the
paper's d_ref verification in scripts/compare_impossibility_vs_refusal_direction.py).

Complements subspace_overlap_v2_energies.py (which needs no model because the
verified d_ref vector is cached). Here we re-run verify_behavior to get the
verified prompt subsets, extract their last-token states, and build
Vt_ref = top-k right singular vectors of (verified harmful - mean(verified harmless)).

Usage: python subspace_overlap_v2_angles.py --model mistral
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from numpy.linalg import norm, svd, qr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import compare_impossibility_vs_refusal_direction as cmp  # noqa: E402

CELLS = {
    "mistral":   [("math800", 15), ("code800", 15)],
    "qwen3_14b": [("math800", 25), ("code800", 24)],
}
KS = (5, 10)
N_RAND_SUB = 200


def build_imp_subspace(model, dataset, layer):
    sig = ROOT / "experiments" / "signals" / f"{dataset}_{model}_allL" / "signals"
    reps = np.load(sig / "reps_last_token_all_layers.npy", mmap_mode="r")
    with (sig / "meta.jsonl").open(encoding="utf-8") as handle:
        meta = [json.loads(line) for line in handle]
    labels = np.array([m["answerable"] for m in meta])
    X = np.array(reps[:, layer, :], dtype=np.float64)
    cd = ROOT / "experiments" / "cached_directions"
    V_A = np.load(cd / f"{model}_{dataset}_L{layer}_d_imp_V_A.npy").astype(np.float64)

    A_idx = np.where(labels == "A")[0]
    U_idx = np.where(labels == "U")[0]
    r = np.random.RandomState(42)
    pA = r.permutation(len(A_idx)); pU = r.permutation(len(U_idx))
    trA = A_idx[pA[:len(A_idx) // 2]]
    trU = U_idx[pU[:len(U_idx) // 2]]

    R_null = X - X @ V_A.T @ V_A
    Xdev = R_null[trU] - R_null[trA].mean(0)
    _, _, Vt = svd(Xdev, full_matrices=False)
    return Vt, X.shape[1]


def principal_angles(Va, Vb):
    s = svd(Va @ Vb.T, compute_uv=False)
    return np.clip(s, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CELLS))
    args = ap.parse_args()
    model_name = args.model

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if model_name in ("qwen3_14b",) else torch.float16
    print(f"[load] {model_name} dtype={dtype} device={device}", flush=True)
    tok = AutoTokenizer.from_pretrained(cmp.MODEL_PATHS[model_name], use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cmp.MODEL_PATHS[model_name], dtype=dtype).to(device)
    model.eval()

    print("[verify] behavior verification (paper protocol)", flush=True)
    v_harm_idx = cmp.verify_behavior(model, tok, cmp.HARMFUL_PROMPTS, device, expect_refusal=True)
    v_harmless_idx = cmp.verify_behavior(model, tok, cmp.HARMLESS_PROMPTS, device, expect_refusal=False)
    print(f"  harmful {len(v_harm_idx)}/{len(cmp.HARMFUL_PROMPTS)} verified refused; "
          f"harmless {len(v_harmless_idx)}/{len(cmp.HARMLESS_PROMPTS)} verified complied", flush=True)
    harm_p = [cmp.HARMFUL_PROMPTS[i] for i in v_harm_idx]
    harmless_p = [cmp.HARMLESS_PROMPTS[i] for i in v_harmless_idx]

    layers_needed = sorted({L for _, L in CELLS[model_name]})
    harm, harmless = {}, {}
    for L in layers_needed:
        print(f"[extract] verified refusal-set states at layer {L}", flush=True)
        harm[L] = cmp.extract_last_token_hidden(model, tok, harm_p, L, device)
        harmless[L] = cmp.extract_last_token_hidden(model, tok, harmless_p, L, device)

    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    out = []
    rng = np.random.default_rng(0)
    for dataset, L in CELLS[model_name]:
        print(f"[cell] {model_name} / {dataset} / L{L}", flush=True)
        Vt_imp, D = build_imp_subspace(model_name, dataset, L)

        # sanity: verified mean-diff should match cached verified d_ref closely
        d_ref_v = harm[L].mean(0) - harmless[L].mean(0)
        d_ref_v = d_ref_v / (norm(d_ref_v) + 1e-15)
        d_ref_cached = np.load(
            ROOT / "experiments" / "cached_directions" / f"{model_name}_L{L}_d_ref_full.npy"
        ).astype(np.float64)
        d_ref_cached /= norm(d_ref_cached)
        cos_recon = float(np.dot(d_ref_v, d_ref_cached))

        Hdev = (harm[L] - harmless[L].mean(0)).astype(np.float64)
        _, _, Vt_ref = svd(Hdev, full_matrices=False)

        cell = dict(model=model_name, dataset=dataset, layer=L, hidden_dim=D,
                    n_harmful_verified=len(harm_p), n_harmless_verified=len(harmless_p),
                    cos_reconstructed_vs_cached_dref=cos_recon,
                    provenance="behavior-verified prompt subsets (paper protocol re-run)")
        for k in KS:
            Vk = Vt_imp[:k]
            Vrefk = Vt_ref[:k]
            cosang = principal_angles(Vk, Vrefk)
            rnd_first, rnd_msq = [], []
            for _ in range(N_RAND_SUB):
                Q, _ = qr(rng.standard_normal((D, k)))
                s = principal_angles(Vk, Q[:, :k].T)
                rnd_first.append(s[0]); rnd_msq.append(float((s ** 2).mean()))
            cell[f"k{k}"] = dict(
                principal_cosines=[float(x) for x in cosang],
                first_cos=float(cosang[0]),
                first_angle_deg=float(np.degrees(np.arccos(cosang[0]))),
                mean_sq_cos=float((cosang ** 2).mean()),
                rand_first_cos_mean=float(np.mean(rnd_first)),
                rand_first_cos_p95=float(np.percentile(rnd_first, 95)),
                rand_mean_sq_cos_mean=float(np.mean(rnd_msq)),
            )
        out.append(cell)
        print(json.dumps({k: v for k, v in cell.items() if k != "principal_cosines"},
                         indent=2, default=str), flush=True)

    with (HERE / f"subspace_overlap_v2_angles_{model_name}.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(out, f, indent=2)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
