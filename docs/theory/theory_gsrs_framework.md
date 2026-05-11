# Generalized Subspace Residual Score (GSRS) — Theoretical Framework

## 1. Definition

Given a representation matrix X ∈ ℝ^{N×D}, define the **Generalized Subspace Residual Score** as:

$$
\text{GSRS}(x; P, w, \phi) = \phi\left( (I - P) x, \; w \right)
$$

Where:
- **P ∈ ℝ^{D×D}**: projection matrix onto a reference subspace S (P = P², P^T = P)
- **w ∈ ℝ^{D}**: scoring direction in the residual space
- **φ: ℝ^D × ℝ^D → ℝ**: scoring function

The three components are **independently specified**, giving a combinatorial family of metrics.

---

## 2. Component Choices

### 2.1 Projection Matrix P

| Choice | Definition | Supervision | Description |
|--------|-----------|-------------|-------------|
| P_pca(A, k) | V_k V_k^T where V_k = top-k PCs of X_A | Requires A labels | Answerable subspace from A-class PCA |
| P_pca(all, k) | V_k V_k^T where V_k = top-k PCs of X_all | **Unsupervised** | Dominant subspace from all data |
| P_unembed | U_k U_k^T from SVD of W_unembed | **Unsupervised** | Reasoning subspace (cf. HARP) |
| I | Identity (no projection) | None | Full space, no subspace step |
| 0 | Zero (full space is residual) | None | Trivially: residual = original |

### 2.2 Scoring Direction w

| Choice | Definition | Supervision | Description |
|--------|-----------|-------------|-------------|
| d̂_means | (μ_U - μ_A) / ‖μ_U - μ_A‖ | Requires A+U labels | Difference-in-means direction |
| d̂_t | t_j / ‖t‖ where t_j = (μ_U - μ_A)_j / se_j | Requires A+U labels | Welch t-statistic weighted |
| d̂_fisher | Σ_w^{-1}(μ_U - μ_A) | Requires A+U labels | Fisher LDA direction |
| c_A / ‖c_A‖ | μ_A / ‖μ_A‖ | Requires A labels | Centroid direction (for distance-from-A) |
| r / ‖r‖ | self-normalized | None | Unit residual (for CosNSRT) |

### 2.3 Scoring Function φ

| Choice | Definition | Description |
|--------|-----------|-------------|
| φ_dot(r, w) | r · w | Linear projection (NSRT) |
| φ_cos(r, w) | (r · w) / ‖r‖ | Cosine similarity (CosNSRT) |
| φ_norm(r, w) | ‖r‖ | Residual magnitude only (Norm) |
| φ_maha(r, w) | √(r^T Σ^{-1} r) | Mahalanobis distance |
| φ_cos_centroid(r, w) | r · w / ‖r‖ | Cosine distance to A centroid |

---

## 3. Unification: Every Metric as a GSRS Instance

### 3.1 NSRT
```
P = P_pca(A, k)          # top-k PCs of A-class
w = d̂_means              # mean-difference direction in residual space
φ = φ_dot                # dot product
```
Score: `NSRT(x) = ((I - V_k V_k^T) x) · d̂`

### 3.2 CosNSRT
```
P = P_pca(A, k)
w = d̂_means
φ = φ_cos                # cosine similarity
```
Score: `CosNSRT(x) = ((I - V_k V_k^T) x) · d̂ / ‖(I - V_k V_k^T) x‖`

**Relation to NSRT**: CosNSRT = NSRT / ‖residual‖. Removes magnitude confound.

### 3.3 T-NSRT
```
P = P_pca(A, k)
w = d̂_t                  # t-statistic weighted direction
φ = φ_dot
```
Score: `T-NSRT(x) = ((I - V_k V_k^T) x) · d̂_t`

**Relation to NSRT**: T-NSRT reweights dimensions by signal-to-noise ratio.

### 3.4 Fisher LDA (in null-space)
```
P = P_pca(A, k)
w = d̂_fisher             # Σ_w^{-1}(μ_U - μ_A)
φ = φ_dot
```
Score: `Fisher(x) = ((I - V_k V_k^T) x) · Σ_w^{-1}(μ_U - μ_A)`

**Relation to NSRT**: Fisher = NSRT with Mahalanobis-whitened direction. Same projection, different direction weighting.

### 3.5 Norm (unsupervised baseline)
```
P = P_pca(A, k)
w = ∅ (unused)
φ = φ_norm               # just residual magnitude
```
Score: `Norm(x) = ‖(I - V_k V_k^T) x‖`

**Relation to NSRT**: Norm = NSRT without directional selectivity. Uses ‖r‖ instead of r · d̂.

### 3.6 own_dist (legacy centroid-distance metric)
```
P = I (identity)          # NO subspace projection
w = c_A / ‖c_A‖          # A centroid direction
φ = φ_cos_centroid        # cosine distance to centroid
```
Score: `own_dist(x) = 1 - cos(x, μ_A)` in FULL representation space.

**Relation to NSRT**: own_dist is GSRS with P=I (no null-space step).
The AUC gap (0.567 → 0.897 = +33pp) decomposes into:
- Null-space projection (P=I → P_pca): **+22.5pp**
- Directed scoring (cos_centroid → d̂_means dot): **+10.7pp**

### 3.7 HARP (concurrent work)
```
P = P_unembed(k)          # SVD of unembedding layer
w = trained classifier     # learned direction
φ = φ_dot
```
Score: `HARP(x) = ((I - U_k U_k^T) x) · w_trained`

**Relation to NSRT**: Same GSRS structure, different P definition (data-independent from unembedding layer vs data-dependent from PCA).

### 3.8 Mahalanobis distance (from A centroid)
```
P = I (identity)
w = ∅
φ = φ_maha               # Mahalanobis distance
```
Score: `Maha(x) = √((x - μ_A)^T Σ^{-1} (x - μ_A))`

**Relation to NSRT**: No subspace projection. Full-space Mahalanobis. Our experiments show this gives AUC ~0.69 — projection is crucial.

### 3.9 CLA (Cross-Layer Agreement)
```
For each layer l in {l₀-1, l₀, l₀+1}:
  P_l = P_pca(A, k) at layer l
  w_l = d̂_means at layer l
  φ = φ_dot
```
Score: `CLA(x) = mean_l [NSRT_l(x)]`

**Relation to NSRT**: CLA = average of NSRT across neighboring layers. Multi-layer extension.

---

## 4. Theoretical Decomposition

### 4.1 The Projection-Direction-Normalization Decomposition

Any GSRS score can be understood through three independent effects:

1. **Projection effect** (P): How much task-irrelevant variance is removed
   - Measured by: AUC(Norm with P) - AUC(Norm with I)
   - Our data: +22.5pp (dominant effect)

2. **Direction effect** (w): How well the scoring direction aligns with the A/U separation
   - Measured by: AUC(NSRT) - AUC(Norm)
   - Our data: +10.9pp

3. **Normalization effect** (φ): Whether magnitude confounds are removed
   - Measured by: AUC(CosNSRT) - AUC(NSRT)
   - Our data: +1.1pp average, but +5.7pp on hard datasets (mathtrap)

**Total NSRT improvement = Projection + Direction + Normalization**
`0.897 - 0.567 = 0.330 = 0.225 + 0.109 - 0.004 (rounding)`

### 4.2 Why Null-Space Projection Dominates

Theorem (informal): If the answerable class lies on a low-rank manifold of effective dimension k ≪ D, then the null-space residual r = (I - V_k V_k^T)x captures the deviation from this manifold. Unanswerable inputs that trigger different internal representations will have larger ‖r‖ and different r directions than answerable inputs projected near the manifold.

Empirical evidence:
- First 10 PCs capture >85% of A-class variance but give AUC ~0.51 (no signal)
- Null-space (remaining ~15% variance) gives AUC 0.897
- Signal is CONCENTRATED in the orthogonal complement of the answerable subspace

### 4.3 CosNSRT as Optimal for Mixed Signal Types

When the A/U separation has two components:
- **Magnitude component**: ‖r_U‖ > ‖r_A‖ (unanswerable = further from manifold)
- **Direction component**: angle(r_U, d̂) > angle(r_A, d̂) (unanswerable = different direction)

NSRT = ‖r‖ · cos(r, d̂) conflates both. CosNSRT = cos(r, d̂) isolates direction.

On datasets where magnitude is informative (math800, code800): NSRT ≈ CosNSRT
On datasets where direction is more informative (falseqa, mathtrap): CosNSRT > NSRT (+5.7pp)

This explains the dataset-dependent performance gap.

---

## 5. Ablation Matrix (from v39 results)

| Metric | P | w | φ | Avg AUC |
|--------|---|---|---|---------|
| own_dist | I | centroid | cos | 0.567 |
| Norm | PCA(A,k) | — | norm | 0.788 |
| Fisher | PCA(A,k) | Σ^{-1}d | dot | 0.852 |
| NSRT | PCA(A,k) | d̂_means | dot | 0.897 |
| T-NSRT | PCA(A,k) | d̂_t | dot | 0.898 |
| CLA | PCA(A,k)×3 | d̂_means×3 | dot→avg | 0.899 |
| CosNSRT | PCA(A,k) | d̂_means | cos | **0.908** |
| Mahalanobis | I | — | maha | 0.690 |

**Reading the table by components:**
- P: own_dist(I)→Norm(PCA) = +22.1pp. Projection is the #1 driver.
- w: Norm(none)→NSRT(d̂) = +10.9pp. Direction is #2.
- φ: NSRT(dot)→CosNSRT(cos) = +1.1pp avg (+5.7pp on hard). Normalization is #3.
- Fisher vs NSRT: Σ^{-1} whitening actually HURTS by 4.5pp. Simple mean-difference direction is better than Fisher in null-space.

---

## 6. What This Framework Buys for EMNLP

1. **Unification narrative**: "We propose GSRS, a family of subspace residual scores parameterized by (P, w, φ). We show that prior methods (Mahalanobis, cosine distance, linear probes) are special cases, and identify the null-space projection as the dominant factor (+22pp)."

2. **Principled ablation**: The three-component decomposition gives a clean ablation story, not ad-hoc metric comparisons.

3. **Design space**: GSRS defines a design space. We explored it systematically and found CosNSRT (P=PCA, w=d̂, φ=cos) optimal. This is how KLE positioned itself — "we define a general family, show SE is a special case, and find the optimal instance."

4. **Against Lavi et al.**: They fix P=I and innovate on w (causal steering). We fix w=d̂_means and innovate on P (null-space projection). Same GSRS framework, orthogonal contributions. Can even cite: "Lavi et al. (2026) can be viewed as GSRS(P=I, w=w_causal, φ=dot)."

---

## 7. Open Questions for Experiments

### 7.1 Unsupervised P (no A labels needed)
Can we define P from all-sample PCA instead of A-only PCA?
- Hypothesis: if A-class dominates (typically >50% of data), all-sample PCA ≈ A-only PCA
- Test: P_pca(all, k) vs P_pca(A, k), measure AUC gap
- If gap < 2pp → we can claim semi-unsupervised NSRT

### 7.2 Spectral Gap k Selection
Can spectral gap of all-sample PCA auto-select k?
- Plot eigenvalue spectrum, find the "elbow" or largest gap
- Compare auto-k vs oracle k (the k that maximizes AUC)
- If they match → unsupervised k selection, big novelty

### 7.3 Cross-Layer GSRS Trajectory
Instead of CLA (simple average), analyze the full NSRT(x) trajectory across layers:
- Trajectory smoothness: Σ |NSRT_l(x) - NSRT_{l-1}(x)|
- Convergence: does NSRT stabilize at deep layers?
- Feature: extract trajectory shape features (slope, curvature, saturation point)

### 7.4 Kernel GSRS
Replace φ_dot with φ_kernel(r, w) = K(r, centroid_U) - K(r, centroid_A) for nonlinear scoring.
Prior result: kernel PCA didn't help → signal is linear. But kernel scoring (not projection) may differ.
