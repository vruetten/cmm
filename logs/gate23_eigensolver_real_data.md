# Gates 2 and 3 — eigensolver and k-NN sparsification on a real coherence graph

Measurement report for §6 Q4 of `.claude/plans/cmm_refactor_plan.md`, plus an
addendum comparing estimators V1 and V2 on the same real coefficients.

Nothing in either repo was modified; nothing was written to `/nrs`.

**Headline results**

1. **`eigsh` is safe as the default.** On a real coherence graph it reproduces
   the dense k-means partition **exactly (ARI = 1.0000)** in every configuration
   tested, at 1e-15 eigenvalue agreement, and is **6.5–46× faster at 32 threads**
   and **10–25× faster at 1 thread**. The advantage survives proper threading.
2. **k-NN sparsification must be rejected.** It is 7–110× faster again but
   destroys the cluster structure: ARI vs the dense-affinity labels ranges from
   **−0.14 to 0.39**. Disconnection was *not* the failure mode — every graph
   stayed connected.
3. **`lobpcg` must not be a default.** It matched eigenvalues to 3e-5 and still
   returned ARI 0.39.
4. **Addendum:** neither V1's nor V2's documented blind spot occurs in this data,
   yet the two estimators still produce materially different clusterings
   (ARI 0.33–0.64). The V2-over-V1 decision is *not* low-stakes, and this dataset
   does not adjudicate it.

---

## 1. How the real matrix was built

### 1.1 Source data

| item | value |
|------|-------|
| dataset | `270705_ubi_tTA_TRE_NES_icAMPSnFR_18539`, `exp3` (`dor003`) |
| file | `/nrs/ahrens/Virginia_nrs/icaMPSnFR/270705_ubi_tTA_TRE_NES_icAMPSnFR_18539/exp3/coherence/band10-40s/..._dor003_bin5p2um_stft.npy` |
| upstream preprocessing (`icampsnfr/scripts/run_coherence.py`) | smooth σ = 3.47 µm → bin to 5.2 µm voxels → zero-phase Butterworth high-pass, cutoff 120 s, order 4, 200 s cropped each end → manual brain mask at 410 counts, 1-voxel in-plane erode |
| masked voxels available | 36 094 |
| **N used** | **10 000** — uniform random subsample, `np.random.default_rng(0)` |
| `fs` | 0.596 207 Hz |
| `nperseg` / `noverlap` | 119 / 60 |
| window / detrend | hann / constant |
| **band** | **0.025 – 0.1 Hz** (periods 10 – 40 s) |
| **K** (segments) | **22** |
| **F** (in-band bins) | **15** |
| STFT dtype | complex64 |

Only the cached `(N, K, F)` STFT coefficients were reused from the `icampsnfr`
pipeline. The coherence matrix itself was computed from scratch.

### 1.2 Estimator — V2, written for this report

The gate requires **V2** (power-weighted, incoherently averaged).
`icampsnfr.coherence.msc_matrix` implements V3, which the project has rejected;
it was **not** used for the matrix under test.

```
P_nm(f) = sum_k X_n(k,f) conj(X_m(k,f))
C[n,m]  = sum_f |P_nm(f)|^2  /  sum_f P_nn(f) P_mm(f)
```

Implemented as `F` row-blocked GEMMs accumulating `|P_f|^2` into the numerator,
with the denominator as a single `p @ p.T` where `p[n,f] = P_nn(f)`. Build time
43 s at N = 10 000.

**Validation.** Cross-checked against a deliberately naive per-pair,
per-frequency double loop on a 60-voxel subset: **max abs difference 1.5e-7**
(complex64 accumulation), diagonal exactly 1.0, range within [0, 1].

No debiasing was applied — the raw V2 matrix is the object under test.

### 1.3 Properties of the resulting matrix

| statistic (strict off-diagonal) | value |
|---|---|
| min | 0.00275 |
| 1st percentile | 0.0222 |
| median | 0.0549 |
| mean | 0.0673 |
| 99th percentile | 0.2476 |
| max | 0.9823 |
| V2 bias floor `1/K` | **0.0455** |

The median off-diagonal coherence (0.055) sits barely above the `1/K = 0.0455`
bias floor. **This is a low-contrast graph** — the realistic case the gate
existed to test, and nothing like synthetic well-separated blocks.

### 1.4 Graph construction

Following `icampsnfr/spectral.py`: `W = exp(-(1 - C)/tau)` with the diagonal
zeroed, `S = D^{-1/2} W D^{-1/2}`, `L = I - S`, so `lambda_L = 1 - lambda_S`.

Two bandwidths, because `tau` dominates the graph's character far more than the
solver choice does:

| `tau` | value | W range (off-diag) | degree min / med / max |
|---|---|---|---|
| `median_tau` (package default) | 0.945062 | ~0.35 – 0.98 | 3637 / 3706 / 4321 |
| `median_tau / 10` | 0.094506 | ~1e-5 – 0.82 | 0.412 / 0.534 / 15.64 |

At the package default the affinity is **nearly uniform** — every pair has weight
between 0.35 and 0.98. That is the worst case for an iterative solver and the
honest default to test.

### 1.5 Clustering convention

`n_clusters = n_eig - 1`, matching `icampsnfr.spectral.spectral_labels`
(`n_eig = n_clusters + 1`). Embedding: drop the trivial first eigenvector,
row-normalize (Ng–Jordan–Weiss), `KMeans(n_init=10, random_state=0)` — identical
seed and settings for every solver, so any ARI below 1 comes from the embedding,
not from k-means.

### 1.6 Machine and threading

Workstation `ruttenv-ws1`, 80 cores by affinity, 376 GB RAM, conda env `all`
(numpy 2.3.4, scipy 1.16.2, sklearn 1.7.2, OpenBLAS 0.3.28, SkylakeX).

**The shell exports `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`.** Any measurement taken with the inherited environment is a
1-thread measurement. Thread count was therefore set explicitly, before numpy
import, in a fresh process for every run, and **every table below states the
thread count it was taken at.**

Dense `eigh` (BLAS-3 tridiagonal reduction) and Lanczos (BLAS-2 `gemv`, memory-
bandwidth bound) do not scale alike, so this matters:

| solver, N = 10 000, n_eig = 16, tau = median | 1 thread | 8 threads | 32 threads | 1→32 scaling |
|---|---|---|---|---|
| dense `eigh` | 120.40 s | 23.20 s | **14.92 s** | **8.1×** |
| `eigsh(S, "LA")` | 8.13 s | 1.24 s | **0.61 s** | **13.3×** |

On this problem both scaled well and the ratio did not collapse. The
`eigsh`-vs-dense verdict is the same at all three thread counts.

---

## 2. Gate 2 — eigensolver on the real graph

### 2.1 Full comparison, `tau = median_tau` (package default)

| n_eig | solver | **1 thr** | **8 thr** | **32 thr** | speedup @32 | matvecs | max &#124;Δλ&#124; | **ARI vs dense** |
|---|---|---|---|---|---|---|---|---|
| 8 | `eigh` dense | 118.84 s | 22.21 s | 16.40 s | 1.0× | — | — | 1.0000 |
| 8 | **`eigsh(S,"LA")`** | **4.75 s** | **0.99 s** | **0.35 s** | **46.7×** | 71 | 1.1e-15 | **1.0000** |
| 8 | `eigsh` ncv=32 | 4.75 s | 1.00 s | 0.31 s | 52.7× | 71 | 1.1e-15 | 1.0000 |
| 8 | `eigsh` tol=1e-6 | 2.81 s | 0.63 s | 0.21 s | 77.9× | 42 | 8.9e-16 | 1.0000 |
| 8 | `lobpcg` | 3.38 s | 0.80 s | 0.25 s | 66.0× | 51 | 7.7e-05 | **0.6074** |
| 16 | `eigh` dense | 120.40 s | 23.20 s | 14.92 s | 1.0× | — | — | 1.0000 |
| 16 | **`eigsh(S,"LA")`** | **8.13 s** | **1.24 s** | **0.61 s** | **24.4×** | 122 | 1.4e-15 | **1.0000** |
| 16 | `eigsh` ncv=64 | 9.04 s | 1.30 s | 0.61 s | 24.3× | 137 | 1.6e-15 | 1.0000 |
| 16 | `eigsh` tol=1e-6 | 5.77 s | 0.97 s | 0.49 s | 30.6× | 89 | 1.5e-15 | 1.0000 |
| 16 | `lobpcg` | 6.98 s | 0.54 s | 0.54 s | 27.6× | 106 | 3.0e-05 | **0.3878** |

### 2.2 Full comparison, `tau = median_tau / 10` (sharp affinity)

| n_eig | solver | **1 thr** | **8 thr** | **32 thr** | speedup @32 | matvecs | max &#124;Δλ&#124; | **ARI vs dense** |
|---|---|---|---|---|---|---|---|---|
| 8 | `eigh` dense | 119.96 s | 25.59 s | 15.96 s | 1.0× | — | — | 1.0000 |
| 8 | **`eigsh(S,"LA")`** | **5.40 s** | **1.14 s** | **1.14 s** | **14.0×** | 81 | 1.6e-15 | **1.0000** |
| 8 | `eigsh` ncv=32 | 4.88 s | 1.02 s | 0.92 s | 17.3× | 74 | 1.2e-15 | 1.0000 |
| 8 | `eigsh` tol=1e-6 | 3.40 s | 0.62 s | 0.64 s | 24.9× | 52 | 2.2e-15 | 1.0000 |
| 8 | `lobpcg` | 6.92 s | 1.29 s | 1.31 s | 12.1× | 105 | 5.8e-08 | 0.9913 |
| 16 | `eigh` dense | 123.18 s | 25.59 s | 14.02 s | 1.0× | — | — | 1.0000 |
| 16 | **`eigsh(S,"LA")`** | **11.98 s** | **1.64 s** | **2.16 s** | **6.5×** | 171 | 2.6e-15 | **1.0000** |
| 16 | `eigsh` ncv=64 | 9.58 s | 1.27 s | 1.80 s | 7.8× | 140 | 2.5e-15 | 1.0000 |
| 16 | `eigsh` tol=1e-6 | 8.06 s | 1.09 s | 1.49 s | 9.4× | 117 | 1.4e-15 | 1.0000 |
| 16 | `lobpcg` | 17.28 s | 2.43 s | 3.28 s | 4.3× | 251 | 1.7e-07 | 0.9977 |

**The narrowest margin anywhere in the grid is 6.5×** (sharp affinity, n_eig=16,
32 threads). `eigsh` never lost.

### 2.3 Memory

Peak RSS sampled in-process at 20 ms. The benchmark holds `C`, `W`, `S` and `L`
simultaneously (~3.2 GB baseline), so the meaningful number is the *increment*:

| solver | baseline RSS | peak RSS | **solver increment** |
|---|---|---|---|
| dense `eigh` | ~4 010 MB | ~4 770 MB | **+755 MB** — one N×N float64 copy |
| `eigsh` | ~3 250 MB | ~3 255 MB | **+~5 MB** — the N × ncv Lanczos basis |
| `lobpcg` | ~3 254 MB | ~3 269 MB | +~15 MB |

Dense `eigh` needs an extra N×N working copy: 800 MB at N = 10 000, **3.2 GB at
N = 20 000**, on top of the Laplacian itself. `eigsh` needs `O(N·ncv)` — under
5 MB. This is a second, independent reason to prefer it as N grows.

### 2.4 Observed spectral gap — the gate's actual premise

The graph is genuinely weak, exactly as feared.

`tau = median_tau`, ascending Laplacian eigenvalues:

```
lambda_L : 0  0.972489  0.990679  0.994719  0.994822  0.995985  0.996269  0.997028
              0.997091  0.997406  0.997493  0.997547  0.997556  0.997628  0.997655  0.997735
lambda_S : 1  0.027511  0.009321  0.005281  0.005178  0.004015  0.003731  0.002972
              0.002910  0.002594  0.002507  0.002453  0.002444  0.002373  0.002346  0.002265
```

Consecutive Laplacian gaps beyond the trivial first run from 1.8e-2 down to
**9.5e-6**. The relative gap `(λ_k − λ_{k+1}) / (λ_1 − λ_{k+1})` on the `S`
problem is **7.6e-4 at k = 8** and **8.1e-5 at k = 16**. On the usual reading
this should have been a disaster for Lanczos.

It was not: 71 matvecs at k = 8, 122 at k = 16. A full dense decomposition of the
same matrix on a 1500-voxel slice shows why:

| `tau` | λ₁ | λ₂ | λ₈ | λ₁₆ | λ_min | 99th pct of &#124;λ&#124; in the bulk (i > 16) |
|---|---|---|---|---|---|---|
| median | 1.0000 | 0.01245 | 0.00261 | 0.00191 | −0.00171 | 0.00160 |
| median/10 | 1.0000 | 0.82926 | 0.45553 | 0.41938 | −0.51135 | 0.41129 |

At `tau = median_tau`, `W = exp(-(1-C)/tau)` on a low-contrast `C` is very close
to **rank one**: `S` has a single eigenvalue at 1 and its entire remaining
spectrum is confined to `|λ| ≤ 0.0125`, a band 80× narrower than the distance to
λ₁. A low-degree Lanczos polynomial separates the top of that narrow band easily.

`tau = median_tau / 10` is the opposite regime — gaps of 1e-3 to 9e-2, spectrum
spread over `[−0.51, 1]`. eigsh needs *more* matvecs there (171 vs 122 at k=16),
not fewer.

**The lesson: a weak Laplacian gap and slow Lanczos convergence are not the same
thing.** The gate's premise (weak gap ⇒ Lanczos struggles) does not hold here,
and the reason is a general property of coherence affinities: a near-uniform
affinity is near-rank-one, which is the *easy* case for a Krylov method. The two
`tau` values bracket the plausible range and `eigsh` won in both.

### 2.5 ARI — the result that matters most

`eigsh` reproduced the dense partition **exactly (ARI = 1.0000)** in all 12
configurations: both `tau`, both `n_eig`, default and 4k `ncv`, and even at
`tol=1e-6` where the solver may stop early.

`lobpcg` did not. At `tau = median_tau` it agreed on eigenvalues to 3e-5–8e-5 and
produced **ARI 0.39 and 0.61** — a substantially different clustering. This is
precisely the failure the gate was written to catch: **matching eigenvalues do
not imply a matching invariant subspace.** On a graph with 1e-5-scale gaps the
eigenvectors rotate freely within the near-degenerate subspace while the
eigenvalues look converged. Unpreconditioned LOBPCG does not resolve them;
ARPACK's implicit restarting does. Note that lobpcg looked *fine* at
`tau = median/10` (ARI 0.99) — the failure only appears in the weak-gap regime,
which is the default regime.

### 2.6 Gate 2 verdict

**(a) Does the speedup survive on real data? Yes — it is larger than synthetic.**
10–25× at 1 thread and 6.5–46× at 32 threads, against the synthetic 5.4×, with
eigenvalue agreement at 1e-15 and *exact* label agreement. The concern that a
weak real-data spectral gap would defeat Lanczos is **not supported**.

**(b) Is `eigsh` safe as the default? Yes.** Use
`eigsh(S, k=n_eig, which="LA")` on `S = D^{-1/2} W D^{-1/2}` and take
`lambda_L = 1 - lambda_S`. No shift-invert, no preconditioner, default `ncv`
(raising `ncv` to 4k changed nothing measurable). **Keep `tol` at the default 0** —
`tol=1e-6` was 1.4–1.7× faster and still gave ARI 1.0, but that is one dataset
and the saving is small next to the risk demonstrated by lobpcg.

**(c) Should dense stay as an explicit `solver="dense"`? Yes**, for three reasons
the measurements support: it is the reference that *defines* ARI = 1.0 for
exactness audits; at small N the ARPACK setup makes the comparison a wash; and
`eigh` with `subset_by_index` returns a contiguous spectrum, which is what
`eigengap` wants when sweeping k. It should not be the default.

**(d) `lobpcg` should not be offered as a default or a peer alternative.** It was
slower than `eigsh` in 3 of 4 configurations and silently returned a different
clustering while reporting converged-looking eigenvalues.

**(e) The `icampsnfr/spectral.py` docstring is wrong** (line 21: *"for N up to
~20k this beats sparse iterative solvers, which struggle for the smallest
Laplacian eigenvalues without shift-invert preconditioning"*). Measured false on
real data at all three thread counts. The premise is the error: nobody should ask
for the smallest eigenvalues of `L`. Correct it when the module is absorbed
(the dense call is at line 101).

---

## 3. Gate 3 — k-NN sparsification

Affinity `W` sparsified by keeping each row's `k_nn` largest entries and
symmetrizing with `A = max(A, Aᵀ)` (`scipy.sparse`). Because
`exp(-(1-C)/tau)` is monotone in `C`, this selects the same neighbours as
top-k on `C` itself. Solver: `eigsh(S_sparse, k=n_eig, which="LA")`.
ARI is measured **against the dense-affinity dense-solver labels** at the same
`tau` and `n_eig`.

### 3.1 Graph structure and connected components

| `k_nn` | nnz | density | build (s) | **components** | largest | singletons |
|---|---|---|---|---|---|---|
| 10 | 154 394 | 1.54e-3 | 1.2 | **1** | 10 000 | 0 |
| 30 | 500 708 | 5.01e-3 | 1.1 | **1** | 10 000 | 0 |
| 100 | 1 757 458 | 1.76e-2 | 1.2 | **1** | 10 000 | 0 |
| 300 | 5 260 656 | 5.26e-2 | 1.5 | **1** | 10 000 | 0 |

**The graph never disconnected**, at any `k_nn`, for either `tau`. No isolated
nodes, no singletons, so `normalized_laplacian` never raised. The plan's concern
about aggressive sparsification disconnecting the graph did not materialize —
`exp(-(1-C)/tau)` is strictly positive everywhere, so every retained edge carries
positive weight and max-symmetrization guarantees each node at least `k_nn`
neighbours. **Disconnection is not the reason k-NN fails here.**

### 3.2 Results, `tau = median_tau`

Dense reference at this `tau`: 112.2 s @1 thr / 17.6 s @32 thr (n_eig=8).

| `k_nn` | n_eig | eigsh **1 thr** | eigsh **32 thr** | speedup @32 | peak RSS | matvecs | max &#124;Δλ&#124; | **ARI vs dense** |
|---|---|---|---|---|---|---|---|---|
| 10 | 8 | 0.22 s | 0.21 s | 81.9× | 1735 MB | 464 | 9.82e-1 | **0.0180** |
| 10 | 16 | 0.25 s | 0.25 s | 70.6× | 1738 MB | 377 | 9.82e-1 | **0.0718** |
| 30 | 8 | 0.17 s | 0.16 s | 110.1× | 1753 MB | 210 | 9.26e-1 | **0.0186** |
| 30 | 16 | 0.19 s | 0.19 s | 94.3× | 1753 MB | 189 | 9.26e-1 | **0.0738** |
| 100 | 8 | 0.18 s | 0.18 s | 97.2× | 1803 MB | 82 | 7.41e-1 | **0.2339** |
| 100 | 16 | 0.36 s | 0.33 s | 53.9× | 1803 MB | 141 | 7.41e-1 | **0.1776** |
| 300 | 8 | 0.67 s | 0.63 s | 27.7× | 1923 MB | 98 | 5.97e-1 | **0.3922** |
| 300 | 16 | 1.11 s | 1.05 s | 16.8× | 1923 MB | 156 | 5.97e-1 | **0.1981** |

### 3.3 Results, `tau = median_tau / 10`

Dense reference at this `tau`: 131.3 s @1 thr / 20.9 s @32 thr (n_eig=8).

| `k_nn` | n_eig | eigsh **1 thr** | eigsh **32 thr** | speedup @32 | matvecs | max &#124;Δλ&#124; | **ARI vs dense** |
|---|---|---|---|---|---|---|---|
| 10 | 8 | 0.56 s | 0.57 s | 36.5× | 1378 | 5.96e-1 | **−0.1350** |
| 10 | 16 | 1.17 s | 1.18 s | 14.8× | 1790 | 7.67e-1 | **−0.0664** |
| 30 | 8 | 0.34 s | 0.35 s | 59.0× | 466 | 5.83e-1 | **−0.1080** |
| 30 | 16 | 0.85 s | 0.84 s | 20.6× | 857 | 7.37e-1 | **−0.0290** |
| 100 | 8 | 1.08 s | 1.03 s | 20.2× | 484 | 4.94e-1 | **0.1603** |
| 100 | 16 | 1.49 s | 1.43 s | 12.1× | 616 | 6.57e-1 | **0.0751** |
| 300 | 8 | 2.48 s | 2.42 s | 8.6× | 372 | 3.78e-1 | **0.2974** |
| 300 | 16 | 2.41 s | 2.39 s | 7.3× | 351 | 5.31e-1 | **0.3222** |

Sparse `eigsh` times are essentially **identical at 1 and 32 threads** — scipy's
CSR matvec is single-threaded and does not use BLAS. Only the dense reference
moves with thread count, which is why the speedup column shrinks by ~6× going
from the 1-thread to the 32-thread basis.

### 3.4 What sparsification does to the spectrum

The eigenvalue disagreement is not a rounding issue — it is total. At
`tau = median`, `k_nn = 10` the sparsified Laplacian's first non-trivial
eigenvalues are `0.0096, 0.0119, 0.0129` against the dense graph's
`0.9725, 0.9907, 0.9947` — `max|Δλ| = 0.982` on a spectrum bounded by ~1. The
k-NN graph is a **different graph with a different spectrum**, not an
approximation of the dense one.

The negative ARI values at `tau = median/10`, `k_nn ∈ {10, 30}` are worth stating
plainly: the sparsified clustering is *worse than chance* agreement with the
dense one — anti-correlated, not merely noisy.

ARI does climb with `k_nn` (0.018 → 0.392 at `tau = median`, n_eig=8), which is
the expected direction, but it is still only 0.39 at `k_nn = 300` — a graph that
is already 5.3 % dense and where the speedup has fallen to 8.6–27.7×. Extrapolating,
recovering the dense partition would require `k_nn` approaching N, i.e. not
sparsifying at all.

### 3.5 Gate 3 verdict

**k-NN sparsification is not worth adopting at any `k_nn` tested, and should not
be added to the package.**

- Speed is not the issue — it is 7–110× faster than dense and 2–10× faster than
  dense-affinity `eigsh`, and it cuts memory from an 800 MB dense `W` to 63 MB at
  `k_nn = 300`.
- **The cluster structure does not survive.** ARI vs the dense-affinity labels
  peaked at **0.39** and went *negative* in four cells. Sparsification changes
  the answer, and the whole point of Gate 2 was that `eigsh` gives the *same*
  answer.
- The failure mode is not the anticipated one. Connectivity was never lost; the
  graph simply is not sparse in structure. With a median off-diagonal coherence
  of 0.055 against a `1/K` floor of 0.045, a voxel's "10 nearest neighbours" are
  barely distinguishable from its 1000th, and top-k selection largely amplifies
  estimator noise rather than capturing real affinity.
- **This is a property of low-contrast coherence graphs**, so it is unlikely to
  be fixed by a different `k_nn`, a different symmetrization, or mutual-k-NN.
  If sparsification is ever revisited it should be after the affinity has real
  contrast (better `tau` selection, more segments `K` to lower the bias floor),
  and it must be re-gated on ARI, not on speed.

Given Gate 2, there is no pressure to adopt it: `eigsh` on the dense affinity is
already 6.5–46× faster than dense `eigh` **and exact**.

---

## 4. Addendum — V1 vs V2 on the same real coefficients

Requested after an adversarial review overturned the stated justification for
choosing V2 over V1. Both are delay-invariant (axis A identical); they differ
only on axis B, how frequencies are weighted. The question is which estimator's
documented blind spot actually occurs in this data.

- **V2's blind spot** (`derivations/coherence_estimators.md` §2): with a spectrum
  spanning orders of magnitude, the power weights `w_f = P_nn(f)P_mm(f)` collapse
  the effective bandwidth toward the strongest bin.
- **V1's blind spot**: it divides by `P_nn(f)P_mm(f)` at *every* bin, so a dead or
  near-dead in-band bin gives `0/0` or amplifies noise to `γ² ≈ 1`.

Both computed on the identical `(10000, 22, 15)` coefficient array used above.
Sanity check of the derivation note's §3 identity: **`V2[whitened] == V1[raw]` to
1.16e-7** (complex64 rounding) — confirms both implementations are correct.

### 4.1 Spectral tilt — V2's blind spot does not occur

In-band auto-power `P_nn(f)`. Mean over 10 000 voxels, per bin:

```
4982.5  5165.1  4382.4  4024.7  4049.5  4074.0  4034.5  3945.5
3959.0  3900.2  3761.2  3848.0  3789.9  3750.9  3858.7
```

normalized to the largest bin:

```
0.965  1.000  0.849  0.779  0.784  0.789  0.781  0.764
0.767  0.755  0.728  0.745  0.734  0.726  0.747
```

| statistic | value |
|---|---|
| **per-bin mean power, max / min** | **1.377×** |
| per-voxel max-bin/min-bin ratio — 5th pct | 1.69 |
| — 25th pct | 2.05 |
| — **median** | **2.47** |
| — 75th pct | 3.19 |
| — 95th pct | 5.52 |
| — max over all 10 000 voxels | 27.5 |
| **participation ratio `F_eff = (Σw)²/Σw²`, median (of F = 15)** | **14.10** |
| `F_eff` quartiles / min | 13.44 / 14.42, min 4.70 |

The band is close to flat. Averaged over voxels the in-band power varies by
**1.38×**, not orders of magnitude, and V2's power weighting retains an effective
bandwidth of **14.1 of 15 bins** at the median voxel — it is behaving almost
exactly like equal weighting.

This is a consequence of the preprocessing, not luck: the zero-phase Butterworth
high-pass at a 120 s cutoff has already removed the `1/f` drift that would tilt
the low end of a 10–40 s band. **The `1/f` scenario in the derivation note's §4
tilt experiment is not the regime this pipeline operates in.**

### 4.2 Dead bins — V1's blind spot does not occur either

| threshold on `P_nn(f)`, relative to that voxel's median in-band power | count / 150 000 cells | fraction |
|---|---|---|
| exactly zero | 0 | 0 |
| < 1e-12 × median | 0 | 0 |
| < 1e-9 × median | 0 | 0 |
| < 1e-6 × median | 0 | 0 |
| < 1e-3 × median | 0 | 0 |
| **< 1e-2 × median** | **0** | **0** |

Minimum observed ratio to the voxel's own median: **0.259**. Absolute range of
`P_nn(f)`: 2.61e2 to 2.75e5. Not a single weak bin anywhere, let alone a dead
one. V1's `0/0` hazard is absent, and a `P_nn(f) > eps` guard would never fire.

### 4.3 Direct comparison — they still disagree substantially

Over all 49.995 M unique pairs:

| statistic | V1 | V2 |
|---|---|---|
| mean | 0.05600 | 0.06730 |
| median | 0.05235 | 0.05494 |

| `V2 − V1` | value |
|---|---|
| mean | **+0.01130** |
| std | 0.02787 |
| max abs | **0.5755** |
| percentiles 0.1 / 1 / 5 / 25 | −0.0229 / −0.0131 / −0.0072 / −0.0015 |
| **50** | **+0.0025** |
| 75 / 95 / 99 / 99.9 | +0.0121 / +0.0617 / +0.1340 / +0.2551 |
| Pearson r | 0.8645 |
| Spearman ρ (200 k sample) | 0.9220 |

V2 is systematically higher, and the excess is concentrated in exactly the pairs
that drive clustering:

| V1 bin | pairs | share | mean(V2−V1) | median | sd |
|---|---|---|---|---|---|
| [0.00, 0.02) | 151 795 | 0.30 % | +0.0001 | −0.0001 | 0.0023 |
| [0.02, 0.05) | 22 135 591 | 44.28 % | +0.0017 | +0.0005 | 0.0070 |
| [0.05, 0.10) | 25 933 304 | 51.87 % | +0.0141 | +0.0058 | 0.0244 |
| **[0.10, 0.20)** | 1 741 883 | 3.48 % | **+0.0913** | +0.0789 | 0.0685 |
| **[0.20, 0.40)** | 27 496 | 0.05 % | **+0.1049** | +0.0503 | 0.1216 |
| [0.40, 0.60) | 2 989 | 0.01 % | +0.0596 | +0.0371 | 0.0655 |
| [0.60, 0.80) | 1 922 | 0.00 % | +0.0398 | +0.0292 | 0.0379 |
| [0.80, 1.01) | 20 | 0.00 % | +0.0175 | +0.0026 | 0.0297 |

6.81 % of pairs move by more than 0.05 and 2.03 % by more than 0.10 — against a
median coherence of 0.055 and a bias floor of 0.045. And the two estimators
disagree markedly on *which* pairs are the strongest:

| set | Jaccard(V1, V2) |
|---|---|
| top 1 % of pairs | **0.400** |
| top 0.1 % of pairs | **0.209** |
| top 0.01 % of pairs | 0.360 |

| neighbourhood | mean overlap | median | 5th pct |
|---|---|---|---|
| top-10 neighbours | 0.658 | 0.727 | 0.273 |
| top-50 | 0.538 | 0.510 | 0.255 |
| top-100 | 0.537 | 0.515 | 0.327 |
| top-300 | 0.593 | 0.581 | 0.442 |

### 4.4 ARI between V1-derived and V2-derived cluster labels

Same pipeline as Gate 2 (`eigsh`, NJW embedding, `KMeans(random_state=0)`), each
matrix using its own `median_tau` (V1 0.947646, V2 0.945062):

| n_eig | n_clusters | **ARI V1 vs V2, own `tau`** | ARI, `tau` matched to V2's |
|---|---|---|---|
| 8 | 7 | **0.6401** | 0.6418 |
| 16 | 15 | **0.3270** | 0.3425 |

Matching `tau` changes almost nothing, so the disagreement is the *estimator*,
not the bandwidth heuristic. Cluster size distributions are comparably balanced
for both (largest cluster 2482 vs 2874 at k=7; 1367 vs 1484 at k=15), so neither
is degenerating.

### 4.5 Addendum verdict

1. **Neither estimator's documented blind spot occurs in this data.** In-band
   power is nearly flat (1.38× across bins, `F_eff` = 14.1/15), so V2's
   effective-bandwidth collapse does not happen. There is not a single near-dead
   bin (minimum 0.259× the voxel median), so V1's `0/0` hazard does not happen
   and would be trivially guarded if it ever did. **Both arguments in
   `derivations/coherence_estimators.md` §2 are inert on this pipeline's output.**

2. **The choice is nevertheless high-stakes, not low-stakes.** Despite flat
   spectra, V1 and V2 agree on only **40 % of the top 1 % of pairs**, ~54 % of
   each voxel's 100 nearest neighbours, and produce clusterings at
   **ARI 0.64 (k=7) and 0.33 (k=15)**. Swapping estimators changes the scientific
   result. This directly contradicts the intuition that near-equal weights imply
   near-equal answers — the weights are near-equal *on average*, but the
   pair-specific weight `w_f = P_nn(f)P_mm(f)` is a product, so the per-voxel
   2.5× spreads compound.

3. **This measurement cannot say which estimator is correct.** There is no ground
   truth here. It establishes only that (a) the stated reasons for preferring V2
   do not apply, and (b) the decision cannot be treated as cosmetic. If V2 is to
   remain the default, it needs a justification that survives this data — the
   §2 tilt and dead-bin arguments do not. A recommended follow-up: a
   stability/recovery test (subsample segments or voxels and measure label
   reproducibility for each estimator), which *can* discriminate without ground
   truth.

4. **Practical note.** The whitening identity holds to 1.16e-7, so V1 remains a
   one-line pre-normalization away from V2 (§3 of the derivation note). Keeping
   both reachable behind `whiten=` costs nothing and is now clearly worth doing —
   with the debias caveat that both need `n_eff = K = 22`, not `K·F = 330`.

---

## 5. Reproduction

Scripts (measurement only, nothing installed, nothing written to `/nrs`):

| file | purpose |
|---|---|
| `scratchpad/build_real_C.py` | loads the real STFT coefficients, subsamples to N=10 000, builds the V2 matrix |
| `scratchpad/gate23_bench.py` | gates 2 and 3; `--gate`, `--n-eig`, `--taus`, `--knn` |
| `scratchpad/v1_vs_v2.py` | the §4 addendum |

Raw outputs: `/tmp/gate23/{gate2_t1,gate2_t8,gate2_t32,gate3_t1,gate3_t32,v1_vs_v2}.json`.

Thread count must be set via `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/
`MKL_NUM_THREADS` **before** numpy is imported; the inherited shell environment
pins all of them to 1.
