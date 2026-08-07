# Phase 3 plan — clustering: `clustering.py`, `kmeans.py`, `metrics.py`

Branch `refactor/phase-1-spectral-core`. Phases 1 (`src/cmm/spectral.py`, f0cf4b2) and
2 (`src/cmm/coherence.py`, bb253fd) are done and committed. This phase delivers the two
clustering routes on top of `Stft` / `CoherenceMatrix`, plus the metrics both need.

## 0. Scope, and how it maps onto the master plan

This "Phase 3" **merges rows 3 and 4 of the master phase table** (`kmeans.py` and
`clustering.py`) and **pulls `metrics.py` forward from row 6**. Rationale:

- The gate that row 4 was waiting on — "Lanczos speedup + agreement confirmed on a real
  coherence graph before dense `eigh` is dropped" — is **already cleared**
  (`logs/gate23_eigensolver_real_data.md`: ARI = 1.0000 in all 12 configurations,
  6.5–46x at 32 threads). Nothing blocks `clustering.py`.
- Both routes need silhouette/ARI in their test suites, so `metrics.py` cannot wait for
  row 6. Only the *metrics module* moves forward; packaging/CI/README stay in Phase 6.

**Out of Phase 3** (confirmed against the master plan):

| Item | Where it stays |
|---|---|
| `phase.py`, `wavelet.py`, `wiener_filter` repair (Q1) | Phase 5 |
| packaging, `pyproject`/`setup.py` reconciliation, CI, README | Phase 6 |
| `icampsnfr` migration — including fixing its `debias_msc` call (`n_eff = n_seg * n_freqs`, wrong by a factor F; `coherence.py:218`) and correcting its `spectral.py:21` docstring in place | Phase 7 |
| `examples/`, `archive/` retirement | Phase 8 |
| hierarchical clustering (`icampsnfr/cluster.py` linkage machinery) | never absorbed (per master plan §3); only its `_distance`/`_subsample_silhouette`/`cluster_summary` logic comes in, via `metrics.py` |
| projection-matrix machinery (`utils.build_fft_trial_projection_matrices`), centroid backprojection | deferred; the old backproject path is broken (C4, C6) and the new API does not use it. Resolved when `cmm.py` is deleted (chunk 5) / Phase 5 |
| the V1-vs-V2 stability discriminator measurement (CLAUDE.md "Open") | not Phase 3 work, but chunk 1 delivers the tooling (`label_stability`) it needs |

**Deletable at the end of Phase 3** (separate pure-deletion commit, nothing before its
replacement is green): `src/cmm/cmm.py`, `cmm_funcs.py`, `power_iteration.py`,
`ana_funcs.py` — all currently dead (jax not installed, none import). **Not deletable**:
`spectral_funcs.py`, `utils.py`, `toy_data.py` (`icampsnfr` imports `spectral_funcs`;
`toy_data` imports `utils`) — those wait for Phase 7/8.

**Constraints honored throughout**: numpy/scipy/scikit-learn only (`kmeans.py` uses no
sklearn at all; mne is not needed this phase); no jax, no dask; fail-fast with named
values; arrays/dataclasses in, arrays/dataclasses out, no file I/O; N=10k scale target;
pure refactors and numerically-moving changes never share a commit; no timings inside
tests, and the one measurement script (Gate 4) sets thread env vars before numpy import
and records the thread count.

## 1. The k-means spec, extracted from the old code

The old `CMM` (`cmm.py` + `cmm_funcs.py`) is the only spec and is known-broken, so the
algorithm is restated here explicitly; the plan implements *this*, not the code.

**What the old code actually does** (verified by reading, formulas re-derived, not copied):

1. **Normalize** per signal and frequency: `x̂[n,:,f] = X[n,:,f] / ||X[n,:,f]||` — each
   signal's segment-vector at each frequency becomes a unit vector in C^K
   (`initialize_coefs`, `cmm.py:116-117`).
2. **Centroid update** per cluster m, per frequency f (`compute_cluster_centroid_eigh`,
   `cmm_funcs.py:18-32`): form the K×K Hermitian gram
   `P_f^(m) = Σ_{n∈m} x̂[n,:,f] x̂[n,:,f]^H` and take its **top eigenvector** `v[m,:,f]`
   (unit norm). This is the direction maximizing `Σ_{n∈m} |v^H x̂[n,:,f]|²` — the top
   left singular vector of the cluster's coefficient stack at f.
3. **Allocation** (`allocate_data_to_clusters`): per-frequency magnitude-squared
   coherence between centroid m and signal n, averaged equally over frequency, argmax
   over m. Because both `v` and `x̂` are unit-norm per frequency, the coherence ratio
   reduces *exactly* to `γ²(m,n,f) = |v[m,:,f]^H x̂[n,:,f]|²` — the ratio's
   normalization cancels (this is also why the legacy coherence path was the one
   scipy-exact piece of the old code).

**The objective**, stated once and put in the module docstring:

```
J(c, V) = Σ_n (1/F) Σ_f |v[c(n),:,f]^H x̂[n,:,f]|²        (0 ≤ J ≤ N)
```

Both steps are **exact block-coordinate ascent** on J: the eigenvector is the exact
maximizer of the centroid subproblem, and argmax allocation cannot decrease any point's
term. Therefore J is monotonically non-decreasing and converges. **Monotonicity is the
central correctness handle** — it is a theorem about the algorithm, independent of the
broken reference implementation, and it becomes a test asserted on every fitted result.

Two observations that shape the API:

- The historical objective is **V1-flavored**: per-frequency normalization + equal
  frequency weighting — exactly the `whiten=True` estimator of `coherence.py`. A
  power-weighted (V2-analogue) variant is the same machinery with *fixed* per-signal
  weights `w[n,f] = p[n,f] / Σ_f p[n,f]` (weighted grams, weighted score); fixed weights
  preserve exact block-coordinate ascent, hence monotonicity. Proposed: implement both
  behind `whiten:`, **default `whiten=True`** (the historical spec, the only variant
  with a track record). Note this default is deliberately opposite to
  `coherence_matrix`'s (`whiten=False`) — flagged as an open question.
- The old **power-iteration** centroid solver never wins (master plan Q4: 0.25–1.04x vs
  a batched GEMM+eigh at every size measured; converged in 4–5 iterations, so the loss
  is arithmetic intensity, not convergence). Replacement: one batched
  `np.linalg.eigh` over the `(M, F, K, K)` gram stack. `power_iteration.py` is deleted.

**What is explicitly not carried over** (each was a confirmed bug or dead weight):
C4/C6 backprojection, C5 sign-inverted silhouette (replaced by
`metrics.coherence_silhouette` on `to_distance`), C7 wrong-cluster indexing in
`compute_cluster_coherence`, C8 dead branch, S2 silent empty-cluster shrink, S6 unclosed
`Pool`/unbound `ite`, H1 import-time seeding, the jax/dask plumbing, and the
time-domain-KMeans init (it needs raw traces an `Stft` does not carry; callers who want
it pass explicit initial labels — see `init=` below).

**Cost check at target scale** (N=10⁴, K=22, F=15, M≤16): centroid step
O(N K² F) ≈ 7·10⁷, allocation O(M N K F) ≈ 5·10⁷ flops per iteration; the `(M,N,F)`
score array is ~20 MB. Plain single-process numpy suffices; no thread pool in this
module.

## 2. Work chunks

### Chunk 1 — `src/cmm/metrics.py`

**Files**: `src/cmm/metrics.py` (new), `tests/test_metrics.py` (new).

**API sketch**:

```python
__all__ = ["coherence_silhouette", "pairwise_ari", "label_stability", "cluster_summary"]

def coherence_silhouette(
    values: np.ndarray, labels: np.ndarray, *,
    use_sqrt: bool = True, sample_size: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Silhouette of a labeling against a coherence matrix.

    Converts via to_distance() first — passing the similarity itself to
    sklearn's metric="precomputed" silently negates the answer (bug C5).
    Rejects a CoherenceMatrix container (same TypeError as to_distance).
    Raises if labels has < 2 distinct values, naming the count.
    sample_size subsamples rows for N in the tens of thousands (from
    icampsnfr/cluster.py::_subsample_silhouette, but raising instead of
    warn-and-NaN on degenerate subsamples, per fail-fast).
    """

def pairwise_ari(labelings: Sequence[np.ndarray]) -> np.ndarray
    # (R, R) symmetric, unit diagonal; sklearn.metrics.adjusted_rand_score pairwise.

def label_stability(labelings: Sequence[np.ndarray]) -> dict
    # {"mean_ari", "min_ari", "n_runs"} over the strict off-diagonal.
    # This is the tooling for the V1-vs-V2 resampling discriminator (CLAUDE.md "Open")
    # and for kmeans restart diagnostics.

def cluster_summary(values, labels, *, subsample_silhouette=2000,
                    use_sqrt=True, rng=None) -> dict
    # Absorbed from icampsnfr/cluster.py::cluster_summary (sizes, singleton
    # fraction, mean intra/inter coherence, subsampled silhouette), linkage-free.
```

**Tests** (`tests/test_metrics.py`, mirroring the existing class layout):
- *Reference/identity*: `coherence_silhouette(C, labels)` equals
  `sklearn.metrics.silhouette_score(to_distance(C), labels, metric="precomputed")`
  exactly.
- *Sign pin (C5 regression)*: on a planted 2-block coherence matrix with true labels,
  `coherence_silhouette > 0.5`, while sklearn fed the raw similarity returns a
  *negative* score — asserting the old bug's direction so it can never come back.
- Shuffled labels score ≈ 0 or below; monotone: sharper block contrast ⇒ higher score.
- Subsampling: seeded rng deterministic; `sample_size >= n` equals full computation.
- `pairwise_ari`: identical labelings → 1.0; invariance under label-id permutation →
  1.0; independent random labelings → ≈ 0; output symmetric, unit diagonal.
- `label_stability`: known mixed set gives expected mean/min.
- `cluster_summary`: on planted blocks `mean_intra_C > mean_inter_C`; singleton
  fraction correct; keys stable.
- *Fail-fast*: single distinct label raises naming the count; label/matrix length
  mismatch raises with both shapes; `CoherenceMatrix` container rejected
  (`match="CoherenceMatrix.values"`).

**Done**: tests green; module imports without sklearn side effects at package import
(sklearn imported inside `metrics.py` only); no other module touched.
**Risk**: low. Thin wrappers over sklearn plus an absorbed, already-correct summary.

### Chunk 2 — `src/cmm/clustering.py` (absorb `icampsnfr/spectral.py`, switch to `eigsh`)

**Files**: `src/cmm/clustering.py` (new), `tests/test_clustering.py` (new).
`icampsnfr` itself is **not** modified (Phase 7).

**API sketch** (names per master plan §3 public API):

```python
Solver = Literal["eigsh", "dense"]

@dataclass(frozen=True)
class SpectralEmbedding:
    vectors: np.ndarray        # (N, n_eig), ascending-Laplacian order, sign-fixed
    eigenvalues: np.ndarray    # (n_eig,) ascending eigenvalues of L
    tau: float
    laplacian: Literal["sym", "rw"]
    solver: str
    def metadata(self) -> dict: ...

def median_tau(c_nn: np.ndarray) -> float                      # ported verbatim
def affinity_gaussian(c_nn: np.ndarray, tau: float) -> np.ndarray   # ported verbatim
def normalized_laplacian(w_nn: np.ndarray, kind: str = "sym") -> np.ndarray
    # dense L, the audit path; keeps the isolated-node raise with count + advice

def spectral_embedding(
    c: CoherenceMatrix | np.ndarray, n_eig: int, *,
    tau: float | None = None,          # None -> median_tau(c)
    laplacian: Literal["sym", "rw"] = "sym",
    solver: Solver = "eigsh",
    random_state: int | None = 0,
) -> SpectralEmbedding

def eigengap(eigenvalues, *, min_k: int = 2) -> tuple[np.ndarray, int]  # ported verbatim

def cluster_embedding(emb: SpectralEmbedding | np.ndarray, n_clusters: int, *,
                      drop_first=True, row_normalize=True,
                      random_state: int = 0, n_init: int = 10) -> np.ndarray

def spectral_labels(c, n_clusters, *, tau=None, n_eig=None, laplacian="sym",
                    solver="eigsh", row_normalize=True, random_state=0) -> np.ndarray
```

Key decisions, all pre-measured:

- **`solver="eigsh"` default** per Gate 2: `S = D^{-1/2} W D^{-1/2}`,
  `eigsh(S, k=n_eig, which="LA")`, `λ_L = 1 − λ_S`, reordered ascending. No
  shift-invert, no preconditioner, default `ncv`, **`tol=0`** (Gate 2 §2.6b: the
  `tol=1e-6` saving is small next to the lobpcg-demonstrated risk). `lobpcg` and k-NN
  sparsification are **not** offered (Gate 2d, Gate 3).
- **`solver="dense"` kept** for audits and contiguous-spectrum eigengap sweeps
  (Gate 2 §2.6c): builds `L` via `normalized_laplacian`, `eigh(subset_by_index=(0, n_eig-1))`.
- **`laplacian="rw"`** served from the same `S` solve: `u_rw = D^{-1/2} u_sym`
  (eigenvalues identical) — so both kinds get the fast solver.
- **Deterministic seeding**: ARPACK's start vector is otherwise random —
  `v0 = default_rng(random_state).standard_normal(n)`; eigenvector **sign fixed**
  (flip so the largest-|·| component is positive, sklearn's convention) so embeddings
  are reproducible across runs and comparable across solvers; sklearn `KMeans` gets the
  same `random_state`.
- **Labels 0-based ints** (sklearn convention; `icampsnfr` used 1-based — Phase 7
  adapts at its call sites). Flagged as open question.
- **Docstring corrected**: the absorbed module must state the *measured* result — a
  weak Laplacian gap does not imply slow Lanczos (near-uniform affinity ⇒ near-rank-one
  S ⇒ isolated λ₁; Gate 2 §2.4) — replacing the measured-false claim at
  `icampsnfr/spectral.py:21`. Cite `logs/gate23_eigensolver_real_data.md`.
- Accepts `CoherenceMatrix` or a bare `(N, N)` array (unwraps `.values`); the k-sweep
  pattern (embed once at `n_eig = max_k + 1`, cluster per k) preserved.

**Tests** (`tests/test_clustering.py`):
- *Naive reference*: explicit per-entry loop building `L` matches
  `normalized_laplacian` for both kinds; `affinity_gaussian` symmetric, zero diagonal,
  strictly monotone in C.
- *Solver agreement* (the load-bearing layer): on planted-block matrices — dense vs
  eigsh eigenvalues `rtol=1e-10`; embedding subspaces agree via principal angles
  (compare subspaces, not vectors, to be robust to rotation within near-degenerate
  blocks); end-to-end labels ARI == 1.0; `rw`-vs-`sym` eigenvalue equality; `rw`
  eigenvectors satisfy `u_rw = D^{-1/2} u_sym` up to scale/sign.
- *External cross-validation*: `spectral_labels` vs
  `sklearn.cluster.SpectralClustering(affinity="precomputed")` on the same `W`:
  ARI > 0.95, k ∈ {2, 3, 5} (master plan L1 row).
- *Eigengap*: recovers true k on block-structured C for k ∈ {2, 3, 5}; `min_k`
  semantics (ported tests).
- *Determinism*: two identical calls give identical labels (seeded `v0` + KMeans seed);
  embed-once-cluster-per-k equals one-shot per k.
- *Properties*: `median_tau` known-value check; eigenvalues nonnegative within 1e-12;
  first eigenvalue ≈ 0 for a connected graph.
- *Fail-fast*: `tau <= 0`; `n_eig` out of `[1, N]`; isolated nodes (count named);
  non-square input; NaN in C — each asserting type + message substring.
- `@pytest.mark.slow`: N=2000 low-contrast matrix (structured signal barely above a
  1/K-scale floor, miniature of the Gate 2 regime): eigsh-vs-dense labels ARI == 1.0.

**Done**: tests green in < 60 s default selection; behavior identical to the
`icampsnfr` dense path on shared inputs (which Gate 2 already proved for the solver
swap); `icampsnfr` untouched; committed as one absorption commit whose message states:
numerically identical to dense (ARI 1.0, dλ ~1e-15, measured), API-visible changes only
(0-based labels, solver/seed parameters, result dataclass).
**Risk**: low–medium. The one flake hazard is eigenvector comparison in near-degenerate
subspaces — mitigated by comparing subspaces/labels, not raw vectors, and by
well-separated fixtures for exact-label assertions.

### Chunk 3 — `src/cmm/kmeans.py` (coherence k-means, no jax)

**Files**: `src/cmm/kmeans.py` (new), `tests/test_kmeans.py` (new).

**API sketch**:

```python
@dataclass(frozen=True)
class CoherenceKMeansResult:
    labels: np.ndarray             # (N,) int, 0-based
    centroids: np.ndarray          # (M, K, F) complex128, unit norm per (m, f)
    centroid_strength: np.ndarray  # (M, F) top-eigenvalue share of cluster energy, in (0, 1]
    scores: np.ndarray             # (M, N) frequency-averaged coherence to each centroid
    objective: np.ndarray          # (n_iter + 1,) trace of J — monotone by construction
    n_iter: int
    converged: bool                # labels stabilized before max_iter
    n_reseeds: int                 # empty-cluster reseeds that occurred
    whiten: bool
    random_state: int | None
    stft_metadata: dict
    def metadata(self) -> dict: ...

# free functions first (each independently testable), class second:
def normalize_coefs(coefs_nkf) -> np.ndarray
    # x-hat; raises on any zero-power (n, f) cell, pointing at finite_power_mask()
def centroid_update(xhat_nkf, labels, n_clusters, *, weights_nf=None
                    ) -> tuple[np.ndarray, np.ndarray]   # (M,K,F) centroids, (M,F) strength
    # batched np.linalg.eigh over the (M, F, K, K) Hermitian gram stack
def centroid_scores(xhat_nkf, centroids_mkf, *, weights_nf=None) -> np.ndarray  # (M, N)

def coherence_kmeans(
    s: Stft | np.ndarray,              # Stft preferred; bare (N, K, F) accepted
    n_clusters: int, *,
    whiten: bool = True,               # True = historical V1-flavored objective
    init: Literal["++", "random"] | np.ndarray = "++",   # or explicit (N,) labels
    n_init: int = 10,
    max_iter: int = 300,
    on_empty: Literal["reseed", "raise"] = "reseed",
    random_state: int | None = 0,
) -> CoherenceKMeansResult

class CoherenceKMeans:   # thin fit/predict wrapper holding params; no maths in the class
```

Design points:

- **Objective and updates exactly as §1**; `whiten=False` uses fixed weights
  `w[n,f] = p[n,f]/Σ_f p[n,f]` (monotonicity preserved). Module docstring states J,
  the ascent argument, and the whitening identity linking scores to coherence.
- **Init**: `"++"` — D²-style seeding on `1 − score` against already-chosen centroids,
  fully driven by `default_rng(random_state)`; `"random"` — distinct data points as
  seeds; **explicit labels array** — the escape hatch that lets a caller reproduce the
  historical time-domain-KMeans initialization externally (the library cannot: it takes
  an `Stft`, not raw traces, and file/trace handling belongs to the caller).
- **Restarts**: `n_init` independent runs, return the best final J; deterministic
  given `random_state` (child seeds spawned from one rng).
- **Convergence**: labels unchanged between iterations (the old criterion, kept) or
  `max_iter`; `converged` recorded, never printed.
- **Empty cluster** (fixes S2): `"reseed"` (default) re-seeds the cluster from the
  currently worst-scoring point, deterministically, and counts it in `n_reseeds` —
  recovery is visible, k never silently shrinks; `"raise"` raises naming the cluster
  index and iteration. Default flagged as an open question (fail-fast purists would
  raise; reseed-with-record is standard k-means practice and not silent).
- **No sklearn, no threading** in this module (§1 cost check); complex128 throughout
  (the K×K eigenproblems are microscopic; no reason to spend the c64 error budget here).

**Correctness strategy** — this is the riskiest chunk; the only spec is broken code, so
correctness is established by triangulation, none of it copied from `cmm.py`:

1. **Naive reference in the test file** (the pattern that carried Phase 2): centroids
   via an explicit per-cluster, per-frequency Python loop with `scipy.linalg.eigh`;
   allocation computed *through `cmm.spectral.coherence`* — the scipy-pinned path — by
   wrapping each centroid as a one-signal `Stft` and taking pairwise per-frequency MSC
   against members. Production must match both to `rtol=1e-10`. This independently
   validates the `|v^H x̂|² == coherence` reduction rather than assuming it.
2. **Analytic toys with known structure** (generator inline in `conftest.py` until
   `synthetic.py` exists in Phase 5/6; design per master plan L3 — independent
   band-limited source per group plus noise, *not* phase-shifted sinusoids):
   recovery ARI > 0.9 vs ground truth for k ∈ {2, 3, 5} across an SNR sweep; a single
   perfectly coherent cluster gives scores ≈ 1 and `centroid_strength` ≈ 1.
3. **Invariance tests**: labels exactly invariant under per-signal amplitude rescaling;
   **delay invariance** — delayed copies co-cluster (the estimator's whole point);
   permutation equivariance of labels; a duplicated signal lands with its twin.
4. **Objective monotonicity**: `np.diff(result.objective) >= -1e-10` asserted on every
   fitted result in the suite — the theorem-level check that does not depend on any
   reference implementation.
5. **Step optimality**: for a fixed assignment, 100 random unit perturbations of each
   centroid never beat the eigenvector's score; each point's assigned score is its row
   max of `scores`.
6. **Cross-route agreement**: `coherence_kmeans` vs `spectral_labels` ARI > 0.9 on the
   well-separated generator (master plan L3), tying the two routes to each other.

**Test list** (`tests/test_kmeans.py`): `TestAgainstNaiveReference` (centroids, scores,
full fit vs naive loop), `TestObjective` (monotone trace; bounded in [0, N]; step
optimality ×2), `TestRecovery` (k ∈ {2,3,5} × SNR; route agreement),
`TestInvariances` (rescale, delay, permutation, duplicate), `TestDeterminism`
(identical seed → identical result including trace; `n_init` returns the best-J run —
verified ≥ every single-init run; explicit-labels init honored on iteration 0),
`TestDegenerate` (`n_clusters=1`; `n_clusters == N`; forced empty cluster → reseed
fires and `n_reseeds` counts it; `on_empty="raise"` names cluster and iteration),
`TestFailFast` (`n_clusters > N` naming both; zero-power signal pointing at
`finite_power_mask`; non-finite coefs; wrong rank; bad `init` string; `max_iter < 1`),
`TestProvenance` (metadata records weighting, seed, `n_iter`, `converged`, stft
provenance; result frozen).

**Done**: all green; objective documented in the docstring; no jax/dask/sklearn
imports; committed alone. Commit message lists the deliberate behavioral deviations
from old `CMM` (init, 0-based labels, empty-cluster handling, no backprojection, no
import-time seeding) — nothing stored is invalidated because the old class has been
unimportable dead code since jax was removed.
**Risk**: medium–high (broken spec). Mitigated by the six-way triangulation above;
residual risk is toy-generator tuning (SNR thresholds), handled by seeding and
generous-but-meaningful ARI bounds.

### Chunk 4 — Gate 4 (the only remaining measurement) and defaults

Everything architectural is already measured (Gates 1–3). One small gate remains, and
it is **post-implementation, non-blocking** — init strategy and restart count are
parameters, not architecture:

- **Gate 4 — k-means init and restarts on real data.** Script in the session
  scratchpad, run on the cached `(10000, 22, 15)` STFT coefficients from Gate 2
  (`/nrs/...bin5p2um_stft.npy`, read-only): `init ∈ {"++", "random"}` ×
  `n_init ∈ {1, 5, 10, 20}` × 10 seeds, at k ∈ {7, 15}. Record final J spread,
  pairwise label ARI across seeds (`metrics.label_stability`), wall time; thread env
  set before numpy import and recorded. Report to `logs/gate4_kmeans_init.md`; freeze
  the shipped defaults (`init`, `n_init`) accordingly. Expected < 1 h of machine time.
- Informative addendum (no decision attached): both routes on the same real matrix,
  route-agreement ARI reported for context.

**Done**: report written, defaults confirmed or amended in a one-line follow-up commit
that states which numbers move (none, unless the default changes).
**Risk**: low.

### Chunk 5 — exports, plan bookkeeping, deletion

**Files**: `src/cmm/__init__.py`, `.claude/plans/cmm_refactor_plan.md`, deletions.

- `__init__.py` currently exports nothing from the new modules. Add the master-plan §3
  re-exports for `spectral`, `coherence`, `clustering`, `kmeans`, `metrics` via a
  module-level `__getattr__` (PEP 562) so `import cmm` stays side-effect-free and does
  not drag sklearn in — honoring "no side effects, no heavy imports". Add
  `tests/test_api.py` (import surface, laziness, `__all__` completeness).
- Update the master plan's phase table (rows 3/4 merged and done, metrics moved) and
  the CLAUDE.md dead-code paragraph once the files are gone.
- **Pure-deletion commit** (separate; nothing numeric moves): `cmm.py`,
  `cmm_funcs.py`, `power_iteration.py`, `ana_funcs.py`.

**Done**: suite green, `icampsnfr`'s imports (`cmm.spectral_funcs`) still work —
run its test suite to confirm nothing in this phase touched the spectral core.
**Risk**: low.

## 3. Ordering and rationale

1 (metrics) → 2 (clustering) → 3 (kmeans) → 4 (gate 4) → 5 (exports/deletion).
Metrics first because both routes' tests consume it; clustering before kmeans because
it is the low-risk port with its gate already cleared, and it provides the second,
independent clustering route that chunk 3's agreement tests triangulate against; the
gate runs only once the thing it measures exists. Five chunks ⇒ at least five commits,
each green, refactor-only and numerics-moving commits never mixed.

## Open questions — DECIDED by the user, 2026-08-06

1. **k-means `whiten` default**: `whiten=True` — the historical V1-flavored objective,
   the only variant with a track record. The asymmetry with `coherence_matrix`'s
   `whiten=False` is accepted; both defaults are one flag away from the other mode.
2. **Empty-cluster default**: `on_empty="reseed"` — deterministic reseed from the
   worst-scoring point, counted in `result.n_reseeds`; `"raise"` available.
3. **Label convention**: 0-based ints everywhere; icampsnfr's 1-based call sites are
   adapted in Phase 7.
4. **Scope**: confirmed — kmeans+clustering merged into this phase, `metrics.py`
   pulled forward.
