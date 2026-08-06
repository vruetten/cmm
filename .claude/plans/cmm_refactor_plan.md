# cmm refactor plan

Bring `cmm` up to a shippable standard: a lean, importable spectral core with a
proper test suite validated against scipy/sklearn, plus two clustering routes on
top of it (coherence k-means, coherence spectral clustering). Absorb the
rewritten spectral/coherence/phase/wavelet code from `icampsnfr`.

Status: plan only. Nothing has been changed in either repo.

---

## 1. Audit findings

Everything below was read or measured, not inferred. Line numbers are against
the current working tree.

### 1.1 Critical — wrong results or hard failure

**C1. The core model does not import.**
`cmm.cmm`, `cmm.cmm_funcs`, `cmm.power_iteration`, `cmm.ana_funcs` all raise
`ModuleNotFoundError: No module named 'jax'` in the `all` env. The `CMM` class
is currently dead code. Only `cmm.spectral_funcs`, `cmm.utils`, `cmm.toy_data`
import — which is exactly the subset `icampsnfr` uses.

**C2. PSD normalization disagrees with scipy.** `spectral_funcs.py:484-489`
divides by `kn * (win**2).sum()` for every scaling and then by `fs` for
`"density"`. Measured against `scipy.signal.welch` (hann, nperseg=64, fs=10):

| `scaling`   | median(cmm / scipy) | should be |
|-------------|---------------------|-----------|
| `"density"` | 0.5                 | 1.0       |
| `"spectrum"`| 21.33               | 1.0       |

Two separate errors. `"spectrum"` uses the wrong denominator — scipy uses
`win.sum()**2`, cmm uses `(win**2).sum()`. And the one-sided ×2 doubling of
non-DC/non-Nyquist bins is missing entirely, which is the residual 0.5. The
missing doubling is *not* a constant factor: DC and Nyquist end up 2× high
**relative to** interior bins, so it distorts spectrum shape, not just scale.

This propagates into `icampsnfr/frc.py`, `spectrum.py`, `traces.py`, all of
which call `estimate_spectrum`. Ratio quantities there (FRC, `prominence`)
divide it out; absolute power values and the DC/Nyquist bins do not.

Coherence itself is **exact** — max abs difference vs `scipy.signal.coherence`
is 8.9e-16 — because the normalization cancels in the ratio. The coherence path
is the one part of this package currently known-good.

**C3. Wrong frequency axis when coefficients are passed in.**
`spectral_funcs.py:432` computes `rfftfreq(f * 2 - 1, ...)` (with a `# to check`
comment) where it should be `rfftfreq(2 * (f - 1), ...)` for even `nperseg`, and
it ignores that `freq_minmax` has already trimmed the axis. Measured at
nperseg=64, fs=10: returns 0.1538 Hz spacing against a true 0.15625 Hz, top bin
4.923 Hz against a true 5.0 Hz. Every `CMM` call path sets `x_in_coefs=True`.

**C4. Project → backproject is not an inverse.** `utils.py:119-123`:
```python
scale   = 1.0 / win.sum() ** 2
DFT_ff  = build_DFT_matrix(...) * scale
wd_tf   = win[:, None] * DFT_ff
iwd_tf  = win[:, None] * 1 / DFT_ff / nperseg   # <- 1/scale, not scale
```
The elementwise reciprocal inverts the forward `scale` instead of applying it,
and the analysis window is applied on both the forward and inverse legs with no
COLA/WOLA normalization. Measured round-trip magnitude ratio: **0.044**. Every
`backproject_centroids` output is wrong by this factor.

**C5. Silhouette is sign-inverted.** `cmm.py:312-316` passes the coherence
matrix — a *similarity* — to `silhouette_score(..., metric="precomputed")`,
which requires a *distance*. `icampsnfr/cluster.py:142` does this correctly
(`1 - sqrt(C)`).

**C6. `ymt_coh` is always empty.** `cmm.py:302-303` appends to `ymt_coh_all`
twice; `ymt_coh` is never appended to, so `np.array([]).squeeze()` is what gets
returned and stored in `store_results()`.

**C7. Wrong cluster indexed in `compute_cluster_coherence`.** `cmm.py:451-453`
indexes `coherence_mnf[ind]` with the enumeration counter but compares against
label `l`. These diverge as soon as any cluster is empty — which C-S2 below
makes routine.

**C8. Dead branch raises.** `cmm.py:223`, the `average=False` path of
`allocate_data_to_clusters`, reads `self.coherence_mnk`, which is never assigned.

**C9. Mutated mutable default.** `freq_minmax=[0, np.inf]` is a shared default
across ~8 functions, and `CMM.__init__` (`cmm.py:43-45`, `66-69`) *mutates it in
place*. That mutates the caller's list, and poisons the shared default for the
rest of the process when the default was taken.

**C10. In-place mutation of a returned array.** `cmm.py:290` does
`freqs[notidx] = 0` on the array `estimate_spectrum` returned.

### 1.2 Serious — silent-continue, against fail-fast

**S1.** `spectral_funcs.py:117-119` — `result = result.real` on the one-sided
path silently discards the imaginary part of complex input rather than raising.
**S2.** `cmm.py:154-155` — an empty cluster prints `"no data point allocated"`
and keeps its stale centroid. The cluster can never recover and `k` silently
shrinks.
**S3.** `estimate_wiener_r` — `lam_f` is computed and discarded; `H` uses raw
`alpha_reg` as an *absolute* offset, contradicting both its own docstring
("relative diagonal loading") and the `# KEY FIX #1` comment above it.
`freq_taper_alpha` and the `tukey` import are unused. A debug `print` is live.
`n_time = 2 * (wn - 1)` uses the *in-band* bin count, so the impulse response
length is wrong whenever `freq_minmax` is restrictive.
**S4.** Dask graphs leak through the public API into `store_results()` and on
into `np.save`.
**S5.** `utils.load_data` calls `exit()` on an unknown extension.
**S6.** `optimize()` never closes its `Pool`; `ite` is unbound if `itemax == 0`.

### 1.3 Hygiene

**H1.** `np.random.seed(0)` and `reload(sf)` run at `cmm.cmm` import time.
**H2.** `pyproject.toml` declares `dependencies = []` while the code imports
numpy, scipy, sklearn, jax, dask, pandas, tifffile, zarr. A conflicting
`setup.py` sits beside it (MIT vs BSD-3, `packages=["src"]`, hardcoded `0.1`).
**H3.** `utils.py` imports pandas/tifffile/zarr at module scope, so importing
`make_chunks` drags in three heavy deps.
**H4.** `tests/unit_test.py` imports from `archive.*`, not from `src/`. Coverage
of the shipped package is zero. `demos/test_*.py` are named like tests but are
scripts.
**H5.** `filterwarnings = ["error"]` in pytest config vs a live `warnings.warn`
in `_triage_segments`.
**H6.** Dead parameters (`scaling`, `axis` in `compute_spectral_coefs`), `abs`
shadows the builtin, `nperseg=None` silently becomes 256.
**H7.** `intialize_clusters` typo; `compute_average_phase_shift` called twice in
`analyse_results`.
**H8.** `power_iteration` has no iteration cap.
**H9.** `cardio_data/`, `ependymal_data/`, `demos/`, `archive/` sit in the repo
root beside the package.

---

## 2. Benchmark findings — the dependency question

Measured before proposing any removal, per your instruction. **Caveat: run under
1-CPU affinity on an 80-core node.** Every number below needs re-running on a
proper multicore allocation before it decides anything.

### jax

Provides nothing measurable. It is not installed in `all`; the two `jnp.einsum`
calls (`cmm.py:116`, `cmm_funcs.py:22`) operate on numpy arrays and their
results are immediately consumed by numpy/scipy, so each is a host→device→host
round trip with no fused work. `power_iteration.py` is jax-only but is not on
any live call path. **Proposed: drop. Low risk.**

### dask — it earns its place on the N×N path

`estimate_spectrum(alltoall=True)` forms an `(N, N, F)` cross-spectrum that
cannot be materialized (11.5 GB at N=6000, F=20; 34.6 GB at K=30, F=60). dask
never builds it.

Same estimator, dask lazy einsum vs a naive blocked-numpy accumulation:

| N | K | F | dask | blocked numpy | ratio |
|---|---|---|------|---------------|-------|
| 1000 | 10 | 20 | 778 ms | 791 ms | 0.98× |
| 3000 | 10 | 20 | 2.59 s | 7.37 s | 0.35× |
| 6000 | 10 | 20 | 7.24 s | 29.9 s | 0.24× |
| 6000 | 30 | 60 | 22.9 s | 99.9 s | 0.23× |

**dask wins 4×.** Naive blocking is not a replacement.

But there is an algebraic shortcut. If coefficients are pre-normalized so
`sum_k |X̂(n,k,f)|² = 1` per `(n,f)` — which `CMM` **already computes** as
`coefs_xnkf_normalized` — then `MSC = |Σ_k X̂ conj(X̂)|²` with no denominator,
and the f-mean accumulates in place:

| N | K | F | dask | normalized-blocked c128 | c64 |
|---|---|---|------|------------------------|-----|
| 1000 | 10 | 20 | 758 ms | 188 ms (4.0×) | 80 ms (9.5×) |
| 3000 | 10 | 20 | 2.61 s | 2.19 s (1.2×) | 883 ms (3.0×) |
| 6000 | 10 | 20 | 7.40 s | 16.1 s (0.46×) | 4.13 s (1.8×) |
| 6000 | 30 | 60 | 22.9 s | 44.1 s (0.52×) | 14.1 s (1.6×) |

c128 is bit-identical to dask; c64 differs by 7e-8, six orders below anything
that matters for a quantity in [0, 1]. The c64 path wins everywhere here and
avoids up to 35 GB of intermediates — but only under 1-CPU affinity.

Two smaller paths, for completeness:

- **Cluster allocation** (`M`=8 clusters × N cells): numpy wins below N≈20k
  (3.7× at N=2000), dask wins at N=100k (1.3×). Sub-second either way.
- **Centroid gram** (`einsum("fkn,fln->fkl")`): at K=30, F=60, N=20000 —
  `np.einsum` 3.70 s, batched `matmul` 1.69 s, `dask.einsum` 1.35 s. The
  eigendecomposition itself is noise (5 ms). Batched `np.linalg.eigh` replaces
  the per-frequency `scipy.linalg.eigh` loop at 0.7 ms vs 2.1 ms.

### Proposed dependency policy

> **SUPERSEDED by gate 1 — see `logs/gate1_dependency_benchmark.md`.** The
> measurements above were invalid. The shell exports `OMP_NUM_THREADS=1`,
> `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1`; numpy/scipy BLAS obeys them
> while dask's scheduler takes its worker count from `cpu_count()` = 80 and
> ignores them. Every "dask wins" number above is **1-thread numpy against
> 80-thread dask**. Verified directly: `os.sched_getaffinity(0)` returns all 80
> CPUs with no cgroup quota. BLAS scales 89.8 → 1535 GFLOPS from 1 to 32 threads,
> degrading beyond 32.
>
> Corrected result: **drop dask.** Blocked numpy with row blocks distributed over
> a `ThreadPoolExecutor` (c64) is fastest in all six 32-thread cells tested —
> 4.2x at N=1000, 7.3x at N=6000, 4.6x at N=20000 (K=10, F=20), narrowing to
> 2.1x at N=20000, K=30, F=60. Agreement holds: c128 vs dask 8.9e-16, c64 vs
> dask 3.4e-7.
>
> The "dask never materializes (N,N,F)" argument also fails twice: blocked numpy
> does not materialize it either and measures *lower* peak RSS, and `chunks="auto"`
> — what the original benchmark used — emits a single (N,N,F) chunk (11.5 GB at
> N=6000), so it was never streaming. dask's only real contribution was task
> parallelism over blocks, which is ~20 lines of `concurrent.futures`.
>
> Caveats carried forward:
>
> - **Where this ran.** All of it — the original benchmarks and gate 1 — ran on
>   `ruttenv-ws1.hhmi.org`, the 80-core workstation. **Nothing ever ran on the
>   Janelia cluster**; LSF is not installed on this host, so earlier wording in
>   this plan about "a real multicore allocation" was wrong and no such run
>   happened. Load average during gate 1 was 30.6/24.9/20.8 with 5 users and two
>   of my own agents active, so the timings carry contention noise. The 2.1–7.3x
>   direction is robust to that; the exact digits are not.
> - **Where the thread cap comes from is unknown.** `OMP_NUM_THREADS=1` is not
>   set in `/etc/profile.d`, `~/.bashrc`, `~/.bash_profile`, or the conda env's
>   `activate.d`. It is most likely injected by the agent harness, in which case
>   an interactive shell may have all 80 threads and was never capped. Check
>   `echo $OMP_NUM_THREADS` in a normal terminal before generalizing. This does
>   not change the conclusion — the finding that blocked numpy needs *block-level*
>   thread-pool parallelism (BLAS threading alone makes it slower at N=1000,
>   since the elementwise |S|² accumulation is single-threaded) holds at any BLAS
>   thread count.
> - The margin narrows to 2.1x at the largest cell and dask scales better with
>   K·F than numpy does. Re-measure before assuming this holds at much larger K·F.
> - The full grid was not completed; see §7 of the gate report for exact gaps.
>
> **To settle this properly:** re-run the two decisive cells (N=20000 at K=30/F=60
> and K=10/F=20) on an idle machine or a real `bsub` allocation. That requires a
> session with LSF access, which this one did not have.

- **Core: numpy, scipy, scikit-learn.** Hard deps, all three genuinely used.
- **dask: drop.** Replaced by blocked numpy over a thread pool. No `backend=`
  argument needed, which also removes S4 (dask graphs leaking into return
  values) by construction.
- **jax: drop.** Confirmed absent from the env. Note the audit undercounted —
  there are **four** `jnp.einsum` sites (`cmm.py:116,291,297`, `cmm_funcs.py:22`),
  plus `power_iteration.py` and a dead `foldxy` in `utils.py:148` with no callers.
  All operate on numpy arrays consumed by numpy/scipy: pure host round-trips.
- **pandas / tifffile / zarr: drop** with `utils.load_data`. I/O belongs in
  caller scripts — this is exactly the split `icampsnfr` already uses.
- **Thread hygiene is now a package concern.** Since the environment caps BLAS at
  one thread, `cmm` must document this and the test/benchmark suite must record
  the thread count with every timing. A timing without it is uninterpretable.

---

## 3. Target architecture

```
src/cmm/
  __init__.py       public API re-export; no side effects, no heavy imports
  spectral.py       windows, segmentation, STFT coefficients, PSD/CSD,
                    pairwise coherence.  <- the module other packages import
  coherence.py      N x N coherence matrix, band-averaging, debias, distances
  kmeans.py         CoherenceKMeans — the coherence mixture model
  clustering.py     spectral clustering: affinity, Laplacian, embedding,
                    eigengap, k-means on the embedding
  phase.py          analytic signal, phase/amplitude, phase-triggered average
  wavelet.py        complex Morlet, single-frequency instantaneous phase
  metrics.py        coherence silhouette, cluster summary
  synthetic.py      toy-data generators shared by tests, demos and README
examples/           the current cardio_data/ ependymal_data/ demos/ scripts
tests/              see section 4
```

Dependency direction is strictly one-way:
`spectral → coherence → {kmeans, clustering} → metrics`, with `phase`, `wavelet`
and `synthetic` depending only on `spectral`. No module imports a sibling
further right. `synthetic` is importable without sklearn.

### Design rules

**Visible parameters.** A frozen `SpectralParams(fs, nperseg, noverlap, window,
detrend, band_hz, nfft, onesided)` dataclass. Every result object carries the
params that produced it and exposes `.metadata()` — the pattern
`icampsnfr.spectrum.SpectrumResult` already uses, so a stored output always
records how it was made. No hidden `self.detrend = False` as in `cmm.py:59`.

**No mutable defaults.** `band_hz: tuple[float, float] | None = None`, never a
list, never mutated (fixes C9, C10).

**Fail fast.** Every degenerate case raises with a message naming the offending
value and the fix: complex input on a one-sided transform (S1), an empty cluster
(S2), `nperseg > T`, a band outside `(0, fs/2)`, isolated graph nodes, non-finite
traces. No `print`-and-continue, no `exit()`.

**No leakage.** Public functions return numpy arrays, always. `backend="dask"`
is an internal execution choice, invisible in the return type.

**Functions first, class second.** `CoherenceKMeans` is a thin `fit`/`predict`
wrapper over free functions that are independently callable and testable — the
class holds state and iteration, not maths.

**Two clustering routes, one matrix.** Both `kmeans.py` and `clustering.py`
consume the same `coherence_matrix` output, so they are directly comparable and
share all upstream tests. (Hierarchical/linkage clustering from
`icampsnfr/cluster.py` is **out of scope** per your selection — but its correct
`1 - sqrt(C)` distance conversion is being taken into `metrics.py` regardless,
since that is what fixes C5.)

### What is absorbed from icampsnfr

| From | Into | Notes |
|------|------|-------|
| `spectral.py` — all 6 functions | `cmm/clustering.py` | verbatim modulo naming |
| `coherence.py` — `bandpass_stft`, `msc_matrix`, `debias_msc` | `cmm/spectral.py`, `cmm/coherence.py` | the blocked matmul becomes the numpy backend |
| `phase.py` — all | `cmm/phase.py` | `sort_by_rastermap` stays behind (rastermap dep) |
| `wavelet.py` — all | `cmm/wavelet.py` | |
| `cluster.py::_distance`, `_subsample_silhouette` | `cmm/metrics.py` | fixes C5 |

Voxel masking, anatomy, binning, I/O, detrending and plotting stay in
`icampsnfr` — they are imaging-specific and out of scope for a spectral package.

### Public API

```python
from cmm import (
    SpectralParams, stft_coefficients, psd, csd, coherence,     # spectral
    coherence_matrix, debias_coherence, coherence_distance,     # coherence
    CoherenceKMeans, coherence_kmeans,                          # kmeans
    affinity_gaussian, normalized_laplacian, spectral_embedding,
    eigengap, cluster_embedding, spectral_labels, median_tau,   # clustering
    analytic_signal, phase_amplitude, phase_triggered_average,  # phase
    morlet_wavelet, wavelet_analytic,                           # wavelet
    coherence_silhouette, cluster_summary,                      # metrics
)
```

---

## 4. Test suite

Four layers. The point of layer 1 is that we never again have to trust our own
arithmetic about window normalization — scipy decides.

```
tests/
  conftest.py                   fixtures: seeded rng, synthetic datasets, param grids
  test_spectral_vs_scipy.py     L1  external cross-validation
  test_spectral_properties.py   L2  analytic identities and invariants
  test_coherence_matrix.py      L2  estimator correctness, debias, backend agreement
  test_kmeans.py                L3  recovery, determinism, degenerate clusters
  test_clustering.py            L3  vs sklearn.cluster.SpectralClustering
  test_phase.py                 L2/L3
  test_wavelet.py               L2/L3
  test_metrics.py               silhouette direction and scale
  test_failfast.py              L4  every documented error path raises
  test_api.py                   import surface, no side effects, params round-trip
```

### L1 — cross-validation against validated packages

Parameterized over `window ∈ {hann, hamming, boxcar, tukey}`, `nperseg ∈ {32,
64, 128}`, `noverlap ∈ {0, nperseg//2, 3*nperseg//4}`, `detrend ∈ {constant,
linear, False}`, `scaling ∈ {density, spectrum}`, one- and two-sided, `nfft >
nperseg`, real and complex input:

| cmm | reference | tolerance |
|-----|-----------|-----------|
| `psd` | `scipy.signal.welch` | `rtol=1e-12` |
| `csd` | `scipy.signal.csd` | `rtol=1e-12` |
| `coherence` | `scipy.signal.coherence` | `rtol=1e-12` (currently 9e-16 — pin it) |
| `stft_coefficients` | `scipy.signal.stft` | `rtol=1e-12` after scaling alignment |
| `analytic_signal` | `scipy.signal.hilbert` on a `sosfiltfilt` reference | `rtol=1e-10` |
| `morlet_wavelet` | `scipy.signal.morlet2` | shape + centre frequency |
| `spectral_labels` | `sklearn.cluster.SpectralClustering(affinity="precomputed")` | `ARI > 0.95` |
| `cluster_embedding` | `sklearn.manifold.SpectralEmbedding` | subspace angle |

This layer is what would have caught C2 and C3 on day one.

Available in the `all` env: scipy 1.16.2, sklearn 1.7.2. **Not** available:
`mne`, `nitime`, `spectrum`, `statsmodels`, `pywt`. See open question Q2 —
adding `mne` as a test-only extra would give a genuinely independent multitaper
coherence reference rather than a second opinion from the same scipy codebase.

### L2 — analytic ground truth and invariants

- **Parseval**: `sum(psd) * df == var(x)` for a boxcar window, no overlap.
- **MSC = 1** for `y(t) = a·x(t - τ)` at every in-band frequency, any `a`, `τ`.
- **MSC invariance**: unchanged under per-signal affine rescaling and time shift.
- **MSC ≈ 0** for independent signals, Monte-Carlo over seeds against the
  known `1/n_eff` bias floor; `debias_coherence` provably reduces it.
- **Matrix identities**: `diag(C) == 1`, `C == C.T` exactly, `0 ≤ C ≤ 1`,
  permutation equivariance (`C[p][:,p] == coherence_matrix(x[p])`).
- **Backend agreement**: numpy vs dask to `rtol=1e-10`; c64 vs c128 to `1e-6`.
- **Round-trip**: `backproject(project(x)) ≈ x` to `rtol=1e-10` — the test that
  pins C4, and the reason C4 must be fixed rather than documented.
- **Frequency axis**: `freqs` from a coefficient-input call equals `freqs` from
  the raw-signal call, exactly. Pins C3.
- **Phase-triggered average**: on a pure sine seed, recovers the known
  phase-amplitude profile; bin counts sum to T.

### L3 — algorithmic recovery

Shared synthetic generator (`cmm.synthetic`): `k` regions, one independent
band-limited source each, plus per-signal noise — the design from
`icampsnfr/tests/test_coherence.py`, which correctly avoids phase-shifted
sinusoids (MSC ≈ 1 between them, so they cannot discriminate).

- Both routes recover the true partition at the true `k`: cluster purity > 0.95,
  `ARI > 0.9` vs ground truth, swept over SNR and `k ∈ {2, 3, 5}`.
- `eigengap` recovers `k` on a block-structured matrix with known `k`.
- k-means and spectral clustering agree with each other (`ARI > 0.9`).
- Determinism: identical labels for a fixed `random_state`, across repeated runs
  and across backends.
- Convergence: labels stabilize; the reported iteration count is the one at
  which they did.
- **Degenerate cases** (currently silent, per S2): an empty cluster, a
  constant/zero trace, `k > n`, `k == 1`, duplicate rows.

### L4 — fail-fast

One test per documented raise. Every `raise` in the package must have a test
asserting both the exception type and a substring of its message, matching your
preference for testing the message a user actually reads.

### Test hygiene

- No test imports `archive/` (H4). `archive/` is deleted.
- `filterwarnings = ["error"]` stays; no library code warns on a valid path (H5).
- Runtime target < 60 s for the default suite; anything slower is marked
  `@pytest.mark.slow` and deselected by default.
- CI on GitHub Actions across 3.10–3.13, plus a ruff and a mypy job. Coverage
  reported, with `src/cmm` gated at a floor (target ≥ 90 %).

---

## 5. Execution phases

Each phase ends green and committed; nothing is deleted before its replacement
passes.

| # | Phase | Deliverable | Gate |
|---|-------|-------------|------|
| 0 | Benchmark on a real allocation | numbers reported, dep policy confirmed | **your sign-off before any removal** |
| 1 | **DONE** — `spectral.py` + L1/L2 tests | scipy-exact PSD/CSD/coherence/STFT | 86 tests green; C2, C3, S1, C9, H4, H5, H6 fixed; every diff vs baseline attributed |
| 2 | `coherence.py` + backends | `coherence_matrix`, debias, distances | backends agree to 1e-10 |
| 3 | `kmeans.py` | `CoherenceKMeans`, jax-free | C1, C5–C8, S2 fixed; recovery tests pass |
| 4 | `clustering.py` | absorbed from `icampsnfr/spectral.py`, Lanczos solver | vs sklearn `SpectralClustering`; Lanczos speedup + agreement confirmed on a **real** coherence graph before dense `eigh` is dropped |
| 5 | `phase.py`, `wavelet.py`, `wiener_filter` | absorbed + Q1 repair | tests ported and passing; `wiener_filter` gets its first-ever tests |
| 6 | `metrics.py`, packaging, README, CI | one `pyproject.toml`, `setup.py` deleted | H2 fixed, CI green |
| 7 | Update `icampsnfr` | 4 import sites patched | `icampsnfr` suite green |
| 8 | Retire scripts | `examples/`, `archive/` deleted | — |

Phase 4 is the point at which the projection-matrix machinery in `utils.py` gets
resolved: `build_fft_trial_projection_matrices` is used only by `CMM` and
`toy_data`, and once C4 is fixed the round-trip test decides whether it stays at
all or whether the STFT path subsumes it.

### README

Rewritten around the two routes, both runnable end to end from
`cmm.synthetic`: quickstart, the two clustering algorithms side by side with
guidance on when each applies, a parameter reference (`fs`, `nperseg`,
`noverlap`, `band_hz`, `k`, `tau`, `laplacian`), the MSC definition and its
`1/n_eff` bias, and a validation section stating what is checked against scipy
and sklearn and to what tolerance. The current badge block is stale (no CI, no
PyPI release) — badges go or become real.

---

## 6. Decisions, and the gates that remain

Q1–Q5 are all resolved. What is left are three *measurement* gates that must
clear before code is removed: the dependency benchmark on a real allocation
(§2), the Lanczos confirmation on a real coherence graph (Q4), and the k-NN
sparsification test (Q4).

**Q1 — RESOLVED: repair into a documented `wiener_filter`.** Phase 5 work item.
The directed $x \to y$ measure it provides cannot be expressed by V2, which is
symmetric. Repair means: use `lam_f` as the relative diagonal loading it claims
to be (currently computed and discarded, with `alpha_reg` applied as an absolute
offset instead); take the irfft length from the *full-band* bin count rather
than the in-band count; drop `freq_taper_alpha`, the `tukey` import and the live
`print`; and stream rather than `H.compute()` the whole $(N,M,F)$ array. Returns
`H`, `h`, and the variance-explained scalar `r`. Needs its own tests — it has
never had any.

**Q2 — RESOLVED: all four reference layers.** scipy for windowing and
normalization (already exact to 9e-16); a deliberately naive pair-and-frequency
loop written into the test suite for readability, which the vectorized, blocked
and multi-backend paths must match — this is what catches chunking and einsum
bugs, the failure mode most likely in this package; analytic ground truth
(MSC=1 for a scaled/delayed copy, the $1/K$ null floor by Monte Carlo, Parseval,
permutation equivariance); and `mne` as a **test-only** extra for an independent
multitaper implementation. The mne check is necessarily loose — multitaper will
not agree with Welch to tight tolerance — so it is a sanity bound, not a pin.

**Q3 — RESOLVED: V2 everywhere.** See `derivations/coherence_estimators.md`
for the full four-way comparison and the measurements behind this.

`coherence_matrix` computes the power-weighted, incoherently-averaged estimator

$$C_{nm} = \frac{\sum_f |P_{nm}(f)|^2}{\sum_f P_{nn}(f) P_{mm}(f)}$$

as its default and only recommended form. `icampsnfr.msc_matrix` (which
averages *coherently* over frequency) is retired: it assumes a constant phase
relationship across the band, which any propagating signal violates — it scores
a 10 s lag at 0.269 where the signals are in fact perfectly coherent.

`whiten=True` remains available as a one-line pre-normalization; it reproduces
cmm's historical per-frequency-mean estimator exactly (verified `rtol=1e-12`),
which is needed to reproduce any pre-refactor cmm result.

Three consequences that become work items:

- **Phase 2** must set `debias_msc`'s `n_eff = n_seg`, **not** `n_seg *
  n_freqs`. Under incoherent averaging the bias floor is $1/K$ *independent of
  bandwidth* (measured: 0.125 at K=8 for every F from 2 to 40). The existing
  call under-corrects by a factor of F.
- **Phase 7** must regenerate every `icampsnfr` coherence artifact — cached
  matrices, dashboard configs, cluster labels, figures. They are all on V3.
- **README/docs** must state that segment count is the only bias lever under
  this estimator, since widening the band no longer lowers the floor. This
  should drive `nperseg`/`noverlap` guidance.

Cost is not a concern: V2 needs F GEMMs to V3's one, but in the blocked
formulation at N=6000, K=10, F=20 that is 4.13 s vs 1.80 s — still 1.8×
*faster* than the current dask path (7.40 s), with no (N,N,F) intermediate.

**Q4 — RESOLVED: delete `power_iteration.py`; switch the *Laplacian* solver to
Lanczos.** Two separate eigenproblems; the scaling concern applies to the second.

*Centroid (K×K).* Matrix-free power iteration never wins, measured at the sizes
that occur — n=20000 rows at K ∈ {10,30,60,120,240}: 0.28×, 0.57×, 1.04×, 0.67×,
0.95×; n=100000 rows at K ∈ {30,120}: 0.25×, 0.56×. It converges in 4–5
iterations even at K=240, so the loss is not convergence — the gram is a single
batched GEMM at full BLAS throughput while power iteration is 4–7 *sequential*
low-arithmetic-intensity passes. Delete. Batched `np.linalg.eigh` replaces the
per-frequency `scipy.linalg.eigh` loop (0.7 ms vs 2.1 ms).

*Laplacian (N×N).* This is the 10k-row problem, and dense `eigh` scales as
$N^3$ where Lanczos scales as $N^2$:

| N | dense `eigh` | `eigsh(S, which="LA")` | speedup | eigenvalue agreement |
|---|---|---|---|---|
| 2000 | 0.59 s | 0.27 s | 2.2× | 1.2e-15 |
| 4000 | 6.95 s | 3.46 s | 2.0× | 1.6e-15 |
| 8000 | 58.4 s | 13.2 s | 4.4× | 2.1e-15 |
| 10000 | 112.9 s | 20.8 s | **5.4×** | 3.1e-15 |

The reformulation is the point: do **not** ask for the smallest eigenvalues of
$L$. Ask for the largest of $S = D^{-1/2} W D^{-1/2}$, with
$\lambda_L = 1 - \lambda_S$. Lanczos converges on largest eigenvalues with no
shift-invert and no preconditioner. The docstring of `icampsnfr/spectral.py`
claims the opposite ("for N up to ~20k this beats sparse iterative solvers");
that claim is measured false and must be corrected when the module is absorbed.
LOBPCG was also tested (24.4 s at N=10000) — competitive but it hit its
iteration cap, so `eigsh` is the default.

**Gate:** these are synthetic well-separated blocks. Lanczos convergence depends
on the spectral gap, so Phase 4 must confirm the speedup and the agreement on a
real coherence graph before dense `eigh` is removed. Keep dense as an explicit
`solver="dense"` option for small N and for exactness audits.

**Also worth testing in Phase 4:** the affinity is dense, so each Lanczos matvec
is $O(N^2)$. Sparsifying to a k-NN graph would make matvecs $O(N k_{nn})$ —
standard practice for spectral clustering at scale, and a much larger win than
5.4× if the cluster structure survives it. Not assumed; to be measured.

**Q5 — RESOLVED: complex64 matmul and float32 matrix by default, `dtype=` to
opt out.** 1.6–9.5× faster, halves the N×N output (3.2 GB → 1.6 GB at N=20000),
and the 7e-8 error sits six orders below the $1/K \approx 0.1$ bias floor that
dominates the estimate anyway. float64 stays available for numerical audits, and
the test suite pins c64-vs-c128 agreement at 1e-6.
