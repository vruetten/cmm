# cmm — coherence-based clustering of timeseries

[![CI](https://github.com/vrutten/cmm/actions/workflows/ci.yml/badge.svg)](https://github.com/vrutten/cmm/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](LICENSE)

Cluster `(N, T)` timeseries by *coherence* — correlation per frequency band,
invariant to per-signal gain and to time delays — instead of by amplitude
similarity. Two clustering routes share one spectral core:

- **Coherence k-means** (`cmm.kmeans`): centroids are per-frequency unit
  vectors in segment space; a signal's score against a centroid *is* its mean
  magnitude-squared coherence with that centroid. Exact block-coordinate
  ascent, so the objective is provably monotone.
- **Coherence spectral clustering** (`cmm.clustering`): Gaussian affinity on
  the coherence matrix, normalized graph Laplacian, Lanczos eigenvectors,
  k-means in the embedding.

Alongside: PSD/CSD/coherence estimation pinned to scipy (`cmm.spectral`), the
band-averaged N×N coherence matrix (`cmm.coherence`), cluster quality metrics
(`cmm.metrics`), narrowband phase tools (`cmm.phase`, `cmm.wavelet`), a
directed Wiener filter (`cmm.wiener`), and toy-data generators
(`cmm.synthetic`).

Requires Python ≥ 3.10; depends on numpy, scipy, scikit-learn only.

```
pip install git+https://github.com/vrutten/cmm.git
```

## Quickstart

Both routes, end to end, on synthetic data with known groups:

```python
from cmm import (
    SpectralParams, stft, coherence_matrix,
    coherence_kmeans, spectral_labels, grouped_traces,
)

# 24 signals in 3 groups; each group shares a band-limited source.
x, truth = grouped_traces([8, 8, 8], t=3072, fs=8.0, snr=2.0, seed=0)

params = SpectralParams(fs=8.0, nperseg=256, band_hz=(0.2, 1.5))
s = stft(x, params)          # (N, K, F) complex coefficients + provenance

# Route 1: coherence k-means
km = coherence_kmeans(s, 3, random_state=0)
km.labels                    # (N,) int32, 0-based
km.scores                    # (k, N) mean coherence of each signal w/ each centroid
km.objective                 # monotone trace, bounded by N

# Route 2: coherence spectral clustering
c = coherence_matrix(s)      # N x N, band-averaged
labels = spectral_labels(c, 3, random_state=0)
```

On this example both routes recover the planted partition exactly
(ARI = 1.0 against `truth`). Choosing `k` and checking stability:

```python
from cmm import spectral_embedding, eigengap, label_stability

emb = spectral_embedding(c, n_eig=8, random_state=0)
gaps, k_hat = eigengap(emb.eigenvalues)          # k_hat == 3 here

runs = [coherence_kmeans(s, 3, random_state=i).labels for i in range(5)]
label_stability(runs)        # {'mean_ari': 1.0, 'min_ari': 1.0, 'n_runs': 5}
```

## Which route when

- **k-means** works in coefficient space — it never forms the N×N matrix, so
  it scales past the point where the matrix fits in memory, and its centroids
  are interpretable objects (a per-frequency phase/weight pattern across
  segments, with `centroid_strength` saying how one-dimensional each cluster
  is per frequency). Its objective favors compact, centroid-shaped clusters.
- **Spectral clustering** sees the full pairwise structure, so it can follow
  clusters that are connected rather than compact, and the eigenvalue
  spectrum gives a principled `k` (eigengap). It costs the N×N matrix plus an
  eigensolve (Lanczos on `D^-1/2 W D^-1/2`; measured 24x faster than dense at
  N=10000 with identical labels — dense stays available as `solver="dense"`).

They agree on well-separated structure (tested: ARI > 0.9 between routes on
planted groups); where they disagree, the disagreement is informative.

## The estimator, and its bias

Coherence between two signals at frequency `f`, from `K` Welch segments:

```math
\gamma^2(f) = \frac{|S_{xy}(f)|^2}{S_{xx}(f)\,S_{yy}(f)} \in [0, 1]
```

The band-averaged matrix uses the power-weighted, incoherently-averaged form

```math
C_{nm} = \frac{\sum_f |P_{nm}(f)|^2}{\sum_f P_{nn}(f)\,P_{mm}(f)}
```

Taking the modulus per frequency *before* averaging is what makes it
delay-invariant: a propagating signal puts a `2πfτ` phase ramp on the
cross-spectrum, and averaging coherently across frequency cancels exactly the
signals one is usually looking for. `coherence_matrix(s, whiten=True)`
pre-normalizes each `(signal, frequency)` cell to unit power, which weights
every frequency bin equally and reproduces the historical cmm estimator
exactly. The k-means route defaults to `whiten=True` for the same
backward-compatibility reason; see `derivations/coherence_estimators.md` for
the full comparison.

**The bias floor is `1/K`, independent of bandwidth.** Two independent
signals do not score 0 — they score about `1/n_segments`. Adding frequency
bins reduces the variance of the estimate but not this bias, so:

- more/shorter segments (smaller `nperseg`, more overlap) lower the floor at
  the cost of frequency resolution — this is the only lever;
- `debias(c)` subtracts the floor analytically (`n_eff = K`, never `K·F`);
- a coherence map that looks uniformly elevated is telling you about `K`,
  not about your data.

## Parameter reference

| Parameter | Where | Meaning |
|---|---|---|
| `fs` | `SpectralParams` | sampling rate, Hz |
| `nperseg` | `SpectralParams` | segment length; sets frequency resolution `fs/nperseg` and, with `noverlap`, the segment count `K` — the coherence bias floor is `1/K` |
| `noverlap` | `SpectralParams` | overlap between segments; default `nperseg // 2` |
| `band_hz` | `SpectralParams` | `(low, high)` band retained after the transform; the frequency axis travels with the coefficients |
| `window`, `detrend`, `nfft`, `onesided` | `SpectralParams` | scipy semantics, validated against scipy |
| `whiten` | `coherence_matrix`, `coherence_kmeans` | equal-weight frequencies (historical V1 estimator) vs power-weighted (V2) |
| `n_clusters` | both routes | `k`; `eigengap` suggests one from the Laplacian spectrum |
| `tau` | `spectral_embedding` | Gaussian affinity scale `exp(-(1-C)/tau)`; default `median_tau` |
| `laplacian` | `spectral_embedding` | `"sym"` (default) or `"rw"` (served from the same symmetric solve) |
| `on_empty` | `coherence_kmeans` | `"reseed"` (default, deterministic, counted in `n_reseeds`) or `"raise"` |

Every result object is a frozen dataclass carrying the parameters that
produced it (`.metadata()`), with read-only arrays.

## What is validated, and how hard

The test suite (293 tests) pins this package to external references rather
than to its own arithmetic:

- `psd` / `csd` / per-frequency `coherence` match `scipy.signal.welch` /
  `csd` / `coherence` to **rtol 1e-12** across windows, overlaps, detrending,
  scalings, one/two-sided, odd/even `nfft`, and band selection.
- `coherence_matrix` matches a deliberately naive pair-by-pair loop; the
  `1/K` floor is verified by Monte Carlo and `debias` against it.
- k-means correctness is triangulated: centroids against a per-frequency
  `scipy.linalg.eigh` loop, scores against the scipy-pinned coherence path,
  objective monotonicity (a theorem for this update), 100-perturbation local
  optimality, planted-structure recovery at `k ∈ {2, 3, 5}`, and delay /
  rescale / permutation invariances.
- spectral clustering against `sklearn.cluster.SpectralClustering`
  (ARI > 0.95) and dense-vs-Lanczos solver agreement (identical labels,
  eigenvalues to 1e-15).
- the Wiener filter recovers a known FIR filter end to end.
- every documented `raise` has a test asserting its message.

## Development

```
pip install -e .[test]
pytest            # warnings are errors; coverage gate at 90%
ruff check src tests
```

Layout: `src/cmm/` (library), `tests/`, `examples/` (analysis scripts, not
part of the package), `derivations/` (estimator notes), `logs/` (measurement
gates and baselines). `spectral_funcs.py`, `utils.py` and `toy_data.py` are
frozen legacy modules kept only until the downstream `icampsnfr` migration;
new code must not import them.
