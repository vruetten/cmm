"""Band-averaged coherence matrix over N signals.

Computes the **V2** estimator — power-weighted, incoherently averaged over
frequency:

.. math::
    C_{nm} = \\frac{\\sum_f |P_{nm}(f)|^2}{\\sum_f P_{nn}(f) P_{mm}(f)},
    \\qquad P_{nm}(f) = \\sum_k X_n(k,f) \\overline{X_m(k,f)}

Taking the modulus per-frequency *before* summing makes this delay-invariant: a
lag puts a ``2*pi*f*tau`` ramp on the cross-spectrum, and an estimator that
sums coherently over frequency cancels it, scoring a 10 s lag at 0.269 on
perfectly coherent signals. Clustering a travelling wave with such an estimator
splits it by lag. See ``derivations/coherence_estimators.md``.

``whiten=True`` normalizes each signal to unit power per frequency first, which
turns the power weighting into equal weighting and reproduces the historical
per-frequency-mean estimator (V1) *exactly*. Which of the two is right for a
given dataset is not settled — both of the arguments originally given for
preferring V2 were measured inert on real data, yet the two disagree at cluster
ARI 0.33-0.64. Report :func:`effective_bins` and look at it.

Two numbers must travel with any matrix from here:

* ``n_eff = K``, the segment count. Under incoherent averaging the null bias
  floor is ``1/K`` **independent of bandwidth** — adding frequency bins reduces
  variance but not bias. :func:`debias` takes ``K``, not ``K*F``.
* ``f_eff``, the participation ratio of the frequency weights. When it falls
  well below ``F`` the nominal band is not the band being measured; see
  ``derivations/spectral_tilt_and_band_collapse.md``.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from cmm.spectral import Stft

__all__ = [
    "CoherenceMatrix",
    "coherence_matrix",
    "debias",
    "to_distance",
    "effective_bins",
    "finite_power_mask",
]


@dataclass(frozen=True)
class CoherenceMatrix:
    """An ``(N, N)`` band-averaged coherence matrix and how it was made.

    Attributes:
        values: ``(N, N)`` symmetric, unit diagonal, in ``[0, 1]``.
        n_eff: segment count K — the effective sample size for the bias floor.
        f_eff: ``(N, N)`` participation ratio of the frequency weights per pair,
            or a scalar ``float(F)`` when ``whiten=True`` (equal weights).
        whiten: whether equal (True, V1) or power (False, V2) weighting was used.
        stft_metadata: provenance of the transform this came from.
    """

    values: np.ndarray
    n_eff: int
    f_eff: np.ndarray | float
    whiten: bool
    stft_metadata: dict

    @property
    def n_signals(self) -> int:
        return self.values.shape[0]

    @property
    def bias_floor(self) -> float:
        """Expected coherence between independent signals: ``1 / K``.

        Independent of bandwidth. If this is not small compared with the
        contrast in ``values``, the matrix is mostly floor.
        """
        return 1.0 / self.n_eff

    def metadata(self) -> dict:
        f_eff = self.f_eff
        summary = (
            float(f_eff)
            if np.isscalar(f_eff)
            else {
                "median": float(np.median(f_eff)),
                "min": float(np.min(f_eff)),
            }
        )
        return {
            "estimator": "V1 (equal weight)" if self.whiten else "V2 (power weight)",
            "n_signals": self.n_signals,
            "n_eff": self.n_eff,
            "bias_floor": self.bias_floor,
            "f_eff": summary,
            "stft": self.stft_metadata,
        }


def finite_power_mask(s: Stft, *, rtol: float = 0.0) -> np.ndarray:
    """Boolean ``(N,)`` marking signals with non-zero in-band power.

    A signal with zero power has undefined coherence with everything, so
    :func:`coherence_matrix` refuses to guess. Filter with this first::

        keep = finite_power_mask(s)
        C = coherence_matrix(replace(s, coefs=s.coefs[keep]))

    ``rtol`` optionally also excludes signals whose total in-band power is below
    ``rtol`` times the median across signals — useful for imaging data where a
    masked-out voxel is near-zero rather than exactly zero.
    """
    power = np.einsum("nkf,nkf->n", s.coefs, s.coefs.conj()).real
    ok = power > 0
    if rtol > 0 and ok.any():
        ok &= power > rtol * np.median(power[ok])
    return ok


def effective_bins(power_nf: np.ndarray) -> np.ndarray:
    """Participation ratio of the V2 frequency weights, per pair.

    ``F_eff = (sum_f w_f)**2 / sum_f w_f**2`` with ``w_f = P_nn(f) P_mm(f)``.
    Equals ``F`` for a flat spectrum and falls toward 1 as one bin dominates.
    Compare against ``power_nf.shape[1]``: below ~0.8 of it the nominal band is
    not what is being measured.
    """
    p = np.asarray(power_nf, dtype=np.float64)
    s1 = p @ p.T                    # sum_f w_f
    s2 = (p * p) @ (p * p).T        # sum_f w_f**2
    out = np.zeros_like(s1)
    np.divide(s1 * s1, s2, out=out, where=s2 > 0)
    return out


def coherence_matrix(
    s: Stft,
    *,
    whiten: bool = False,
    dtype: np.dtype | type = np.complex64,
    block: int = 512,
    workers: int | None = None,
    compute_f_eff: bool = True,
) -> CoherenceMatrix:
    """Band-averaged coherence between every pair of signals in ``s``.

    ``whiten=False`` (default) gives V2, power-weighted. ``whiten=True`` gives
    V1, equal-weighted, exactly.

    ``dtype`` sets the matmul precision. The default ``complex64`` is 1.6-9.5x
    faster and halves the output, at ~7e-8 error — six orders below the ``1/K``
    bias floor that dominates the estimate. Pass ``complex128`` for a numerical
    audit.

    Work is blocked over rows and distributed with a thread pool. Note that
    BLAS threading alone is *not* enough: the elementwise ``|S|**2``
    accumulation is single-threaded and dominates, so a pure-BLAS
    parallelization measured *slower* than serial at N=1000. Note also that
    this environment may export ``OMP_NUM_THREADS=1``, which caps BLAS but not
    this pool.

    The ``(N, N, F)`` intermediate is never materialized (11.5 GB at N=6000,
    F=20); only ``(block, N)`` exists at a time.
    """
    coefs = np.asarray(s.coefs)
    if coefs.ndim != 3:
        raise ValueError(f"expected (N, K, F) coefficients, got shape {coefs.shape}")
    n, _, n_freqs = coefs.shape
    if n_freqs < 1:
        raise ValueError("no frequency bins; check band_hz")

    power_nf = np.einsum("nkf,nkf->nf", coefs, coefs.conj()).real
    dead = ~(power_nf > 0).all(axis=1)
    if dead.any():
        raise ValueError(
            f"{int(dead.sum())} of {n} signals have zero power in at least one "
            f"in-band frequency (first: {int(np.flatnonzero(dead)[0])}); "
            "coherence is undefined there. Filter with finite_power_mask() "
            "before calling."
        )

    if whiten:
        work = (coefs / np.sqrt(power_nf)[:, None, :]).astype(dtype, copy=False)
        weights_nf = np.ones_like(power_nf)
    else:
        work = coefs.astype(dtype, copy=False)
        weights_nf = power_nf

    # (F, N, K), contiguous, so each frequency's gemm reads a packed block.
    x_fnk = np.ascontiguousarray(np.moveaxis(work, 2, 0))
    x_fkn = np.ascontiguousarray(x_fnk.conj().transpose(0, 2, 1))

    den = weights_nf @ weights_nf.T          # sum_f P_nn(f) P_mm(f), one gemm
    # Output precision follows the compute precision: complex64 -> float32,
    # complex128 -> float64. Storing a float64 computation as float32 would
    # cap agreement with a reference implementation at ~1e-7 and make the
    # "pass complex128 for an audit" option meaningless.
    real_dtype = np.empty(0, dtype=dtype).real.dtype
    out = np.empty((n, n), dtype=real_dtype)

    def fill(start: int) -> None:
        stop = min(start + block, n)
        acc = np.zeros((stop - start, n), dtype=np.float64)
        for f in range(n_freqs):
            sxy = x_fnk[f, start:stop] @ x_fkn[f]      # (b, N) cross-spectrum
            acc += sxy.real.astype(np.float64) ** 2 + sxy.imag.astype(np.float64) ** 2
        np.divide(acc, den[start:stop], out=acc)
        out[start:stop] = acc.astype(real_dtype)

    starts = range(0, n, block)
    if workers is None:
        workers = min(32, (os.cpu_count() or 1))
    if workers > 1 and len(starts) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fill, starts))
    else:
        for start in starts:
            fill(start)

    # Symmetrize: the blocked accumulation is exact in exact arithmetic but
    # float32 rounding leaves C[i,j] and C[j,i] differing in the last bit, and
    # downstream eigensolvers insist on exact symmetry.
    out += out.T
    out *= 0.5
    np.fill_diagonal(out, 1.0)

    f_eff = effective_bins(weights_nf) if compute_f_eff else float(n_freqs)
    return CoherenceMatrix(
        values=out,
        n_eff=int(s.n_segments),
        f_eff=float(n_freqs) if whiten else f_eff,
        whiten=whiten,
        stft_metadata=s.metadata(),
    )


def debias(c: CoherenceMatrix | np.ndarray, n_eff: int | None = None) -> np.ndarray:
    """Remove the ``1/n_eff`` null floor and clamp to ``[0, 1]``.

    ``(C - b) / (1 - b)`` with ``b = 1/n_eff``. For an incoherent estimator
    ``n_eff`` is the **segment count K**, not ``K*F``: the bias floor does not
    improve with bandwidth. Passing ``K*F`` under-corrects by a factor of F.
    A :class:`CoherenceMatrix` supplies the right value itself.
    """
    if isinstance(c, CoherenceMatrix):
        n_eff = c.n_eff if n_eff is None else n_eff
        values = c.values
    else:
        if n_eff is None:
            raise ValueError("n_eff is required when passing a bare array")
        values = np.asarray(c)
    if n_eff < 1:
        raise ValueError(f"n_eff must be >= 1, got {n_eff}")
    if n_eff == 1:
        return np.clip(values, 0.0, 1.0)
    bias = 1.0 / n_eff
    return np.clip((values - bias) / (1.0 - bias), 0.0, 1.0)


def to_distance(values: np.ndarray, *, use_sqrt: bool = True) -> np.ndarray:
    """Dissimilarity from a coherence matrix, with an exactly zero diagonal.

    Coherence is a squared quantity; ``sqrt(C)`` behaves like a correlation and
    ``1 - sqrt(C)`` is closer to a metric. Neither is a true metric on a general
    coherence matrix, so anything assuming squared Euclidean distances (Ward
    linkage) is not valid on the result.

    This exists because passing a *similarity* where a *distance* is expected
    silently inverts the answer — the pre-refactor ``compute_model_silhouette``
    did exactly that with ``metric="precomputed"``.
    """
    if isinstance(values, CoherenceMatrix):
        raise TypeError("pass CoherenceMatrix.values, not the container")
    v = np.asarray(values, dtype=np.float64)
    v = np.clip(0.5 * (v + v.T), 0.0, 1.0)
    d = 1.0 - (np.sqrt(v) if use_sqrt else v)
    np.fill_diagonal(d, 0.0)
    return d
