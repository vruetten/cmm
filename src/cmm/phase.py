"""Instantaneous phase / amplitude of narrowband oscillations, and their
phase-triggered averages.

Given per-signal traces and a target band, the two operations are:

1. :func:`analytic_signal` — Butterworth bandpass then Hilbert transform per
   signal. The complex output ``z(t) = x_filt(t) + i·H[x_filt](t)`` carries
   the instantaneous phase (``angle(z)``) and envelope (``|z|``) directly.
   The complex signal is the useful cache: everything downstream reads from
   it, so a heavy dataset only has to pay the filter/Hilbert cost once.

2. :func:`phase_triggered_average` — amplitude-weighted average of the whole
   trace matrix, binned by the phase of a *seed* signal. If the seed is a
   mean over a group of signals (a "reference oscillator"), the returned
   ``(n_bins, N)`` frames are a stroboscopic view of the oscillation
   everywhere else, one frame per phase of the seed's cycle.

Nothing here is coherence-specific — the module operates on any ``(N, T)``
trace matrix and any target band, so it composes with any upstream
preprocessing that produces detrended traces.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

__all__ = [
    "analytic_signal",
    "bin_centers",
    "phase_amplitude",
    "phase_triggered_average",
    "valid_row_mask",
]


def analytic_signal(
    traces_nt: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
    *,
    order: int = 4,
    dtype: np.dtype | type = np.complex64,
) -> np.ndarray:
    """Butterworth bandpass + Hilbert per signal; returns ``(N, T)`` complex.

    Zero-phase filtering (forward-backward) is required here: a downstream
    phase reading would be off by the filter's group delay on a forward-only
    path.

    ``order`` is the prototype order, and the realised filter is steeper than
    that number suggests, in two compounding steps. ``btype="bandpass"``
    doubles it, so ``order=4`` designs 8 poles; the forward-backward pass then
    applies ``|H|**2``, doubling the attenuation again. Each band edge ends up
    rolling off at 8th-order-equivalent for the default ``order=4``.

    Measured at a representative slow-oscillation setting (``fs≈0.6 Hz``,
    10-40 s band), that costs ~64 s of impulse-response ringing (±32 s,
    symmetric because the filter is zero-phase) in exchange for ~93 dB of
    out-of-band rejection. Ringing length is set by the *bandwidth*, not by
    ``order``: dropping to ``order=2`` only shortens it to ~34 s and gives up
    47 dB. Callers that care about the contaminated edges should trim after
    this returns rather than soften the filter.

    Second-order sections are used rather than transfer-function coefficients
    because a low band edge near DC (``lo/nyquist`` small) degrades ``(b, a)``
    conditioning as poles cluster toward ``z = 1``.

    The filter and Hilbert transform both run in float64; ``dtype`` only sets
    the precision of the returned array. The default ``complex64`` halves the
    on-disk size of a cache at a cost of ~4e-8 relative error, which is flat
    in record length (measured 3.6e-8 at T=416 and 3.9e-8 at T=4847) and some
    six orders of magnitude below typical measurement noise. Pass
    ``complex128`` when a downstream step genuinely needs the extra bits;
    :func:`phase_triggered_average` will then carry them through.
    """
    if not np.issubdtype(np.dtype(dtype), np.complexfloating):
        raise ValueError(f"dtype must be a complex type, got {np.dtype(dtype)}")
    lo, hi = band_hz
    if not (0 < lo < hi < fs / 2):
        raise ValueError(
            f"band {band_hz} outside (0, fs/2={fs / 2:g}); check fs and band_hz"
        )
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    x = np.asarray(traces_nt, dtype=np.float64)  # the filter is stable in float64
    x_filt = sosfiltfilt(sos, x, axis=1)
    return hilbert(x_filt, axis=1).astype(dtype)


def phase_amplitude(analytic_nt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split analytic signal into phase in ``[-π, π)`` and non-negative amplitude."""
    z = np.asarray(analytic_nt)
    return np.angle(z).astype(np.float32), np.abs(z).astype(np.float32)


def phase_triggered_average(
    analytic_nt: np.ndarray,
    seed_rows: np.ndarray,
    *,
    n_bins: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude-weighted average of every row, binned by seed phase.

    The seed's analytic signal is the mean of ``analytic_nt`` over
    ``seed_rows``. Its instantaneous phase indexes timepoints into ``n_bins``
    bins uniformly spaced on ``[-π, π)``. For each bin ``k``:

    ``frames[k, i] = Σ_{t ∈ bin_k} A_seed(t) · Re[z(t, i)] / Σ_{t ∈ bin_k} A_seed(t)``

    Rows in ``analytic_nt`` are the *filtered* signals (``Re[z]``), not the
    raw traces, so what appears in the frames is the oscillation content —
    baseline drift has already been suppressed by the bandpass.

    Returns ``(frames, counts)``: ``counts`` is the number of samples that
    landed in each bin (integer), and ``frames`` is ``(n_bins, N)`` in the
    real precision of ``analytic_nt`` — float32 for a complex64 input, float64
    for complex128 — so a caller that paid for the extra bits keeps them.
    Bins with zero samples get NaN in ``frames``.

    The weighted sums always accumulate in float64 regardless of input
    precision, since a bin can hold thousands of terms.
    """
    z = np.asarray(analytic_nt)
    if z.ndim != 2:
        raise ValueError(f"expected (N, T), got shape {z.shape}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    seed_rows = np.asarray(seed_rows, dtype=np.int64).ravel()
    if seed_rows.size == 0:
        raise ValueError("seed_rows is empty; no reference signal to phase-lock to")

    N, T = z.shape
    real_dtype = z.real.dtype
    x_filt = z.real

    # Fail loudly on non-finite seed rows, and do it *before* averaging them:
    # `np.digitize` maps NaN to the same index as +pi, so the wrap below would
    # silently file it under bin 0, and an infinite seed can still yield a
    # finite angle (angle(inf+0j) == 0) while producing an infinite weight.
    # The filters are IIR/FFT-based, so one bad sample poisons a whole row.
    seed_block = z[seed_rows]
    finite = np.isfinite(seed_block)
    if not finite.all():
        bad = [int(r) for r in seed_rows[~finite.all(axis=1)]]
        raise ValueError(
            f"seed signal is not finite at {int((~finite.all(axis=0)).sum())}/{T} "
            f"timepoints; {len(bad)} of {seed_rows.size} seed rows contain "
            f"non-finite samples (first few: {bad[:10]}). A single NaN in a trace "
            "spreads across the whole row through the bandpass and Hilbert steps, "
            "so drop those signals upstream rather than averaging over them."
        )

    # Seed analytic signal: mean over the selected rows. Averaging complex
    # values preserves both the shared amplitude and the shared phase — a
    # cluster oscillating in-phase gives a strong seed, one that's incoherent
    # gives a weak seed (amplitude near zero), which then contributes little
    # via the amplitude weighting. That's the right behaviour.
    seed_z = seed_block.mean(axis=0)

    seed_phase = np.angle(seed_z).astype(np.float64)
    seed_amp = np.abs(seed_z).astype(np.float64)

    # Bin edges are chosen so bin 0 starts at -π; last edge is +π; each bin
    # spans 2π/n_bins radians. Right-edge of the last bin becomes +π which
    # numpy's digitize treats as out-of-range; wrap that to bin 0.
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(seed_phase, edges) - 1
    bin_idx = np.where(bin_idx == n_bins, 0, bin_idx)

    # Every timepoint must land in exactly one bin. A phase below -pi would
    # give index -1, which matches no k in the loop below and would be dropped
    # without a trace — shrinking the denominator rather than raising.
    if not ((bin_idx >= 0) & (bin_idx < n_bins)).all():
        stray = np.unique(bin_idx[(bin_idx < 0) | (bin_idx >= n_bins)])
        raise ValueError(
            f"{int(((bin_idx < 0) | (bin_idx >= n_bins)).sum())} timepoints fell "
            f"outside bins 0..{n_bins - 1} (stray indices {stray.tolist()}); "
            "seed phase is outside [-pi, pi]"
        )

    frames = np.zeros((n_bins, N), dtype=real_dtype)
    weights = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)
    for k in range(n_bins):
        sel = bin_idx == k
        counts[k] = int(sel.sum())
        if counts[k] == 0:
            continue
        w = seed_amp[sel]
        wsum = float(w.sum())
        if wsum <= 0:
            continue
        # (T_k,) weights times (N, T_k) filtered volume -> (N,)
        frames[k] = (x_filt[:, sel] * w).sum(axis=1) / wsum
        weights[k] = wsum

    if int(counts.sum()) != T:
        raise AssertionError(
            f"binning lost samples: counts.sum()={int(counts.sum())} but T={T}"
        )

    # Empty bins get NaN so a downstream colour map treats them as missing
    # rather than a strong signal at zero.
    for k in range(n_bins):
        if counts[k] == 0:
            frames[k, :] = np.nan
    return frames, counts


def bin_centers(n_bins: int) -> np.ndarray:
    """Phase-bin centres in ``[-π, π)`` — useful as the frame time axis."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def valid_row_mask(
    matrix_ntbins: np.ndarray,
    *,
    rel_var_threshold: float = 1e-8,
) -> np.ndarray:
    """Identify rows of an ``(N, n_bins)`` phase matrix worth sorting/displaying.

    A row is "valid" if it has at least one finite bin AND its bin-to-bin
    variance (ignoring NaNs) is above ``rel_var_threshold`` times the global
    variance across all finite entries of the matrix. Rows failing either
    check are all-NaN, all-constant, or so nearly-flat that they carry no
    phase-locked signal and would just clutter a sorted display.

    Returns a boolean array of length ``N``. The shape / row indexing of the
    input matrix is preserved: row ``i`` in the output corresponds to row
    ``i`` in the input, so callers can safely intersect this mask with
    external signal indices.
    """
    x = np.asarray(matrix_ntbins)
    if x.ndim != 2:
        raise ValueError(f"expected (N, n_bins), got shape {x.shape}")
    finite = np.isfinite(x)
    has_data = finite.any(axis=1)

    # Global variance across all finite entries — the scale against which
    # per-row variance is judged. If the whole matrix is flat there's nothing
    # to sort, so every row is deemed invalid.
    finite_vals = x[finite]
    if finite_vals.size == 0:
        return np.zeros(x.shape[0], dtype=bool)
    global_var = float(np.var(finite_vals))
    if global_var == 0.0:
        return np.zeros(x.shape[0], dtype=bool)

    # Per-row variance ignoring NaNs. `nanvar` warns when a row is all-NaN;
    # suppress that path with `has_data` — those rows are already ruled out.
    row_var = np.full(x.shape[0], 0.0, dtype=np.float64)
    if has_data.any():
        row_var[has_data] = np.nanvar(x[has_data], axis=1)

    return has_data & (row_var > rel_var_threshold * global_var)
