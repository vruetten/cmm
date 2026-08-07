"""Instantaneous phase / amplitude via complex Morlet wavelet convolution.

Alternative to :func:`cmm.phase.analytic_signal` (Butterworth bandpass +
Hilbert). Convolves each per-signal trace with a complex Morlet wavelet at a
target frequency, producing the same ``(N, T)`` complex signal — ``arg`` is
instantaneous phase, ``abs`` is instantaneous amplitude at that frequency.

Complex Morlet::

    ψ(t) = π^(-1/4) · exp(2πi·f·t) · exp(-t² / (2 σ_t²))

with ``σ_t = n_cycles / (2π f)``. The Gaussian envelope carries ``n_cycles``
cycles of the carrier within one standard deviation, so a larger ``n_cycles``
gives narrower frequency selectivity (``Δf ≈ f / n_cycles``) at the cost of
wider time support (``≈ 4 σ_t``).

Trade-off vs the Butterworth + Hilbert path:

* Wavelet has no impulse-response ringing — the Gaussian envelope is smooth
  and finite in both domains. Butterworth+Hilbert at a 10-40 s band rings
  for ~±32 s (see the docstring of :func:`cmm.phase.analytic_signal`), which
  contaminates each trace end.
* Wavelet gives phase at *one* frequency by construction. Butterworth+Hilbert
  gives phase of "whatever the bandpass let through", which mixes any
  frequencies still present in the passband.
* Both are zero-phase (a real Gaussian envelope makes ψ symmetric around
  its peak modulo the carrier), so no group-delay correction is needed.

As a worked example, a 30 s target period with ``freq_hz = 1/30``,
``n_cycles = 6`` gives σ_t ≈ 28.6 s (support ~115 s) and Δf ≈ 0.006 Hz —
narrower than a 10-40 s bandpass and centred on the oscillation of interest.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

__all__ = ["morlet_wavelet", "wavelet_analytic"]


def morlet_wavelet(fs: float, freq_hz: float, n_cycles: float = 6.0,
                   *, dtype: np.dtype | type = np.complex128) -> np.ndarray:
    """Sampled complex Morlet at ``freq_hz`` with ``n_cycles`` under the envelope.

    Length is truncated at ±4 σ_t (>99.99 % of the Gaussian mass), so the
    kernel length in samples is roughly ``8 · n_cycles · fs / (2π · freq_hz)``.

    Normalised so that ``|W| ≈ A`` for an input ``A · cos(2π f t)`` at ``f =
    freq_hz`` — matches the amplitude convention of
    :func:`cmm.phase.analytic_signal` (Hilbert of a real signal returns
    ``x + i·H[x]`` whose magnitude equals the envelope, not twice it, when the
    signal is already narrowband).
    """
    if freq_hz <= 0 or freq_hz >= fs / 2:
        raise ValueError(
            f"freq_hz={freq_hz} must satisfy 0 < f < fs/2 = {fs / 2}"
        )
    if n_cycles < 2:
        raise ValueError(
            f"n_cycles={n_cycles} too small; Morlet needs >=2 for admissibility"
        )
    sigma_t = n_cycles / (2 * np.pi * freq_hz)
    # ±4σ covers >99.99 %; round up so kernel length is odd (centre-aligned).
    half = int(np.ceil(4 * sigma_t * fs))
    t = np.arange(-half, half + 1) / fs
    envelope = np.exp(-0.5 * (t / sigma_t) ** 2)
    carrier = np.exp(2j * np.pi * freq_hz * t)
    psi = carrier * envelope
    # Normalise so a unit-amplitude input sine gives |W| = 1. For a pure sine
    # x(t) = A cos(2π f t) = A/2 (e^{i2πft} + e^{-i2πft}), convolution with
    # ψ*(τ) = π^(-1/4) e^{-i2πfτ} envelope(τ) yields at zero lag
    #     ∫ (A/2)(e^{i2πfτ} + e^{-i2πfτ}) · e^{-i2πfτ} · envelope(τ) dτ
    #     = (A/2) · sum(envelope) + (rapidly-oscillating small term)
    # (in the discrete sum), so |W| ≈ A/2 · sum(envelope) at the carrier. To
    # get |W| = A we divide by (sum(envelope) / 2). This is more direct than
    # the π^(-1/4) analytical normaliser and correct for the discrete case.
    psi *= 2.0 / envelope.sum()
    return psi.astype(dtype)


def wavelet_analytic(
    traces_nt: np.ndarray,
    fs: float,
    freq_hz: float,
    *,
    n_cycles: float = 6.0,
    dtype: np.dtype | type = np.complex64,
) -> np.ndarray:
    """Complex Morlet convolution at ``freq_hz`` for every row of ``traces_nt``.

    Returns ``(N, T)`` in ``dtype``. ``dtype`` only sets the returned
    precision; the convolution runs in float64 / complex128 for numerical
    stability.

    Edges: `fftconvolve(..., mode='same')` extends with zeros, which biases
    the wavelet magnitude toward zero within one kernel-half of each end.
    Callers that care about clean phase at the edges should either crop
    the first / last ``4 · σ_t · fs`` samples or ensure the trace was
    already edge-cropped upstream.

    Also, a per-signal NaN would spread across the entire row through the
    Gaussian tail of ``fftconvolve``. We raise on non-finite input rather
    than silently propagating; an upstream variance filter should have
    already removed any dead signals, so a NaN here indicates a bug worth
    catching.
    """
    if not np.issubdtype(np.dtype(dtype), np.complexfloating):
        raise ValueError(f"dtype must be a complex type, got {np.dtype(dtype)}")
    x = np.asarray(traces_nt)
    if x.ndim != 2:
        raise ValueError(f"expected (N, T) traces, got shape {x.shape}")
    if not np.isfinite(x).all():
        n_bad_rows = int((~np.isfinite(x)).any(axis=1).sum())
        raise ValueError(
            f"traces_nt contains non-finite values in {n_bad_rows}/{x.shape[0]} rows; "
            "FFT convolution would spread them across the whole row"
        )

    psi = morlet_wavelet(fs, freq_hz, n_cycles=n_cycles, dtype=np.complex128)
    # Convolution with ψ*(-t) is cross-correlation; scipy's fftconvolve
    # convolves x * psi, so pre-conjugate + time-reverse to get the intended
    # W(t) = ∫ x(τ) ψ*(τ - t) dτ. For a symmetric Gaussian carrier this
    # differs only in the sign convention of phase; the standard MNE / MATLAB
    # convention is to use ψ directly (not ψ*), giving phase that increases
    # in time for a positive-frequency sinusoid. Match that.
    x64 = x.astype(np.float64, copy=False)
    W = fftconvolve(x64, psi[np.newaxis, :], mode="same", axes=1)
    return W.astype(dtype)
