"""Windowed Fourier analysis: STFT coefficients, PSD, CSD, pairwise coherence.

This is the module other packages import. Everything downstream — the coherence
matrix, both clustering routes — is built on :func:`stft`.

Conventions follow :mod:`scipy.signal` exactly, and the test suite pins that
agreement to ``rtol=1e-12`` against ``welch``, ``csd`` and ``coherence``. In
particular:

* ``scaling="density"`` divides by ``fs * (win**2).sum()`` and ``"spectrum"`` by
  ``win.sum()**2`` — these are *different* normalizations, not a labelling
  choice, and the pre-refactor code applied the density formula under both names.
* One-sided spectra double every bin except DC and, for even ``nfft``, Nyquist.
  Omitting that does not merely rescale a spectrum; it leaves the end bins too
  high *relative* to the interior.
* Segments are detrended first, then windowed, then transformed.

The cross-spectrum convention is scipy's: ``csd(x, y) = mean_k conj(X) * Y``.
Note this is the conjugate of the ``P_nm = sum_k X_n conj(X_m)`` convention used
in ``derivations/coherence_estimators.md``; magnitudes agree, phase sign flips.

Parameters live in :class:`SpectralParams` and travel with the result, so a
stored output records how it was made. There is no ``x_in_coefs`` flag: once you
hold an :class:`Stft` you pass the :class:`Stft`, and the frequency axis is
carried rather than reconstructed from an array length.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.signal import detrend as _scipy_detrend
from scipy.signal.windows import get_window

__all__ = [
    "SpectralParams",
    "Stft",
    "stft",
    "psd",
    "csd",
    "coherence",
    "segment",
]

Scaling = Literal["density", "spectrum"]


@dataclass(frozen=True)
class SpectralParams:
    """Everything needed to reproduce a spectral estimate.

    Attributes:
        fs: Sampling frequency, Hz.
        nperseg: Samples per segment.
        noverlap: Samples of overlap between segments. ``None`` means
            ``nperseg // 2``. More overlap buys more segments but they are not
            independent — see ``derivations/coherence_estimators.md`` §5 for why
            the segment count, not the bandwidth, sets the coherence bias floor.
        window: Window name, ``(name, param)`` tuple, or an explicit array of
            length ``nperseg``.
        detrend: ``"constant"``, ``"linear"``, or ``False``. Applied per segment
            before windowing.
        band_hz: Optional ``(low, high)`` in Hz, inclusive, retained after the
            transform. ``None`` keeps every bin.
        nfft: Transform length; ``None`` means ``nperseg``. Larger zero-pads.
        onesided: Return only non-negative frequencies. Requires real input.
    """

    fs: float
    nperseg: int
    noverlap: int | None = None
    window: str | tuple | np.ndarray = "hann"
    detrend: str | Literal[False] = "constant"
    band_hz: tuple[float, float] | None = None
    nfft: int | None = None
    onesided: bool = True

    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ValueError(f"fs must be positive, got {self.fs}")
        if self.nperseg < 1:
            raise ValueError(f"nperseg must be >= 1, got {self.nperseg}")
        if self.resolved_noverlap >= self.nperseg:
            raise ValueError(
                f"noverlap={self.resolved_noverlap} must be < nperseg={self.nperseg}"
            )
        if self.resolved_noverlap < 0:
            raise ValueError(f"noverlap must be >= 0, got {self.resolved_noverlap}")
        if self.resolved_nfft < self.nperseg:
            raise ValueError(
                f"nfft={self.resolved_nfft} must be >= nperseg={self.nperseg}"
            )
        if self.detrend not in ("constant", "linear", False):
            raise ValueError(
                f"detrend must be 'constant', 'linear' or False, got {self.detrend!r}"
            )
        if self.band_hz is not None:
            lo, hi = self.band_hz
            if not lo < hi:
                raise ValueError(f"band_hz must be (low, high) with low < high, got {self.band_hz}")
            if lo < 0:
                raise ValueError(f"band_hz low must be >= 0, got {lo}")

    @property
    def resolved_noverlap(self) -> int:
        return self.nperseg // 2 if self.noverlap is None else int(self.noverlap)

    @property
    def resolved_nfft(self) -> int:
        return self.nperseg if self.nfft is None else int(self.nfft)

    @property
    def step(self) -> int:
        return self.nperseg - self.resolved_noverlap

    def n_segments(self, n_times: int) -> int:
        """Number of whole segments that fit in ``n_times`` samples."""
        if n_times < self.nperseg:
            raise ValueError(
                f"nperseg={self.nperseg} exceeds the record length {n_times}; "
                "shorten nperseg or use a longer recording"
            )
        return (n_times - self.resolved_noverlap) // self.step

    def with_(self, **kwargs) -> SpectralParams:
        """Copy with fields replaced — params are frozen, so this is the way."""
        return replace(self, **kwargs)


@dataclass(frozen=True)
class Stft:
    """Complex short-time Fourier coefficients and the axis they live on.

    Attributes:
        coefs: ``(N, K, F)`` complex — signals, segments, retained frequencies.
        freqs: ``(F,)`` Hz, matching ``coefs``' last axis. Carried, never
            recomputed from ``F`` — reconstructing it was a live bug.
        params: the :class:`SpectralParams` that produced this.
        window: ``(nperseg,)`` window actually applied.
        onesided_factor: ``(F,)`` of 1.0 or 2.0 — the one-sided power doubling
            for each retained bin, kept aligned through band selection so
            :func:`psd` cannot mis-apply it.
    """

    coefs: np.ndarray
    freqs: np.ndarray
    params: SpectralParams
    window: np.ndarray
    onesided_factor: np.ndarray

    @property
    def n_signals(self) -> int:
        return self.coefs.shape[0]

    @property
    def n_segments(self) -> int:
        """K. This — not the number of frequency bins — sets the coherence bias floor."""
        return self.coefs.shape[1]

    @property
    def n_freqs(self) -> int:
        return self.coefs.shape[2]

    def scale(self, scaling: Scaling) -> float:
        """scipy's normalization constant for the requested scaling."""
        win = self.window
        if scaling == "density":
            return 1.0 / (self.params.fs * (win * win).sum())
        if scaling == "spectrum":
            return 1.0 / win.sum() ** 2
        raise ValueError(f"scaling must be 'density' or 'spectrum', got {scaling!r}")

    def metadata(self) -> dict:
        """Everything needed to reproduce this, for embedding in saved outputs."""
        p = self.params
        return {
            "fs": float(p.fs),
            "nperseg": int(p.nperseg),
            "noverlap": int(p.resolved_noverlap),
            "nfft": int(p.resolved_nfft),
            "window": p.window if isinstance(p.window, str) else "array",
            "detrend": p.detrend,
            "onesided": bool(p.onesided),
            "band_hz": None if p.band_hz is None else [float(b) for b in p.band_hz],
            "n_signals": self.n_signals,
            "n_segments": self.n_segments,
            "n_freqs": self.n_freqs,
            "freq_hz": self.freqs.tolist(),
        }


def resolve_window(window, nperseg: int) -> np.ndarray:
    """Window array of length ``nperseg``.

    Raises rather than warning on a length mismatch — a silently resized window
    changes every normalization downstream.
    """
    if isinstance(window, (str, tuple)):
        return np.asarray(get_window(window, nperseg), dtype=np.float64)
    win = np.asarray(window, dtype=np.float64)
    if win.ndim != 1:
        raise ValueError(f"window array must be 1-D, got shape {win.shape}")
    if win.size != nperseg:
        raise ValueError(f"window length {win.size} != nperseg {nperseg}")
    return win


def segment(x_nt: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    """``(N, K, nperseg)`` overlapping view of ``(N, T)``.

    A strided view, so it costs nothing; the caller must not write to it. Any
    trailing samples that do not fill a whole segment are dropped, matching
    scipy.
    """
    x = np.asarray(x_nt)
    if x.ndim != 2:
        raise ValueError(f"expected (N, T), got shape {x.shape}")
    step = nperseg - noverlap
    n_seg = (x.shape[-1] - noverlap) // step
    if n_seg < 1:
        raise ValueError(
            f"nperseg={nperseg} exceeds the record length {x.shape[-1]}"
        )
    shape = x.shape[:-1] + (n_seg, nperseg)
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def stft(x_nt: np.ndarray, params: SpectralParams) -> Stft:
    """Windowed Fourier coefficients of ``(N, T)`` traces (1-D is promoted).

    Detrend, window, transform — in that order, matching scipy. No scaling is
    applied here; :func:`psd` and :func:`csd` apply it, because the correct
    constant depends on what you are estimating.
    """
    x = np.asarray(x_nt)
    if x.ndim == 1:
        x = x[None]
    if x.ndim != 2:
        raise ValueError(f"expected (N, T) or (T,), got shape {x.shape}")
    if np.iscomplexobj(x) and params.onesided:
        raise ValueError(
            "onesided=True requires real input; complex input needs "
            "onesided=False. (The pre-refactor code silently discarded the "
            "imaginary part here.)"
        )
    if not np.isfinite(x).all():
        n_bad = int((~np.isfinite(x)).sum())
        raise ValueError(
            f"input contains {n_bad} non-finite samples; a single NaN spreads "
            "across a whole segment through detrending and the transform, so "
            "drop or interpolate those signals upstream"
        )

    n_times = x.shape[-1]
    params.n_segments(n_times)  # raises if nperseg > T
    win = resolve_window(params.window, params.nperseg)
    nfft = params.resolved_nfft

    seg = segment(x, params.nperseg, params.resolved_noverlap)
    if params.detrend is not False:
        seg = _scipy_detrend(seg, type=params.detrend, axis=-1)
    else:
        seg = np.array(seg, copy=True)  # own the memory before windowing
    seg = seg * win

    if params.onesided:
        coefs = np.fft.rfft(seg, n=nfft, axis=-1)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / params.fs)
        factor = np.full(freqs.shape, 2.0)
        factor[0] = 1.0
        if nfft % 2 == 0:
            factor[-1] = 1.0  # unpaired Nyquist bin
    else:
        coefs = np.fft.fft(seg, n=nfft, axis=-1)
        freqs = np.fft.fftfreq(nfft, d=1.0 / params.fs)
        factor = np.ones(freqs.shape)

    if params.band_hz is not None:
        lo, hi = params.band_hz
        keep = (np.abs(freqs) >= lo) & (np.abs(freqs) <= hi)
        if not keep.any():
            raise ValueError(
                f"band_hz={params.band_hz} selects no frequency bins; the axis "
                f"spans {freqs.min():.6g}-{freqs.max():.6g} Hz at "
                f"{params.fs / nfft:.6g} Hz resolution"
            )
        coefs, freqs, factor = coefs[..., keep], freqs[keep], factor[keep]

    return Stft(coefs=coefs, freqs=freqs, params=params, window=win,
                onesided_factor=factor)


def psd(s: Stft, scaling: Scaling = "density") -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density per signal: ``(N, F)`` real, and the frequencies.

    Matches ``scipy.signal.welch`` to ``rtol=1e-12``.
    """
    p = (s.coefs * s.coefs.conj()).real.mean(axis=1)
    p = p * s.scale(scaling) * s.onesided_factor
    return p, s.freqs


def csd(
    sx: Stft, sy: Stft | None = None, scaling: Scaling = "density", *, pairwise: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-spectral density, scipy's ``conj(X) * Y`` convention.

    ``sy=None`` means the auto case. With ``pairwise=True`` the two inputs must
    have the same number of signals and the result is ``(N, F)`` — signal *i*
    against signal *i*. Otherwise every pair is formed and the result is
    ``(N, M, F)``.

    Matches ``scipy.signal.csd`` to ``rtol=1e-12``.
    """
    sy = sx if sy is None else sy
    _check_compatible(sx, sy)
    if pairwise:
        if sx.n_signals != sy.n_signals:
            raise ValueError(
                f"pairwise=True needs equal signal counts, got "
                f"{sx.n_signals} and {sy.n_signals}"
            )
        p = np.einsum("nkf,nkf->nf", sx.coefs.conj(), sy.coefs, optimize=True)
    else:
        p = np.einsum("nkf,mkf->nmf", sx.coefs.conj(), sy.coefs, optimize=True)
    p = p / sx.n_segments
    p = p * sx.scale(scaling) * sx.onesided_factor
    return p, sx.freqs


def coherence(
    sx: Stft, sy: Stft | None = None, *, pairwise: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude-squared coherence per frequency, in ``[0, 1]``.

    ``|Pxy|**2 / (Pxx * Pyy)``. Every normalization constant cancels in the
    ratio, which is why this was the one estimate the pre-refactor code got
    right. Matches ``scipy.signal.coherence`` to ``rtol=1e-12``.

    Note the ``1/K`` bias floor: for independent signals this does not go to
    zero but to ``1 / n_segments``, independent of bandwidth. See
    ``derivations/coherence_estimators.md`` §5.
    """
    sy = sx if sy is None else sy
    pxy, freqs = csd(sx, sy, pairwise=pairwise)
    pxx, _ = psd(sx)
    pyy, _ = psd(sy)
    num = (pxy * pxy.conj()).real
    den = pxx * pyy if pairwise else pxx[:, None, :] * pyy[None, :, :]
    # A bin where either signal has exactly zero power has undefined coherence.
    # Return NaN rather than letting 0/0 contribute noise to a band average --
    # a degenerate bin should be visible, not quietly averaged in. Note this
    # only catches exact zeros; a bin at dust level (detrending drives the DC
    # bin of a boxcar-windowed segment to ~1e-32) still divides, and the result
    # there is numerically meaningless in any implementation.
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out, freqs


def _check_compatible(sx: Stft, sy: Stft) -> None:
    if sx.n_segments != sy.n_segments:
        raise ValueError(
            f"segment counts differ ({sx.n_segments} vs {sy.n_segments}); the "
            "two signals must be transformed with the same parameters and length"
        )
    if sx.freqs.shape != sy.freqs.shape or not np.array_equal(sx.freqs, sy.freqs):
        raise ValueError("frequency axes differ; transform both with the same params")
