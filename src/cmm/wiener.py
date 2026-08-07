"""Directed Wiener filter between two sets of signals.

For each source signal ``x_n`` and each target ``y_m`` this estimates the
linear filter that best predicts the target from the source,

    Ŷ_m(f) = H[n, m, f] · X_n(f),      H = S_yx / (S_xx + λ(f)),

where ``S_yx(f) = ⟨conj(X) · Y⟩`` and ``S_xx(f) = ⟨|X|²⟩`` are Welch averages
over segments and ``λ(f) = alpha_reg · median_n S_xx(n, f)`` is a *relative*
diagonal loading — scaled to the data, so the same ``alpha_reg`` means the
same thing across datasets and frequencies. All window/segment normalization
constants cancel in the ratio, including inside ``λ``.

Alongside the filter this returns the fraction of each target's band power the
filter explains,

    r[n, m] = Σ_f C_nm(f) · S_yy(m, f) / Σ_f S_yy(m, f),

with ``C`` the per-frequency magnitude-squared coherence. This follows from
the Wiener residual spectrum ``S_err = S_yy · (1 − C)``: coherence weighted by
where the target actually has power, not by bin count. Unlike the coherence
matrix (:func:`cmm.coherence_matrix`), this quantity is directional in
normalization — it asks "how much of *y* does *x* explain", which is not
symmetric in x and y.

This module replaces the pre-refactor ``estimate_wiener_r``, which computed
its loading as an absolute offset (contradicting its own docstring), took the
impulse-response length from the in-band bin count (wrong whenever a band was
selected), and had never been tested. Every number moves relative to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spectral import SpectralParams, Stft, _check_compatible

__all__ = ["WienerFilter", "wiener_filter"]


@dataclass(frozen=True)
class WienerFilter:
    """Result of :func:`wiener_filter`.

    Attributes:
        gain: ``(N, M, F)`` complex — ``H[n, m, f]`` such that
            ``Ŷ_m(f) = H · X_n(f)``. On the retained (possibly band-limited)
            frequency axis ``freqs``.
        impulse: ``(N, M, nfft)`` — time-domain filter ``h``, the inverse
            transform of ``gain`` embedded in the *full* frequency grid (bins
            outside a selected band are zero). Real for a one-sided transform;
            complex when the transform was two-sided, because complex input
            signals can genuinely need a complex filter. Entry ``t`` is the
            coefficient at lag ``lags[t]``; a causal relationship concentrates
            weight at small non-negative lags.
        r: ``(N, M)`` real — fraction of target band power explained,
            in ``[0, 1]`` up to estimator noise. Subject to the same ``1/K``
            coherence bias floor as everything else built on K segments.
        freqs: ``(F,)`` Hz, the axis ``gain`` lives on.
        lags: ``(nfft,)`` seconds, the axis ``impulse`` lives on, in FFT order:
            ``0, 1/fs, …`` then the negative lags.
        alpha_reg: the relative loading actually used.
        params: the :class:`cmm.spectral.SpectralParams` of the inputs.
    """

    gain: np.ndarray
    impulse: np.ndarray
    r: np.ndarray
    freqs: np.ndarray
    lags: np.ndarray
    alpha_reg: float
    params: SpectralParams

    def metadata(self) -> dict:
        """Provenance for embedding in saved outputs."""
        p = self.params
        return {
            "alpha_reg": float(self.alpha_reg),
            "n_sources": int(self.gain.shape[0]),
            "n_targets": int(self.gain.shape[1]),
            "n_freqs": int(self.gain.shape[2]),
            "nfft": int(self.impulse.shape[-1]),
            "fs": float(p.fs),
            "nperseg": int(p.nperseg),
            "noverlap": int(p.resolved_noverlap),
            "detrend": p.detrend,
            "band_hz": None if p.band_hz is None else [float(b) for b in p.band_hz],
        }


def _full_grid_indices(s: Stft) -> tuple[np.ndarray, np.ndarray]:
    """Positions of the retained bins on the full (un-banded) frequency grid.

    The impulse response must be transformed on the full grid: taking the
    inverse transform of just the in-band bins reinterprets them as a
    contiguous spectrum starting at DC, which was the pre-refactor bug.
    """
    nfft = s.params.resolved_nfft
    if s.params.onesided:
        full = np.fft.rfftfreq(nfft, d=1.0 / s.params.fs)
    else:
        full = np.fft.fftfreq(nfft, d=1.0 / s.params.fs)
    # Band selection kept exact elements of this same array, so exact
    # equality matching is safe — and asserted.
    idx = np.array([int(np.nonzero(full == f)[0][0]) for f in s.freqs])
    if not np.array_equal(full[idx], s.freqs):
        raise AssertionError("retained frequencies not found on the full grid")
    return idx, full


def wiener_filter(
    sx: Stft, sy: Stft, *, alpha_reg: float = 1e-3
) -> WienerFilter:
    """Wiener filter predicting every signal in ``sy`` from every signal in ``sx``.

    Both inputs must come from :func:`cmm.spectral.stft` with the same
    parameters and record length (checked). ``alpha_reg`` sets the diagonal
    loading ``λ(f) = alpha_reg · median_n S_xx(n, f)``; ``0`` disables it and
    gives the exact ratio.

    Memory: materializes ``(N, M, F)`` complex gains and ``(N, M, nfft)``
    impulse responses — size the source set N accordingly (it is typically a
    handful of seeds against many targets).
    """
    _check_compatible(sx, sy)
    if alpha_reg < 0:
        raise ValueError(f"alpha_reg must be >= 0, got {alpha_reg}")

    x = sx.coefs
    y = sy.coefs
    # Raw segment sums; every normalization constant cancels in H, C and r.
    pxx = np.einsum("nkf,nkf->nf", x, x.conj(), optimize=True).real
    pyy = np.einsum("mkf,mkf->mf", y, y.conj(), optimize=True).real
    for name, p in (("sx", pxx), ("sy", pyy)):
        if (p <= 0).any():
            n_bad = int((p <= 0).any(axis=1).sum())
            raise ValueError(
                f"{n_bad}/{p.shape[0]} signals in {name} have a frequency bin "
                "with zero power; the filter and coherence are undefined there. "
                "Drop them first — cmm.coherence.finite_power_mask finds them."
            )

    pxy = np.einsum("nkf,mkf->nmf", x.conj(), y, optimize=True)  # ⟨conj(X)·Y⟩ = S_yx

    lam = alpha_reg * np.median(pxx, axis=0)  # (F,), relative loading
    gain = pxy / (pxx[:, None, :] + lam[None, None, :])

    # Per-frequency MSC, unregularized — the exact quantity the residual
    # spectrum S_yy·(1 − C) is built from.
    msc = (pxy * pxy.conj()).real / (pxx[:, None, :] * pyy[None, :, :])
    # Weight by where the target's power actually is, including the one-sided
    # doubling so paired bins count what they physically carry.
    w = pyy * sy.onesided_factor[None, :]
    r = (msc * w[None, :, :]).sum(axis=-1) / w.sum(axis=-1)[None, :]

    # Impulse response on the full grid: in-band gains at their true bin
    # positions, zero outside the band, then one inverse transform of length
    # nfft — never of the in-band bin count.
    idx, full = _full_grid_indices(sx)
    nfft = sx.params.resolved_nfft
    n, m = gain.shape[:2]
    gain_full = np.zeros((n, m, full.size), dtype=gain.dtype)
    gain_full[..., idx] = gain
    if sx.params.onesided:
        impulse = np.fft.irfft(gain_full, n=nfft, axis=-1)
    else:
        impulse = np.fft.ifft(gain_full, n=nfft, axis=-1)

    lag_samples = ((np.arange(nfft) + nfft // 2) % nfft) - nfft // 2
    lags = lag_samples / sx.params.fs

    for a in (gain, impulse, r, lags):
        a.flags.writeable = False
    return WienerFilter(
        gain=gain, impulse=impulse, r=r, freqs=sx.freqs, lags=lags,
        alpha_reg=float(alpha_reg), params=sx.params,
    )
