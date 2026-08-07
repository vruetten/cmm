"""Synthetic timeseries with known cluster structure.

Shared by the test suite, the README and demo scripts, so there is one source
of truth for "traces with ground-truth groups".

The design point that matters: groups share an independent **band-limited
noise source**, not a phase-shifted sinusoid. A sinusoid is coherent with
*everything* at its own frequency — two groups built from phase-shifted
copies of the same tone have pairwise coherence 1 and cannot discriminate
any estimator or clustering route. Band-limited noise sources drawn
independently per group are mutually incoherent in expectation, so recovery
of the partition is a real test.

Depends only on numpy; importable without scipy or scikit-learn.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bandlimited_source", "grouped_traces"]


def bandlimited_source(
    rng: np.random.Generator,
    t: int,
    fs: float,
    band: tuple[float, float] = (0.3, 1.2),
) -> np.ndarray:
    """One unit-variance noise trace with power only inside ``band`` (Hz).

    White Gaussian noise, transformed, zeroed outside the band, transformed
    back — the phase spectrum stays random, so two calls give incoherent
    sources even in the same band.
    """
    lo, hi = band
    freqs = np.fft.rfftfreq(t, 1 / fs)
    if not (0 <= lo < hi):
        raise ValueError(f"band must be (low, high) with 0 <= low < high, got {band}")
    keep = (freqs >= lo) & (freqs <= hi)
    if not keep.any():
        raise ValueError(
            f"band {band} Hz selects no frequency bins on a {t}-sample record "
            f"at fs={fs} (resolution {fs / t:.6g} Hz)"
        )
    spec = np.fft.rfft(rng.standard_normal(t))
    spec[~keep] = 0.0
    src = np.fft.irfft(spec, t)
    return src / src.std()


def grouped_traces(
    sizes,
    t: int = 3072,
    fs: float = 8.0,
    snr: float = 2.0,
    seed: int = 0,
    band: tuple[float, float] = (0.3, 1.2),
) -> tuple[np.ndarray, np.ndarray]:
    """``(N, t)`` traces in ``len(sizes)`` groups, plus 0-based true labels.

    Every signal is independent white noise; each group additionally shares
    ``snr`` times its own :func:`bandlimited_source`. Within-group coherence
    therefore concentrates in ``band`` while between-group coherence sits at
    the estimator's bias floor.

    ``sizes`` gives the number of signals per group, e.g. ``[8, 8, 8]``.
    """
    sizes = [int(s) for s in sizes]
    if len(sizes) < 1 or any(s < 1 for s in sizes):
        raise ValueError(f"sizes must be positive group sizes, got {sizes}")
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(sizes)), sizes)
    x = rng.standard_normal((labels.size, t))
    for g in range(len(sizes)):
        src = bandlimited_source(rng, t, fs, band=band)
        x[labels == g] += snr * src
    return x, labels
