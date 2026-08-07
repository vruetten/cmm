"""Phase 5: complex Morlet wavelet analytic signal.

Ported from the icampsnfr suite. Verifies that the Morlet-based analytic
signal recovers phase, amplitude, and instantaneous frequency on synthetic
signals.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmm.wavelet import morlet_wavelet, wavelet_analytic


def test_morlet_wavelet_shape_and_normalization() -> None:
    fs = 10.0
    f = 1.0
    n_cycles = 6.0
    psi = morlet_wavelet(fs, f, n_cycles=n_cycles)
    # ~4 sigma_t on each side, odd length
    sigma_t = n_cycles / (2 * np.pi * f)
    expected_half = int(np.ceil(4 * sigma_t * fs))
    assert psi.shape == (2 * expected_half + 1,)
    assert psi.dtype == np.complex128
    # Envelope must be centred (peak at midpoint)
    assert np.argmax(np.abs(psi)) == expected_half


def test_morlet_wavelet_bad_freq_rejected() -> None:
    with pytest.raises(ValueError, match="0 < f < fs/2"):
        morlet_wavelet(fs=2.0, freq_hz=1.5)   # above Nyquist
    with pytest.raises(ValueError, match="0 < f < fs/2"):
        morlet_wavelet(fs=2.0, freq_hz=0.0)
    with pytest.raises(ValueError, match="n_cycles"):
        morlet_wavelet(fs=10.0, freq_hz=1.0, n_cycles=1.0)


def test_pure_sine_phase_and_amplitude() -> None:
    """Pure sine at target frequency: recovered phase should match the
    analytical phase within a few degrees away from the edges."""
    fs = 5.0
    f = 0.1                           # 10 s period
    T = int(fs * 200)                 # 200 s recording
    t = np.arange(T) / fs
    A = 3.7
    phi0 = 0.7
    x = A * np.cos(2 * np.pi * f * t + phi0)
    W = wavelet_analytic(x[np.newaxis, :], fs=fs, freq_hz=f, n_cycles=6.0)
    assert W.shape == (1, T)
    assert W.dtype == np.complex64

    # Trim edges: 4 sigma_t on each side, where sigma_t = 6 / (2π f) ≈ 9.55 s.
    edge = int(np.ceil(4 * 6 / (2 * np.pi * f) * fs))
    core = slice(edge, T - edge)

    amp = np.abs(W[0, core])
    # Amplitude should be A within a few percent
    np.testing.assert_allclose(amp.mean(), A, rtol=0.02)
    np.testing.assert_allclose(amp.std() / A, 0.0, atol=0.02)

    # Phase: cos(2π f t + phi0) has instantaneous phase (2π f t + phi0);
    # wavelet returns phase with the same convention (MNE-like), so
    # unwrapped phase should be linear in t with slope 2π f and intercept
    # phi0. Wrap into [-π, π] and check angular distance to ground truth.
    got_phase = np.angle(W[0, core])
    expected = 2 * np.pi * f * t[core] + phi0
    diff = np.angle(np.exp(1j * (got_phase - expected)))
    max_deg = float(np.max(np.abs(diff)) * 180 / np.pi)
    assert max_deg < 3.0, f"max phase error {max_deg:.2f} deg exceeds 3 deg"


def test_multi_voxel_batching() -> None:
    """Batched application should give per-row results identical to
    single-row calls (no cross-row leakage in FFT convolution)."""
    fs = 5.0
    f = 0.2
    T = int(fs * 100)
    rng = np.random.default_rng(0)
    N = 4
    t = np.arange(T) / fs
    x = np.stack([
        1.0 * np.cos(2 * np.pi * f * t + 0.3),
        2.5 * np.cos(2 * np.pi * f * t + 1.2),
        0.7 * np.cos(2 * np.pi * f * t - 0.9),
        rng.standard_normal(T) * 0.01,   # noise-only row
    ])
    W_batch = wavelet_analytic(x, fs=fs, freq_hz=f, n_cycles=6.0)
    for i in range(N):
        W_i = wavelet_analytic(x[i : i + 1], fs=fs, freq_hz=f, n_cycles=6.0)
        np.testing.assert_allclose(W_batch[i], W_i[0], rtol=1e-5, atol=1e-6)


def test_chirp_phase_tracks_linear_sweep() -> None:
    """Chirp with frequency linear in t: where the sweep passes through the
    wavelet's centre frequency, the phase must advance at that frequency."""
    fs = 20.0
    T = int(fs * 300)
    t = np.arange(T) / fs
    f0 = 0.05
    f1 = 0.20
    phi = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t ** 2 / t[-1])
    x = np.cos(phi)
    f_target = 0.5 * (f0 + f1)   # 0.125 Hz, halfway through the sweep
    W = wavelet_analytic(x[np.newaxis, :], fs=fs, freq_hz=f_target, n_cycles=6.0)
    # Instantaneous freq = f0 + (f1 - f0) t / t_end; equals f_target at t_mid.
    t_mid = t[-1] * (f_target - f0) / (f1 - f0)
    i_mid = int(t_mid * fs)
    window = slice(i_mid - int(2 * fs), i_mid + int(2 * fs))
    unwrapped = np.unwrap(np.angle(W[0, window]))
    d_phase = np.diff(unwrapped)
    # dt = 1/fs, so d_phase should ≈ 2π f_target / fs
    expected_d = 2 * np.pi * f_target / fs
    np.testing.assert_allclose(d_phase.mean(), expected_d, rtol=0.05)


def test_rejects_nan_input() -> None:
    """Non-finite input must fail loudly — a single NaN spreads across the
    whole row via FFT convolution."""
    fs = 5.0
    x = np.random.default_rng(0).standard_normal((3, 100))
    x[1, 50] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        wavelet_analytic(x, fs=fs, freq_hz=0.1)


def test_rejects_wrong_ndim() -> None:
    with pytest.raises(ValueError, match=r"\(N, T\)"):
        wavelet_analytic(np.zeros(100), fs=5.0, freq_hz=0.1)
    with pytest.raises(ValueError, match=r"\(N, T\)"):
        wavelet_analytic(np.zeros((2, 3, 4)), fs=5.0, freq_hz=0.1)


def test_rejects_real_dtype() -> None:
    with pytest.raises(ValueError, match="complex type"):
        wavelet_analytic(np.zeros((1, 100)), fs=5.0, freq_hz=0.1, dtype=np.float32)
