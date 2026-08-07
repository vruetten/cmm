"""Phase 5: the directed Wiener filter — its first-ever tests.

The pre-refactor ``estimate_wiener_r`` was never tested; these tests define
the repaired semantics. The flagship check is end-to-end: generate ``y`` by
passing white noise through a known FIR filter and demand the filter back.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmm.spectral import SpectralParams, stft
from cmm.wiener import wiener_filter

FS = 10.0
T = 16384
P = SpectralParams(fs=FS, nperseg=256, noverlap=128, detrend=False)


def _white(seed, t_samples=T, n=1):
    x = np.random.default_rng(seed).standard_normal((n, t_samples))
    return x


def _fir_pair(seed=0, g=(0.5, -0.3, 0.2)):
    """White noise x and y = g * x (convolution), transient discarded."""
    g = np.asarray(g, dtype=np.float64)
    pad = 4 * g.size
    x_full = np.random.default_rng(seed).standard_normal(T + pad)
    y_full = np.convolve(x_full, g)[: x_full.size]
    return x_full[pad:][None], y_full[pad:][None], g


class TestFirRecovery:
    def test_impulse_response_recovers_the_true_filter(self):
        x, y, g = _fir_pair()
        wf = wiener_filter(stft(x, P), stft(y, P), alpha_reg=0.0)
        h = wf.impulse[0, 0]
        assert h.shape == (P.nperseg,)
        np.testing.assert_allclose(h[: g.size], g, atol=0.02)
        # Everything outside the true support is noise-level.
        assert np.abs(h[g.size :]).max() < 0.02

    def test_gain_matches_the_true_transfer_function(self):
        x, y, g = _fir_pair()
        wf = wiener_filter(stft(x, P), stft(y, P), alpha_reg=0.0)
        j = np.arange(g.size)
        true_gain = (g[None, :] * np.exp(-2j * np.pi * wf.freqs[:, None] * j / FS)).sum(
            axis=1
        )
        np.testing.assert_allclose(wf.gain[0, 0], true_gain, atol=0.02)

    def test_noiseless_relation_explains_nearly_all_target_power(self):
        x, y, _ = _fir_pair()
        wf = wiener_filter(stft(x, P), stft(y, P))
        assert wf.r[0, 0] > 0.95, wf.r


class TestDelay:
    def test_pure_delay_peaks_at_the_right_lag(self):
        d = 5
        x_full = _white(1, T + d)[0]
        x, y = x_full[d:][None], x_full[:-d][None]  # y(t) = x(t - d)
        wf = wiener_filter(stft(x, P), stft(y, P), alpha_reg=0.0)
        assert int(np.argmax(np.abs(wf.impulse[0, 0]))) == d
        assert wf.lags[d] == pytest.approx(d / FS)
        np.testing.assert_allclose(np.abs(wf.gain[0, 0]), 1.0, atol=0.05)
        assert wf.r[0, 0] > 0.95

    def test_lags_axis_is_fft_ordered(self):
        x = _white(2)
        wf = wiener_filter(stft(x, P), stft(x, P))
        n = P.nperseg
        assert wf.lags[0] == 0.0
        assert wf.lags[1] == pytest.approx(1 / FS)
        assert wf.lags[-1] == pytest.approx(-1 / FS)
        assert wf.lags.shape == (n,)


class TestRSemantics:
    def test_r_is_bounded_and_independent_signals_sit_near_the_bias_floor(self):
        x = _white(3)
        y = _white(4)
        sx, sy = stft(x, P), stft(y, P)
        wf = wiener_filter(sx, sy)
        k = sx.n_segments
        assert 0.0 <= wf.r[0, 0] <= 1.0
        # Same 1/K coherence bias floor as everything else; loose factor for
        # overlap-induced segment correlation.
        assert wf.r[0, 0] < 3.0 / k, (wf.r, 1.0 / k)

    def test_r_and_msc_weighting_are_scale_invariant(self):
        x, y, _ = _fir_pair(seed=5)
        wf1 = wiener_filter(stft(x, P), stft(y, P))
        wf2 = wiener_filter(stft(x, P), stft(7.5 * y, P))
        np.testing.assert_allclose(wf2.r, wf1.r, rtol=1e-12)
        # The filter itself must scale with the target.
        np.testing.assert_allclose(wf2.gain, 7.5 * wf1.gain, rtol=1e-12)

    def test_two_sided_transform_gives_the_same_r_and_impulse(self):
        # The one-sided doubling in the r weights is exactly what makes the
        # one- and two-sided answers agree for real input.
        x, y, _ = _fir_pair(seed=6)
        p2 = P.with_(onesided=False)
        wf1 = wiener_filter(stft(x, P), stft(y, P), alpha_reg=0.0)
        wf2 = wiener_filter(stft(x, p2), stft(y, p2), alpha_reg=0.0)
        np.testing.assert_allclose(wf2.r, wf1.r, rtol=1e-10)
        assert np.iscomplexobj(wf2.impulse)  # two-sided may serve complex input
        np.testing.assert_allclose(wf2.impulse.imag, 0.0, atol=1e-12)
        np.testing.assert_allclose(wf2.impulse.real, wf1.impulse, atol=1e-12)


class TestRegularization:
    def test_loading_only_shrinks_the_gain(self):
        x, y, _ = _fir_pair(seed=7)
        wf0 = wiener_filter(stft(x, P), stft(y, P), alpha_reg=0.0)
        wf1 = wiener_filter(stft(x, P), stft(y, P), alpha_reg=0.5)
        assert (np.abs(wf1.gain) <= np.abs(wf0.gain) + 1e-15).all()
        assert np.abs(wf1.gain).mean() < np.abs(wf0.gain).mean()

    def test_negative_alpha_raises(self):
        x = _white(8)
        s = stft(x, P)
        with pytest.raises(ValueError, match="alpha_reg must be >= 0"):
            wiener_filter(s, s, alpha_reg=-1e-3)


class TestBandLimited:
    def test_in_band_gain_is_unchanged_by_band_selection(self):
        # Each bin's gain depends only on that bin (alpha_reg=0), so selecting
        # a band must not move the retained bins.
        x, y, _ = _fir_pair(seed=9)
        pb = P.with_(band_hz=(0.5, 4.5))
        wf_full = wiener_filter(stft(x, P), stft(y, P), alpha_reg=0.0)
        wf_band = wiener_filter(stft(x, pb), stft(y, pb), alpha_reg=0.0)
        keep = (wf_full.freqs >= 0.5) & (wf_full.freqs <= 4.5)
        np.testing.assert_allclose(wf_band.gain[0, 0], wf_full.gain[0, 0][keep],
                                   rtol=1e-12)

    def test_impulse_length_is_the_full_nfft_and_the_delay_survives(self):
        # The pre-refactor code took the transform length from the in-band bin
        # count, which shifts and rescales every lag. On the full grid the
        # delay peak must stay at d samples even from a band-limited estimate.
        d = 5
        x_full = _white(10, T + d)[0]
        x, y = x_full[d:][None], x_full[:-d][None]
        pb = P.with_(band_hz=(0.3, 4.7))
        wf = wiener_filter(stft(x, pb), stft(y, pb), alpha_reg=0.0)
        assert wf.impulse.shape[-1] == P.nperseg  # not the in-band bin count
        assert int(np.argmax(np.abs(wf.impulse[0, 0]))) == d


class TestFailFast:
    def test_zero_power_signal_raises(self):
        x = _white(11, n=2)
        x[1] = 0.0
        s = stft(x, P)
        with pytest.raises(ValueError, match="zero power"):
            wiener_filter(s, s)

    def test_incompatible_transforms_raise(self):
        x = _white(12)
        s1 = stft(x, P)
        s2 = stft(x, P.with_(window="hamming"))
        with pytest.raises(ValueError, match="windows differ"):
            wiener_filter(s1, s2)


class TestProvenance:
    def test_arrays_are_frozen_and_metadata_is_complete(self):
        x, y, _ = _fir_pair(seed=13)
        wf = wiener_filter(stft(x, P), stft(y, P))
        for a in (wf.gain, wf.impulse, wf.r, wf.lags):
            assert not a.flags.writeable
        md = wf.metadata()
        for key in ("alpha_reg", "n_sources", "n_targets", "n_freqs", "nfft",
                    "fs", "nperseg", "noverlap", "detrend", "band_hz"):
            assert key in md, key
        assert md["alpha_reg"] == wf.alpha_reg
