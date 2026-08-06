"""Layer 2: analytic ground truth, invariants, and every documented failure.

Layer 1 pins us to scipy. This layer pins the things scipy cannot check for us:
identities with a known closed form, invariances the estimator must have, and
the requirement that every degenerate input raises with a message naming the
problem rather than returning something plausible.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cmm.spectral import (
    SpectralParams,
    coherence,
    csd,
    psd,
    segment,
    stft,
)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


class TestAnalyticIdentities:
    def test_parseval_density_integrates_to_variance(self, rng):
        # One boxcar segment, no detrend: sum(PSD) * df == mean(x**2).
        t = 1024
        x = rng.standard_normal((3, t))
        p = SpectralParams(
            fs=8.0, nperseg=t, noverlap=0, window="boxcar", detrend=False
        )
        pxx, f = psd(stft(x, p))
        df = f[1] - f[0]
        assert_allclose(pxx.sum(axis=1) * df, (x**2).mean(axis=1), rtol=1e-12)

    def test_spectrum_scaling_recovers_sine_amplitude(self):
        # scaling='spectrum' at a bin-centred sine gives A**2/2.
        fs, t, amp, f0 = 100.0, 1024, 3.0, 100.0 * 8 / 256  # exactly on a bin
        tt = np.arange(t) / fs
        x = amp * np.sin(2 * np.pi * f0 * tt)[None]
        p = SpectralParams(fs=fs, nperseg=256, noverlap=0, detrend=False)
        pxx, f = psd(stft(x, p), scaling="spectrum")
        assert_allclose(pxx[0].max(), amp**2 / 2, rtol=1e-6)
        assert f[np.argmax(pxx[0])] == pytest.approx(f0)

    def test_coherence_is_one_for_a_scaled_copy(self, rng):
        x = rng.standard_normal((1, 1024))
        y = -4.25 * x
        p = SpectralParams(fs=10.0, nperseg=128)
        c, _ = coherence(stft(x, p), stft(y, p), pairwise=True)
        assert_allclose(c[0][1:], 1.0, rtol=1e-10)

    def test_coherence_stays_high_for_a_delayed_copy(self, rng):
        # A delay rotates the cross-spectrum phase but not its magnitude, so
        # per-frequency MSC is delay-invariant. This is the property the whole
        # estimator choice in derivations/coherence_estimators.md turns on.
        #
        # It is not exactly 1 for finite windowed segments: offsetting by `lag`
        # means the two tapered segments do not cover identical samples, and the
        # mismatch costs most at the lowest bins. Measured at this seed and
        # configuration: min 0.972, mean 0.990. The bounds below sit under those
        # with margin -- they guard against a collapse, not against drift.
        x = rng.standard_normal(4096)
        lag = 7
        p = SpectralParams(fs=10.0, nperseg=256)
        c, _ = coherence(
            stft(x[lag:][None], p), stft(x[:-lag][None], p), pairwise=True
        )
        assert c[0][1:].min() > 0.95
        assert c[0][1:].mean() > 0.98

    def test_delay_appears_as_a_linear_cross_spectrum_phase_ramp(self, rng):
        # The sharper statement of the same physics: for y(t) = x(t - tau) the
        # cross-spectrum phase is exactly -2*pi*f*tau. Magnitude tests can pass
        # on a broken phase convention; this one cannot.
        x = rng.standard_normal(8192)
        lag, fs = 7, 10.0
        p = SpectralParams(fs=fs, nperseg=512)
        pxy, f = csd(
            stft(x[lag:][None], p), stft(x[:-lag][None], p), pairwise=True
        )
        phase = np.unwrap(np.angle(pxy[0]))
        slope = np.polyfit(f, phase, 1)[0]
        assert_allclose(slope, -2 * np.pi * lag / fs, rtol=0.02)

    def test_coherence_is_invariant_to_affine_rescaling(self, rng):
        x = rng.standard_normal((2, 2048))
        p = SpectralParams(fs=10.0, nperseg=128)
        base, _ = coherence(stft(x, p))
        scaled = np.stack([13.0 * x[0] + 5.0, -0.02 * x[1] - 3.0])
        other, _ = coherence(stft(scaled, p))
        assert_allclose(base, other, rtol=1e-9, atol=1e-12)

    @pytest.mark.parametrize("k_target", [4, 8, 16])
    def test_null_coherence_sits_at_one_over_k(self, k_target):
        # For independent signals MSC does not go to zero; it goes to 1/K,
        # independent of bandwidth. Widening the band does not clean this up --
        # only more segments do. See derivations/coherence_estimators.md §5.
        rng = np.random.default_rng(7)
        nperseg = 64
        p = SpectralParams(fs=10.0, nperseg=nperseg)
        t = nperseg * (k_target + 1) // 2
        k_actual = p.n_segments(t)
        acc = [
            coherence(stft(rng.standard_normal((2, t)), p))[0][0, 1, 1:].mean()
            for _ in range(40)
        ]
        assert_allclose(np.mean(acc), 1.0 / k_actual, rtol=0.25)


class TestInvariants:
    def test_coherence_lies_in_the_unit_interval(self, rng):
        x = rng.standard_normal((6, 1024))
        p = SpectralParams(fs=10.0, nperseg=128)
        c, _ = coherence(stft(x, p))
        finite = c[np.isfinite(c)]
        assert finite.min() >= 0.0
        assert finite.max() <= 1.0 + 1e-12

    def test_auto_coherence_diagonal_is_one(self, rng):
        x = rng.standard_normal((5, 1024))
        p = SpectralParams(fs=10.0, nperseg=128)
        c, _ = coherence(stft(x, p))
        assert_allclose(np.einsum("nnf->nf", c)[:, 1:], 1.0, rtol=1e-10)

    def test_csd_is_hermitian_under_signal_exchange(self, rng):
        x = rng.standard_normal((4, 1024))
        p = SpectralParams(fs=10.0, nperseg=128)
        c, _ = csd(stft(x, p))
        assert_allclose(c, np.conj(np.swapaxes(c, 0, 1)), rtol=1e-10, atol=1e-14)

    def test_permuting_signals_permutes_the_matrix(self, rng):
        x = rng.standard_normal((5, 1024))
        perm = np.array([3, 0, 4, 1, 2])
        p = SpectralParams(fs=10.0, nperseg=128)
        a, _ = coherence(stft(x, p))
        b, _ = coherence(stft(np.ascontiguousarray(x[perm]), p))
        assert_allclose(b, a[np.ix_(perm, perm)], rtol=1e-12, atol=1e-14)

    def test_one_dimensional_input_is_promoted(self, rng):
        x = rng.standard_normal(1024)
        p = SpectralParams(fs=10.0, nperseg=128)
        s = stft(x, p)
        assert s.coefs.shape[0] == 1
        assert_allclose(s.coefs, stft(x[None], p).coefs, rtol=0, atol=0)

    def test_segment_view_matches_an_explicit_loop(self, rng):
        x = rng.standard_normal((2, 100))
        got = segment(x, nperseg=16, noverlap=6)
        step = 16 - 6
        for k in range(got.shape[1]):
            assert_allclose(got[:, k], x[:, k * step : k * step + 16], rtol=0, atol=0)

    def test_segment_handles_non_contiguous_input(self, rng):
        # Voxel matrices arrive as transposes; a strided view of a strided view
        # must still read the right samples.
        x = np.ascontiguousarray(rng.standard_normal((100, 2))).T
        assert not x.flags["C_CONTIGUOUS"]
        assert_allclose(segment(x, 16, 6), segment(np.ascontiguousarray(x), 16, 6))


class TestProvenance:
    def test_params_are_frozen(self):
        p = SpectralParams(fs=10.0, nperseg=64)
        with pytest.raises(Exception):
            p.fs = 20.0  # type: ignore[misc]

    def test_with_returns_a_modified_copy(self):
        p = SpectralParams(fs=10.0, nperseg=64)
        q = p.with_(nperseg=128)
        assert p.nperseg == 64 and q.nperseg == 128 and q.fs == 10.0

    def test_metadata_round_trips_the_parameters(self, rng):
        p = SpectralParams(fs=10.0, nperseg=64, band_hz=(0.5, 3.0), detrend="linear")
        m = stft(rng.standard_normal((2, 512)), p).metadata()
        assert m["fs"] == 10.0
        assert m["nperseg"] == 64
        assert m["noverlap"] == 32
        assert m["detrend"] == "linear"
        assert m["band_hz"] == [0.5, 3.0]
        assert len(m["freq_hz"]) == m["n_freqs"]

    def test_stft_carries_the_frequency_axis_rather_than_deriving_it(self, rng):
        # The pre-refactor bug (C3) was reconstructing freqs from the array
        # length. Band selection makes that impossible to do correctly.
        p = SpectralParams(fs=10.0, nperseg=64, band_hz=(1.0, 3.0))
        s = stft(rng.standard_normal((2, 512)), p)
        assert s.freqs.size == s.n_freqs
        assert s.freqs.min() >= 1.0 and s.freqs.max() <= 3.0
        expected = np.fft.rfftfreq(64, d=1 / 10.0)
        assert_allclose(s.freqs, expected[(expected >= 1.0) & (expected <= 3.0)],
                        rtol=0, atol=0)


class TestFailFast:
    """Every documented raise, checked by type and by message."""

    def test_nperseg_longer_than_record(self, rng):
        p = SpectralParams(fs=10.0, nperseg=256)
        with pytest.raises(ValueError, match="exceeds the record length"):
            stft(rng.standard_normal((2, 100)), p)

    def test_noverlap_not_less_than_nperseg(self):
        with pytest.raises(ValueError, match="must be < nperseg"):
            SpectralParams(fs=10.0, nperseg=64, noverlap=64)

    def test_negative_noverlap(self):
        with pytest.raises(ValueError, match="noverlap must be >= 0"):
            SpectralParams(fs=10.0, nperseg=64, noverlap=-1)

    def test_nfft_smaller_than_nperseg(self):
        with pytest.raises(ValueError, match="must be >= nperseg"):
            SpectralParams(fs=10.0, nperseg=64, nfft=32)

    def test_non_positive_fs(self):
        with pytest.raises(ValueError, match="fs must be positive"):
            SpectralParams(fs=0.0, nperseg=64)

    def test_unknown_detrend(self):
        with pytest.raises(ValueError, match="detrend must be"):
            SpectralParams(fs=10.0, nperseg=64, detrend="quadratic")

    def test_inverted_band(self):
        with pytest.raises(ValueError, match="low < high"):
            SpectralParams(fs=10.0, nperseg=64, band_hz=(3.0, 1.0))

    def test_band_selecting_no_bins(self, rng):
        p = SpectralParams(fs=10.0, nperseg=64, band_hz=(4.999, 4.9999))
        with pytest.raises(ValueError, match="selects no frequency bins"):
            stft(rng.standard_normal((2, 512)), p)

    def test_complex_input_with_onesided_raises_instead_of_discarding(self, rng):
        # S1: the pre-refactor code silently took .real here.
        z = rng.standard_normal((2, 512)) + 1j * rng.standard_normal((2, 512))
        with pytest.raises(ValueError, match="onesided=True requires real input"):
            stft(z, SpectralParams(fs=10.0, nperseg=64))

    def test_non_finite_input_raises(self, rng):
        x = rng.standard_normal((2, 512))
        x[0, 17] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            stft(x, SpectralParams(fs=10.0, nperseg=64))

    def test_window_array_of_wrong_length(self, rng):
        p = SpectralParams(fs=10.0, nperseg=64, window=np.ones(32))
        with pytest.raises(ValueError, match="window length"):
            stft(rng.standard_normal((2, 512)), p)

    def test_mismatched_segment_counts(self, rng):
        p = SpectralParams(fs=10.0, nperseg=64)
        a = stft(rng.standard_normal((2, 512)), p)
        b = stft(rng.standard_normal((2, 1024)), p)
        with pytest.raises(ValueError, match="segment counts differ"):
            csd(a, b)

    def test_mismatched_frequency_axes(self, rng):
        x = rng.standard_normal((2, 512))
        a = stft(x, SpectralParams(fs=10.0, nperseg=64))
        b = stft(x, SpectralParams(fs=10.0, nperseg=64, band_hz=(1.0, 3.0)))
        with pytest.raises(ValueError, match="frequency axes differ"):
            csd(a, b)

    def test_pairwise_with_unequal_signal_counts(self, rng):
        p = SpectralParams(fs=10.0, nperseg=64)
        a = stft(rng.standard_normal((2, 512)), p)
        b = stft(rng.standard_normal((3, 512)), p)
        with pytest.raises(ValueError, match="equal signal counts"):
            csd(a, b, pairwise=True)

    def test_unknown_scaling(self, rng):
        s = stft(rng.standard_normal((2, 512)), SpectralParams(fs=10.0, nperseg=64))
        with pytest.raises(ValueError, match="scaling must be"):
            psd(s, scaling="rms")  # type: ignore[arg-type]

    def test_bad_input_rank(self, rng):
        with pytest.raises(ValueError, match=r"expected \(N, T\)"):
            stft(rng.standard_normal((2, 3, 4)), SpectralParams(fs=10.0, nperseg=2))
