"""Layer 1: cross-validation of cmm.spectral against scipy.signal.

scipy decides, not our arithmetic. This layer exists because the pre-refactor
code's window normalization was wrong by 21.3x under ``scaling="spectrum"`` and
missing the one-sided doubling under both scalings, and nothing caught it for
years. The parameter sweep below is deliberately wider than any single call
site needs.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import signal as sps

from cmm.spectral import SpectralParams, coherence, csd, psd, stft

RTOL = 1e-12

WINDOWS = ["hann", "hamming", "boxcar", ("tukey", 0.25)]
NPERSEGS = [32, 64, 128]
DETRENDS = ["constant", "linear", False]
SCALINGS = ["density", "spectrum"]


def assert_spectra_close(ours, ref, rtol=RTOL):
    """Compare with an absolute floor scaled to the reference's magnitude.

    A pure relative comparison is meaningless for bins that are numerically
    zero: detrending drives the DC bin of a boxcar-windowed segment to ~1e-32,
    where two correct implementations differ by 5 % of dust.
    """
    ref = np.asarray(ref)
    assert_allclose(ours, ref, rtol=rtol, atol=rtol * np.abs(ref).max())


def _sp_kwargs(p: SpectralParams, *, onesided: bool = True):
    kw = dict(
        fs=p.fs,
        window=p.window,
        nperseg=p.nperseg,
        noverlap=p.resolved_noverlap,
        nfft=p.nfft,
        detrend=p.detrend,
        axis=-1,
    )
    if onesided:  # scipy.signal.coherence does not accept return_onesided
        kw["return_onesided"] = p.onesided
    return kw


@pytest.fixture(scope="module")
def traces():
    rng = np.random.default_rng(0)
    n, t = 5, 1024
    tt = np.arange(t) / 20.0
    x = rng.standard_normal((n, t))
    x[0] += 3 * np.sin(2 * np.pi * 1.3 * tt)
    x[1] += 3 * np.sin(2 * np.pi * 1.3 * tt + 0.7)  # coherent with x[0]
    x += np.linspace(0, 2, t)  # a trend for detrend='linear' to bite on
    return x


@pytest.fixture(scope="module")
def traces_rev(traces):
    return np.ascontiguousarray(traces[::-1])


class TestPsdVsWelch:
    @pytest.mark.parametrize("window", WINDOWS)
    @pytest.mark.parametrize("scaling", SCALINGS)
    def test_matches_welch_across_windows_and_scalings(self, traces, window, scaling):
        p = SpectralParams(fs=20.0, nperseg=64, noverlap=32, window=window)
        ours, f = psd(stft(traces, p), scaling=scaling)
        f_sp, ref = sps.welch(traces, scaling=scaling, **_sp_kwargs(p))
        assert_allclose(f, f_sp, rtol=RTOL)
        assert_spectra_close(ours, ref)

    @pytest.mark.parametrize("nperseg", NPERSEGS)
    @pytest.mark.parametrize("noverlap_frac", [0.0, 0.5, 0.75])
    def test_matches_welch_across_segmentation(self, traces, nperseg, noverlap_frac):
        p = SpectralParams(
            fs=20.0, nperseg=nperseg, noverlap=int(noverlap_frac * nperseg)
        )
        ours, f = psd(stft(traces, p))
        f_sp, ref = sps.welch(traces, scaling="density", **_sp_kwargs(p))
        assert_allclose(f, f_sp, rtol=RTOL)
        assert_spectra_close(ours, ref)

    @pytest.mark.parametrize("detrend", DETRENDS)
    def test_matches_welch_across_detrend(self, traces, detrend):
        p = SpectralParams(fs=20.0, nperseg=64, detrend=detrend)
        ours, _ = psd(stft(traces, p))
        _, ref = sps.welch(traces, scaling="density", **_sp_kwargs(p))
        assert_spectra_close(ours, ref)

    def test_matches_welch_with_zero_padding(self, traces):
        p = SpectralParams(fs=20.0, nperseg=64, nfft=256)
        ours, f = psd(stft(traces, p))
        f_sp, ref = sps.welch(traces, scaling="density", **_sp_kwargs(p))
        assert_allclose(f, f_sp, rtol=RTOL)
        assert_spectra_close(ours, ref)

    @pytest.mark.parametrize("nfft", [65, 129])
    def test_matches_welch_odd_nfft_has_no_nyquist_bin(self, traces, nfft):
        # With odd nfft every bin except DC is doubled; with even nfft the last
        # bin is unpaired Nyquist and must not be. Both paths need covering.
        p = SpectralParams(fs=20.0, nperseg=64, nfft=nfft)
        ours, _ = psd(stft(traces, p))
        _, ref = sps.welch(traces, scaling="density", **_sp_kwargs(p))
        assert_spectra_close(ours, ref)

    def test_matches_welch_twosided_complex_input(self):
        rng = np.random.default_rng(3)
        z = rng.standard_normal((3, 512)) + 1j * rng.standard_normal((3, 512))
        p = SpectralParams(fs=20.0, nperseg=64, onesided=False)
        ours, f = psd(stft(z, p))
        f_sp, ref = sps.welch(z, scaling="density", **_sp_kwargs(p))
        assert_allclose(f, f_sp, rtol=RTOL)
        assert_spectra_close(ours, ref)


class TestCsdVsScipy:
    @pytest.mark.parametrize("scaling", SCALINGS)
    def test_pairwise_matches_scipy_csd(self, traces, traces_rev, scaling):
        p = SpectralParams(fs=20.0, nperseg=64)
        ours, f = csd(stft(traces, p), stft(traces_rev, p), scaling=scaling, pairwise=True)
        f_sp, ref = sps.csd(traces, traces_rev, scaling=scaling, **_sp_kwargs(p))
        assert_allclose(f, f_sp, rtol=RTOL)
        assert_spectra_close(ours, ref)

    def test_auto_csd_equals_psd(self, traces):
        p = SpectralParams(fs=20.0, nperseg=64)
        s = stft(traces, p)
        c, _ = csd(s, pairwise=True)
        pxx, _ = psd(s)
        assert_spectra_close(c.real, pxx)
        assert_allclose(c.imag, 0.0, atol=1e-18)

    def test_alltoall_diagonal_is_the_pairwise_result(self, traces):
        p = SpectralParams(fs=20.0, nperseg=64)
        s = stft(traces, p)
        full, _ = csd(s)
        diag = np.einsum("nnf->nf", full)
        pair, _ = csd(s, pairwise=True)
        assert_spectra_close(diag, pair)

    def test_alltoall_offdiagonal_matches_scipy_directly(self, traces):
        # The diagonal cannot tell which operand is conjugated (conj(X)*X is
        # symmetric in that choice), and Hermitian symmetry survives a global
        # swap too. Only a distinct pair against scipy pins the phase sign,
        # which is what delay direction is inferred from.
        p = SpectralParams(fs=20.0, nperseg=64)
        full, f = csd(stft(traces, p))
        f_sp, ref = sps.csd(traces[0], traces[2], scaling="density", **_sp_kwargs(p))
        assert_allclose(f, f_sp, rtol=RTOL)
        assert_spectra_close(full[0, 2], ref)


class TestCoherenceVsScipy:
    @pytest.mark.parametrize("window", WINDOWS)
    @pytest.mark.parametrize("nperseg", NPERSEGS)
    def test_matches_scipy_coherence(self, traces, traces_rev, window, nperseg):
        p = SpectralParams(fs=20.0, nperseg=nperseg, window=window)
        ours, f = coherence(stft(traces, p), stft(traces_rev, p), pairwise=True)
        f_sp, ref = sps.coherence(traces, traces_rev, **_sp_kwargs(p, onesided=False))
        assert_allclose(f, f_sp, rtol=RTOL)
        # detrend='constant' removes the mean, so the DC bin carries no signal.
        # With a boxcar window it is driven to ~1e-32 and both implementations
        # are dividing dust by dust; demanding they agree there tests nothing.
        # Every other bin must match to 1e-12.
        informative = f > 0
        assert_allclose(ours[:, informative], ref[:, informative], rtol=RTOL, atol=1e-12)

    @pytest.mark.parametrize("window", WINDOWS)
    def test_degenerate_dc_bin_stays_in_range(self, traces, traces_rev, window):
        # Whatever the DC bin contains, it must remain a coherence: finite and
        # within [0, 1]. This is the guard that the 0/0 path cannot emit inf.
        p = SpectralParams(fs=20.0, nperseg=64, window=window)
        ours, _ = coherence(stft(traces, p), stft(traces_rev, p), pairwise=True)
        dc = ours[:, 0]
        assert np.isfinite(dc).all()
        assert ((dc >= 0.0) & (dc <= 1.0)).all()

    def test_exactly_zero_power_gives_nan_not_noise(self):
        # A constant trace has zero power at every bin after detrending.
        p = SpectralParams(fs=20.0, nperseg=64)
        x = np.zeros((2, 512))
        x[1] = np.random.default_rng(0).standard_normal(512)
        c, _ = coherence(stft(x, p), pairwise=False)
        assert np.isnan(c[0, 0]).all(), "dead-vs-dead should be undefined, not 0/0 noise"
        assert np.isnan(c[0, 1]).all(), "dead-vs-live should be undefined too"
        assert np.isfinite(c[1, 1]).all()

    def test_scaling_cancels_in_the_ratio(self, traces):
        # The reason coherence was correct pre-refactor while the PSD was not.
        p = SpectralParams(fs=20.0, nperseg=64)
        c, _ = coherence(stft(traces, p))
        c2, _ = coherence(stft(traces, p.with_(fs=p.fs * 7.0)))
        assert_allclose(c, c2, rtol=1e-12, atol=1e-14)


class TestStftCoefficientsVsScipy:
    def test_coefficients_match_scipy_stft_up_to_its_scaling(self, traces):
        p = SpectralParams(fs=20.0, nperseg=64, noverlap=32, detrend=False)
        s = stft(traces, p)
        f_sp, _, z = sps.stft(
            traces, fs=p.fs, window=p.window, nperseg=p.nperseg,
            noverlap=p.resolved_noverlap, detrend=False, return_onesided=True,
            boundary=None, padded=False, scaling="spectrum",
        )
        # scipy.signal.stft applies 1/win.sum() and returns (..., F, K).
        ours = np.moveaxis(s.coefs, -1, -2) / s.window.sum()
        assert_allclose(s.freqs, f_sp, rtol=RTOL)
        assert_allclose(ours, z, rtol=1e-11, atol=1e-13 * np.abs(z).max())


class TestRegressionsAgainstTheAudit:
    """Each of these failed before the rewrite, with the measured value named."""

    def test_spectrum_scaling_is_not_the_density_formula(self, traces):
        # C2: pre-refactor "spectrum" was off by win.sum()**2/(win**2).sum(),
        # measured at 21.3x for hann/64.
        p = SpectralParams(fs=20.0, nperseg=64)
        s = stft(traces, p)
        spec, _ = psd(s, scaling="spectrum")
        dens, _ = psd(s, scaling="density")
        win = s.window
        expected = (p.fs * (win * win).sum()) / win.sum() ** 2
        assert_allclose(spec / dens, expected, rtol=RTOL)

    def test_onesided_doubling_is_applied_to_interior_bins_only(self, traces):
        # C2: the missing factor of 2 left DC and Nyquist 2x high *relative* to
        # the interior, so this is a shape error, not a scale error.
        p = SpectralParams(fs=20.0, nperseg=64)
        one, f1 = psd(stft(traces, p))
        two, f2 = psd(stft(traces, p.with_(onesided=False)))
        # fftfreq places Nyquist at -fs/2, so match on |f|.
        nyq = int(np.argmin(np.abs(np.abs(f2) - p.fs / 2)))
        assert_spectra_close(one[:, 0], two[:, 0])            # DC not doubled
        assert_spectra_close(one[:, -1], two[:, nyq])         # Nyquist not doubled
        interior = 5
        assert_spectra_close(one[:, interior], 2 * two[:, interior])
        assert f1[-1] == pytest.approx(p.fs / 2)

    def test_frequency_axis_survives_band_selection(self, traces):
        # C3: the old code rebuilt freqs from the coefficient count as
        # rfftfreq(2F-1), giving 0.1538 Hz spacing where the truth was 0.15625.
        p = SpectralParams(fs=20.0, nperseg=64)
        full = stft(traces, p)
        band = stft(traces, p.with_(band_hz=(1.0, 5.0)))
        keep = (full.freqs >= 1.0) & (full.freqs <= 5.0)
        assert_allclose(band.freqs, full.freqs[keep], rtol=0, atol=0)
        assert_allclose(band.coefs, full.coefs[..., keep], rtol=0, atol=0)

    def test_band_selected_psd_equals_the_slice_of_the_full_psd(self, traces):
        p = SpectralParams(fs=20.0, nperseg=64)
        full, f = psd(stft(traces, p))
        band, fb = psd(stft(traces, p.with_(band_hz=(1.0, 5.0))))
        keep = (f >= 1.0) & (f <= 5.0)
        assert_spectra_close(band, full[:, keep])
        assert_allclose(fb, f[keep], rtol=0, atol=0)
