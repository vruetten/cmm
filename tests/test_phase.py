"""Phase 5: analytic signal, phase/amplitude, phase-triggered average.

Ported from the icampsnfr suite, where these functions were developed;
`sort_by_rastermap` stays behind there (rastermap dependency).
"""

from __future__ import annotations

import numpy as np
import pytest

from cmm import phase

FS = 1.0
T = 4096
BAND_HZ = (1 / 40.0, 1 / 10.0)


def _monochromatic(freq_hz, phase0=0.0, amp=1.0, fs=FS, t_samples=T):
    """A pure cosine at `freq_hz` with initial phase `phase0`."""
    t = np.arange(t_samples) / fs
    return amp * np.cos(2 * np.pi * freq_hz * t + phase0)


class TestAnalyticSignal:
    def test_shape_and_dtype(self):
        traces = np.random.default_rng(0).standard_normal((5, T)).astype(np.float32)
        z = phase.analytic_signal(traces, fs=FS, band_hz=BAND_HZ)
        assert z.shape == traces.shape
        assert z.dtype == np.complex64

    def test_out_of_band_signal_is_suppressed(self):
        # 0.001 Hz is well below the pass band lower edge (0.025 Hz).
        traces = _monochromatic(0.001)[None, :]
        z = phase.analytic_signal(traces, fs=FS, band_hz=BAND_HZ)
        # Envelope should be tiny; measure in the middle to avoid filtfilt edge effects.
        env_mid = np.abs(z[0, T // 4 : 3 * T // 4])
        assert env_mid.max() < 0.05, env_mid.max()

    def test_in_band_amplitude_and_phase_are_recovered(self):
        # 30 s period = 1/30 Hz, sits in the middle of the 10-40 s band.
        f = 1 / 30.0
        phi0 = 0.7
        traces = _monochromatic(f, phase0=phi0)[None, :]
        z = phase.analytic_signal(traces, fs=FS, band_hz=BAND_HZ)
        # Amplitude estimate away from the edges.
        env_mid = np.abs(z[0, T // 4 : 3 * T // 4])
        assert 0.9 < env_mid.mean() < 1.1
        # Phase should track 2πft + phi0 (mod 2π). Compare a mid sample.
        t_mid = T // 2
        expected = (2 * np.pi * f * (t_mid / FS) + phi0) % (2 * np.pi)
        got = np.angle(z[0, t_mid]) % (2 * np.pi)
        diff = min(abs(got - expected), 2 * np.pi - abs(got - expected))
        assert diff < 0.05, (got, expected, diff)

    def test_band_outside_nyquist_raises(self):
        traces = np.zeros((1, T))
        with pytest.raises(ValueError, match="outside"):
            phase.analytic_signal(traces, fs=FS, band_hz=(0.1, 0.6))

    def test_real_dtype_rejected(self):
        with pytest.raises(ValueError, match="complex type"):
            phase.analytic_signal(np.zeros((1, T)), fs=FS, band_hz=BAND_HZ,
                                  dtype=np.float64)


class TestPhaseAmplitude:
    def test_bounds(self):
        z = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex64)
        p, a = phase.phase_amplitude(z)
        # numpy uses the closed interval [-pi, pi] (angle(-1) == pi exactly).
        # Add a float32 slop on top since the values are stored in float32.
        eps = 1e-6
        assert p.min() >= -np.pi - eps and p.max() <= np.pi + eps
        assert np.allclose(a, 1.0)


class TestPhaseTriggeredAverage:
    def test_traveling_wave_is_recovered(self):
        # Two "voxels": both oscillate at 1/30 Hz but with a phase offset of π/2.
        # Seed = voxel 0. The phase-triggered average should show voxel 1
        # leading/lagging voxel 0 by π/2 in the movie.
        f = 1 / 30.0
        v0 = _monochromatic(f, phase0=0.0)
        v1 = _monochromatic(f, phase0=+np.pi / 2)
        traces = np.stack([v0, v1])
        z = phase.analytic_signal(traces, fs=FS, band_hz=BAND_HZ)

        frames, counts = phase.phase_triggered_average(
            z, seed_rows=np.array([0]), n_bins=16
        )
        assert frames.shape == (16, 2)
        assert (counts > 0).all()

        centers = phase.bin_centers(16)
        # Ideal shapes at each seed phase φ:
        #   v0 = cos(2πft + 0)      → cos(φ)
        #   v1 = cos(2πft + π/2)    → cos(φ + π/2) = -sin(φ)
        target0 = np.cos(centers)
        target1 = np.cos(centers + np.pi / 2)
        corr0 = np.corrcoef(frames[:, 0], target0)[0, 1]
        corr1 = np.corrcoef(frames[:, 1], target1)[0, 1]
        assert corr0 > 0.98, corr0
        assert corr1 > 0.98, corr1

    def test_empty_seed_raises(self):
        z = np.zeros((3, T), dtype=np.complex64)
        with pytest.raises(ValueError, match="seed_rows is empty"):
            phase.phase_triggered_average(z, seed_rows=np.array([], dtype=np.int64))

    def test_every_timepoint_lands_in_exactly_one_bin(self):
        # counts must account for all T samples. A phase that escaped the bin
        # range would be dropped silently and shrink the denominator.
        z = phase.analytic_signal(
            np.random.default_rng(4).standard_normal((4, T)), fs=FS, band_hz=BAND_HZ
        )
        for n_bins in (2, 16, 37):
            _, counts = phase.phase_triggered_average(
                z, seed_rows=np.array([0]), n_bins=n_bins
            )
            assert counts.sum() == T, (n_bins, counts.sum())

    def test_nan_in_a_seed_trace_raises_instead_of_filling_bin_zero(self):
        # np.digitize maps NaN to the same index as +pi, so the wrap would
        # file every NaN timepoint under bin 0 rather than failing.
        traces = np.random.default_rng(5).standard_normal((4, T))
        traces[1, T // 2] = np.nan
        z = np.asarray(traces, dtype=np.complex128)  # skip the filter: it raises first
        with pytest.raises(ValueError, match="not finite"):
            phase.phase_triggered_average(z, seed_rows=np.array([0, 1]), n_bins=16)

    def test_non_finite_outside_the_seed_is_tolerated(self):
        traces = np.random.default_rng(6).standard_normal((4, T))
        z = np.asarray(traces, dtype=np.complex128)
        z[3, 10] = np.nan  # row 3 is not a seed row
        _, counts = phase.phase_triggered_average(z, seed_rows=np.array([0]), n_bins=16)
        assert counts.sum() == T

    def test_infinite_seed_raises_even_though_its_angle_is_finite(self):
        # angle(inf + 0j) == 0.0, so a seed_phase-only check would pass this
        # while the amplitude weight is inf and would poison the mean.
        z = phase.analytic_signal(
            np.random.default_rng(7).standard_normal((3, T)), fs=FS, band_hz=BAND_HZ
        ).astype(np.complex128)
        z[:, 5] = complex(np.inf, 0.0)
        assert np.isfinite(np.angle(z[:, 5])).all()  # the guard cannot rely on angle
        with pytest.raises(ValueError, match="not finite"):
            phase.phase_triggered_average(z, seed_rows=np.array([0, 1]), n_bins=16)

    def test_random_seed_suppresses_phase_locked_target(self):
        # White-noise seed whose bandpassed analytic signal has phase that is
        # essentially uncorrelated with the target sinusoid. Averaging the
        # target's cosine by these random phase bins should cancel most of
        # the oscillation.
        f = 1 / 30.0
        target = _monochromatic(f, phase0=0.0)
        noise = np.random.default_rng(2).standard_normal(target.size).astype(np.float32)
        traces = np.stack([target, noise])
        z = phase.analytic_signal(traces, fs=FS, band_hz=BAND_HZ)

        coh, _ = phase.phase_triggered_average(z, seed_rows=np.array([0]), n_bins=16)
        inc, _ = phase.phase_triggered_average(z, seed_rows=np.array([1]), n_bins=16)
        coh_range = np.nanmax(coh[:, 0]) - np.nanmin(coh[:, 0])
        inc_range = np.nanmax(inc[:, 0]) - np.nanmin(inc[:, 0])
        # Not exactly zero because a finite-length noise sample has residual
        # coherence with any target, but well under half the coherent amplitude.
        assert inc_range < 0.5 * coh_range, (coh_range, inc_range)

    def test_precision_follows_input(self):
        z = phase.analytic_signal(
            np.random.default_rng(8).standard_normal((2, T)), fs=FS, band_hz=BAND_HZ,
            dtype=np.complex128,
        )
        frames, _ = phase.phase_triggered_average(z, seed_rows=np.array([0]))
        assert frames.dtype == np.float64
        frames32, _ = phase.phase_triggered_average(
            z.astype(np.complex64), seed_rows=np.array([0])
        )
        assert frames32.dtype == np.float32

    def test_bad_ndim_raises(self):
        with pytest.raises(ValueError, match=r"expected \(N, T\)"):
            phase.phase_triggered_average(
                np.zeros(T, dtype=np.complex64), seed_rows=np.array([0])
            )

    def test_too_few_bins_raises(self):
        z = np.ones((2, T), dtype=np.complex64)
        with pytest.raises(ValueError, match="n_bins must be >= 2"):
            phase.phase_triggered_average(z, seed_rows=np.array([0]), n_bins=1)


class TestBinCenters:
    def test_bin_edges_are_uniform_on_the_circle(self):
        centers = phase.bin_centers(16)
        assert centers.shape == (16,)
        assert centers.min() >= -np.pi
        assert centers.max() < np.pi
        assert np.allclose(np.diff(centers), 2 * np.pi / 16)


class TestValidRowMask:
    def test_all_nan_and_flat_rows_are_invalid(self):
        rng = np.random.default_rng(3)
        n_bins = 16
        # 5 varying rows (real signal): cosines with different phase offsets.
        centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
        varying = np.stack(
            [np.cos(centers + i * 0.4) for i in range(5)]
        ).astype(np.float32)
        varying = varying + 0.01 * rng.standard_normal(varying.shape).astype(np.float32)
        nans = np.full((5, n_bins), np.nan, dtype=np.float32)
        matrix = np.concatenate([varying, nans], axis=0)

        mask = phase.valid_row_mask(matrix)
        assert mask.shape == (10,)
        assert mask.dtype == bool
        assert mask[:5].all()
        assert not mask[5:].any()

    def test_all_nan_matrix_yields_all_invalid(self):
        matrix = np.full((7, 4), np.nan, dtype=np.float32)
        mask = phase.valid_row_mask(matrix)
        assert not mask.any()

    def test_flat_constant_row_is_invalid(self):
        # A row of constant 3.0 has zero variance — flagged invalid alongside
        # a genuinely varying row.
        matrix = np.stack([
            np.array([0.1, 0.2, -0.3, 0.5], dtype=np.float32),
            np.full(4, 3.0, dtype=np.float32),
        ])
        mask = phase.valid_row_mask(matrix)
        assert mask[0]
        assert not mask[1]

    def test_bad_ndim_raises(self):
        with pytest.raises(ValueError, match=r"expected \(N, n_bins\)"):
            phase.valid_row_mask(np.zeros(8, dtype=np.float32))
