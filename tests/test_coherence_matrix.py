"""Phase 2: the band-averaged coherence matrix.

The layer that matters most here is the naive reference implementation. The
production path is blocked, threaded, float32 and reorganized in memory; a
deliberately slow pair-by-pair loop written for readability is what proves
none of that changed the answer.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cmm.coherence import (
    CoherenceMatrix,
    coherence_matrix,
    debias,
    effective_bins,
    finite_power_mask,
    to_distance,
)
from cmm.spectral import SpectralParams, stft


# --------------------------------------------------------------------------
# Naive references. Slow, obvious, and written from the formulas in
# derivations/coherence_estimators.md rather than from the implementation.
# --------------------------------------------------------------------------
def naive_v2(coefs: np.ndarray) -> np.ndarray:
    """C[n,m] = sum_f |P_nm(f)|^2 / sum_f P_nn(f) P_mm(f), pair by pair."""
    n, _, nf = coefs.shape
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            num = den = 0.0
            for f in range(nf):
                pij = np.vdot(coefs[j, :, f], coefs[i, :, f])  # sum_k Xi conj(Xj)
                pii = np.vdot(coefs[i, :, f], coefs[i, :, f]).real
                pjj = np.vdot(coefs[j, :, f], coefs[j, :, f]).real
                num += abs(pij) ** 2
                den += pii * pjj
            out[i, j] = num / den
    return out


def naive_v1(coefs: np.ndarray) -> np.ndarray:
    """C[n,m] = mean_f |P_nm(f)|^2 / (P_nn(f) P_mm(f)), pair by pair."""
    n, _, nf = coefs.shape
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            acc = 0.0
            for f in range(nf):
                pij = np.vdot(coefs[j, :, f], coefs[i, :, f])
                pii = np.vdot(coefs[i, :, f], coefs[i, :, f]).real
                pjj = np.vdot(coefs[j, :, f], coefs[j, :, f]).real
                acc += abs(pij) ** 2 / (pii * pjj)
            out[i, j] = acc / nf
    return out


@pytest.fixture(scope="module")
def small_stft():
    rng = np.random.default_rng(11)
    n, t = 9, 768
    tt = np.arange(t) / 4.0
    x = rng.standard_normal((n, t))
    for i in range(0, n, 3):  # three groups sharing a source
        x[i : i + 3] += 2.0 * np.sin(2 * np.pi * 0.35 * tt + i)
    p = SpectralParams(fs=4.0, nperseg=128, band_hz=(0.1, 1.0))
    return stft(x, p)


class TestAgainstNaiveReference:
    def test_v2_matches_the_naive_loop(self, small_stft):
        got = coherence_matrix(small_stft, dtype=np.complex128)
        assert_allclose(got.values, naive_v2(small_stft.coefs), rtol=1e-10, atol=1e-12)

    def test_v1_matches_the_naive_loop(self, small_stft):
        got = coherence_matrix(small_stft, whiten=True, dtype=np.complex128)
        assert_allclose(got.values, naive_v1(small_stft.coefs), rtol=1e-10, atol=1e-12)

    def test_whitening_identity_holds(self, small_stft):
        # coherence_estimators.md §3: V2 on whitened coefficients IS V1.
        v1 = coherence_matrix(small_stft, whiten=True, dtype=np.complex128).values
        d = np.einsum("nkf,nkf->nf", small_stft.coefs, small_stft.coefs.conj()).real
        pre = dataclasses.replace(
            small_stft, coefs=small_stft.coefs / np.sqrt(d)[:, None, :]
        )
        v2_of_whitened = coherence_matrix(pre, dtype=np.complex128).values
        assert_allclose(v1, v2_of_whitened, rtol=1e-10, atol=1e-12)

    def test_v1_and_v2_actually_differ(self, small_stft):
        # If these agreed, the whole estimator discussion would be moot.
        v1 = coherence_matrix(small_stft, whiten=True, dtype=np.complex128).values
        v2 = coherence_matrix(small_stft, dtype=np.complex128).values
        assert np.abs(v1 - v2).max() > 1e-3


class TestBlockingAndThreading:
    @pytest.mark.parametrize("block", [1, 2, 3, 5, 64])
    def test_block_size_does_not_change_the_answer(self, small_stft, block):
        # Not bit-exact: BLAS picks a different internal summation order for
        # differently-shaped gemms, worth 2-5e-16 here. Anything larger would
        # mean the blocking itself is wrong, not the rounding.
        ref = coherence_matrix(small_stft, dtype=np.complex128, block=1024).values
        got = coherence_matrix(small_stft, dtype=np.complex128, block=block).values
        assert_allclose(got, ref, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("workers", [1, 2, 8])
    def test_worker_count_is_bit_exact(self, small_stft, workers):
        # Stronger than the block-size case and non-negotiable: each row block
        # is written by exactly one thread, so scheduling must not be
        # observable in the output at all.
        ref = coherence_matrix(
            small_stft, dtype=np.complex128, block=2, workers=1
        ).values
        got = coherence_matrix(
            small_stft, dtype=np.complex128, block=2, workers=workers
        ).values
        assert_allclose(got, ref, rtol=0, atol=0)

    def test_float32_matches_float64_within_its_precision(self, small_stft):
        c64 = coherence_matrix(small_stft, dtype=np.complex64).values
        c128 = coherence_matrix(small_stft, dtype=np.complex128).values
        assert np.abs(c64 - c128).max() < 1e-6


class TestMatrixProperties:
    def test_symmetric_unit_diagonal_and_bounded(self, small_stft):
        c = coherence_matrix(small_stft)
        assert_allclose(np.diag(c.values), 1.0, rtol=0, atol=0)
        assert_allclose(c.values, c.values.T, rtol=0, atol=0)
        assert c.values.min() >= 0.0
        assert c.values.max() <= 1.0 + 1e-6

    def test_permutation_equivariance(self, small_stft):
        perm = np.array([4, 0, 7, 2, 8, 1, 5, 3, 6])
        base = coherence_matrix(small_stft, dtype=np.complex128).values
        permuted = coherence_matrix(
            dataclasses.replace(small_stft, coefs=small_stft.coefs[perm]),
            dtype=np.complex128,
        ).values
        assert_allclose(permuted, base[np.ix_(perm, perm)], rtol=1e-12, atol=1e-12)

    def test_invariant_to_per_signal_rescaling(self, small_stft):
        # Coherence measures shared structure, not amplitude.
        rng = np.random.default_rng(2)
        gains = rng.uniform(0.01, 100.0, size=small_stft.n_signals)
        base = coherence_matrix(small_stft, dtype=np.complex128).values
        scaled = coherence_matrix(
            dataclasses.replace(
                small_stft, coefs=small_stft.coefs * gains[:, None, None]
            ),
            dtype=np.complex128,
        ).values
        assert_allclose(scaled, base, rtol=1e-9, atol=1e-11)

    def test_identical_signals_have_coherence_one(self, small_stft):
        dup = dataclasses.replace(
            small_stft, coefs=np.concatenate([small_stft.coefs, small_stft.coefs[:1]])
        )
        c = coherence_matrix(dup, dtype=np.complex128).values
        assert_allclose(c[0, -1], 1.0, rtol=1e-10)

    def test_single_signal_gives_a_one_by_one_identity(self, small_stft):
        s = dataclasses.replace(small_stft, coefs=small_stft.coefs[:1])
        c = coherence_matrix(s, dtype=np.complex128)
        assert c.values.shape == (1, 1)
        assert c.values[0, 0] == 1.0

    def test_single_frequency_bin_reduces_to_per_bin_msc(self, small_stft):
        # The boundary where "band-averaged" degenerates: with one bin the
        # frequency weighting has nothing to weight, so V1 and V2 coincide
        # and both equal the plain per-bin ratio.
        s = dataclasses.replace(
            small_stft,
            coefs=small_stft.coefs[..., :1],
            freqs=small_stft.freqs[:1],
            onesided_factor=small_stft.onesided_factor[:1],
        )
        v2 = coherence_matrix(s, dtype=np.complex128).values
        v1 = coherence_matrix(s, whiten=True, dtype=np.complex128).values
        assert_allclose(v2, naive_v2(s.coefs), rtol=1e-10, atol=1e-12)
        assert_allclose(v1, v2, rtol=1e-10, atol=1e-12)

    def test_delay_invariance(self):
        # The property that ruled out coherent averaging over frequency.
        rng = np.random.default_rng(5)
        x = rng.standard_normal(4096)
        lag = 11
        p = SpectralParams(fs=10.0, nperseg=256, band_hz=(0.2, 2.0))
        s = stft(np.stack([x[lag:], x[:-lag]]), p)
        c = coherence_matrix(s, dtype=np.complex128).values
        assert c[0, 1] > 0.95, c[0, 1]


class TestBiasFloorAndDebias:
    @pytest.mark.parametrize("k_target", [4, 8, 16])
    def test_null_floor_is_one_over_k_not_one_over_kf(self, k_target):
        # The whole reason debias() takes K and not K*F.
        rng = np.random.default_rng(3)
        nperseg = 64
        t = nperseg * (k_target + 1) // 2
        p = SpectralParams(fs=10.0, nperseg=nperseg, band_hz=(0.5, 4.0))
        offdiag = []
        for _ in range(30):
            s = stft(rng.standard_normal((4, t)), p)
            c = coherence_matrix(s, dtype=np.complex128)
            v = c.values[~np.eye(4, dtype=bool)]
            offdiag.append(v.mean())
        k = stft(rng.standard_normal((4, t)), p).n_segments
        assert_allclose(np.mean(offdiag), 1.0 / k, rtol=0.3)

    def test_bias_floor_property_reports_one_over_k(self, small_stft):
        c = coherence_matrix(small_stft)
        assert c.bias_floor == pytest.approx(1.0 / small_stft.n_segments)
        assert c.n_eff == small_stft.n_segments

    def test_debias_uses_the_matrices_own_n_eff(self, small_stft):
        c = coherence_matrix(small_stft)
        assert_allclose(debias(c), debias(c.values, n_eff=c.n_eff))

    def test_debias_maps_the_floor_to_zero(self):
        c = np.array([[1.0, 0.25], [0.25, 1.0]])
        assert debias(c, n_eff=4)[0, 1] == pytest.approx(0.0)

    def test_debias_clamps_and_is_monotone(self):
        c = np.array([[1.0, 0.01], [0.01, 1.0]])
        out = debias(c, n_eff=10)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_debias_with_wrong_n_eff_under_corrects(self):
        # Passing K*F instead of K is the live bug in icampsnfr's call site.
        c = np.full((2, 2), 0.1)
        np.fill_diagonal(c, 1.0)
        right = debias(c, n_eff=10)[0, 1]
        wrong = debias(c, n_eff=10 * 15)[0, 1]
        assert right < wrong
        assert right == pytest.approx(0.0, abs=1e-12)

    def test_debias_rejects_bare_array_without_n_eff(self):
        with pytest.raises(ValueError, match="n_eff is required"):
            debias(np.eye(2))

    def test_debias_matches_the_formula_via_bias_floor(self, small_stft):
        # A second, independent line of defense for the K-not-K*F rule: build
        # the expectation from the container's bias_floor property, which does
        # not share debias()'s own n_eff resolution path.
        c = coherence_matrix(small_stft)
        b = c.bias_floor
        expected = np.clip((c.values - b) / (1.0 - b), 0.0, 1.0)
        assert_allclose(debias(c), expected, rtol=0, atol=0)


class TestEffectiveBins:
    def test_flat_spectrum_uses_every_bin(self):
        p = np.ones((4, 12))
        assert_allclose(effective_bins(p), 12.0, rtol=1e-12)

    def test_one_dominant_bin_collapses_to_one(self):
        p = np.zeros((2, 10))
        p[:, 0] = 1.0
        p[:, 1:] = 1e-8
        assert effective_bins(p)[0, 1] < 1.05

    def test_power_law_tilt_reduces_effective_bins(self):
        # derivations/spectral_tilt_and_band_collapse.md §2.
        f = np.linspace(1.0, 4.0, 9)
        flat = effective_bins(np.ones((2, 9)))[0, 1]
        tilted = effective_bins(np.stack([f**-2.0, f**-2.0]))[0, 1]
        assert tilted < 0.5 * flat

    def test_whitened_matrix_reports_full_bin_count(self, small_stft):
        c = coherence_matrix(small_stft, whiten=True)
        assert c.f_eff == float(small_stft.n_freqs)


class TestDistance:
    def test_diagonal_is_exactly_zero(self, small_stft):
        d = to_distance(coherence_matrix(small_stft).values)
        assert_allclose(np.diag(d), 0.0, rtol=0, atol=0)

    def test_distance_is_reversed_coherence(self):
        c = np.array([[1.0, 0.25], [0.25, 1.0]])
        assert to_distance(c, use_sqrt=True)[0, 1] == pytest.approx(0.5)
        assert to_distance(c, use_sqrt=False)[0, 1] == pytest.approx(0.75)

    def test_more_coherent_pairs_are_closer(self, small_stft):
        c = coherence_matrix(small_stft).values
        d = to_distance(c)
        iu = np.triu_indices_from(c, k=1)
        assert np.corrcoef(c[iu], d[iu])[0, 1] < -0.99

    def test_rejects_the_container_by_mistake(self, small_stft):
        c = coherence_matrix(small_stft)
        with pytest.raises(TypeError, match="CoherenceMatrix.values"):
            to_distance(c)  # type: ignore[arg-type]


class TestFailFast:
    def test_zero_power_signal_raises_with_guidance(self, small_stft):
        coefs = small_stft.coefs.copy()
        coefs[3] = 0.0
        with pytest.raises(ValueError, match="finite_power_mask"):
            coherence_matrix(dataclasses.replace(small_stft, coefs=coefs))

    def test_finite_power_mask_identifies_the_dead_signal(self, small_stft):
        coefs = small_stft.coefs.copy()
        coefs[3] = 0.0
        keep = finite_power_mask(dataclasses.replace(small_stft, coefs=coefs))
        assert not keep[3]
        assert keep.sum() == small_stft.n_signals - 1

    def test_filtering_then_computing_succeeds(self, small_stft):
        coefs = small_stft.coefs.copy()
        coefs[3] = 0.0
        s = dataclasses.replace(small_stft, coefs=coefs)
        keep = finite_power_mask(s)
        c = coherence_matrix(dataclasses.replace(s, coefs=s.coefs[keep]))
        assert c.n_signals == small_stft.n_signals - 1
        assert np.isfinite(c.values).all()

    @pytest.mark.parametrize("bad", [0, 1])
    def test_debias_rejects_degenerate_n_eff(self, bad):
        # n_eff=1 is not merely small: every raw coherence is trivially 1.0
        # at a single segment, so "debiasing" it would return garbage as if
        # it were valid output.
        with pytest.raises(ValueError, match="n_eff must be >= 2"):
            debias(np.eye(2), n_eff=bad)

    def test_single_dead_bin_raises_not_just_a_fully_dead_signal(self, small_stft):
        # The realistic case: a signal alive everywhere except one in-band
        # bin. A weakened check (.any for .all) passes every other test in
        # this file — this is the one that pins the condition's precision.
        coefs = small_stft.coefs.copy()
        coefs[2, :, 1] = 0.0
        with pytest.raises(ValueError, match="zero power in at least one"):
            coherence_matrix(dataclasses.replace(small_stft, coefs=coefs))

    def test_mask_applies_the_same_per_bin_criterion(self, small_stft):
        # Whatever finite_power_mask keeps, coherence_matrix must accept —
        # otherwise its documented filter-then-compute workflow still raises.
        coefs = small_stft.coefs.copy()
        coefs[2, :, 1] = 0.0
        s = dataclasses.replace(small_stft, coefs=coefs)
        keep = finite_power_mask(s)
        assert not keep[2]
        c = coherence_matrix(dataclasses.replace(s, coefs=s.coefs[keep]))
        assert c.n_signals == small_stft.n_signals - 1
        assert np.isfinite(c.values).all()

    def test_wrong_coefficient_rank(self, small_stft):
        with pytest.raises(ValueError, match=r"expected \(N, K, F\)"):
            coherence_matrix(dataclasses.replace(small_stft, coefs=np.ones((3, 4))))


class TestProvenance:
    def test_metadata_records_the_estimator_and_the_floor(self, small_stft):
        m = coherence_matrix(small_stft).metadata()
        assert m["estimator"].startswith("V2")
        assert m["n_eff"] == small_stft.n_segments
        assert m["bias_floor"] == pytest.approx(1.0 / small_stft.n_segments)
        assert "median" in m["f_eff"]
        assert m["stft"]["band_hz"] == [0.1, 1.0]

    def test_whitened_metadata_says_v1(self, small_stft):
        md = coherence_matrix(small_stft, whiten=True).metadata()
        assert md["estimator"].startswith("V1")

    def test_metadata_f_eff_schema_is_stable(self, small_stft):
        # Same keys however the matrix was made, so a saved-JSON consumer
        # never branches on compute_f_eff or whiten.
        for kwargs in ({}, {"whiten": True}, {"compute_f_eff": False}):
            m = coherence_matrix(small_stft, **kwargs).metadata()
            assert set(m["f_eff"]) == {"median", "min"}
        m = coherence_matrix(small_stft, whiten=True).metadata()
        assert m["f_eff"]["median"] == float(small_stft.n_freqs)

    def test_result_is_immutable(self, small_stft):
        c = coherence_matrix(small_stft)
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.n_eff = 3  # type: ignore[misc]
        assert isinstance(c, CoherenceMatrix)
