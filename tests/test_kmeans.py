"""Phase 3 chunk 3: coherence k-means.

The only spec is known-broken code, so correctness is triangulated instead of
copied: a naive per-cluster eigh loop, allocation validated *through the
scipy-pinned cmm.spectral.coherence* (proving the |v^H x_hat|^2 == coherence
reduction rather than assuming it), objective monotonicity (a theorem about
block-coordinate ascent, independent of any reference implementation),
planted-structure recovery, invariances, and cross-route agreement with
spectral clustering.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import eigh as scipy_eigh
from sklearn.metrics import adjusted_rand_score

from cmm.clustering import spectral_labels
from cmm.coherence import coherence_matrix
from cmm.kmeans import (
    CoherenceKMeans,
    CoherenceKMeansResult,
    centroid_scores,
    centroid_update,
    coherence_kmeans,
    normalize_coefs,
)
from cmm.spectral import SpectralParams, coherence, stft
from cmm.synthetic import bandlimited_source, grouped_traces


PARAMS = SpectralParams(fs=8.0, nperseg=256, band_hz=(0.2, 1.5))


@pytest.fixture(scope="module")
def grouped():
    x, truth = grouped_traces([8, 8, 8])
    return stft(x, PARAMS), truth


class TestAgainstNaiveReference:
    def test_centroids_match_a_scipy_eigh_loop(self, grouped):
        s, truth = grouped
        xhat = normalize_coefs(s.coefs)
        cen, strength = centroid_update(xhat, truth, 3)
        n, k, f = xhat.shape
        for m in range(3):
            members = xhat[truth == m]
            for fi in range(0, f, 7):
                g = np.zeros((k, k), dtype=np.complex128)
                for x_nk in members[..., fi]:
                    g += np.outer(x_nk, x_nk.conj()) / f
                vals, vecs = scipy_eigh(g)
                v_naive = vecs[:, -1]
                overlap = abs(np.vdot(v_naive, cen[m, :, fi]))
                assert overlap == pytest.approx(1.0, abs=1e-10)
                assert strength[m, fi] == pytest.approx(
                    vals[-1] / vals.sum(), rel=1e-10
                )

    def test_scores_are_coherence_via_the_scipy_pinned_path(self, grouped):
        # The reduction the whole algorithm rests on: because centroid and
        # x_hat are unit vectors per frequency, the score IS the mean MSC
        # computed by cmm.spectral.coherence (itself pinned to scipy), and
        # MSC's scale-invariance means the raw signal works on that side.
        s, truth = grouped
        xhat = normalize_coefs(s.coefs)
        cen, _ = centroid_update(xhat, truth, 3)
        scores = centroid_scores(xhat, cen)
        for m in (0, 2):
            s_cen = dataclasses.replace(s, coefs=cen[m][None])
            for n_idx in (0, 11, 23):
                s_sig = dataclasses.replace(s, coefs=s.coefs[n_idx][None])
                gamma2, _ = coherence(s_cen, s_sig, pairwise=True)
                assert scores[m, n_idx] == pytest.approx(
                    float(gamma2[0].mean()), rel=1e-10
                )

    def test_full_fit_matches_a_naive_loop(self, grouped):
        s, truth = grouped
        xhat = normalize_coefs(s.coefs)
        n = xhat.shape[0]
        rng = np.random.default_rng(9)
        init = rng.integers(0, 3, n)
        while np.unique(init).size < 3:  # ensure every cluster is seeded
            init = rng.integers(0, 3, n)

        labels = init.copy()
        for _ in range(50):
            cen, _ = centroid_update(xhat, labels, 3)
            sc = centroid_scores(xhat, cen)
            new = sc.argmax(axis=0)
            if np.array_equal(new, labels):
                break
            labels = new

        got = coherence_kmeans(s, 3, init=init, max_iter=50)
        assert np.array_equal(got.labels, labels)
        assert got.objective[-1] == pytest.approx(
            float(sc[labels, np.arange(n)].sum()), rel=1e-12
        )


class TestObjective:
    @pytest.mark.parametrize("whiten", [True, False])
    def test_trace_is_monotone_and_bounded(self, grouped, whiten):
        s, _ = grouped
        r = coherence_kmeans(s, 3, whiten=whiten, n_init=3)
        assert r.n_reseeds == 0
        assert (np.diff(r.objective) >= -1e-10).all()
        assert 0.0 <= r.objective[-1] <= s.n_signals + 1e-9

    def test_centroid_is_the_argmax_no_perturbation_beats_it(self, grouped):
        s, truth = grouped
        xhat = normalize_coefs(s.coefs)
        cen, _ = centroid_update(xhat, truth, 3)
        rng = np.random.default_rng(1)
        m, fi = 1, 4
        members = xhat[truth == m][..., fi]
        base = (np.abs(members @ cen[m, :, fi].conj()) ** 2).sum()
        k = members.shape[1]
        for _ in range(100):
            v = cen[m, :, fi] + 0.2 * (
                rng.standard_normal(k) + 1j * rng.standard_normal(k)
            )
            v /= np.linalg.norm(v)
            perturbed = (np.abs(members @ v.conj()) ** 2).sum()
            assert perturbed <= base + 1e-10

    def test_assigned_score_is_the_row_max(self, grouped):
        s, _ = grouped
        r = coherence_kmeans(s, 3)
        assigned = r.scores[r.labels, np.arange(r.labels.size)]
        assert_allclose(assigned, r.scores.max(axis=0), rtol=0, atol=0)


class TestRecovery:
    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_recovers_planted_groups(self, k):
        x, truth = grouped_traces([6] * k, seed=k)
        r = coherence_kmeans(stft(x, PARAMS), k)
        assert adjusted_rand_score(r.labels, truth) > 0.9

    def test_recovery_survives_lower_snr(self):
        x, truth = grouped_traces([8, 8, 8], snr=1.0, seed=3)
        r = coherence_kmeans(stft(x, PARAMS), 3)
        assert adjusted_rand_score(r.labels, truth) > 0.9

    def test_perfectly_coherent_cluster_scores_near_one(self):
        rng = np.random.default_rng(4)
        src = bandlimited_source(rng, 3072, 8.0)
        gains = rng.uniform(0.5, 2.0, 6)
        x = gains[:, None] * src[None, :]
        r = coherence_kmeans(stft(x, PARAMS), 1)
        assigned = r.scores[0]
        assert assigned.min() > 0.99
        assert r.centroid_strength.min() > 0.99

    def test_agrees_with_the_spectral_route(self, grouped):
        s, truth = grouped
        km = coherence_kmeans(s, 3)
        spec = spectral_labels(coherence_matrix(s, whiten=True), 3)
        assert adjusted_rand_score(km.labels, spec) > 0.9
        assert adjusted_rand_score(km.labels, truth) > 0.9


class TestInvariances:
    @pytest.mark.parametrize("whiten", [True, False])
    def test_labels_exactly_invariant_to_per_signal_rescaling(self, grouped, whiten):
        # Gains cancel exactly in both x_hat and the power-fraction weights,
        # so the entire run — including every random draw — is identical.
        s, _ = grouped
        gains = np.random.default_rng(2).uniform(0.01, 100.0, s.n_signals)
        scaled = dataclasses.replace(s, coefs=s.coefs * gains[:, None, None])
        a = coherence_kmeans(s, 3, whiten=whiten)
        b = coherence_kmeans(scaled, 3, whiten=whiten)
        assert np.array_equal(a.labels, b.labels)

    def test_delayed_copies_co_cluster(self):
        # The estimator's whole point: a delay is a per-frequency phase and
        # must not split a group.
        rng = np.random.default_rng(5)
        t = 3072
        groups = []
        for g in range(2):
            src = bandlimited_source(rng, t + 32, 8.0)
            for lag in (0, 5, 11):
                trace = src[lag : lag + t] * 2.0 + 0.5 * rng.standard_normal(t)
                groups.append((g, trace))
        truth = np.array([g for g, _ in groups])
        x = np.stack([tr for _, tr in groups])
        r = coherence_kmeans(stft(x, PARAMS), 2)
        assert adjusted_rand_score(r.labels, truth) == 1.0

    def test_permutation_equivariance(self, grouped):
        s, _ = grouped
        perm = np.random.default_rng(6).permutation(s.n_signals)
        a = coherence_kmeans(s, 3)
        b = coherence_kmeans(
            dataclasses.replace(s, coefs=s.coefs[perm]), 3
        )
        assert adjusted_rand_score(a.labels[perm], b.labels) == 1.0

    def test_duplicated_signal_lands_with_its_twin(self, grouped):
        s, _ = grouped
        dup = dataclasses.replace(
            s, coefs=np.concatenate([s.coefs, s.coefs[:1]])
        )
        r = coherence_kmeans(dup, 3)
        assert r.labels[-1] == r.labels[0]


class TestDeterminism:
    def test_same_seed_same_everything(self, grouped):
        s, _ = grouped
        a = coherence_kmeans(s, 3, random_state=7)
        b = coherence_kmeans(s, 3, random_state=7)
        assert np.array_equal(a.labels, b.labels)
        assert_allclose(a.objective, b.objective, rtol=0, atol=0)
        assert_allclose(a.centroids, b.centroids, rtol=0, atol=0)

    def test_restarts_return_at_least_the_single_run(self, grouped):
        # spawn(n) children are ordered, so the n_init=1 run is the first
        # child of the n_init=5 pool and the max can only improve on it.
        s, _ = grouped
        single = coherence_kmeans(s, 3, n_init=1, random_state=0)
        multi = coherence_kmeans(s, 3, n_init=5, random_state=0)
        assert multi.objective[-1] >= single.objective[-1] - 1e-12

    def test_explicit_init_is_honored(self, grouped):
        s, truth = grouped
        r = coherence_kmeans(s, 3, init=truth.astype(int))
        # perfect init on separated data: nothing should move
        assert adjusted_rand_score(r.labels, truth) == 1.0
        assert r.converged


class TestDegenerate:
    def test_one_cluster(self, grouped):
        s, _ = grouped
        r = coherence_kmeans(s, 1)
        assert (r.labels == 0).all()
        assert r.converged

    def test_every_signal_its_own_cluster(self):
        rng = np.random.default_rng(8)
        x = rng.standard_normal((5, 1024))
        s = stft(x, SpectralParams(fs=8.0, nperseg=128, band_hz=(0.2, 1.5)))
        r = coherence_kmeans(s, 5)
        assert np.unique(r.labels).size == 5
        assert r.objective[-1] == pytest.approx(5.0, abs=1e-9)

    def test_empty_cluster_is_reseeded_and_counted(self):
        # Duplicate signals let two seeds carry identical content; argmax
        # ties then empty a cluster, which must be repaired, not dropped.
        rng = np.random.default_rng(0)
        a = bandlimited_source(rng, 2048, 8.0)
        b = bandlimited_source(rng, 2048, 8.0)
        x = np.stack([a, a, a, b]) + 0.001 * 0
        s = stft(x, SpectralParams(fs=8.0, nperseg=256, band_hz=(0.2, 1.5)))
        seed_hit = None
        for seed in range(30):
            r = coherence_kmeans(
                s, 3, init="random", n_init=1, max_iter=4, random_state=seed
            )
            assert np.unique(r.labels).size == 3  # k never shrinks
            if r.n_reseeds > 0:
                seed_hit = seed
                break
        assert seed_hit is not None, "no seed produced an empty cluster"
        with pytest.raises(ValueError, match="became empty at iteration"):
            coherence_kmeans(
                s, 3, init="random", n_init=1, max_iter=4,
                random_state=seed_hit, on_empty="raise",
            )


class TestFailFast:
    def test_too_many_clusters(self, grouped):
        s, _ = grouped
        with pytest.raises(ValueError, match=r"n_clusters must be in \[1, 24\]"):
            coherence_kmeans(s, 25)

    def test_zero_power_cell_points_at_the_mask(self, grouped):
        s, _ = grouped
        coefs = s.coefs.copy()
        coefs[2, :, 1] = 0.0
        with pytest.raises(ValueError, match="finite_power_mask"):
            coherence_kmeans(dataclasses.replace(s, coefs=coefs), 3)

    def test_non_finite_coefs(self):
        bad = np.ones((3, 4, 5), dtype=complex)
        bad[1, 2, 3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            normalize_coefs(bad)

    def test_wrong_rank(self):
        with pytest.raises(ValueError, match=r"expected \(N, K, F\)"):
            normalize_coefs(np.ones((3, 4)))

    def test_bad_init_string(self, grouped):
        s, _ = grouped
        with pytest.raises(ValueError, match="init must be"):
            coherence_kmeans(s, 3, init="kmeans++")  # type: ignore[arg-type]

    def test_bad_max_iter_and_on_empty_and_n_init(self, grouped):
        s, _ = grouped
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            coherence_kmeans(s, 3, max_iter=0)
        with pytest.raises(ValueError, match="on_empty must be"):
            coherence_kmeans(s, 3, on_empty="ignore")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="n_init must be >= 1"):
            coherence_kmeans(s, 3, n_init=0)

    def test_init_labels_validated(self, grouped):
        s, truth = grouped
        with pytest.raises(ValueError, match="init labels shape"):
            coherence_kmeans(s, 3, init=truth[:5].astype(int))
        with pytest.raises(ValueError, match=r"must lie in \[0, 3\)"):
            coherence_kmeans(s, 3, init=truth.astype(int) + 1)
        with pytest.raises(ValueError, match="leave cluster 1 empty"):
            coherence_kmeans(s, 3, init=np.zeros(s.n_signals, dtype=int))

    def test_centroid_update_refuses_an_empty_cluster(self, grouped):
        s, truth = grouped
        xhat = normalize_coefs(s.coefs)
        with pytest.raises(ValueError, match="cluster 3 has no members"):
            centroid_update(xhat, truth, 4)


class TestProvenance:
    def test_metadata_records_the_run(self, grouped):
        s, _ = grouped
        r = coherence_kmeans(s, 3, random_state=5)
        m = r.metadata()
        assert m["n_clusters"] == 3
        assert m["weighting"].startswith("equal")
        assert m["random_state"] == 5
        assert m["converged"] is True
        assert m["stft"]["band_hz"] == [0.2, 1.5]
        assert isinstance(r, CoherenceKMeansResult)

    def test_result_is_frozen(self, grouped):
        s, _ = grouped
        r = coherence_kmeans(s, 3)
        with pytest.raises(Exception):
            r.n_iter = 0  # type: ignore[misc]

    def test_class_wrapper_fit_predict_roundtrip(self, grouped):
        s, _ = grouped
        km = CoherenceKMeans(3).fit(s)
        assert np.array_equal(km.predict(s), km.result_.labels)
