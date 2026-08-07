"""Phase 3 chunk 2: spectral clustering on a coherence matrix.

The load-bearing layer is solver agreement: Gate 2 proved eigsh and dense give
identical clusters on a real N=10000 graph; these tests pin that equivalence
structurally (eigenvalues, subspaces via principal angles, end-to-end labels)
so a solver regression cannot ship quietly.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import subspace_angles
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

from cmm.clustering import (
    SpectralEmbedding,
    affinity_gaussian,
    cluster_embedding,
    eigengap,
    median_tau,
    normalized_laplacian,
    spectral_embedding,
    spectral_labels,
)


def noisy_blocks(sizes, intra=0.55, inter=0.08, noise=0.03, seed=0):
    """A coherence-like block matrix with symmetric noise and known labels."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(sizes)), sizes)
    n = labels.size
    c = np.full((n, n), inter)
    for lbl in range(len(sizes)):
        idx = labels == lbl
        c[np.ix_(idx, idx)] = intra
    c = c + noise * rng.standard_normal((n, n))
    c = np.clip(0.5 * (c + c.T), 0.0, 1.0)
    np.fill_diagonal(c, 1.0)
    return c, labels


class TestNaiveReference:
    @pytest.mark.parametrize("kind", ["sym", "rw"])
    def test_laplacian_matches_a_per_entry_loop(self, kind):
        c, _ = noisy_blocks([6, 6])
        w = affinity_gaussian(c, 0.3)
        d = w.sum(axis=1)
        n = w.shape[0]
        ref = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                delta = 1.0 if i == j else 0.0
                if kind == "sym":
                    ref[i, j] = delta - w[i, j] / np.sqrt(d[i] * d[j])
                else:
                    ref[i, j] = delta - w[i, j] / d[i]
        assert_allclose(normalized_laplacian(w, kind), ref, rtol=1e-12, atol=1e-15)

    def test_affinity_is_symmetric_zero_diagonal_and_monotone(self):
        c, _ = noisy_blocks([8, 8])
        w = affinity_gaussian(c, 0.3)
        assert_allclose(w, w.T, rtol=0, atol=0)
        assert_allclose(np.diag(w), 0.0, rtol=0, atol=0)
        # strictly increasing in coherence: order of off-diagonal entries kept
        iu = np.triu_indices_from(c, k=1)
        cs = np.clip(0.5 * (c + c.T), 0, 1)
        order_c = np.argsort(cs[iu])
        assert (np.diff(w[iu][order_c]) >= 0).all()


class TestSolverAgreement:
    def test_eigenvalues_match_dense(self):
        c, _ = noisy_blocks([20, 20, 20])
        a = spectral_embedding(c, 6, tau=0.3, solver="eigsh")
        b = spectral_embedding(c, 6, tau=0.3, solver="dense")
        assert_allclose(a.eigenvalues, b.eigenvalues, rtol=1e-10, atol=1e-12)

    def test_embedding_subspaces_agree(self):
        # Compare subspaces, not raw vectors: within a near-degenerate block
        # the individual eigenvectors are only defined up to rotation.
        c, _ = noisy_blocks([20, 20, 20])
        a = spectral_embedding(c, 4, tau=0.3, solver="eigsh")
        b = spectral_embedding(c, 4, tau=0.3, solver="dense")
        assert subspace_angles(a.vectors, b.vectors).max() < 1e-7

    @pytest.mark.parametrize("laplacian", ["sym", "rw"])
    def test_end_to_end_labels_identical_across_solvers(self, laplacian):
        c, truth = noisy_blocks([25, 25, 25])
        la = spectral_labels(c, 3, tau=0.3, laplacian=laplacian, solver="eigsh")
        lb = spectral_labels(c, 3, tau=0.3, laplacian=laplacian, solver="dense")
        assert adjusted_rand_score(la, lb) == 1.0
        assert adjusted_rand_score(la, truth) == 1.0

    def test_rw_eigenvalues_equal_sym_eigenvalues(self):
        c, _ = noisy_blocks([15, 15])
        sym = spectral_embedding(c, 5, tau=0.3, laplacian="sym")
        rw = spectral_embedding(c, 5, tau=0.3, laplacian="rw")
        assert_allclose(rw.eigenvalues, sym.eigenvalues, rtol=0, atol=0)

    def test_rw_vectors_are_scaled_sym_vectors(self):
        c, _ = noisy_blocks([15, 15])
        w = affinity_gaussian(c, 0.3)
        d = w.sum(axis=1)
        sym = spectral_embedding(c, 4, tau=0.3, laplacian="sym")
        rw = spectral_embedding(c, 4, tau=0.3, laplacian="rw")
        expected = sym.vectors / np.sqrt(d)[:, None]
        expected /= np.linalg.norm(expected, axis=0, keepdims=True)
        for j in range(4):
            cos = abs(float(expected[:, j] @ rw.vectors[:, j]))
            assert cos == pytest.approx(1.0, abs=1e-10)

    def test_low_contrast_miniature_of_the_gate2_regime(self):
        # Structured signal barely above a floor, N in the hundreds: the
        # regime where lobpcg matched eigenvalues yet produced ARI 0.39.
        c, truth = noisy_blocks(
            [200, 200, 200], intra=0.12, inter=0.05, noise=0.01, seed=7
        )
        la = spectral_labels(c, 3, solver="eigsh")
        lb = spectral_labels(c, 3, solver="dense")
        assert adjusted_rand_score(la, lb) == 1.0
        assert adjusted_rand_score(la, truth) > 0.99


class TestAgainstSklearn:
    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_matches_sklearn_spectral_clustering(self, k):
        c, truth = noisy_blocks([15] * k, seed=k)
        tau = 0.3
        ours = spectral_labels(c, k, tau=tau)
        ref = SpectralClustering(
            n_clusters=k, affinity="precomputed", random_state=0, n_init=10
        ).fit_predict(affinity_gaussian(c, tau))
        assert adjusted_rand_score(ours, ref) > 0.95
        assert adjusted_rand_score(ours, truth) > 0.95


class TestEigengap:
    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_recovers_the_planted_k(self, k):
        c, _ = noisy_blocks([15] * k, seed=k)
        emb = spectral_embedding(c, n_eig=k + 4, tau=0.3, solver="dense")
        _, suggested = eigengap(emb.eigenvalues)
        assert suggested == k

    def test_min_k_excludes_the_trivial_gap(self):
        w = [0.0, 0.9, 1.0, 1.05]
        _, k1 = eigengap(w, min_k=1)
        _, k2 = eigengap(w, min_k=2)
        assert k1 == 1
        assert k2 == 2

    def test_gaps_span_the_full_spectrum(self):
        gaps, _ = eigengap([0.0, 0.1, 0.5, 0.6])
        assert_allclose(gaps, [0.1, 0.4, 0.1], rtol=1e-12)


class TestDeterminism:
    def test_repeated_calls_are_identical(self):
        c, _ = noisy_blocks([20, 20, 20])
        a = spectral_labels(c, 3)
        b = spectral_labels(c, 3)
        assert np.array_equal(a, b)

    def test_embed_once_then_cluster_per_k_matches_one_shot(self):
        c, _ = noisy_blocks([15, 15, 15, 15])
        max_k = 4
        emb = spectral_embedding(c, n_eig=max_k + 1, tau=0.3)
        for k in (2, 3, 4):
            swept = cluster_embedding(emb, k)
            oneshot = spectral_labels(c, k, tau=0.3, n_eig=max_k + 1)
            assert np.array_equal(swept, oneshot)

    def test_labels_are_zero_based_int(self):
        c, _ = noisy_blocks([12, 12, 12])
        labels = spectral_labels(c, 3, tau=0.3)
        assert labels.min() == 0
        assert labels.max() == 2
        assert np.issubdtype(labels.dtype, np.integer)


class TestProperties:
    def test_median_tau_known_value(self):
        c = np.array([[1.0, 0.8], [0.8, 1.0]])
        assert median_tau(c) == pytest.approx(0.2)

    def test_eigenvalues_nonnegative_and_first_is_zero(self):
        c, _ = noisy_blocks([15, 15])
        emb = spectral_embedding(c, 5, tau=0.3, solver="dense")
        assert emb.eigenvalues.min() > -1e-12
        assert emb.eigenvalues[0] == pytest.approx(0.0, abs=1e-10)
        assert (np.diff(emb.eigenvalues) >= -1e-14).all()

    def test_embedding_carries_its_recipe(self):
        c, _ = noisy_blocks([10, 10])
        emb = spectral_embedding(c, 3, laplacian="rw", solver="dense")
        m = emb.metadata()
        assert m["laplacian"] == "rw" and m["solver"] == "dense"
        assert m["tau"] == pytest.approx(median_tau(c))
        assert m["n_eig"] == 3 and len(m["eigenvalues"]) == 3
        assert isinstance(emb, SpectralEmbedding)


class TestFailFast:
    def test_non_positive_tau(self):
        c, _ = noisy_blocks([6, 6])
        with pytest.raises(ValueError, match="tau must be > 0"):
            affinity_gaussian(c, 0.0)

    def test_n_eig_out_of_range(self):
        c, _ = noisy_blocks([6, 6])
        with pytest.raises(ValueError, match=r"n_eig must be in \[1, 12\]"):
            spectral_embedding(c, 13, tau=0.3)

    def test_eigsh_refuses_the_full_spectrum(self):
        c, _ = noisy_blocks([6, 6])
        with pytest.raises(ValueError, match="solver='dense'"):
            spectral_embedding(c, 12, tau=0.3, solver="eigsh")

    def test_isolated_nodes_are_counted(self):
        # exp(-(1-C)/tau) underflows to exactly zero at tiny tau, isolating
        # every node.
        c, _ = noisy_blocks([6, 6])
        with pytest.raises(ValueError, match="isolated nodes"):
            spectral_embedding(c, 3, tau=1e-5)

    def test_non_square_matrix(self):
        with pytest.raises(ValueError, match="square"):
            spectral_embedding(np.ones((4, 5)), 2, tau=0.3)

    def test_non_finite_matrix(self):
        c, _ = noisy_blocks([6, 6])
        c[2, 3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            spectral_embedding(c, 3, tau=0.3)

    def test_unknown_solver_and_laplacian(self):
        c, _ = noisy_blocks([6, 6])
        with pytest.raises(ValueError, match="solver must be"):
            spectral_embedding(c, 3, tau=0.3, solver="lobpcg")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="laplacian must be"):
            spectral_embedding(c, 3, tau=0.3, laplacian="left")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unknown Laplacian kind"):
            normalized_laplacian(affinity_gaussian(c, 0.3), "ward")

    def test_cluster_embedding_rejects_bad_k(self):
        with pytest.raises(ValueError, match="n_clusters must be >= 1"):
            cluster_embedding(np.ones((5, 3)), 0)

    def test_median_tau_needs_two_signals(self):
        with pytest.raises(ValueError, match="at least 2 signals"):
            median_tau(np.ones((1, 1)))
