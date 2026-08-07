"""Phase 3 chunk 1: clustering metrics.

The load-bearing test is the sign pin: the pre-refactor silhouette passed the
coherence similarity to sklearn where a distance is expected, negating every
score (C5). Both directions are asserted so the inversion cannot come back.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.metrics import adjusted_rand_score, silhouette_score

from cmm.coherence import CoherenceMatrix, to_distance
from cmm.metrics import (
    cluster_summary,
    coherence_silhouette,
    label_stability,
    pairwise_ari,
)


def planted_blocks(sizes, intra=0.6, inter=0.1):
    """A block coherence matrix with known labels: intra > inter, unit diagonal."""
    labels = np.repeat(np.arange(len(sizes)), sizes)
    n = labels.size
    c = np.full((n, n), inter)
    for lbl in range(len(sizes)):
        idx = labels == lbl
        c[np.ix_(idx, idx)] = intra
    np.fill_diagonal(c, 1.0)
    return c, labels


class TestCoherenceSilhouette:
    def test_equals_sklearn_on_the_converted_distance(self):
        c, labels = planted_blocks([8, 8, 8])
        ours = coherence_silhouette(c, labels)
        ref = silhouette_score(to_distance(c), labels, metric="precomputed")
        assert ours == ref

    def test_sign_pin_against_the_inverted_legacy_metric(self):
        # C5: feeding the similarity itself where sklearn wants a distance.
        # True labels on a block similarity must score well through the correct
        # path and *negative* through the legacy one — pin both directions.
        c, labels = planted_blocks([10, 10])
        assert coherence_silhouette(c, labels) > 0.5
        legacy = c.copy()
        np.fill_diagonal(legacy, 0.0)
        assert silhouette_score(legacy, labels, metric="precomputed") < 0.0

    def test_shuffled_labels_score_near_zero(self):
        c, labels = planted_blocks([12, 12])
        shuffled = np.random.default_rng(1).permutation(labels)
        assert coherence_silhouette(c, shuffled) < 0.1
        assert coherence_silhouette(c, shuffled) < coherence_silhouette(c, labels)

    def test_sharper_contrast_scores_higher(self):
        weak, labels = planted_blocks([10, 10], intra=0.3, inter=0.1)
        sharp, _ = planted_blocks([10, 10], intra=0.8, inter=0.1)
        assert coherence_silhouette(sharp, labels) > coherence_silhouette(weak, labels)

    def test_subsample_is_deterministic_by_default(self):
        c, labels = planted_blocks([30, 30, 30])
        a = coherence_silhouette(c, labels, sample_size=40)
        b = coherence_silhouette(c, labels, sample_size=40)
        assert a == b

    def test_subsample_covering_everything_equals_the_full_score(self):
        c, labels = planted_blocks([10, 10])
        full = coherence_silhouette(c, labels)
        assert coherence_silhouette(c, labels, sample_size=labels.size) == full
        assert coherence_silhouette(c, labels, sample_size=10 * labels.size) == full

    def test_different_seeds_sample_differently(self):
        rng = np.random.default_rng(4)
        n = 60
        c = np.clip((rng.uniform(0.0, 0.6, (n, n)) + 0.2 * np.eye(n)), 0, 1)
        c = 0.5 * (c + c.T)
        np.fill_diagonal(c, 1.0)
        labels = np.repeat([0, 1, 2], n // 3)
        scores = {
            coherence_silhouette(c, labels, sample_size=20, rng=s) for s in range(5)
        }
        assert len(scores) > 1


class TestPairwiseAri:
    def test_identical_labelings_score_one(self):
        lab = np.repeat([0, 1, 2], 10)
        a = pairwise_ari([lab, lab.copy(), lab.copy()])
        assert_allclose(a, 1.0, rtol=0, atol=0)

    def test_invariant_to_label_id_permutation(self):
        lab = np.repeat([0, 1, 2], 10)
        renamed = (lab + 1) % 3
        assert pairwise_ari([lab, renamed])[0, 1] == pytest.approx(1.0)

    def test_independent_labelings_score_near_zero(self):
        rng = np.random.default_rng(2)
        a = rng.integers(0, 4, 400)
        b = rng.integers(0, 4, 400)
        assert abs(pairwise_ari([a, b])[0, 1]) < 0.1

    def test_symmetric_with_unit_diagonal(self):
        rng = np.random.default_rng(3)
        labs = [rng.integers(0, 3, 50) for _ in range(4)]
        a = pairwise_ari(labs)
        assert_allclose(a, a.T, rtol=0, atol=0)
        assert_allclose(np.diag(a), 1.0, rtol=0, atol=0)


class TestLabelStability:
    def test_known_mixture(self):
        l1 = np.repeat([0, 1], 20)
        l2 = np.repeat([0, 1, 2, 3], 10)
        r = adjusted_rand_score(l1, l2)
        got = label_stability([l1, l1.copy(), l2])
        assert got["n_runs"] == 3
        assert got["min_ari"] == pytest.approx(r)
        assert got["mean_ari"] == pytest.approx((1.0 + 2 * r) / 3)


class TestClusterSummary:
    def test_planted_blocks_report_their_contrast(self):
        c, labels = planted_blocks([10, 10, 10], intra=0.6, inter=0.1)
        s = cluster_summary(c, labels)
        assert s["k"] == 3
        assert s["n_signals"] == 30
        assert s["mean_intra_C"] == pytest.approx(0.6)
        assert s["mean_inter_C"] == pytest.approx(0.1)
        assert s["mean_intra_C"] > s["mean_inter_C"]
        assert s["silhouette"] > 0.5

    def test_singleton_fraction(self):
        c, labels = planted_blocks([1, 10, 10])
        s = cluster_summary(c, labels)
        assert s["singleton_frac"] == pytest.approx(1 / 3)
        assert s["size_min"] == 1 and s["size_max"] == 10

    def test_keys_are_stable(self):
        c, labels = planted_blocks([8, 8])
        assert set(cluster_summary(c, labels)) == {
            "k", "n_signals", "size_min", "size_median", "size_max",
            "singleton_frac", "mean_intra_C", "mean_inter_C", "silhouette",
        }


class TestFailFast:
    def test_single_label_raises_naming_the_count(self):
        c, _ = planted_blocks([10])
        with pytest.raises(ValueError, match="got 1 distinct"):
            coherence_silhouette(c, np.zeros(10, dtype=int))

    def test_label_length_mismatch_names_both_shapes(self):
        c, labels = planted_blocks([8, 8])
        with pytest.raises(ValueError, match=r"labels shape \(15,\)"):
            coherence_silhouette(c, labels[:15])

    def test_rejects_the_container(self):
        c, labels = planted_blocks([8, 8])
        box = CoherenceMatrix(
            values=c, n_eff=8, f_eff=3.0, whiten=False, stft_metadata={}
        )
        with pytest.raises(TypeError, match="CoherenceMatrix.values"):
            coherence_silhouette(box, labels)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="CoherenceMatrix.values"):
            cluster_summary(box, labels)  # type: ignore[arg-type]

    def test_non_square_matrix(self):
        with pytest.raises(ValueError, match="square"):
            coherence_silhouette(np.ones((4, 5)), np.zeros(4, dtype=int))

    def test_degenerate_subsample_raises_not_nan(self):
        # One label owns all but a single signal; a tiny subsample that misses
        # the singleton has nothing to compare and must say so.
        c, _ = planted_blocks([20, 1])
        labels = np.array([0] * 20 + [1])
        with pytest.raises(ValueError, match="distinct label"):
            coherence_silhouette(c, labels, sample_size=3, rng=0)

    def test_tiny_sample_size_rejected(self):
        c, labels = planted_blocks([8, 8])
        with pytest.raises(ValueError, match="sample_size must be >= 2"):
            coherence_silhouette(c, labels, sample_size=1)

    def test_pairwise_ari_needs_two_labelings(self):
        with pytest.raises(ValueError, match="at least 2 labelings"):
            pairwise_ari([np.zeros(5, dtype=int)])

    def test_pairwise_ari_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="labeling 1 has shape"):
            pairwise_ari([np.zeros(5, dtype=int), np.zeros(6, dtype=int)])
