"""Quality metrics for clusterings of a coherence matrix.

Silhouette here always goes through :func:`cmm.coherence.to_distance` first.
The pre-refactor ``compute_model_silhouette`` passed the coherence matrix — a
*similarity* — straight to sklearn's ``metric="precomputed"``, which expects a
distance, silently negating every score (bug C5). The conversion is not left
to the caller precisely so that mistake has nowhere to live.

ARI utilities compare labelings to each other, not to ground truth — which is
what real data offers. :func:`label_stability` is the tool for the open
V1-vs-V2 estimator question (CLAUDE.md): prefer whichever estimator gives more
reproducible labels under resampling.

sklearn is imported by this module only; ``import cmm`` never drags it in.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

from cmm.coherence import CoherenceMatrix, to_distance

__all__ = [
    "cluster_summary",
    "coherence_silhouette",
    "label_stability",
    "pairwise_ari",
]


def _check_matrix_and_labels(values, labels_n) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(values, CoherenceMatrix):
        raise TypeError("pass CoherenceMatrix.values, not the container")
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] != v.shape[1]:
        raise ValueError(
            f"expected a square (N, N) coherence matrix, got shape {v.shape}"
        )
    labels = np.asarray(labels_n)
    if labels.shape != (v.shape[0],):
        raise ValueError(
            f"labels shape {labels.shape} does not match the {v.shape} matrix; "
            f"expected ({v.shape[0]},)"
        )
    return v, labels


def coherence_silhouette(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    use_sqrt: bool = True,
    sample_size: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Silhouette of a labeling against a coherence matrix, higher is better.

    Converts to a distance via :func:`to_distance` before calling sklearn —
    feeding the similarity itself to ``metric="precomputed"`` negates the
    answer (bug C5). Pass ``CoherenceMatrix.values``, not the container.

    ``sample_size`` scores a random row subsample instead of the full matrix
    (the full computation is O(N^2) and prohibitive at tens of thousands of
    signals). The default ``rng=None`` seeds deterministically; pass a seed or
    Generator to vary the subsample. A subsample that loses all but one label
    raises rather than returning NaN.
    """
    v, lab = _check_matrix_and_labels(values, labels)
    n_distinct = np.unique(lab).size
    if n_distinct < 2:
        raise ValueError(
            f"silhouette needs at least 2 clusters, got {n_distinct} distinct label(s)"
        )
    if sample_size is not None:
        if sample_size < 2:
            raise ValueError(f"sample_size must be >= 2, got {sample_size}")
        if sample_size < v.shape[0]:
            gen = (
                rng
                if isinstance(rng, np.random.Generator)
                else np.random.default_rng(0 if rng is None else rng)
            )
            idx = gen.choice(v.shape[0], size=sample_size, replace=False)
            v = v[np.ix_(idx, idx)]
            lab = lab[idx]
            sub_distinct = np.unique(lab).size
            if sub_distinct < 2:
                raise ValueError(
                    f"the {sample_size}-row subsample contains {sub_distinct} "
                    "distinct label(s); increase sample_size or pass "
                    "sample_size=None"
                )
    d = to_distance(v, use_sqrt=use_sqrt)
    return float(silhouette_score(d, lab, metric="precomputed"))


def pairwise_ari(labelings: Sequence[np.ndarray]) -> np.ndarray:
    """``(R, R)`` adjusted Rand index between every pair of labelings.

    Symmetric with unit diagonal. ARI is invariant to label-id permutation, so
    two runs that found the same partition under different numbering score 1.
    """
    labs = [np.asarray(lab) for lab in labelings]
    if len(labs) < 2:
        raise ValueError(f"need at least 2 labelings to compare, got {len(labs)}")
    n = labs[0].size
    for i, lab in enumerate(labs):
        if lab.ndim != 1 or lab.size != n:
            raise ValueError(
                f"labeling {i} has shape {lab.shape}; expected ({n},) matching "
                "the first"
            )
    r = len(labs)
    out = np.ones((r, r))
    for i in range(r):
        for j in range(i + 1, r):
            out[i, j] = out[j, i] = adjusted_rand_score(labs[i], labs[j])
    return out


def label_stability(labelings: Sequence[np.ndarray]) -> dict:
    """Reproducibility of a set of labelings: mean and min pairwise ARI.

    The discriminator proposed for the V1-vs-V2 estimator choice: run the same
    pipeline over resampled segments or signals and prefer the estimator whose
    labels move less.
    """
    a = pairwise_ari(labelings)
    iu = np.triu_indices_from(a, k=1)
    return {
        "mean_ari": float(a[iu].mean()),
        "min_ari": float(a[iu].min()),
        "n_runs": int(a.shape[0]),
    }


def cluster_summary(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    subsample_silhouette: int = 2000,
    use_sqrt: bool = True,
    rng: np.random.Generator | int | None = None,
) -> dict:
    """Per-k report to judge whether a clustering is too coarse or too fine.

    Cluster sizes (min/median/max), singleton fraction, mean intra- vs
    inter-cluster coherence, and a subsampled silhouette. Absorbed from
    icampsnfr's hierarchical module, minus the linkage machinery.
    """
    v, lab = _check_matrix_and_labels(values, labels)
    unique, counts = np.unique(lab, return_counts=True)
    n_total = lab.size

    intra_sum = 0.0
    intra_count = 0
    for lbl in unique:
        idx = np.flatnonzero(lab == lbl)
        if idx.size < 2:
            continue
        block = v[np.ix_(idx, idx)]
        intra_sum += float(block.sum()) - float(np.trace(block))
        intra_count += idx.size * (idx.size - 1)
    mean_intra = intra_sum / intra_count if intra_count else float("nan")

    total_off = float(v.sum()) - float(np.trace(v))
    total_pairs = n_total * (n_total - 1)
    inter_count = total_pairs - intra_count
    mean_inter = (total_off - intra_sum) / inter_count if inter_count else float("nan")

    return {
        "k": int(unique.size),
        "n_signals": int(n_total),
        "size_min": int(counts.min()),
        "size_median": int(np.median(counts)),
        "size_max": int(counts.max()),
        "singleton_frac": float((counts == 1).mean()),
        "mean_intra_C": mean_intra,
        "mean_inter_C": mean_inter,
        "silhouette": coherence_silhouette(
            v, lab, use_sqrt=use_sqrt, sample_size=subsample_silhouette, rng=rng
        ),
    }
