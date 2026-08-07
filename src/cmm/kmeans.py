"""Coherence k-means: cluster signals by shared per-frequency phase structure.

The objective, stated once. With ``x_hat[n,:,f]`` each signal's segment-vector
normalized to unit length per frequency, per-signal frequency weights
``w[n,f]`` summing to 1 over f, and one unit-norm centroid ``v[m,:,f]`` per
cluster and frequency:

.. math::
    J(c, V) = \\sum_n \\sum_f w[n,f] \\, |v[c(n),:,f]^H x_hat[n,:,f]|^2,
    \\qquad 0 \\le J \\le N

Because both ``v`` and ``x_hat`` are unit vectors, each term *is* the
magnitude-squared coherence between centroid and signal at that frequency —
the ratio's normalization cancels, which is why the legacy allocation through
a full coherence computation reduces to this inner product (the test suite
verifies that reduction against ``cmm.spectral.coherence`` rather than
assuming it).

Both steps are exact block-coordinate ascent on J: the centroid update takes
the top eigenvector of each cluster's per-frequency Gram matrix (the exact
maximizer of the centroid subproblem), and allocation argmaxes each signal's
score. J therefore never decreases — asserted on every fitted result in the
tests — except across an empty-cluster reseed, which deliberately moves a
point and may dent the trace (visible in ``n_reseeds``).

``whiten=True`` (default) is the historical objective: equal frequency
weighting, the V1-flavored choice every pre-refactor result used.
``whiten=False`` weights frequencies by each signal's power fraction,
``w[n,f] = p[n,f] / sum_f p[n,f]`` — the V2 analogue. Fixed weights preserve
the ascent property either way.

The old power-iteration centroid solver measured 0.25-1.04x against a batched
``eigh`` at every size (master plan Q4), so it was not carried over. No jax,
no threading: at N=10^4, K=22, F=15 an iteration is ~10^8 flops and plain
numpy suffices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from cmm.spectral import Stft

__all__ = [
    "CoherenceKMeans",
    "CoherenceKMeansResult",
    "centroid_scores",
    "centroid_update",
    "coherence_kmeans",
    "normalize_coefs",
]


@dataclass(frozen=True)
class CoherenceKMeansResult:
    """A fitted coherence k-means model and how it got there.

    Attributes:
        labels: ``(N,)`` int32, 0-based.
        centroids: ``(M, K, F)`` complex128, unit norm per ``(m, f)``.
        centroid_strength: ``(M, F)`` top-eigenvalue share of each cluster's
            per-frequency energy, in ``(0, 1]`` — 1 means the cluster is a
            single coherent mode at that frequency.
        scores: ``(M, N)`` frequency-averaged coherence of every signal to
            every centroid; ``scores[labels[n], n]`` is signal n's fit.
        objective: J after the initial assignment and after each iteration —
            non-decreasing except across reseeds.
        n_iter: iterations run.
        converged: labels stopped changing before ``max_iter``.
        n_reseeds: empty-cluster reseeds that occurred (0 in a clean run).
        whiten: True = equal frequency weights (historical V1 flavor).
        random_state: the seed the winning restart derived from.
        stft_metadata: provenance of the transform, when an Stft was given.
    """

    labels: np.ndarray
    centroids: np.ndarray
    centroid_strength: np.ndarray
    scores: np.ndarray
    objective: np.ndarray
    n_iter: int
    converged: bool
    n_reseeds: int
    whiten: bool
    random_state: int | None
    stft_metadata: dict

    @property
    def n_clusters(self) -> int:
        return self.centroids.shape[0]

    def metadata(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "n_signals": int(self.labels.size),
            "objective": float(self.objective[-1]),
            "n_iter": self.n_iter,
            "converged": self.converged,
            "n_reseeds": self.n_reseeds,
            "weighting": "equal (V1 flavor)" if self.whiten else "power (V2 flavor)",
            "random_state": self.random_state,
            "stft": self.stft_metadata,
        }


def normalize_coefs(coefs_nkf: np.ndarray) -> np.ndarray:
    """Unit-normalize each signal's segment-vector per frequency: x_hat.

    Raises on any zero-power ``(n, f)`` cell — a zero vector has no direction
    to normalize and undefined coherence with everything.
    """
    x = np.asarray(coefs_nkf, dtype=np.complex128)
    if x.ndim != 3:
        raise ValueError(f"expected (N, K, F) coefficients, got shape {x.shape}")
    if not np.isfinite(x).all():
        n_bad = int((~np.isfinite(x)).sum())
        raise ValueError(f"coefficients contain {n_bad} non-finite entries")
    power = np.einsum("nkf,nkf->nf", x, x.conj()).real
    dead = ~(power > 0).all(axis=1)
    if dead.any():
        raise ValueError(
            f"{int(dead.sum())} of {x.shape[0]} signals have zero power in at "
            f"least one in-band frequency (first: {int(np.flatnonzero(dead)[0])}); "
            "filter with finite_power_mask() before clustering"
        )
    return x / np.sqrt(power)[:, None, :]


def _resolve_weights(weights_nf, n: int, f: int) -> np.ndarray:
    if weights_nf is None:
        return np.full((n, f), 1.0 / f)
    w = np.asarray(weights_nf, dtype=np.float64)
    if w.shape != (n, f):
        raise ValueError(f"weights shape {w.shape} does not match (N, F) = ({n}, {f})")
    return w


def centroid_update(
    xhat_nkf: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    *,
    weights_nf: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cluster, per-frequency top eigenvector of the weighted Gram matrix.

    The exact maximizer of ``sum_{n in m} w[n,f] |v^H x_hat[n,:,f]|^2`` over
    unit ``v``. Returns ``(M, K, F)`` centroids and the ``(M, F)`` strength
    (top-eigenvalue share of the Gram trace).
    """
    xhat = np.asarray(xhat_nkf)
    n, k, f = xhat.shape
    lab = np.asarray(labels)
    w = _resolve_weights(weights_nf, n, f)
    counts = np.bincount(lab, minlength=n_clusters)
    if (counts == 0).any():
        empty = int(np.flatnonzero(counts == 0)[0])
        raise ValueError(
            f"cluster {empty} has no members; reassign or drop it before "
            "updating centroids"
        )
    centroids = np.empty((n_clusters, k, f), dtype=np.complex128)
    strength = np.empty((n_clusters, f))
    for m in range(n_clusters):
        idx = lab == m
        g = np.einsum(
            "nkf,nlf,nf->fkl", xhat[idx], xhat[idx].conj(), w[idx], optimize=True
        )
        vals, vecs = np.linalg.eigh(g)
        centroids[m] = vecs[:, :, -1].T
        strength[m] = vals[:, -1] / vals.sum(axis=1)
    return centroids, strength


def centroid_scores(
    xhat_nkf: np.ndarray,
    centroids_mkf: np.ndarray,
    *,
    weights_nf: np.ndarray | None = None,
) -> np.ndarray:
    """``(M, N)`` frequency-weighted coherence of every signal to every centroid."""
    xhat = np.asarray(xhat_nkf)
    cen = np.asarray(centroids_mkf)
    n, _, f = xhat.shape
    w = _resolve_weights(weights_nf, n, f)
    inner = np.einsum("mkf,nkf->mnf", cen.conj(), xhat, optimize=True)
    mag = inner.real**2 + inner.imag**2
    return np.einsum("mnf,nf->mn", mag, w, optimize=True)


def _fill_empty(labels, scores, n_clusters, on_empty, iteration, n_reseeds):
    counts = np.bincount(labels, minlength=n_clusters)
    for m in np.flatnonzero(counts == 0):
        if on_empty == "raise":
            raise ValueError(
                f"cluster {m} became empty at iteration {iteration}; lower "
                "n_clusters, or use on_empty='reseed'"
            )
        counts = np.bincount(labels, minlength=n_clusters)
        movable = counts[labels] > 1
        if not movable.any():
            raise ValueError(
                f"cluster {m} is empty and every other cluster is a singleton; "
                "n_clusters is too large for this data"
            )
        assigned = scores[labels, np.arange(labels.size)]
        worst = int(np.argmin(np.where(movable, assigned, np.inf)))
        labels[worst] = m
        n_reseeds += 1
    return labels, n_reseeds


def _init_labels(xhat, weights, n_clusters, init, rng):
    n = xhat.shape[0]
    if init == "random":
        seeds = rng.choice(n, size=n_clusters, replace=False)
    elif init == "++":
        # D^2-style seeding on 1 - coherence: each new seed is drawn with
        # probability proportional to how badly the chosen seeds explain it.
        seeds = [int(rng.integers(n))]
        best = centroid_scores(xhat, xhat[seeds[-1:]], weights_nf=weights)[0]
        for _ in range(1, n_clusters):
            d = np.clip(1.0 - best, 0.0, None)
            total = d.sum()
            probs = d / total if total > 0 else np.full(n, 1.0 / n)
            nxt = int(rng.choice(n, p=probs))
            seeds.append(nxt)
            s = centroid_scores(xhat, xhat[nxt : nxt + 1], weights_nf=weights)[0]
            best = np.maximum(best, s)
        seeds = np.asarray(seeds)
    else:
        raise ValueError(f"init must be '++', 'random' or a labels array, got {init!r}")
    scores = centroid_scores(xhat, xhat[seeds], weights_nf=weights)
    return scores.argmax(axis=0), scores


def _single_fit(xhat, weights, n_clusters, init, max_iter, on_empty, rng):
    n = xhat.shape[0]
    n_reseeds = 0
    if isinstance(init, np.ndarray):
        labels = init.astype(np.int64).copy()
    else:
        labels, seed_scores = _init_labels(xhat, weights, n_clusters, init, rng)
        labels, n_reseeds = _fill_empty(
            labels, seed_scores, n_clusters, on_empty, 0, n_reseeds
        )

    objective = []
    converged = False
    centroids = strength = scores = None
    for it in range(max_iter):
        centroids, strength = centroid_update(
            xhat, labels, n_clusters, weights_nf=weights
        )
        scores = centroid_scores(xhat, centroids, weights_nf=weights)
        if it == 0:
            objective.append(float(scores[labels, np.arange(n)].sum()))
        new_labels = scores.argmax(axis=0)
        new_labels, n_reseeds = _fill_empty(
            new_labels, scores, n_clusters, on_empty, it, n_reseeds
        )
        objective.append(float(scores[new_labels, np.arange(n)].sum()))
        if np.array_equal(new_labels, labels):
            converged = True
            break
        labels = new_labels
    return {
        "labels": labels.astype(np.int32),
        "centroids": centroids,
        "centroid_strength": strength,
        "scores": scores,
        "objective": np.asarray(objective),
        "n_iter": it + 1,
        "converged": converged,
        "n_reseeds": n_reseeds,
    }


def coherence_kmeans(
    s: Stft | np.ndarray,
    n_clusters: int,
    *,
    whiten: bool = True,
    init: Literal["++", "random"] | np.ndarray = "++",
    n_init: int = 10,
    max_iter: int = 300,
    on_empty: Literal["reseed", "raise"] = "reseed",
    random_state: int | None = 0,
) -> CoherenceKMeansResult:
    """Cluster ``(N, K, F)`` spectral coefficients by band coherence.

    Accepts an :class:`Stft` (preferred — provenance travels with the result)
    or a bare coefficient array. ``n_init`` independent restarts run and the
    best final objective wins; passing an explicit ``(N,)`` label array as
    ``init`` is the deterministic escape hatch (it forces a single run) and is
    how a caller reproduces the historical time-domain-KMeans initialization,
    which needs raw traces this library deliberately does not take.

    An empty cluster is repaired by deterministically reseeding it from the
    worst-fitting signal and counted in ``n_reseeds`` (``on_empty="reseed"``,
    the default), or raises naming the cluster and iteration
    (``on_empty="raise"``). k never shrinks silently — the pre-refactor
    implementation did exactly that.
    """
    coefs = s.coefs if isinstance(s, Stft) else np.asarray(s)
    meta = s.metadata() if isinstance(s, Stft) else {}
    xhat = normalize_coefs(coefs)
    n, _, f = xhat.shape

    if not 1 <= n_clusters <= n:
        raise ValueError(f"n_clusters must be in [1, {n}], got {n_clusters}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if on_empty not in ("reseed", "raise"):
        raise ValueError(f"on_empty must be 'reseed' or 'raise', got {on_empty!r}")
    if n_init < 1:
        raise ValueError(f"n_init must be >= 1, got {n_init}")

    if whiten:
        weights = None
    else:
        power = np.einsum(
            "nkf,nkf->nf", np.asarray(coefs), np.asarray(coefs).conj()
        ).real
        weights = power / power.sum(axis=1, keepdims=True)

    if isinstance(init, np.ndarray):
        lab = np.asarray(init)
        if lab.shape != (n,):
            raise ValueError(f"init labels shape {lab.shape} != ({n},)")
        present = np.unique(lab)
        if lab.min() < 0 or lab.max() >= n_clusters:
            raise ValueError(
                f"init labels must lie in [0, {n_clusters}), got range "
                f"[{lab.min()}, {lab.max()}]"
            )
        if present.size != n_clusters:
            missing = int(np.setdiff1d(np.arange(n_clusters), present)[0])
            raise ValueError(
                f"init labels leave cluster {missing} empty; every cluster "
                "needs at least one member"
            )
        runs = [
            _single_fit(xhat, weights, n_clusters, lab, max_iter, on_empty,
                        np.random.default_rng(random_state))
        ]
    else:
        master = np.random.default_rng(random_state)
        runs = [
            _single_fit(xhat, weights, n_clusters, init, max_iter, on_empty, child)
            for child in master.spawn(n_init)
        ]

    best = max(runs, key=lambda r: r["objective"][-1])
    return CoherenceKMeansResult(
        **best,
        whiten=whiten,
        random_state=random_state,
        stft_metadata=meta,
    )


class CoherenceKMeans:
    """Thin fit/predict wrapper around :func:`coherence_kmeans`.

    Holds parameters; all the mathematics lives in the free functions.
    """

    def __init__(
        self,
        n_clusters: int,
        *,
        whiten: bool = True,
        init: Literal["++", "random"] | np.ndarray = "++",
        n_init: int = 10,
        max_iter: int = 300,
        on_empty: Literal["reseed", "raise"] = "reseed",
        random_state: int | None = 0,
    ):
        self.n_clusters = n_clusters
        self.whiten = whiten
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.on_empty = on_empty
        self.random_state = random_state

    def fit(self, s: Stft | np.ndarray) -> CoherenceKMeans:
        self.result_ = coherence_kmeans(
            s,
            self.n_clusters,
            whiten=self.whiten,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            on_empty=self.on_empty,
            random_state=self.random_state,
        )
        return self

    def predict(self, s: Stft | np.ndarray) -> np.ndarray:
        """Labels for new signals against the fitted centroids."""
        if not hasattr(self, "result_"):
            raise ValueError("call fit() before predict()")
        coefs = s.coefs if isinstance(s, Stft) else np.asarray(s)
        xhat = normalize_coefs(coefs)
        if self.whiten:
            weights = None
        else:
            power = np.einsum(
                "nkf,nkf->nf", np.asarray(coefs), np.asarray(coefs).conj()
            ).real
            weights = power / power.sum(axis=1, keepdims=True)
        scores = centroid_scores(xhat, self.result_.centroids, weights_nf=weights)
        return scores.argmax(axis=0).astype(np.int32)
