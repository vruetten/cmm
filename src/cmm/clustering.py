"""Coherence spectral clustering: affinity, normalized Laplacian, embedding.

The coherence matrix is a natural similarity in ``[0, 1]``. A Gaussian kernel
on the pseudo-distance ``1 - C`` gives the affinity ``W``, the normalized
Laplacian's smallest eigenvectors give the embedding, and k-means on the
embedding gives labels (0-based).

Two knobs matter:

* ``tau`` — the Gaussian bandwidth. Small ``tau`` makes the graph sparse and
  locally connected (many small clusters); large ``tau`` makes it near-uniform
  (few large clusters). ``tau=None`` uses the median heuristic
  (:func:`median_tau`); sweep 0.5x-2x around it to check stability.
* ``laplacian`` — ``"sym"`` (Ng-Jordan-Weiss) or ``"rw"`` (Shi-Malik).
  Eigenvalues are identical; eigenvectors differ by a ``D^{-1/2}`` factor, and
  both are served from the same symmetric solve.

The default solver is ``eigsh`` on ``S = D^{-1/2} W D^{-1/2}`` with
``which="LA"`` and ``lambda_L = 1 - lambda_S``. Measured on a real N=10000
coherence graph (``logs/gate23_eigensolver_real_data.md``): 24x faster than
dense at 32 threads, eigenvalues to ~1e-15, identical labels (ARI 1.0000 in
all 12 configurations). A weak Laplacian gap does *not* make Lanczos slow — a
near-uniform affinity is near-rank-one, so the whole non-trivial spectrum sits
below an isolated ``lambda_1 = 1``; the old claim to the contrary in
``icampsnfr/spectral.py`` is measured false. ``solver="dense"`` remains for
audits and contiguous-spectrum eigengap sweeps. ``lobpcg`` and k-NN
sparsification were both measured wrong (ARI 0.39) and are deliberately not
offered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from cmm.coherence import CoherenceMatrix

__all__ = [
    "SpectralEmbedding",
    "affinity_gaussian",
    "cluster_embedding",
    "eigengap",
    "median_tau",
    "normalized_laplacian",
    "spectral_embedding",
    "spectral_labels",
]

Solver = Literal["eigsh", "dense"]
Laplacian = Literal["sym", "rw"]


@dataclass(frozen=True)
class SpectralEmbedding:
    """The smallest Laplacian eigenpairs of a coherence graph, and their recipe.

    Attributes:
        vectors: ``(N, n_eig)``, columns in ascending-eigenvalue order,
            unit-norm, sign-fixed (largest-magnitude component positive) so
            repeated runs and both solvers produce comparable embeddings.
        eigenvalues: ``(n_eig,)`` ascending eigenvalues of the Laplacian.
        tau: the Gaussian bandwidth actually used.
        laplacian: ``"sym"`` or ``"rw"``.
        solver: ``"eigsh"`` or ``"dense"``.
    """

    vectors: np.ndarray
    eigenvalues: np.ndarray
    tau: float
    laplacian: str
    solver: str

    @property
    def n_signals(self) -> int:
        return self.vectors.shape[0]

    @property
    def n_eig(self) -> int:
        return self.vectors.shape[1]

    def metadata(self) -> dict:
        return {
            "n_signals": self.n_signals,
            "n_eig": self.n_eig,
            "tau": float(self.tau),
            "laplacian": self.laplacian,
            "solver": self.solver,
            "eigenvalues": self.eigenvalues.tolist(),
        }


def _as_matrix(c) -> np.ndarray:
    v = c.values if isinstance(c, CoherenceMatrix) else np.asarray(c)
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] != v.shape[1]:
        raise ValueError(
            f"expected a square (N, N) coherence matrix, got shape {v.shape}"
        )
    if not np.isfinite(v).all():
        n_bad = int((~np.isfinite(v)).sum())
        raise ValueError(
            f"coherence matrix contains {n_bad} non-finite entries; filter "
            "degenerate signals (finite_power_mask) before clustering"
        )
    return v


def median_tau(c_nn) -> float:
    """Median-heuristic bandwidth: ``median(1 - C)`` over the strict off-diagonal.

    A scale-adaptive starting point, not an answer — sweep 0.5x-2x around it
    and check that the cluster count is stable.
    """
    v = _as_matrix(c_nn)
    if v.shape[0] < 2:
        raise ValueError("need at least 2 signals to define a distance distribution")
    iu = np.triu_indices(v.shape[0], k=1)
    d = 1.0 - np.clip(0.5 * (v + v.T), 0.0, 1.0)
    return float(np.median(d[iu]))


def affinity_gaussian(c_nn, tau: float) -> np.ndarray:
    """``W = exp(-(1 - C) / tau)``, self-loops zeroed.

    ``C`` is symmetrized and clipped to ``[0, 1]`` on entry to tolerate the
    last-bit asymmetries of the row-blocked matmul. The diagonal is zeroed
    after exponentiation so degree sums neighbours only — the standard
    graph-Laplacian convention.
    """
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    v = _as_matrix(c_nn)
    c_sym = np.clip(0.5 * (v + v.T), 0.0, 1.0)
    w = np.exp(-(1.0 - c_sym) / tau)
    np.fill_diagonal(w, 0.0)
    return w


def _degrees(w_nn: np.ndarray) -> np.ndarray:
    d = w_nn.sum(axis=1)
    if np.any(d <= 0):
        n_iso = int(np.sum(d <= 0))
        raise ValueError(
            f"{n_iso} isolated nodes (zero degree) in the affinity graph; "
            "increase tau or drop degenerate rows of C before calling"
        )
    return d


def normalized_laplacian(w_nn: np.ndarray, kind: str = "sym") -> np.ndarray:
    """Dense normalized graph Laplacian, ``"sym"`` or ``"rw"`` — the audit path.

    Note the solvers never eigendecompose the ``"rw"`` form directly: it is
    not symmetric, and symmetrizing it changes the spectrum. Both kinds are
    served from the symmetric solve (``u_rw = D^{-1/2} u_sym``); this function
    exists to write either Laplacian down explicitly for inspection.
    """
    w = np.asarray(w_nn, dtype=np.float64)
    d = _degrees(w)
    n = w.shape[0]
    if kind == "sym":
        d_invhalf = 1.0 / np.sqrt(d)
        return np.eye(n) - (w * d_invhalf[:, None]) * d_invhalf[None, :]
    if kind == "rw":
        return np.eye(n) - w / d[:, None]
    raise ValueError(f"unknown Laplacian kind: {kind!r}; expected 'sym' or 'rw'")


def _fix_signs(u: np.ndarray) -> np.ndarray:
    """Flip each column so its largest-magnitude component is positive."""
    idx = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[idx, np.arange(u.shape[1])])
    signs[signs == 0] = 1.0
    return u * signs


def spectral_embedding(
    c,
    n_eig: int,
    *,
    tau: float | None = None,
    laplacian: Laplacian = "sym",
    solver: Solver = "eigsh",
    random_state: int | None = 0,
) -> SpectralEmbedding:
    """Smallest ``n_eig`` Laplacian eigenpairs of the coherence graph.

    Accepts a :class:`CoherenceMatrix` or a bare ``(N, N)`` array.
    ``tau=None`` uses :func:`median_tau`.

    ``solver="eigsh"`` runs ARPACK on ``S = D^{-1/2} W D^{-1/2}`` for the
    *largest* eigenvalues — the numerically easy end — and maps back via
    ``lambda_L = 1 - lambda_S``. Its start vector is seeded from
    ``random_state`` so repeated runs are identical; ``tol`` stays 0 (an
    inexact Krylov solve is how lobpcg produced matching eigenvalues and
    wrong clusters). ``solver="dense"`` computes the same pairs via
    ``scipy.linalg.eigh``; use it for audits or when ``n_eig`` approaches N.

    For a k-sweep, embed once at ``n_eig = max_k + 1`` and call
    :func:`cluster_embedding` per k — the solve is the expensive step.
    """
    v = _as_matrix(c)
    n = v.shape[0]
    if not 1 <= n_eig <= n:
        raise ValueError(f"n_eig must be in [1, {n}], got {n_eig}")
    tau = median_tau(v) if tau is None else float(tau)
    w_aff = affinity_gaussian(v, tau)
    d = _degrees(w_aff)
    d_invhalf = 1.0 / np.sqrt(d)

    if solver == "eigsh":
        if n_eig >= n:
            raise ValueError(
                f"eigsh (ARPACK) needs n_eig < N, got n_eig={n_eig} at N={n}; "
                "use solver='dense' for the full spectrum"
            )
        s = (w_aff * d_invhalf[:, None]) * d_invhalf[None, :]
        v0 = (
            np.random.default_rng(random_state).standard_normal(n)
            if random_state is not None
            else None
        )
        lam_s, u = eigsh(s, k=n_eig, which="LA", v0=v0, tol=0)
        lam = 1.0 - lam_s
        order = np.argsort(lam)
        lam, u = lam[order], u[:, order]
    elif solver == "dense":
        lap = normalized_laplacian(w_aff, kind="sym")
        lap = 0.5 * (lap + lap.T)
        lam, u = eigh(lap, subset_by_index=(0, n_eig - 1))
    else:
        raise ValueError(f"solver must be 'eigsh' or 'dense', got {solver!r}")

    if laplacian == "rw":
        u = u * d_invhalf[:, None]
        u = u / np.linalg.norm(u, axis=0, keepdims=True)
    elif laplacian != "sym":
        raise ValueError(f"laplacian must be 'sym' or 'rw', got {laplacian!r}")

    return SpectralEmbedding(
        vectors=_fix_signs(u),
        eigenvalues=lam,
        tau=tau,
        laplacian=laplacian,
        solver=solver,
    )


def eigengap(eigenvalues, *, min_k: int = 2) -> tuple[np.ndarray, int]:
    """Consecutive eigenvalue gaps and the ``k`` the eigengap heuristic favours.

    For an ideal graph of ``k`` well-separated components the normalized
    Laplacian has exactly ``k`` near-zero eigenvalues, so the largest jump in
    the ascending spectrum sits between index ``k-1`` and ``k``. Real
    coherence graphs are never that clean — read the winner as a hint about
    which ``k`` the data supports, not as an answer.

    ``min_k=2`` excludes the ``k=1`` gap: a connected graph has one zero
    eigenvalue, so ``lambda_0 -> lambda_1`` is a large jump by construction
    and would win on essentially every dense affinity graph, reporting the
    useless answer "one cluster". Pass ``min_k=1`` to include it, e.g. to
    count connected components.

    Returns ``(gaps, k_suggested)`` where ``gaps[i] = w[i+1] - w[i]`` implies
    ``k = i + 1``; only the argmax search is restricted by ``min_k``.
    """
    w = np.asarray(eigenvalues, dtype=np.float64).ravel()
    if w.size < 2:
        raise ValueError(f"need at least 2 eigenvalues to form a gap, got {w.size}")
    if min_k < 1:
        raise ValueError(f"min_k must be >= 1, got {min_k}")
    gaps = np.diff(w)
    start = min_k - 1
    if start >= gaps.size:
        raise ValueError(
            f"min_k={min_k} leaves no gap to search among {gaps.size} gaps; "
            "compute more eigenvalues"
        )
    return gaps, int(np.argmax(gaps[start:])) + start + 1


def cluster_embedding(
    emb: SpectralEmbedding | np.ndarray,
    n_clusters: int,
    *,
    drop_first: bool = True,
    row_normalize: bool = True,
    random_state: int = 0,
    n_init: int = 10,
) -> np.ndarray:
    """K-means on the spectral embedding; ``(N,)`` int32 labels starting at 0.

    ``drop_first`` skips the trivial constant eigenvector (correct for a
    connected graph). ``row_normalize`` is the Ng-Jordan-Weiss step —
    projecting rows onto the unit sphere stops low-degree nodes with large
    eigenvector components from dragging k-means around.
    """
    u = emb.vectors if isinstance(emb, SpectralEmbedding) else np.asarray(emb)
    if u.ndim != 2:
        raise ValueError(f"expected an (N, n_eig) embedding, got shape {u.shape}")
    if n_clusters < 1:
        raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
    x = u[:, 1:] if drop_first and u.shape[1] > 1 else u
    if x.shape[1] < 1:
        raise ValueError("no eigenvectors left after drop_first; increase n_eig")
    if row_normalize:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x = np.where(norms > 0, x / np.where(norms > 0, norms, 1.0), x)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    return km.fit(x).labels_.astype(np.int32)


def spectral_labels(
    c,
    n_clusters: int,
    *,
    tau: float | None = None,
    n_eig: int | None = None,
    laplacian: Laplacian = "sym",
    solver: Solver = "eigsh",
    row_normalize: bool = True,
    random_state: int = 0,
) -> np.ndarray:
    """One-shot: coherence matrix (or :class:`CoherenceMatrix`) to labels.

    ``n_eig=None`` uses ``n_clusters + 1``, leaving room for the constant
    eigenvector that gets dropped. For a k-sweep, call
    :func:`spectral_embedding` once and :func:`cluster_embedding` per k.
    """
    if n_eig is None:
        n_eig = n_clusters + 1
    emb = spectral_embedding(
        c, n_eig, tau=tau, laplacian=laplacian, solver=solver, random_state=random_state
    )
    return cluster_embedding(
        emb, n_clusters, drop_first=True, row_normalize=row_normalize,
        random_state=random_state,
    )
