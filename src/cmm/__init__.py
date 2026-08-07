"""Coherence-based clustering of timeseries.

Spectral core (:mod:`cmm.spectral`), band-averaged coherence matrix
(:mod:`cmm.coherence`), and two clustering routes — coherence k-means
(:mod:`cmm.kmeans`) and coherence spectral clustering (:mod:`cmm.clustering`)
— with shared metrics (:mod:`cmm.metrics`). Alongside: narrowband phase
tools (:mod:`cmm.phase`, :mod:`cmm.wavelet`).

Everything below is loaded lazily (PEP 562): ``import cmm`` has no side
effects and imports neither scipy nor sklearn until a symbol that needs them
is touched.

One deliberate omission: the per-frequency pairwise :func:`cmm.spectral.coherence`
function is *not* re-exported at the top level, because ``cmm.coherence`` is
the coherence-matrix submodule and giving one name two meanings depending on
import order is how silent bugs start. Use ``cmm.spectral.coherence`` for the
per-frequency curve and :func:`cmm.coherence_matrix` for the band-averaged
matrix.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cmm")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Virginia M.S. Rutten"
__email__ = "vms.rutten@gmail.com"

_SUBMODULES = {
    "spectral", "coherence", "clustering", "kmeans", "metrics",
    "phase", "wavelet",
}

_EXPORTS = {
    # spectral
    "SpectralParams": "spectral",
    "Stft": "spectral",
    "stft": "spectral",
    "psd": "spectral",
    "csd": "spectral",
    "segment": "spectral",
    # coherence
    "CoherenceMatrix": "coherence",
    "coherence_matrix": "coherence",
    "debias": "coherence",
    "to_distance": "coherence",
    "effective_bins": "coherence",
    "finite_power_mask": "coherence",
    # clustering
    "SpectralEmbedding": "clustering",
    "median_tau": "clustering",
    "affinity_gaussian": "clustering",
    "normalized_laplacian": "clustering",
    "spectral_embedding": "clustering",
    "eigengap": "clustering",
    "cluster_embedding": "clustering",
    "spectral_labels": "clustering",
    # kmeans
    "CoherenceKMeansResult": "kmeans",
    "CoherenceKMeans": "kmeans",
    "coherence_kmeans": "kmeans",
    "normalize_coefs": "kmeans",
    "centroid_update": "kmeans",
    "centroid_scores": "kmeans",
    # metrics
    "coherence_silhouette": "metrics",
    "pairwise_ari": "metrics",
    "label_stability": "metrics",
    "cluster_summary": "metrics",
    # phase
    "analytic_signal": "phase",
    "phase_amplitude": "phase",
    "phase_triggered_average": "phase",
    "bin_centers": "phase",
    "valid_row_mask": "phase",
    # wavelet
    "morlet_wavelet": "wavelet",
    "wavelet_analytic": "wavelet",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(import_module(f"cmm.{_EXPORTS[name]}"), name)
    if name in _SUBMODULES:
        return import_module(f"cmm.{name}")
    raise AttributeError(f"module 'cmm' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS) | _SUBMODULES)
