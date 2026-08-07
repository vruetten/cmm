"""Phase 3 chunk 5: the import surface.

``import cmm`` must be side-effect-free and cheap; every advertised name must
resolve; and ``cmm.coherence`` must unambiguously be the submodule, never the
per-frequency function from cmm.spectral.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_bare_import_is_lazy_and_sklearn_free():
    # A fresh interpreter, because this test's own siblings import everything.
    code = (
        "import sys, cmm; "
        "heavy = [m for m in ('sklearn', 'cmm.spectral', 'cmm.coherence', "
        "'cmm.clustering', 'cmm.kmeans', 'cmm.metrics', 'cmm.phase', "
        "'cmm.wavelet', 'cmm.wiener') if m in sys.modules]; "
        "assert not heavy, f'import cmm eagerly loaded {heavy}'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_every_advertised_name_resolves():
    import cmm

    for name in cmm.__all__:
        assert getattr(cmm, name) is not None


def test_top_level_names_are_the_submodule_names():
    import cmm
    from cmm.coherence import coherence_matrix
    from cmm.kmeans import coherence_kmeans
    from cmm.metrics import coherence_silhouette

    assert cmm.coherence_matrix is coherence_matrix
    assert cmm.coherence_kmeans is coherence_kmeans
    assert cmm.coherence_silhouette is coherence_silhouette


def test_cmm_coherence_is_the_submodule_not_the_function():
    import cmm
    import cmm.spectral

    assert cmm.coherence.__name__ == "cmm.coherence"
    assert callable(cmm.spectral.coherence)
    assert "coherence" not in cmm.__all__


def test_unknown_attribute_raises_attribute_error():
    import cmm

    with pytest.raises(AttributeError, match="no attribute 'welch'"):
        _ = cmm.welch


def test_dir_lists_the_public_api():
    import cmm

    listing = dir(cmm)
    assert "coherence_matrix" in listing
    assert "spectral" in listing
    assert "coherence_kmeans" in listing
