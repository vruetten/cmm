"""The shared toy-data generators: the properties every consumer relies on."""

from __future__ import annotations

import numpy as np
import pytest

from cmm.spectral import SpectralParams, stft
from cmm.coherence import coherence_matrix
from cmm.synthetic import bandlimited_source, grouped_traces


def test_bandlimited_source_is_unit_variance_and_in_band():
    rng = np.random.default_rng(0)
    fs, t = 8.0, 4096
    src = bandlimited_source(rng, t, fs, band=(0.3, 1.2))
    assert src.shape == (t,)
    assert src.std() == pytest.approx(1.0)
    spec = np.abs(np.fft.rfft(src)) ** 2
    freqs = np.fft.rfftfreq(t, 1 / fs)
    out_of_band = spec[(freqs < 0.3) | (freqs > 1.2)]
    assert out_of_band.max() < 1e-20 * spec.max()


def test_two_sources_are_incoherent_but_not_identical():
    rng = np.random.default_rng(1)
    a = bandlimited_source(rng, 4096, 8.0)
    b = bandlimited_source(rng, 4096, 8.0)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.1


def test_grouped_traces_deterministic_and_labeled():
    x1, l1 = grouped_traces([4, 5, 6], seed=7)
    x2, l2 = grouped_traces([4, 5, 6], seed=7)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(l1, l2)
    assert x1.shape == (15, 3072)
    np.testing.assert_array_equal(l1, np.repeat([0, 1, 2], [4, 5, 6]))


def test_groups_separate_in_coherence():
    x, labels = grouped_traces([6, 6], seed=2)
    s = stft(x, SpectralParams(fs=8.0, nperseg=256, band_hz=(0.2, 1.5)))
    c = coherence_matrix(s).values
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(labels.size, dtype=bool)
    within = c[same & off_diag].mean()
    between = c[~same].mean()
    assert within > 3 * between, (within, between)


def test_degenerate_inputs_raise():
    with pytest.raises(ValueError, match="sizes must be positive"):
        grouped_traces([])
    with pytest.raises(ValueError, match="sizes must be positive"):
        grouped_traces([3, 0])
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="low < high"):
        bandlimited_source(rng, 1024, 8.0, band=(1.2, 0.3))
    with pytest.raises(ValueError, match="selects no frequency bins"):
        bandlimited_source(rng, 64, 8.0, band=(0.001, 0.002))
