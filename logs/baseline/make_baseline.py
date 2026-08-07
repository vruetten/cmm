"""Freeze the CURRENT (pre-refactor) spectral outputs so changes can be attributed.

This is not a regression target — the current code is known wrong in specific
ways (see CLAUDE.md). It is an *attribution* tool: after the Phase 1 rewrite,
every difference against this snapshot must be explained by a named bug fix. A
difference that cannot be explained is a new bug.

PROVENANCE CAVEAT: this script runs against the *working-tree*
``src/cmm/spectral_funcs.py``, which is ahead of anything committed — the
committed version returns 2 values from ``compute_spectral_coefs`` (this
script unpacks 3) and lacks ``estimate_spectrum``'s ``t=`` keyword. A git sha
alone therefore cannot reproduce ``baseline.npz``; ``baseline_meta.json``
records the sha256 of the actual ``spectral_funcs.py`` used, plus whether the
tree was dirty, so the true input is at least identifiable. To make this
reproducible from history, commit that file's current state.

Run from the repo root:  python logs/baseline/make_baseline.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from cmm.spectral_funcs import compute_coherence, compute_spectral_coefs, estimate_spectrum

OUT = Path(__file__).parent


def fixed_input():
    """Deterministic (N, T) traces with structure at several in-band frequencies."""
    rng = np.random.default_rng(20260806)
    n, t, fs = 12, 2048, 10.0
    tt = np.arange(t) / fs
    x = rng.standard_normal((n, t))
    # a few shared oscillations so coherence is non-trivial
    for i, f0 in enumerate((0.3, 0.7, 1.5)):
        x[i::3] += 2.0 * np.sin(2 * np.pi * f0 * tt + i)
    x += 0.5 * np.cumsum(rng.standard_normal((n, t)), axis=1) / np.sqrt(t)  # drift
    return np.ascontiguousarray(x), fs


def main():
    x, fs = fixed_input()
    nperseg, noverlap = 128, 64
    out = {}

    coefs, freqs, win = compute_spectral_coefs(
        x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant"
    )
    out["coefs"] = np.asarray(coefs)
    out["coefs_freqs"] = np.asarray(freqs)
    out["window"] = np.asarray(win)

    for scaling in ("density", "spectrum"):
        p, f = estimate_spectrum(
            x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant",
            scaling=scaling, alltoall=False, abs=True,
        )
        out[f"psd_{scaling}"] = np.asarray(p)
        out[f"psd_{scaling}_freqs"] = np.asarray(f)

    coh, f = compute_coherence(
        x[:3], x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant"
    )
    out["coherence"] = np.asarray(coh)
    out["coherence_freqs"] = np.asarray(f)

    # Band-limited coefficients — the path icampsnfr uses.
    coefs_b, freqs_b, _ = compute_spectral_coefs(
        x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant",
        freq_minmax=[0.2, 2.0],
    )
    out["coefs_band"] = np.asarray(coefs_b)
    out["coefs_band_freqs"] = np.asarray(freqs_b)

    # The input itself is not stored — `fixed_input()` regenerates it from a
    # fixed seed, so there is one source of truth for it.
    np.savez_compressed(OUT / "baseline.npz", **out)

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    import cmm.spectral_funcs as _sf

    sf_path = Path(_sf.__file__)
    sf_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", str(sf_path)],
            capture_output=True, text=True,
        ).stdout.strip()
    )
    if sf_dirty:
        print(
            "WARNING: spectral_funcs.py is dirty — baseline.npz depends on "
            "uncommitted code; the git sha alone cannot reproduce it"
        )
    meta = {
        "git_sha": sha,
        "spectral_funcs_dirty": sf_dirty,
        "spectral_funcs_sha256": hashlib.sha256(sf_path.read_bytes()).hexdigest()[:16],
        "fs": fs,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "arrays": {
            k: {
                "shape": list(np.shape(v)),
                "dtype": str(np.asarray(v).dtype),
                "sha256": hashlib.sha256(
                    np.ascontiguousarray(v).tobytes()
                ).hexdigest()[:16],
            }
            for k, v in out.items()
        },
    }
    (OUT / "baseline_meta.json").write_text(json.dumps(meta, indent=2))
    for k, v in out.items():
        a = np.asarray(v)
        print(f"  {k:24s} {str(a.shape):18s} {a.dtype}")
    print(f"\ngit sha {sha}")


if __name__ == "__main__":
    main()
