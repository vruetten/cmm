"""Attribute every difference between the new cmm.spectral and the frozen baseline.

The baseline is not a regression target — the old code is wrong in specific,
measured ways. This script asserts that each observed change matches a *named*
bug fix, and fails loudly on any change that does not. An unexplained
difference is a new bug.

Like make_baseline.py, check 6 (the C3 regression proof) calls the legacy
``estimate_spectrum`` with the ``t=`` keyword that only exists in the
*working-tree* ``spectral_funcs.py`` — it crashes against the committed
version. See the provenance caveat in make_baseline.py.

Run from the repo root:  python logs/baseline/compare_to_baseline.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cmm.spectral import SpectralParams, coherence, psd, stft

from make_baseline import fixed_input  # noqa: E402  (same directory)

B = np.load(Path(__file__).parent / "baseline.npz")
FS, NPERSEG, NOVERLAP = 10.0, 128, 64
OK = True


def report(name: str, verdict: bool, detail: str) -> None:
    global OK
    OK &= verdict
    print(f"  [{'OK ' if verdict else 'FAIL'}] {name}\n         {detail}")


def main() -> None:
    x, _ = fixed_input()
    p = SpectralParams(fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP, detrend="constant")
    s = stft(x, p)

    # --- 1. Raw coefficients: no bug touched this path, expect bit-identical.
    d = np.abs(s.coefs - B["coefs"]).max()
    report("stft coefficients unchanged", d < 1e-12,
           f"max|diff| = {d:.3e} (expected ~0: neither applies scaling)")

    d = np.abs(s.freqs - B["coefs_freqs"]).max()
    report("full-band frequency axis unchanged", d == 0.0,
           f"max|diff| = {d:.3e} (the old axis was only wrong for band-limited input)")

    # --- 2. PSD density: C2, missing one-sided doubling. Expect exactly 2x on
    #        interior bins and 1x at DC and Nyquist.
    new, f = psd(s, scaling="density")
    ratio = new / B["psd_density"]
    interior = np.ones(f.size, dtype=bool)
    interior[[0, -1]] = False
    ok = (np.allclose(ratio[:, interior], 2.0, rtol=1e-9)
          and np.allclose(ratio[:, [0, -1]], 1.0, rtol=1e-9))
    report("PSD density: one-sided doubling restored (C2)", ok,
           f"new/old = {ratio[:, interior].mean():.6f} interior, "
           f"{ratio[:, 0].mean():.6f} at DC, {ratio[:, -1].mean():.6f} at Nyquist")

    # --- 3. PSD spectrum: C2, wrong denominator *and* missing doubling.
    #        old used 1/(K*(win**2).sum()); correct is 1/win.sum()**2.
    new_sp, _ = psd(s, scaling="spectrum")
    win = s.window
    predicted = ((win * win).sum() / win.sum() ** 2) * np.where(interior, 2.0, 1.0)
    ratio_sp = new_sp / B["psd_spectrum"]
    ok = np.allclose(ratio_sp, predicted[None], rtol=1e-9)
    report("PSD spectrum: correct denominator + doubling (C2)", ok,
           f"new/old = {ratio_sp[:, interior].mean():.6f} interior, "
           f"predicted {predicted[interior][0]:.6f}")

    # --- 4. Coherence: every normalization cancels, expect unchanged.
    c, _ = coherence(s, stft(x[:3].copy(), p))
    old = np.moveaxis(B["coherence"], 0, 1)  # old was (M, N, F)
    d = np.nanmax(np.abs(c - old))
    report("coherence unchanged", d < 1e-12,
           f"max|diff| = {d:.3e} (scaling cancels in the ratio — this is why "
           "the old coherence was correct despite C2)")

    # --- 5. Band-limited path. The OLD compute_spectral_coefs axis was already
    #        correct, so expect no change here. C3 lives elsewhere (below).
    pb = p.with_(band_hz=(0.2, 2.0))
    sb = stft(x, pb)
    full = np.fft.rfftfreq(NPERSEG, d=1 / FS)
    truth = full[(full >= 0.2) & (full <= 2.0)]
    ok = np.array_equal(sb.freqs, truth) and np.allclose(B["coefs_band_freqs"], truth)
    report("band frequency axis unchanged (old one was already right)", ok,
           "compute_spectral_coefs derived freqs from rfftfreq(nfft) and filtered — correct")

    d = np.abs(sb.coefs - B["coefs_band"]).max()
    report("band-limited coefficients unchanged", d < 1e-12,
           f"max|diff| = {d:.3e}")

    # --- 6. C3 proper: estimate_spectrum(x_in_coefs=True) rebuilt the axis as
    #        rfftfreq(2F-1) from the coefficient count. That is the path CMM
    #        uses internally, and it is the one that was wrong. The new API has
    #        no such path — an Stft carries its axis — so the fix is structural.
    from cmm.spectral_funcs import estimate_spectrum

    _, old_axis = estimate_spectrum(
        np.asarray(B["coefs"]), fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP,
        x_in_coefs=True, alltoall=False, abs=True, detrend="constant",
        t=x.shape[-1],
    )
    err = np.abs(np.asarray(old_axis) - np.asarray(B["coefs_freqs"])).max()
    report("C3 confirmed on the x_in_coefs path, and structurally removed", err > 1e-3,
           f"old axis was wrong by up to {err:.4f} Hz "
           f"(top bin {np.asarray(old_axis)[-1]:.4f} vs true {B['coefs_freqs'][-1]:.4f}); "
           "the new API carries s.freqs instead of reconstructing it")

    print("\n" + ("all differences attributed to named fixes"
                  if OK else "UNEXPLAINED DIFFERENCE — investigate"))
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
