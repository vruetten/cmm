# Adversarial review — Phase 1 spectral core (commit f0cf4b2)

Scope: `src/cmm/spectral.py`, `tests/test_spectral_vs_scipy.py`,
`tests/test_spectral_properties.py`, `logs/baseline/make_baseline.py`,
`logs/baseline/compare_to_baseline.py`, all as committed at f0cf4b2 on
`refactor/phase-1-spectral-core`.

The previously-killed review instance found only that the "21.3x" claim in the
commit message is exact. This pass found two HIGH-severity items beyond that:
one is a real numerical-correctness bug in the shipped API, the other is that
the audit's own attribution pipeline cannot be reproduced from git history —
it silently depended on uncommitted state. Both are demonstrated below with
runnable snippets. `86/86` tests pass and `python -m pytest ...` is green, as
claimed; that claim is not in dispute. What follows is what the green suite
does not tell you.

All mutation testing was performed with `git checkout -- src/cmm/spectral.py`
after each mutation; `git diff --stat src/cmm/spectral.py` is empty at the end
of this review, confirmed above.

---

### 1. [SEVERITY: HIGH] `csd`/`coherence` use only `sx`'s normalization constant — breaks the `[0,1]` bound and the "scaling cancels" invariant whenever `sx` and `sy` come from differently-windowed `Stft`s

**Location**: `src/cmm/spectral.py:326-331` (`csd`), `:349-350` (`coherence`), `:364-371` (`_check_compatible`)

**Bug/bias**: `csd(sx, sy, ...)` scales the cross-spectrum by `sx.scale(scaling)` only (line 330: `p = p * sx.scale(scaling) * sx.onesided_factor`). `coherence` then divides `|pxy|**2` by `psd(sx) * psd(sy)`, where `psd(sy)` uses **`sy`'s own** scale. If `sx` and `sy` were built with different windows (same `nperseg`/`noverlap`/`fs`, which is enough to pass `_check_compatible` — it only checks `n_segments` and `freqs`), the `a = sx.scale` factor embedded in `pxy` does not cancel against the `b = sy.scale` embedded in `pyy`; the result is scaled by `a/b`, which is neither bounded nor unity in general.

**Why it matters**: This directly contradicts the module's own docstring: `coherence`'s docstring says the result is "Magnitude-squared coherence per frequency, in `[0, 1]`" and "Every normalization constant cancels in the ratio." Neither is true once the two `Stft`s disagree on window. Reproduced:

```python
import numpy as np
from cmm.spectral import SpectralParams, stft, coherence

rng = np.random.default_rng(1)
x = rng.standard_normal((1, 2048))
y = 3.0 * x  # perfectly coherent: coherence must be exactly 1 everywhere

sx = stft(x, SpectralParams(fs=10.0, nperseg=128, window="hann"))
sy = stft(y, SpectralParams(fs=10.0, nperseg=128, window=("tukey", 0.1)))
c, f = coherence(sx, sy, pairwise=True)
print(c.min(), c.max())   # -> 0.6788 2.0906  (docstring promises exactly 1.0, always in [0,1])
```
Output: `min 0.6788, max 2.0906`. For independent noise with `hann` vs `boxcar` windows the null floor is inflated by exactly `a/b = 2.667` (measured), turning a should-be-~0.03 null coherence into values up to 0.41. Nothing in the API prevents this: two `stft()` calls with the same `fs`/`nperseg`/`noverlap` but different `window` arguments is a perfectly ordinary, encouraged usage pattern (that's the entire reason `coherence(sx, sy)` takes two separate `Stft` objects instead of one combined call). Downstream, this corrupts any coherence matrix built by transforming different channel groups with different windows (e.g. per-modality preprocessing), silently, with values that can exceed 1.

**Suggested fix**: `_check_compatible` must check that the two `Stft`s share the normalization-relevant parameters, not just segment count and frequency axis:
```python
def _check_compatible(sx: Stft, sy: Stft) -> None:
    if not np.array_equal(sx.window, sy.window):
        raise ValueError(
            "sx and sy were transformed with different windows "
            f"({sx.window.shape} vs {sy.window.shape} content differs); "
            "psd/csd/coherence normalization assumes a shared window"
        )
    ...
```
Also add a Layer-1 test constructing `sx`, `sy` from different windows and asserting either a raise, or (if mixed windows are meant to be legal) that `coherence` stays bounded and matches an independently-normalized reference.

---

### 2. [SEVERITY: HIGH] The baseline-attribution pipeline cannot be reproduced from git history — `make_baseline.py` crashes against the `spectral_funcs.py` actually committed at f0cf4b2, and so does `compare_to_baseline.py`'s single most important check

**Location**: `logs/baseline/make_baseline.py:43-48` and `:20`; `logs/baseline/compare_to_baseline.py:95-101`; cross-file: `logs/baseline/*.py ↔ src/cmm/spectral_funcs.py@f0cf4b2`

**Bug/bias**: `make_baseline.py` (committed in f0cf4b2) calls:
```python
coefs, freqs, win = compute_spectral_coefs(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend="constant")
```
unpacking **three** return values. At the actual commit f0cf4b2 (and at its parent a737118, which `baseline_meta.json`'s `git_sha` field claims is the baseline's provenance), `compute_spectral_coefs` returns only **two** values (`coefs_xnkf, freqs` — verified with `git show f0cf4b2:src/cmm/spectral_funcs.py`) and *does* apply `scaling="spectrum"` in-place (`coefs_xnkf *= scale`, default `scaling="spectrum"`). The three-value-returning, unscaled version only exists in the currently uncommitted, dirty working tree (`git status` shows `M src/cmm/spectral_funcs.py`). I confirmed this by checking out f0cf4b2 into a clean `git worktree` and running the script verbatim:

```
$ git worktree add --detach /tmp/wt f0cf4b2
$ cd /tmp/wt && PYTHONPATH=src python logs/baseline/make_baseline.py
Traceback (most recent call last):
  ...
  File ".../make_baseline.py", line 43, in main
    coefs, freqs, win = compute_spectral_coefs(...)
ValueError: not enough values to unpack (expected 3, got 2)
```

The checked-in `logs/baseline/baseline.npz` (which IS committed as part of f0cf4b2) therefore was not produced by any code that exists at any commit in this history — it was generated against a since-modified, still-uncommitted `spectral_funcs.py`. The `git_sha` recorded in `baseline_meta.json` (`a73711864e7880df94572df3e2877f4d58ab5c79`) is consequently **false provenance**: neither that commit nor f0cf4b2 can reproduce the artifact.

It gets worse: `compare_to_baseline.py`, also run from the same clean worktree against the actually-committed `spectral_funcs.py`, gets through 7 of its 8 checks (they only touch the frozen `.npz` and the new `cmm.spectral`, so they're insulated from this) and then crashes on check #6 — the one titled `"C3 confirmed on the x_in_coefs path, and structurally removed"`, i.e. the regression proof for the headline bug (C3, the "wrong frequency axis" bug) that motivated this entire Phase 1 rewrite:
```
$ PYTHONPATH=src:logs/baseline python logs/baseline/compare_to_baseline.py
  [OK ] stft coefficients unchanged
  ... (6 more OK) ...
Traceback (most recent call last):
  File ".../compare_to_baseline.py", line 97, in main
    _, old_axis = estimate_spectrum(..., t=x.shape[-1])
TypeError: estimate_spectrum() got an unexpected keyword argument 't'
```
The `t=` keyword was added to `estimate_spectrum`/`compute_coherence` only in the dirty, uncommitted edits to `spectral_funcs.py`.

**Why it matters**: The Phase 1 gate in `.claude/plans/cmm_refactor_plan.md` states "every diff vs baseline attributed," and the commit message asserts specific measured ratios (21.3x, 0.5x, etc.) that this script is supposed to re-derive and check on every run. Per the project's own working rule ("Measure, do not assume... Keep it that way"), an attribution artifact whose generating code is not in git history is unverifiable — nobody can re-run the audit from a fresh clone of this exact commit and get anything but a crash. If `spectral_funcs.py`'s dirty edits are ever committed, stashed, or lost, this whole audit trail silently breaks with no warning; conversely, right now, anyone bisecting or auditing this commit in isolation cannot confirm the claimed numbers at all.

**Suggested fix**: Before committing `baseline.npz`, verify `git status --porcelain src/cmm/spectral_funcs.py` is empty, or better, have `make_baseline.py` assert this itself (`subprocess.run(["git","diff","--quiet","--", "src/cmm/spectral_funcs.py"])` and raise if dirty) so the recorded `git_sha` is trustworthy by construction. Either commit the `spectral_funcs.py` state the baseline actually depends on, or regenerate `baseline.npz` from the code that is genuinely at HEAD and re-run both scripts to confirm they pass before merging.

---

### 3. [SEVERITY: MEDIUM-HIGH] Test gap: the all-pairs `csd()` phase/conjugate convention between *distinct* signals is never checked against scipy — a full conjugate-operand swap survives the whole suite

**Location**: `src/cmm/spectral.py:328` (`csd`, `pairwise=False` branch) ↔ `tests/test_spectral_vs_scipy.py::TestCsdVsScipy`, `tests/test_spectral_properties.py::TestInvariants::test_csd_is_hermitian_under_signal_exchange`

**Bug/bias**: `csd(sx)` with `pairwise=False` builds the `(N, M, F)` all-pairs cross-spectrum via `np.einsum("nkf,mkf->nmf", sx.coefs.conj(), sy.coefs, ...)`. Mutation test: swap which operand is conjugated —
```python
p = np.einsum("nkf,mkf->nmf", sx.coefs, sy.coefs.conj(), optimize=True)  # conj on the other operand
```
This flips the phase sign of every off-diagonal entry (equivalent to inverting every inferred delay direction between distinct signal pairs — exactly the quantity `derivations/coherence_estimators.md` calls out as sign-sensitive: "magnitudes agree, phase sign flips"). Running the full suite against this mutation:
```
$ python -m pytest tests/test_spectral_vs_scipy.py tests/test_spectral_properties.py -q
........................................................................ [ 83%]
..............                                                           [100%]
86 passed in 2.65s
```
**86/86 pass. Nothing catches it.** The reason: `test_alltoall_diagonal_is_the_pairwise_result` only checks the diagonal (self-pairs, where `conj(X)*X = X*conj(X)`, insensitive to which operand is conjugated), and `test_csd_is_hermitian_under_signal_exchange` only checks `full == conj(full.T)`, which a *global* conjugate-operand swap still satisfies (swapping which side is conjugated is itself an operation consistent with Hermitian symmetry). `coherence()`'s tests never expose this either, because `|pxy|**2` is phase-blind by construction — coherence is squared-magnitude, so it cannot detect a sign flip in phase; only a direct `csd` vs. `scipy.signal.csd` off-diagonal comparison can.

I verified the *current, unmutated* code is actually correct here (so this is a coverage gap, not a live bug):
```python
import numpy as np
from scipy import signal as sps
from cmm.spectral import SpectralParams, stft, csd

rng = np.random.default_rng(0)
x = rng.standard_normal((3, 2048))
p = SpectralParams(fs=10.0, nperseg=128)
s = stft(x, p)
full, f = csd(s)
f_sp, ref02 = sps.csd(x[0], x[2], fs=p.fs, window=p.window, nperseg=p.nperseg,
                       noverlap=p.resolved_noverlap, detrend=p.detrend, scaling="density")
print(np.abs(full[0, 2] - ref02).max())   # -> 4.39e-17, matches scipy exactly
```

**Why it matters**: `csd()`'s all-to-all path (`pairwise=False`) is a documented, shipped part of the public API (`__all__`), and per `CLAUDE.md` the phase convention is explicitly load-bearing ("Note this is the conjugate of the `P_nm = sum_k X_n conj(X_m)` convention... magnitudes agree, phase sign flips"). A future refactor of this einsum (e.g. someone "simplifying" the two branches to share code) could silently invert phase/delay direction for every non-self signal pair and ship with all 86 tests green.

**Suggested fix**: Add, alongside the existing `pairwise=True` scipy check, a direct off-diagonal check:
```python
def test_alltoall_offdiagonal_matches_scipy_directly(traces):
    p = SpectralParams(fs=20.0, nperseg=64)
    full, f = csd(stft(traces, p))
    f_sp, ref = sps.csd(traces[0], traces[2], scaling="density", **_sp_kwargs(p))
    assert_spectra_close(full[0, 2], ref)
```

---

### 4. [SEVERITY: MEDIUM] Test gap: `band_hz`'s documented "inclusive" lower bound is never exercised at an exact bin boundary — an off-by-one there survives the whole suite

**Location**: `src/cmm/spectral.py:283` (`keep = (np.abs(freqs) >= lo) & (np.abs(freqs) <= hi)`) ↔ `tests/test_spectral_vs_scipy.py::TestRegressionsAgainstTheAudit::test_frequency_axis_survives_band_selection`, `tests/test_spectral_properties.py::TestProvenance::test_stft_carries_the_frequency_axis_rather_than_deriving_it`

**Bug/bias**: Mutation test — change the lower-bound comparison from `>=` to `>`:
```python
keep = (np.abs(freqs) > lo) & (np.abs(freqs) <= hi)  # exclusive lower bound
```
```
$ python -m pytest tests/test_spectral_vs_scipy.py tests/test_spectral_properties.py -q
86 passed in 2.80s
```
**Not caught.** Every test that uses `band_hz` (`(1.0, 5.0)` at `fs=20, nperseg=64`, spacing `0.3125` Hz; and `(0.2, 2.0)` / `(0.5, 3.0)` / `(1.0, 3.0)` elsewhere) picks bounds that never land exactly on a bin — `1.0/0.3125 = 3.2`, not an integer — so the `>=` vs `>` distinction at the lower edge is structurally unreachable by any existing test. I confirmed the documented "inclusive" behavior does currently hold, but only by hand-constructing a boundary case:
```python
p_full = SpectralParams(fs=20.0, nperseg=64)
full = stft(x, p_full)
lo = full.freqs[3]                       # 0.9375 Hz, exactly on a bin
band = stft(x, p_full.with_(band_hz=(float(lo), 5.0)))
print(lo in band.freqs)                  # -> True (correct, and undocumented-by-test)
```

**Why it matters**: `SpectralParams.band_hz`'s docstring explicitly promises `"inclusive"`. A one-character regression here (`>=` → `>`) silently drops the lowest requested bin whenever a caller's `band_hz[0]` happens to land exactly on a frequency bin — a common case for callers computing `band_hz` from `k * fs / nperseg` deliberately to align with the grid.

**Suggested fix**: Add a boundary-exact test:
```python
def test_band_hz_lower_bound_is_inclusive_at_an_exact_bin(traces):
    p = SpectralParams(fs=20.0, nperseg=64)
    full = stft(traces, p)
    lo = float(full.freqs[3])  # exactly on a bin by construction
    band = stft(traces, p.with_(band_hz=(lo, 5.0)))
    assert lo in band.freqs
```

---

### 5. [SEVERITY: MEDIUM] `resolve_window` silently discards the imaginary part of a complex window array — the exact failure mode the module explicitly guards against elsewhere

**Location**: `src/cmm/spectral.py:200-207`

**Bug/bias**: `resolve_window`'s ndarray branch does `win = np.asarray(window, dtype=np.float64)`. If `window` is a complex-dtype ndarray, this cast silently truncates to the real part (with only a `ComplexWarning`, easily filtered/ignored, not an exception):
```python
import warnings, numpy as np
from cmm.spectral import resolve_window
win = np.ones(8) + 1j*np.arange(8)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    out = resolve_window(win, 8)
print(out)                       # [1. 1. 1. 1. 1. 1. 1. 1.]  -- imaginary part gone
print([str(x.message) for x in w])  # ['Casting complex values to real discards the imaginary part']
```

**Why it matters**: `stft()` explicitly raises on exactly this pattern for the *input* array (S1: "onesided=True requires real input... The pre-refactor code silently discarded the imaginary part here") — but the identical hazard for a user-supplied `window` array is not guarded at all. A caller who accidentally passes a complex array (e.g. the output of a Hilbert-transform-based taper, or a slicing bug that leaves a complex dtype from an earlier FFT) gets a plausible-looking but silently wrong window with no exception and an easily-missed warning, directly contradicting the module's own stated fail-fast precedent.

**Suggested fix**:
```python
win = np.asarray(window)
if np.iscomplexobj(win):
    raise ValueError(f"window array must be real, got dtype {win.dtype}")
win = win.astype(np.float64, copy=False)
```

---

### 6. [SEVERITY: MEDIUM] `SpectralParams` silently floors non-integer `noverlap`/`nfft` instead of raising

**Location**: `src/cmm/spectral.py:107-112` (`resolved_noverlap`, `resolved_nfft`)

**Bug/bias**: Both properties do `int(self.noverlap)` / `int(self.nfft)`, which truncates any non-integral value with no warning or error:
```python
from cmm.spectral import SpectralParams
p = SpectralParams(fs=10.0, nperseg=64, noverlap=31.99999)
print(p.resolved_noverlap)   # 31, silently — nearly a full sample of overlap dropped
p2 = SpectralParams(fs=10.0, nperseg=64, nfft=128.7)
print(p2.resolved_nfft)      # 128, silently
```

**Why it matters**: This is a plausible real mistake — e.g. `noverlap = nperseg * 0.5` composed with a non-power-of-two `nperseg`, or `nfft = 2 ** np.ceil(np.log2(nperseg))` returning a numpy float — and it directly conflicts with the project's stated fail-fast stance ("I want to know when unexpected things happen"). Given the module elsewhere goes out of its way to validate inputs precisely (raising on `noverlap >= nperseg`, on `nfft < nperseg`, on non-finite input), silently flooring a fractional segmentation parameter is an inconsistency, not a deliberate design choice.

**Suggested fix**:
```python
@property
def resolved_noverlap(self) -> int:
    if self.noverlap is None:
        return self.nperseg // 2
    if not float(self.noverlap).is_integer():
        raise ValueError(f"noverlap must be an integer, got {self.noverlap}")
    return int(self.noverlap)
```
(same pattern for `resolved_nfft`).

---

### 7. [SEVERITY: LOW-MEDIUM] `Stft` is a frozen dataclass, but its array fields are not read-only — the "provenance" guarantee is cosmetic

**Location**: `src/cmm/spectral.py:132-151`

**Bug/bias**: `@dataclass(frozen=True)` on `Stft` prevents `s.coefs = other_array`, but does not prevent `s.coefs[:] = 0` or `s.coefs *= mask`. None of `coefs`, `freqs`, `window`, `onesided_factor` have `.flags.writeable = False` set.

**Why it matters**: The module's stated design goal is explicit: "Parameters live in `SpectralParams` and travel with the result, so a stored output records how it was made." A frozen dataclass is being used specifically to promise that an `Stft` you're holding is a trustworthy, immutable record of a computation. That promise is false for the arrays themselves — any downstream code (a masking step, an in-place normalization "optimization") can silently corrupt `s.coefs` while `s.params`/`s.metadata()` keep reporting the original, now-incorrect provenance.

**Suggested fix**: In `stft()`'s return, or in `Stft.__post_init__`, set `coefs.flags.writeable = False` (and similarly for `freqs`, `window`, `onesided_factor`) so any accidental in-place write raises `ValueError: assignment destination is read-only` instead of silently succeeding.

---

### 8. [SEVERITY: LOW] Redundant full-array copy on the `detrend=False` fast path defeats the module's own "costs nothing" claim about `segment()`

**Location**: `src/cmm/spectral.py:262-267`

**Bug/bias**:
```python
seg = segment(x, params.nperseg, params.resolved_noverlap)
if params.detrend is not False:
    seg = _scipy_detrend(seg, type=params.detrend, axis=-1)
else:
    seg = np.array(seg, copy=True)  # own the memory before windowing
seg = seg * win
```
The `np.array(seg, copy=True)` in the `else` branch is dead weight: the very next line, `seg = seg * win`, is an out-of-place multiply that already allocates a brand-new, non-strided array (`win` broadcasts against the strided view without requiring pre-ownership). No code between the copy and the multiply reads or writes `seg` in a way that needs it to already be a dense owned array.

**Why it matters**: `segment()`'s own docstring advertises "A strided view, so it costs nothing" as a selling point, and the `detrend=False` path is precisely the case where that "free" property should be realized (no detrend call to also allocate a copy). Instead this path allocates a full `(N, K, nperseg)` array twice — once for the dead copy, once for the windowed result — doubling peak memory and wall time for exactly the configuration meant to be cheapest. Minor in isolation, but the project's CLAUDE.md is unusually benchmarking-conscious (dedicated thread-count and dask-vs-blocked-numpy sections), so an avoidable 2x allocation on the fast path is worth removing before it's measured against something and produces a confusing number.

**Suggested fix**: Delete the `else` branch; `seg = seg * win` (or, if `win` could ever be scalar-broadcast trivially and skip allocation — it can't, `win` is always `(nperseg,)` — this is moot) already produces an owned array.

---

### 9. [SEVERITY: LOW] `nperseg=1` is accepted and produces a fully degenerate, silently-zero transform

**Location**: `src/cmm/spectral.py:83-84` (`SpectralParams.__post_init__`, `nperseg < 1` check)

**Bug/bias**: `nperseg >= 1` is the only lower bound enforced. With `nperseg=1`, every segment is detrended to exactly zero (a 1-point "linear" or "constant" detrend removes the entire, only, sample), so the transform is 100% signal-free by construction:
```python
from cmm.spectral import SpectralParams, stft
import numpy as np
x = np.random.default_rng(0).standard_normal((1, 10))
s = stft(x, SpectralParams(fs=10.0, nperseg=1, detrend="linear"))
print(s.coefs.ravel()[:3])   # ~1e-16, i.e. zero — no exception, no warning
```

**Why it matters**: This isn't a crash, so it's easy to miss in a parameter sweep or an automated `nperseg` search that accidentally tries `nperseg=1` (e.g. an off-by-one in a caller's loop bound) — the function returns a plausible-shaped, all-near-zero result instead of raising on an input that can never produce a meaningful spectral estimate.

**Suggested fix**: Raise for `nperseg < 2` (a single-sample "segment" has no frequency content to estimate, `F=1` bin at DC only, and with `detrend != False` it's identically zero).

---

### 10. [NOTE — test fragility, not a bug] Odd-`nfft` Nyquist-parity handling has exactly one point of coverage

**Location**: `src/cmm/spectral.py:274-275` ↔ `tests/test_spectral_vs_scipy.py::TestPsdVsWelch::test_matches_welch_odd_nfft_has_no_nyquist_bin`

**Observation**: Mutation-tested (always treat the last one-sided bin as unpaired, regardless of `nfft` parity):
```python
factor[-1] = 1.0  # unconditionally, no `if nfft % 2 == 0:` guard
```
Result: `1 failed, 85 passed` — correctly caught, but by exactly one test (`test_matches_welch_odd_nfft_has_no_nyquist_bin`, the only test in the suite using an odd `nfft`; every other parametrization uses `nperseg ∈ {32, 64, 128}` with `nfft=None`, all even). This isn't a live bug — the code is correct and the test does its job — but it means this entire branch has a single point of failure: delete or accidentally skip that one test (e.g. a `pytest -k` filter, a merge conflict dropping it) and the odd/even Nyquist distinction becomes silently untested again. Consider adding odd `nfft` to the `NPERSEGS`/`nfft` matrix more broadly rather than as a single dedicated test.

---

## Mutation-testing kill matrix

All mutations applied to `src/cmm/spectral.py`, tested against `tests/test_spectral_vs_scipy.py tests/test_spectral_properties.py` (86 tests total), then reverted with `git checkout -- src/cmm/spectral.py`. Final state confirmed clean (`git diff --quiet src/cmm/spectral.py` — no output).

| # | Mutation | Result | Caught by |
|---|---|---|---|
| 1 | Drop one-sided doubling (`factor = np.ones(...)` always) | **CAUGHT** | 27 tests, incl. `test_onesided_doubling_is_applied_to_interior_bins_only`, Parseval, sine-amplitude, all `welch`/`csd` comparisons |
| 2 | Swap `"density"`/`"spectrum"` scale formulas in `Stft.scale` | **CAUGHT** | 28 tests, incl. `test_spectrum_scaling_is_not_the_density_formula`, Parseval, sine-amplitude |
| 3 | Off-by-one in `segment()`'s `n_seg` (`+ 1`) | **CAUGHT** | 55 tests (near-total collapse) — also a latent memory-safety note: `n_seg` too large makes `as_strided` construct a view reading past the buffer end; numpy does not bounds-check `as_strided`, so a less-obvious off-by-one here could read adjacent heap memory rather than crashing |
| 4 | Drop `.conj()` in `csd` (both `pairwise=True` and all-pairs einsums) | **CAUGHT** | 20 tests, incl. `test_csd_is_hermitian_under_signal_exchange`, `test_delay_appears_as_a_linear_cross_spectrum_phase_ramp`, all coherence-vs-scipy tests |
| 5 | Nyquist bin unconditionally treated as unpaired (drop the `nfft % 2 == 0` guard) | **CAUGHT** (narrowly — see Finding 10) | exactly 1 test: `test_matches_welch_odd_nfft_has_no_nyquist_bin` |
| 6 | `band_hz` lower bound `>=` → `>` (exclusive) | **NOT CAUGHT** — 86/86 pass | — (see Finding 4) |
| 7 | All-pairs `csd()` (`pairwise=False`): conjugate `sy.coefs` instead of `sx.coefs` | **NOT CAUGHT** — 86/86 pass | — (see Finding 3) |

5 of 7 mutations killed, 2 survived the entire 86-test suite untouched. Both survivors are concrete, addable tests (given above), not fundamental design problems.

---

## What I did not find a HIGH-severity problem in

- The `psd`/`csd`/`coherence` formulas themselves, run with matched-window `Stft`s, agree with `scipy.signal.welch`/`csd`/`coherence` to `rtol=1e-12` across all sampled windows, segmentations, detrends, and `nfft` values — verified by re-running the suite, not just reading it.
- `segment()`'s strided view is correct for both contiguous and transposed/non-contiguous input (`test_segment_handles_non_contiguous_input` genuinely exercises this and passes).
- `stft()`'s complex-input and non-finite-input fail-fast checks (S1) are real and correctly placed before any array is silently coerced.
- `SpectralParams` being a frozen dataclass with no mutable-default containers (fixing C9) is real; there is no `[0, np.inf]`-style shared mutable default anywhere in the class.
- Order of operations (detrend → window → transform) matches the docstring and matches scipy's own `_spectral_helper`, verified via the direct `scipy.signal.stft` coefficient comparison test.

None of that reaches the bar of "no HIGH-severity issues" for the module as a whole, given Findings 1 and 2 above — but the core per-Stft numerics are solid.
