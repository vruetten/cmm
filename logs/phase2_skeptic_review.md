# Adversarial review — Phase 2 coherence-matrix module

Scope: `src/cmm/coherence.py`, `tests/test_coherence_matrix.py`, commit `bb253fd`
(`refactor/phase-1-spectral-core`). Cross-checked against `src/cmm/spectral.py`
(Phase 1) and `derivations/coherence_estimators.md`.

Method: independent re-derivation of V1/V2 from the raw formulas (not reusing
`naive_v1`/`naive_v2`), numerical cross-check against `spectral.py`'s own
`csd`/`psd`/`coherence`, and mutation testing of 7 deliberate bugs injected
into `coherence.py`, each reverted immediately via `git checkout --
src/cmm/coherence.py`. Final state confirmed byte-identical to the pre-review
file and `pytest tests/test_coherence_matrix.py` passes 43/43. No file outside
`coherence.py` was modified during mutation testing; nothing under
`/nrs/ahrens` was touched.

## Verified correct (stated up front, since the burden of proof is on the code)

- **V1/V2 formulas**: independently re-derived from `derivations/` (not the
  test file's `naive_v1`/`naive_v2`) and agree with `coherence_matrix` to
  8.9e-16 / 8.3e-17 — machine precision, not "close."
- **Phase 1 ↔ Phase 2 consistency**: building V2 by hand from
  `spectral.csd`/`spectral.psd`, and V1 from `spectral.coherence().mean(-1)`,
  agrees with `coherence_matrix` to 4.4e-16 / 3.3e-16.
- **Whitening identity** (`C2[whitened X] == C1[X]`): holds to 1e-10 in the
  shipped test, confirmed independently.
- **complex64 precision claim**: measured 5.4e-9 max error at N=200, K=30 —
  ~6e6x below the `1/K` floor, consistent with the docstring's "six orders
  below."
- **"(N,N,F) intermediate never materialized"**: at N=3000, K=30, F=69 the
  hypothetical dense float64 intermediate is 4968 MB; measured peak traced
  Python allocation was 508 MB. Claim holds.
- **`out += out.T` in-place aliasing**: verified against an explicit
  `a + a.T` on this numpy (2.3.4) — bit-identical, not a hazard here.
- **Blocked threaded `fill()` row-slice writes**: disjoint for every
  `(n, block)` combination tested, including `n < block` and `n` not
  divisible by `block`; `stop = min(start + block, n)` is correct and the
  mutation removing `min()` was tested (see Finding 9) — it degrades to a
  loud crash, not silent corruption.

## Findings, ranked

### 1. [SEVERITY: HIGH] `finite_power_mask` does not guarantee `coherence_matrix` will accept its own output
**Location**: `src/cmm/coherence.py:105-122` (mask) vs. `:177-185` (dead check)

**Bug/bias**: `finite_power_mask` sums power over **both** segments and
frequencies in one reduction (`einsum("nkf,nkf->n", ...)`) and keeps a signal
if that *total* is nonzero. `coherence_matrix`'s own guard requires power to
be nonzero at **every individual frequency bin**
(`~(power_nf > 0).all(axis=1)`). A signal that is dead at one in-band bin but
alive at the others passes the mask and is still rejected by
`coherence_matrix`, exactly reproducing the `ValueError` the mask's docstring
tells the caller to use it to avoid:

```python
keep = finite_power_mask(s)
C = coherence_matrix(replace(s, coefs=s.coefs[keep]))  # can still raise
```

Reproduced directly:
```
mask: [True True True True True True]     # signal 2 kept
coherence_matrix raised: 1 of 6 signals have zero power in at least one
in-band frequency (first: 2) ...
post-filter STILL RAISES (docstring workflow broken): <same message>
```

**Why it matters**: this is the one function whose entire purpose is to be
the fail-fast escape hatch for the dead-power case, and it silently fails at
that job for the (arguably more realistic) case of a single dead bin rather
than a totally dead signal — the exact case a masked-out imaging voxel with a
near-DC artifact would produce. The existing test
(`test_finite_power_mask_identifies_the_dead_signal`) only exercises the
trivial case where the whole signal is zero across every `(k,f)`, so both the
weaker `.any`-style total-power check and the intended per-bin check agree —
the bug is invisible to the suite.

**Suggested fix**: make the mask's semantics match the check it exists to
satisfy — per-bin, not total:
```python
def finite_power_mask(s: Stft, *, rtol: float = 0.0) -> np.ndarray:
    power_nf = np.einsum("nkf,nkf->nf", s.coefs, s.coefs.conj()).real
    ok = (power_nf > 0).all(axis=1)
    if rtol > 0 and ok.any():
        total = power_nf.sum(axis=1)
        ok &= total > rtol * np.median(total[ok])
    return ok
```
Add a test with a signal dead at exactly one bin (not all) to close the gap.

---

### 2. [SEVERITY: HIGH] Dead-signal check can silently regress from "any bin" to "all bins" with zero test failures
**Location**: `src/cmm/coherence.py:178` — `dead = ~(power_nf > 0).all(axis=1)`

**Bug/bias**: Mutating `.all(axis=1)` to `.any(axis=1)` — i.e., weakening
"raise if any in-band bin is dead" to "raise only if every in-band bin is
dead" — passes all 43 tests unchanged:
```
=== MUTATION 7: dead-check uses .any() instead of .all() ===
...........................................                              [100%]
43 passed in 3.14s
```
With the mutation live, a signal dead at one bin out of many silently
produces a finite (non-NaN, non-Inf) output instead of raising:
```
c.values[2] = [0.0885 0.0939 1.0 0.0961 0.1035 0.1258]
```

**Why it matters**: this is precisely the class of failure the module's
docstring and `TestFailFast` class exist to prevent, and the only test that
would need to fail to catch a regression here (`test_zero_power_signal_raises_
with_guidance`) zeroes an entire signal, which triggers both `.all()` and
`.any()` identically — so the test asserts the fail-fast *message* exists,
never the fail-fast *condition's precision*. A future edit (e.g. someone
"optimizing" the check, or refactoring `dead` to reuse `finite_power_mask`
verbatim as its docstring invites) could silently weaken this and nothing
would catch it. Note this also interacts with Finding 1: whether `.all()` is
even the right check for `whiten=False` is a separate, real question — V2
tolerates isolated dead bins mathematically (a zero-power bin contributes 0
to both `num` and `den`, not `0/0`); only `whiten=True` genuinely requires it,
since whitening divides by `sqrt(power_nf)` per bin
(`coherence.py:188`) and a zero there produces `inf`/`nan` in `work`
immediately. The check is currently uniform across both modes without
comment on why, and that design choice itself is untested against the
partial-dead-bin case for either branch.

**Suggested fix**: add a test with power zeroed at one bin for one signal
(not all bins), asserting the `ValueError` still fires; and separately verify
(and document) whether `.all()` is intentionally stricter than necessary for
`whiten=False`, or split the check by mode.

---

### 3. [SEVERITY: MEDIUM] `debias(n_eff=1)` silently returns garbage instead of raising
**Location**: `src/cmm/coherence.py:259-260`

**Bug/bias**: With `K = 1` segment, the coherence estimator is degenerate —
every pair, including two independent signals, scores exactly `1.0` by
construction (`γ²(f) = |Xn Xm*|² / (|Xn|²|Xm|²) ≡ 1` for a single sample).
The debias formula `(C - 1/K)/(1 - 1/K)` is `0/0` at `K=1`, and rather than
raising, the code special-cases it to `np.clip(values, 0, 1)` — silently
returning the (meaningless, always-1.0) raw values as if they were valid
debiased output:
```python
out = debias(np.array([[1.0, 1.0], [1.0, 1.0]]), n_eff=1)
# -> [[1. 1.] [1. 1.]]   (garbage in, garbage out, no warning)
```

**Why it matters**: this directly violates the project's stated fail-fast
stance. A caller who accidentally passes `n_eff=1` (e.g., a single-segment
STFT from too-short a recording) gets a coherence matrix that reports every
pair as perfectly coherent, silently, with no signal that the debiasing step
did nothing meaningful. There is no test for this branch at all — it is
untested dead-letter code path discovered only by inspection.

**Suggested fix**: raise instead of silently no-op'ing:
```python
if n_eff <= 1:
    raise ValueError(
        f"debias is undefined at n_eff={n_eff}: with a single segment "
        "every pair's raw coherence is trivially 1.0 and there is no "
        "bias to remove. Use more segments (shorter nperseg or more "
        "overlap) before debiasing."
    )
```

---

### 4. [SEVERITY: MEDIUM] `effective_bins` computed and discarded when `whiten=True`, ~25-30% wasted wall-clock at the default `compute_f_eff=True`
**Location**: `src/cmm/coherence.py:232-236`

**Bug/bias**:
```python
f_eff = effective_bins(weights_nf) if compute_f_eff else float(n_freqs)
...
f_eff=float(n_freqs) if whiten else f_eff,
```
When `whiten=True`, `weights_nf` is all-ones (`coherence.py:189`), and
`effective_bins` is computed on it — two `(N,F)@(F,N)` matmuls — purely to be
thrown away one line later in favor of `float(n_freqs)`.

**Why it matters**: measured directly (N=4000, K=30, F=69, complex64, 8
workers):
```
whiten=True, compute_f_eff=True:  4.44 s
whiten=True, compute_f_eff=False: 3.50 s   (21% faster)
```
`compute_f_eff=True` is the default, so every V1 (`whiten=True`) call pays
this tax silently unless the caller knows to also pass
`compute_f_eff=False` — which is not documented as necessary for V1 and
contradicts the parameter's own docstring implication that it only trades off
diagnostic value, not correctness-irrelevant waste.

**Suggested fix**: skip the computation entirely when whitened:
```python
if whiten:
    f_eff = float(n_freqs)
elif compute_f_eff:
    f_eff = effective_bins(weights_nf)
else:
    f_eff = float(n_freqs)
```

---

### 5. [SEVERITY: MEDIUM] `workers=None` defaults to `os.cpu_count()`, ignoring the actual CPU allocation on a shared cluster node
**Location**: `src/cmm/coherence.py:216-217` — `workers = min(32, (os.cpu_count() or 1))`

**Bug/bias**: `os.cpu_count()` reports the physical core count of the node
(80 here), not what the job was actually allocated (e.g., via LSF/SLURM
`-n`), and does not respect `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS` either
— which is exactly the failure mode `CLAUDE.md` documents as having already
invalidated a round of benchmarking in this repo ("`os.sched_getaffinity(0)`
returns all 80 CPUs and there is no cgroup quota"). Confirmed in this shell:
```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
os.cpu_count(): 80          # -> workers defaults to 32
```

**Why it matters**: on a job allocated, say, 4 cores, `coherence_matrix()`
called with default `workers=None` spawns up to 32 Python threads doing
CPU-bound elementwise `|S|**2` accumulation (explicitly documented as
single-threaded-dominated work, `coherence.py:161-164`) — oversubscribing the
node 8x, degrading this job's own throughput and starving co-scheduled jobs
on a shared Janelia node. This is a correctness-adjacent operational hazard
in the exact category the project has already been burned by once.

**Suggested fix**: prefer an explicit allocation-aware default, e.g.
`len(os.sched_getaffinity(0))` capped by an env var the scheduler sets
(`LSB_DJOB_NUMPROC`/`SLURM_CPUS_PER_TASK`), or — simplest and consistent with
the "measure, do not assume" rule — require `workers` to be passed explicitly
in any batch/cluster context and only default silently for interactive use,
with a comment stating the assumption plainly.

---

### 6. [SEVERITY: LOW/test-gap] `debias()`'s own default `n_eff` resolution is a single-test choke point
**Location**: `src/cmm/coherence.py:250-251`; mutation test

**Bug/bias**: Mutating the `n_eff is None` resolution to use `c.n_eff * F`
instead of `c.n_eff` is caught by exactly one test out of 43
(`test_debias_uses_the_matrices_own_n_eff`):
```
FAILED tests/test_coherence_matrix.py::TestBiasFloorAndDebias::
       test_debias_uses_the_matrices_own_n_eff
1 failed, 42 passed
```

**Why it matters**: this is the exact bug class the module's docstring calls
out by name as "the live bug in icampsnfr's call site" — `n_eff = K*F`
instead of `K`. It is caught, which is good, but by only one test with no
redundancy; if that single test were ever skipped, xfailed, or accidentally
weakened (e.g. loosened tolerance), the regression would be invisible. Given
how central this exact mistake is to the module's stated purpose, it merits
more than one line of defense — e.g. also assert `debias(c).max() <
debias(c, n_eff=c.n_eff * 2).max()` style monotonicity checks, or a direct
equality assertion inside `test_bias_floor_property_reports_one_over_k`.

**Suggested fix**: add a second, independent assertion path (e.g. checking
against `CoherenceMatrix.bias_floor` directly rather than only against
`debias(c.values, n_eff=c.n_eff)`, which shares the same code path for
resolving `n_eff`).

---

### 7. [SEVERITY: LOW] `CoherenceMatrix.metadata()["f_eff"]` has an inconsistent schema depending on `compute_f_eff`
**Location**: `src/cmm/coherence.py:86-94`, interacting with `:232`

**Bug/bias**: `metadata()`'s `f_eff` summary is a `dict` with `median`/`min`
keys when `f_eff` is an `(N,N)` array (the default path), but a bare `float`
when `whiten=True` **or** when `compute_f_eff=False`:
```
compute_f_eff=False -> f_eff summary type: <class 'float'> 29.0
compute_f_eff=True  -> f_eff summary type: <class 'dict'> {'median': ..., 'min': ...}
```
Only the `whiten=True` case is tested
(`test_whitened_metadata_says_v1`/`test_metadata_records_the_estimator_and_
the_floor`); `compute_f_eff=False` with `whiten=False` is untested and
produces the same float-vs-dict divergence for a different reason.

**Why it matters**: any downstream code (e.g. a dashboard or saved-JSON
consumer, which `stft_metadata`/`metadata()` are explicitly designed to
support per the module docstring) that does `m["f_eff"]["median"]`
unconditionally will crash on a `TypeError` for these two configurations
opaquely, far from where the actual choice (`compute_f_eff=False`) was made.

**Suggested fix**: normalize to one schema always, e.g.
```python
summary = {"median": float(np.median(f_eff)), "min": float(np.min(f_eff))} \
    if not np.isscalar(f_eff) else {"median": float(f_eff), "min": float(f_eff)}
```
so callers never have to branch on how the matrix was made.

---

### 8. [SEVERITY: LOW] No test for `N=1` or `F=1` degenerate inputs
**Location**: `tests/test_coherence_matrix.py` (absence)

**Bug/bias**: Inspection and a quick manual check show `coherence_matrix`
handles `N=1` (trivial `(1,1)` output, diagonal forced to 1.0) and `F=1`
(the whole "band" is a single bin, `effective_bins` divides `s1*s1` by `s2`
with only one term, giving `F_eff=1` trivially) without crashing, but neither
is exercised by the suite. `F=1` in particular is worth a real test: it's the
boundary at which "band-averaged" degenerates to a single-frequency MSC, and
it's the shape most likely to trip an axis-order or reduction bug that a
larger `F` would mask (e.g., an accidental `keepdims` mismatch).

**Why it matters**: not a caught bug, but the concrete edge case a future
band_hz narrowing (`band_hz=(f, f)`-adjacent) would produce in real usage,
and it is exactly the kind of boundary the "measure, do not assume" rule asks
for rather than trusting that it "should work."

**Suggested fix**: add
```python
def test_single_frequency_bin_reduces_to_per_bin_msc():
    ...
def test_single_signal_is_a_1x1_identity_matrix():
    ...
```

---

### 9. [Checked, not a bug] Missing `min()` clamp on the block boundary — verified fail-fast, not silently corrupting
**Location**: `src/cmm/coherence.py:207`

Mutating `stop = min(start + block, n)` to `stop = start + block` was tested
directly per the review brief. For any `N` not evenly divisible by `block`,
Python's slice-clamping (`x_fnk[f, start:stop]` silently clamps `stop` to the
array bound) desyncs from the pre-declared `acc = np.zeros((stop - start, n))`
buffer shape, producing an immediate `ValueError: operands could not be
broadcast together with shapes (5,9) (4,9) (5,9)` rather than silent
corruption. For `N` evenly divisible by `block` the mutation is a no-op. This
is worth recording precisely because it is a case where removing an
apparently load-bearing guard turns out to be self-defending by accident of
NumPy's broadcasting rules — not a defense that should be relied on
elsewhere, but not a live risk here either.

---

## Mutation kill matrix

All mutations applied to `src/cmm/coherence.py`, run against
`pytest tests/test_coherence_matrix.py -q`, then reverted with
`git checkout -- src/cmm/coherence.py`. Final diff confirmed empty.

| # | Mutation | Result | Tests failing |
|---|----------|--------|----------------|
| 1 | Swap num/den (`den[start:stop], acc` instead of `acc, den[start:stop]`) | **Killed** | 7 |
| 2 | Drop `.conj()` on `x_fkn` | **Killed** | 5 |
| 3 | Accumulate `real**2` only, drop `imag**2` | **Killed** | 29 |
| 4 | `debias`'s implicit `n_eff` resolution uses `K*F` instead of `K` | **Killed** | 1 (thin — see Finding 6) |
| 5 | Skip symmetrization (`out += out.T; out *= 0.5`) | **Killed** | 1 (exact `rtol=0,atol=0` check) |
| 6 | Drop `min()` on block-stop boundary | **Killed** (crashes, does not corrupt — see Finding 9) | 22 (of which most are collateral `ValueError`s, not targeted assertions) |
| 7 | Dead-signal check `.all(axis=1)` → `.any(axis=1)` | **Survived** | 0 — see Finding 2 |

6 of 7 mutations killed; the one survivor is on the module's core fail-fast
guarantee (Finding 2), and two of the "killed" mutations (4, 5) are killed by
only a single test each, which is a thinner margin than the module's stated
purpose warrants (Finding 6).

## Summary assessment

The numerical core is solid: V1, V2, the whitening identity, and consistency
with the independently-audited Phase 1 spectral core all check out to machine
precision, and the threading/blocking scheme is genuinely race-free with no
silent-corruption failure modes found under direct attack. The real problems
are concentrated exactly where the project's stated priorities say they
should be scrutinized hardest — fail-fast guarantees — and in both cases
(Findings 1, 2) the gap is in the same place: the module's dead-power
handling is tested only for the trivial "entire signal is zero" case, never
for "one bin out of many is zero," which is the realistic failure mode for
real imaging data. Findings 3-5 are secondary but genuine: a silent
garbage-in-garbage-out branch in `debias`, a measured ~25% performance tax
paid needlessly on every default-configured V1 call, and a thread-count
default that reproduces a resource-oversubscription failure mode this
project has already been burned by once.
