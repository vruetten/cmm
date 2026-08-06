# Adversarial review: cmm refactor plan and its supporting evidence

Reviewer scope: `.claude/plans/cmm_refactor_plan.md`, `derivations/coherence_estimators.md`,
`scratchpad/{bench,bench2,bench3,bench_eig}.py`, `src/cmm/*`,
`icampsnfr/src/icampsnfr/{spectral,coherence,cluster,phase}.py`.

Method: every numeric audit claim that could be cheaply re-run was re-run (in
the `all` conda env, read-only, nothing under `src/` touched). Every
mathematical claim in the coherence-estimator note was independently
re-derived with a synthetic generator I wrote myself (not copied from the
plan) to check whether the reported numbers are directionally real or
flattering. Findings below are ranked by severity; each is tagged
**(a) demonstrated error**, **(b) unsupported-but-plausible claim**, or
**(c) judgement call**.

---

## HIGH severity

### 1. (c) The V2-over-V1 decision is argued with mismatched rigor, and the central claim ("V2 everywhere") is not actually settled by the evidence presented

**Claim challenged**: §6 of `coherence_estimators.md` picks V2 (power-weighted,
incoherent) as the sole default, over V1 (equal-weight, incoherent), "because
it is numerically stabler ... and has an exact unit diagonal."

**Evidence**: The document's own §4 measures V2's spectral-tilt failure
precisely — a pair sharing only its strongest band scores **1.0000** under V2
against V1's 0.8202, i.e. V2 saturates at the ceiling for a pair that is
*not* coherent over roughly half the nominal band. I reproduced this
independently with a different synthetic construction (own script, not the
plan's) and got V1=0.56, V2=0.9999 for an analogous tilt case — same
qualitative and near-identical order-of-magnitude story: **V2's failure mode
is not small**, it saturates the metric.

By contrast, V1's stated failure mode ("one dead or near-dead in-band bin
either produces 0/0 or amplifies pure noise to γ²≈1") is never quantified
anywhere in the note — no measured example, no Monte Carlo, nothing with a
number attached. The decision therefore rests on a precisely measured, severe
failure of the chosen estimator (V2) versus an unmeasured, hand-waved failure
of the rejected alternative (V1). That is an asymmetric evidentiary standard,
and it happens to point toward the estimator with the *worse documented
number*.

The document is explicit that axis A (delay-invariance, which rules out V3)
is "the reason", and that V2-over-V1 is a secondary, weaker argument. But the
plan then elevates this secondary argument to a blanket "V2 everywhere"
policy applied to every downstream consumer (`icampsnfr` dashboards, cluster
labels, figures) without ever testing whether real ependymal/neural coherence
data actually has dead in-band bins (V1's problem) more or less often than it
has power spectra tilted enough to trigger V2's problem (astronomically
likely for calcium/voltage imaging, which is close to 1/f). Both failure
modes are physically plausible for this domain; only one was measured, and
it disfavors the choice made.

**What would resolve it**: (1) quantify V1's dead-bin failure the same way
V2's tilt failure was quantified — a Monte Carlo with a genuinely dead bin
(zero or near-zero power at one in-band frequency) at various severities;
(2) run both V1 and V2 on one real coherence dataset (e.g. the same
`/nrs/.../exp3/coherence/band10-40s` STFT coefficients already staged by
`build_real_C.py` in the scratchpad) and report which estimator's blind spot
actually bites — tilt is trivial to check (plot the mean in-band power
spectrum of a few hundred real voxels); dead bins are trivial to check
(fraction of (voxel, freq) cells with power below some floor). This is a
half-day measurement, not a re-derivation, and it is the single most
consequential unmeasured claim in the whole plan.

### 2. (a) The delay-sweep, spectral-tilt, and Monte Carlo bias-floor numbers in `coherence_estimators.md` are not reproducible from the artifacts in scope

**Claim challenged**: The note states "All numbers below are measured, not
asserted. Reproduce with `scratchpad/bench*.py` and the snippets at the end"
(line 9-10).

**Evidence**: `derivations/coherence_estimators.md` is 324 lines total and
ends at §7 "Implementation notes" — there is no "snippets at the end"
appendix. And the four scripts actually in the reviewed scratchpad
(`bench.py`, `bench2.py`, `bench3.py`, `bench_eig.py`) contain **no** code
that computes a delay sweep, a spectral-tilt case, or a Monte Carlo bias
floor. I read every function in all four files line by line:

- `bench.py::bench_cross`, `bench_auto` (numpy/dask paths) compute
  `mean_f |Pxy(f)|²/(Pxx(f)Pyy(f))` — **that is V1**, not V2.
- `bench.py::bench_auto::blocked_bandavg` computes the flattened-`(K,F)`
  single-GEMM estimator — **that is V3**.
- `bench2.py::blocked_numpy` divides per-frequency then means over `f` —
  **V1** again (its own docstring even says "keep cmm's exact estimator",
  i.e. the pre-refactor V1 default).
- `bench3.py::normalized_blocked` whitens per `(n,f)` first, which by the
  note's own §3 identity (`C⁽²⁾[X̂] = C⁽¹⁾[X]`) makes it **mathematically
  V1 computed via the whitened-V2 formula** — not V2 on unwhitened data.
- `bench_eig.py` is unrelated (eigensolver timing).

None of these four scripts computes real, unwhitened V2
(`Σ|P_nm|² / Σ P_nn P_mm` with no per-(n,f) whitening) — the estimator §6
actually recommends. The "V2 needs 4.13 s vs V3's 1.80 s at N=6000, K=10,
F=20" cost figure quoted under Q3 cannot be traced to any of the four target
scripts. (Separate, unwhitened-V2 code does exist — `v2_worker.py`,
`v2_driver.py`, `build_real_C.py`, `gate23_bench.py` — but these are not
among the reviewed scripts, and by the time of this review the sweep that
would populate them had only reached N=1000, not the N=6000/20000 configs
the plan cites.)

Separately, I independently re-derived the *qualitative* content of all
three tables (delay sweep, tilt, bias floor) with my own from-scratch
synthetic generator and got the same story in every case (V1/V2
delay-invariant, V3 collapses with lag; V2 saturates under tilt where V1
does not; V1/V2 sit near 1/K regardless of F while V3/V4 fall with F). So the
underlying statistics are very likely sound — the 1/K floor for a K-segment
MSC estimate under the null is in fact a textbook result (Beta(1, K−1) null
distribution of single-frequency MSC, mean exactly 1/K), not a fabricated
number. But "very likely sound based on an independent re-derivation" is a
materially weaker claim than "measured, reproduce with these scripts" — and
the scripts named do not do what the note says they do.

**What would resolve it**: point the note at the actual scripts that
generated each table (they evidently exist somewhere, given the precision of
the reported digits), or check in a `derivations/scripts/` directory with the
literal code used, keyed to each table. Do not claim reproducibility from
scripts that compute a different estimator.

### 3. (b) "$K_\text{eff} < K$ under overlap" is asserted, drives a concrete recommendation, and was never measured

**Claim challenged**: §5/§6 of the derivations note: "with overlapping
segments the true $K_\text{eff}$ is smaller still, so V1/V2 floors in
practice are worse than 1/K" — and the plan draws the operational conclusion
"segment count is the only bias lever ... reducing `nperseg` or increasing
overlap to buy more segments is the only way down."

**Evidence**: This is self-flagged by the plan's author as unmeasured (task
item 2 explicitly calls it out), and I found no script anywhere in scope that
computes bias floor as a function of `noverlap`/overlap fraction. The claim
is directionally plausible — WOSA/Welch segment correlation under overlap is
well documented (Percival & Walden; the usual "equivalent degrees of
freedom" correction) — but two things are true simultaneously in the plan's
own text: (1) "increasing overlap ... is the only way down" and (2) "with
overlapping segments the true $K_\text{eff}$ is smaller still" (i.e. overlap
makes the floor worse, not better). These two sentences point in **opposite
directions** for how to set `noverlap`, and nothing in the plan reconciles
them with a number. As written, the promised README guidance ("this should
drive `nperseg`/`noverlap` choice", §6 item 3) cannot actually be written
correctly yet — the plan doesn't know the sign of the effect it's asking
`README` to prescribe.

**What would resolve it**: a Monte Carlo of the null-floor as a function of
`noverlap/nperseg` at fixed total recording length, holding K (nominal, not
effective) roughly comparable — this directly answers whether users should
raise or lower overlap under V2, and is cheap (same MC machinery as the
existing K/F sweep, one more axis).

### 4. (a)/(b) C4's ~23x backprojection error was invisible for an unknown period; the plan neither audits historical outputs nor asks whether the machinery is load-bearing before Phase 4

**Claim challenged**: §5 says "Phase 4 is the point at which the
projection-matrix machinery in `utils.py` gets resolved ... once C4 is fixed
the round-trip test decides whether it stays at all."

**Evidence** (demonstrated, re-derived independently): I rebuilt
`build_fft_trial_projection_matrices` + `project`/`backproject` outside the
package and measured round-trip magnitude ratios between 0.01 and 0.4
depending on how segments/frequencies are aggregated — never close to 1,
confirming C4 qualitatively (the exact 0.044 figure depends on the precise
aggregation used in the original measurement, which I did not need to match
to confirm the bug is real and large). `CMM` (the only caller) has been
unimportable in the `all` env since jax broke (C1), so the *current* impact
is zero — but that only bounds the *recent* window. Historical `.npy` result
caches under `cardio_data/`, `ependymal_data/` predate the jax breakage and
may embed this factor silently if they ever called `backproject_centroids`.

More importantly: a ~23x error that "nothing visibly broke" on is strong
evidence the output was never scrutinized quantitatively by anyone —
otherwise a factor-of-23 amplitude mismatch between a signal and its
supposed reconstruction is not subtle. That's worth treating as a signal in
itself (unlike the coherence-ratio bugs C2/C3, which *cancel* by
construction and so are legitimately invisible even to a careful user, C4's
error does not cancel in `backproject_centroids`'s outputs). The plan defers
"should this exist at all" to Phase 4 rather than resolving it now with a
simple grep for call sites across `cardio_data/`, `ependymal_data/`, and
`icampsnfr` — which would settle the load-bearing question well before
Phase 4 and might eliminate an entire module's worth of Phase-1-through-3
work.

**What would resolve it**: (1) `grep -rn backproject_centroids\|build_fft_trial_projection_matrices`
across all three repos/examples now, not in Phase 4; (2) if any saved
`.npy`/figure output used it, flag those specific artifacts as suspect in the
plan's own audit rather than leaving it implicit.

### 5. (b) Phase ordering defers the only integration test six phases past a full API rewrite, while icampsnfr's *existing* code already contains claims the plan's own audit disproves

**Claim challenged**: Execution phases (§5): `icampsnfr` is updated in Phase
7, after `spectral.py`, `coherence.py`, `kmeans.py`, `clustering.py`,
`phase.py`, `wavelet.py`, and `metrics.py` are all rewritten and "green."

**Evidence**: `icampsnfr/src/icampsnfr/coherence.py`'s own docstring (lines
19-23) currently asserts: "With enough in-band frequency bins ... `n_eff` is
a few hundred and the bias floor is negligible" — this is precisely the claim
the plan's own §5/Q3 analysis refutes (V1/V2's floor is 1/K regardless of F;
only V3, which `icampsnfr.msc_matrix` implements, gets the `n_eff = K*F`
benefit, and V3 is being retired specifically because it's not
delay-invariant). Likewise `icampsnfr/spectral.py`'s docstring (lines 20-23)
asserts dense `eigh` "for N up to ~20k ... beats sparse iterative solvers",
which the plan's own Q4 benchmark (5.4x speedup for `eigsh` at N=10000)
measures false. Both of these are live, currently-shipping claims in
downstream user-facing documentation, already known wrong by the plan's
author, and the plan does not fix them until Phase 7 — six phases in which
anyone consulting `icampsnfr`'s docstrings for guidance is reading
demonstrably incorrect advice that the plan's author already possesses the
correction for. There's no reason a documentation-only fix to those two
docstrings needs to wait for the full module migration.

Separately: none of Phases 1-6 gate on whether cmm's *changing* public API
(§3's `from cmm import (...)` list) actually matches what `icampsnfr`'s real
call sites need. If a mismatch surfaces only in Phase 7, phases considered
"done and green" may need rework — the plan's own "nothing is deleted before
its replacement passes" rule doesn't cover "the replacement's public API
doesn't fit its only consumer," which can only be discovered once Phase 7
starts.

**What would resolve it**: (1) fix the two false docstring claims in
`icampsnfr` immediately, independent of the refactor timeline; (2) pull a
smoke-test of `icampsnfr`'s actual import/call sites against the *proposed*
new `cmm` API forward, e.g. as a Phase-0 or Phase-1 dry run, rather than
discovering API mismatches at Phase 7.

---

## MEDIUM severity

### 6. (c) The test-suite runtime target is unfalsifiable as written

**Claim challenged**: §4 "Runtime target < 60 s for the default suite;
anything slower is marked `@pytest.mark.slow` and deselected by default."

**Evidence**: The L1 cross-validation matrix alone is `window(4) ×
nperseg(3) × noverlap(3) × detrend(3) × scaling(2) × sides(2) × nfft(≥2) ×
real/complex(2)` ≈ 1700+ combinations, times up to 8 functions compared
against scipy/sklearn/mne. Combined with an `mne` multitaper reference (which
is meaningfully slower per call than Welch) and L3's SNR × k sweeps (each
requiring an actual k-means/spectral-clustering fit), this plausibly exceeds
60 s by a wide margin if run as literally described. But because "anything
slower is marked slow," the <60s claim as stated has no content: any test
that would violate it can simply be exempted after the fact, and the plan
never specifies which of L1/L2/L3/mne belongs in the unmarked default set.
This isn't necessarily wrong practice (splitting fast/slow suites is
standard), but as a *measurement gate the plan cites as a design constraint*,
it currently constrains nothing.

**What would resolve it**: state explicitly, in the plan, which specific
test files/parametrizations are "default" (and must genuinely fit under 60s)
versus "slow", rather than leaving the split to be decided implicitly by
whatever turns out to be slow.

### 7. (c) Module count / dependency direction: real merge candidates exist, and the stated acyclic ordering is ambiguous in a way that could hide a real cycle

**Claim challenged**: "lean, mean, no excess" (user's stated goal) versus an
8-module layout; "Dependency direction is strictly one-way: spectral →
coherence → {kmeans, clustering} → metrics."

**Evidence**: `kmeans.py` and `clustering.py` both consume the exact same
`coherence_matrix` output and are explicitly designed to "share all upstream
tests" and be "directly comparable" (§3) — that's an argument for one file
with two entry points (`coherence_kmeans`, `spectral_labels`), not two
files, especially under an explicit "no excess" mandate. `phase.py` and
`wavelet.py` are both narrowband instantaneous-phase estimators (Hilbert vs.
Morlet) and could reasonably live together. `metrics.py` is two functions
(`coherence_silhouette`, `cluster_summary`); its natural home is beside
whichever module needs it most, not a standalone file.

On the acyclicity claim: the diagram reads as `kmeans, clustering → metrics`,
i.e. metrics is *downstream* of (imported by, in the plan's own convention)
kmeans/clustering. But today's `CMM.compute_model_silhouette` computes
silhouette *from inside* the clustering class as a convergence/reporting
step — if `CoherenceKMeans.fit` wants the same behavior post-refactor, it
needs to *call into* `metrics.py`, which would make `kmeans.py` depend on
`metrics.py` — the opposite of the stated direction, and a genuine risk of
`metrics.py` needing to import `kmeans.py` for type/labels handling while
`kmeans.py` imports `metrics.py` for scoring. The plan asserts no cycle
exists but doesn't define which direction the arrows encode clearly enough
to check.

**What would resolve it**: an explicit statement of what each arrow means
(import direction vs. conceptual pipeline order), and a decision on whether
`CoherenceKMeans.fit` calls `metrics.coherence_silhouette` internally (if
yes, the diagram's ordering is wrong and needs `metrics` to precede
`kmeans`/`clustering`, not follow them).

### 8. (b) The "recovery" validation criteria are all easy, and the mixture-model's actual optimization step is never reconciled with the "V2 everywhere" decision

**Claim challenged**: §4 L3 "Both routes recover the true partition at the
true k: purity > 0.95, ARI > 0.9" as the model-validation strategy (task item
3: is there any external validation of the mixture model itself?).

**Evidence**: I read `icampsnfr/tests/test_coherence.py`, the design the
plan's L3 explicitly reuses. Its synthetic generator produces
well-separated, block-diagonal ground truth by construction (independent
band-limited sources per region, small noise) — a purity/ARI bar of
0.9-0.95 on data like this is a low bar that almost any reasonable
similarity + clustering combination clears, including ones with real bugs in
edge cases. There is no test against a published/independent reference for
the *coherence-mixture-model* itself (only the spectral primitives get
external validation, per the plan's own honest admission in item 3 of the
task brief), no near-degenerate/overlapping-cluster stress test, and no test
of boundary voxels that are genuinely ambiguous between two sources.

Separately: `CMM`'s actual centroid step (`compute_cluster_centroid_eigh`,
`cmm_funcs.py`) computes, **per frequency independently**, the dominant
eigenvector of a `K×K` gram built by contracting over cells within a
cluster — this is a fundamentally different mathematical object from any of
the four V1-V4 scalar coherence estimators in `coherence_estimators.md`. The
"V2 everywhere" decision is argued entirely in terms of the *scoring*
estimator (`coherence_matrix`, used for assignment/silhouette/affinity); the
plan never states whether or how the per-frequency eigen-centroid step needs
to change to stay consistent with that decision, or whether it's considered
out of scope. If it's out of scope, clusters would be *formed* by an
implicitly equal-weighted-across-frequency criterion (the eigen-decomposition
happens independently per f, with no power weighting across f) while being
*scored/assigned* by a power-weighted, tilt-vulnerable criterion (V2) — a
mismatch between formation and evaluation criteria that the plan doesn't
discuss.

**What would resolve it**: (1) add at least one test with genuinely
overlapping/ambiguous clusters (e.g. two sources sharing a sub-band) where
the "true" answer is itself uncertain, to see whether the model degrades
gracefully; (2) an explicit statement in `kmeans.py`'s design section on
whether/how the eigen-centroid step is or isn't affected by the V-estimator
choice.

### 9. (a) `estimate_wiener_r`'s inline comment actively misdescribes what the code does — worse than a merely stale TODO

**Claim challenged**: S3 in the audit says `lam_f` "is computed and
discarded" and `H` "uses raw `alpha_reg` ... contradicting ... the `# KEY
FIX #1` comment above it" — accurately described, but understated.

**Evidence** (`spectral_funcs.py:273-278`, confirmed by direct read): the
comment block literally says:
```
# ---------- KEY FIX #1: frequency-dependent regularization ----------
# lambda(f) = alpha * median_n Sxx(n,f)  (shape (F,))
lam_f = alpha_reg * da.median(pxx_diag, axis=0)              # (F,)
# Broadcast to (N,1,F) and add to Sxx:
print("alpha_reg", alpha_reg)
H = pxy / (pxx_diag[:, None, :] + alpha_reg)      # (N,M,F)
```
The comment claims a fix that the very next line of code does not perform —
`H`'s denominator adds the scalar `alpha_reg`, not the computed `lam_f`. This
isn't a stale docstring nobody updated; it's an active, specific, technically
detailed claim of correctness sitting directly above the line that
contradicts it, which is exactly the shape of comment most likely to mislead
a future maintainer who trusts the comment over re-deriving the math
themselves. Flagging it only as "computed and discarded" (S3's framing)
undersells how misleading this specific pattern is.

**What would resolve it**: Q1's planned repair (Phase 5) — use `lam_f` as
written in the comment — already fixes this; just worth flagging the comment
itself as a hazard class (comments asserting correctness beside code that
doesn't match) worth grep-checking elsewhere in the codebase before shipping.

---

## LOW severity / judgement calls

### 10. (b) S1's "silently discards the imaginary part" framing may overstate current reachability

**Claim challenged**: S1: `spectral_funcs.py:117-119`, `result = result.real`
on the "onesided" path, framed as silently discarding meaningful imaginary
input.

**Evidence**: `compute_spectral_coefs` (the only current caller of
`myfft_helper`) computes `sides` upstream: `sides = "twosided"` whenever
`np.iscomplexobj(xnt)` is true (lines 49-55), and only passes
`sides="onesided"` to `myfft_helper` when the input is real. Given that
routing, the `.real` cast in the "onesided" branch of `myfft_helper`
currently only ever executes on already-real data (through the one live call
path), making it a no-op in practice today, not a live silent-data-loss bug.
It remains bad defensive practice — worth an assertion rather than an
unconditional cast, especially since the refactor plan intends to expose
`stft_coefficients` (built on this helper) as public API, at which point
`myfft_helper`'s current safety net (its caller's own dispatch logic) may no
longer be guaranteed to hold for all call paths. I'd keep the fail-fast fix
from the plan, but the severity classification ("Serious — silent-continue")
implies live current harm that I could not confirm exists yet.

### 11. (a) Confirmed correct — no dispute here (spot-check results)

Direct execution against scipy in the `all` env reproduced the audit's
headline numbers essentially exactly: median(cmm/scipy) = 0.500000 for
`"density"`, 21.333333 for `"spectrum"` (C2); DC bin ratio 42.667 vs interior
21.333, exactly the claimed 2x shape distortion. Frequency axis: cmm returns
spacing 0.15385 Hz / top bin 4.9231 Hz against true 0.15625 Hz / 5.0 Hz (C3),
matching the audit's figures to 4 decimal places. C5 (silhouette sign
inversion — `cmm.py:315` passes a similarity to `metric="precomputed"`
silhouette, `icampsnfr/cluster.py:142` does `1 - sqrt(C)` correctly), C6
(`ymt_coh` never appended to, `ymt_coh_all` appended twice — `cmm.py:294-303`),
C7 (`ind`/`l` index divergence in `compute_cluster_coherence`), C8
(`self.coherence_mnk` never assigned), C9 (mutable default list mutated
in-place at `cmm.py:43-45`/`66-69`), H2 (`setup.py`: MIT vs pyproject's
BSD-3, `packages=["src"]`, hardcoded `0.1`), H4 (`tests/unit_test.py` imports
`from archive.spectral_funcs import ...` and `from archive.cmm_funcs import
...`), H5 (`filterwarnings = ["error"]` in `pyproject.toml:118` alongside a
live `warnings.warn` in `_triage_segments`), H6 (`scaling` and `axis`
parameters of `compute_spectral_coefs` are accepted but never referenced in
the function body — confirmed by full read; `nperseg=None` silently becomes
256 via `_triage_segments`), and the `icampsnfr` cross-file claims (C5's
counter-example, the docstring claims refuted by Q3/Q4) all check out on
direct inspection or execution. The audit is, on the whole, accurate and not
overstated — this is worth stating plainly since the brief asks for it.

### 12. (b) `power_iteration.py`'s missing iteration cap is a genuine infinite-hang risk, not just a hygiene nit (H8)

**Claim challenged**: H8 files this under "Hygiene": "`power_iteration` has
no iteration cap."

**Evidence** (confirmed by direct read of `power_iteration.py`):
`jax.lax.while_loop(cond_fun, inner_product, init_val)` where `cond_fun`
only checks `diff > 0.001` — no `maxiter`, no fallback. For a matrix with two
eigenvalues of comparable magnitude (plausible for any real coherence-derived
gram matrix with two nearly-tied dominant clusters), the power-iteration
update can fail to converge to a fixed tolerance and the `while_loop` will
spin indefinitely, hanging the JAX process with no user-visible error. That's
a process-level hang, not merely a style issue — I'd have filed this as
Critical/Serious rather than Hygiene, though it's moot in practice since Q4
already recommends deleting the file outright. Worth calling out explicitly
so it doesn't get silently ported into a "we'll keep power iteration as an
option" fallback later.

### 13. (a) The coherence-ratio scale-invariance argument (why C2 cancels in coherence but not in absolute PSD) is mathematically sound, not merely empirically lucky

The audit states coherence is exact (8.9e-16 vs scipy) "because the
normalization cancels in the ratio." I checked this is not a coincidence: the
missing one-sided ×2 doubling and the wrong `(win**2).sum()` denominator are
both *functions of frequency bin index and window alone*, identical across
`Pxy`, `Pxx`, `Pyy` for any pair of signals computed under the same
`nperseg`/`window`/`noverlap`. Since MSC is `|Pxy|²/(Pxx·Pyy)`, any
per-frequency multiplicative factor common to all three cancels exactly,
algebraically — this is a hard guarantee, not an empirical accident, and the
plan can state it with that confidence rather than "measured to be exact."
Worth noting this generalizes: *any* future scale bug that's a pure function
of `(frequency, window, nperseg)` will similarly cancel in coherence but
contaminate absolute-power outputs — a useful invariant for the new test
suite (L2's Parseval/permutation-equivariance tests should also assert this
"scale-cancellation" property explicitly as its own test, not just infer it
from the coherence-vs-scipy comparison).

### 14. (c) The measurement gates referenced by Q3/Q4 as "remaining" are visibly mid-flight in the scratchpad, not close to done

Artifacts outside the four target scripts (`gate23_bench.py`,
`v2_worker.py`, `v2_driver.py`, `build_real_C.py`, `res/driver.log`) show
infrastructure toward exactly the "real coherence graph" and "real data"
gates the plan flags as outstanding (Q3's tilt-on-real-data question, Q4's
Lanczos-on-real-graph gate) — but as observed live during this review, the
sweep driver had only completed N=1000 configurations, nowhere near the
N=6000/20000 scale the plan's own tables cite, and no `gate2`/`gate3` output
against a real `C_v2_real.npy` existed yet. This is consistent with, not
contradictory to, the plan's own "Gate: ... must clear before code is
removed" language — I flag it only to make explicit that, as of this review,
none of Q3's "real data" gap or Q4's "real coherence graph" gate is closed
yet, despite the plan's prose in places reading as though the numeric
conclusions (e.g. "5.4x" Lanczos speedup, "V2 everywhere") are settled.

---

## Summary — most severe first

1. **[HIGH, judgement]** V2-over-V1 decision compares a precisely quantified,
   severe V2 failure (tilt → 1.0 saturation) against an unquantified V1
   failure (dead-bin blowup); the choice is not actually settled by the
   evidence in the note, and was never tested on real data.
2. **[HIGH, demonstrated]** The delay/tilt/bias-floor tables in
   `coherence_estimators.md` are not reproducible from the cited `bench*.py`
   scripts (which compute V1/V3, not V2) or from any "snippets at the end"
   (which don't exist in the file). Independent re-derivation suggests the
   underlying math is sound, but the specific published numbers are
   currently unverifiable as cited.
3. **[HIGH, unsupported]** "$K_\text{eff}<K$ under overlap" drives a concrete
   `nperseg`/`noverlap` recommendation while being unmeasured, and the plan's
   own two sentences on the topic point in opposite directions.
4. **[HIGH, demonstrated+judgement]** C4's ~23x backprojection error went
   unnoticed for an unknown period; the plan defers "does this machinery
   even belong" to Phase 4 instead of settling it now with a call-site audit,
   and doesn't audit historical saved outputs for contamination.
5. **[HIGH, unsupported]** Integration with the only downstream consumer
   (`icampsnfr`) is deferred to Phase 7, six phases after a full rewrite,
   while `icampsnfr`'s own docstrings currently assert claims (n_eff formula,
   "eigh beats sparse solvers") the plan's own audit has already disproved —
   an easy, independent fix being needlessly delayed.
6. **[MEDIUM, judgement]** The <60s test-runtime target is unfalsifiable as
   written (unconstrained "slow" escape hatch against a >1700-combination L1
   matrix plus mne plus SNR/k sweeps).
7. **[MEDIUM, judgement]** Module count/dependency-direction: real merge
   candidates exist (`kmeans`+`clustering`, `phase`+`wavelet`), and the
   stated acyclic ordering (`kmeans, clustering → metrics`) likely runs
   backwards from real implementation needs (silhouette called from inside
   `.fit`), risking an undocumented cycle.
8. **[MEDIUM, unsupported]** L3 recovery tests only check an easy,
   well-separated synthetic case; the mixture model's actual per-frequency
   eigen-centroid step is never reconciled with the "V2 everywhere" decision.
9. **[MEDIUM, demonstrated]** `estimate_wiener_r`'s "KEY FIX #1" comment
   actively misdescribes the code beside it, not merely stale.
10. **[LOW, judgement]** S1 may currently be unreachable dead code given
    upstream dispatch logic — still worth fixing defensively, but the
    "Serious/silent-continue" severity label may overstate present harm.
11. **[LOW, demonstrated]** The bulk of the audit (C2, C3, C5-C9, H2, H4-H6,
    and the cross-file `icampsnfr` claims) is accurate on direct
    re-execution/inspection — no overstatement found there.
12. **[LOW, unsupported]** `power_iteration.py`'s uncapped `while_loop` is a
    genuine hang risk, arguably mis-filed as "Hygiene" rather than
    "Serious" — moot since the file is slated for deletion.
13. **[LOW, demonstrated]** The coherence-ratio scale-cancellation argument
    is an algebraic guarantee, not an empirical coincidence — worth stating
    with that confidence and testing explicitly in L2.
14. **[LOW, judgement]** The "remaining measurement gates" (Q3 real-data,
    Q4 real-graph) are visibly mid-flight in the scratchpad as of this
    review, not close to closed, despite some prose reading as settled.

No demonstrated HIGH-severity error was found in the plan's *architecture* or
*execution-phase* design beyond the ordering/deferral concerns above; the
gating discipline ("nothing deleted before its replacement passes") is sound
in principle. The most consequential open risk is squarely scientific, not
software-engineering: the V2 estimator decision (finding 1) is the one place
where the plan's own evidence, read skeptically, does not clearly support
the conclusion drawn from it.
