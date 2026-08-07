# cmm — project-scoped instructions

Coherence-based clustering of timeseries: coherence k-means (the coherence
mixture model) and coherence spectral clustering, on a shared spectral core
that other packages import. Python, numpy/scipy/scikit-learn.

Currently mid-refactor from a working-but-broken state. Read these before
making decisions — they exist so the same ground is not re-covered:

- `.claude/plans/cmm_refactor_plan.md` — audit, target architecture, phases, gates
- `derivations/coherence_estimators.md` — the four coherence estimators and why V2
- `logs/` — measurement-gate results and reviews
- `journal/` — session log; `/start-session` reads the latest entry

## Decided — do not re-litigate

**The coherence estimator is V2**: power-weighted, incoherently averaged over
frequency.

```
C[n,m] = sum_f |P_nm(f)|^2 / sum_f P_nn(f) P_mm(f),   P_nm(f) = sum_k X_n(k,f) conj(X_m(k,f))
```

Taking the modulus per-frequency *before* summing is what makes it
delay-invariant, and that is the whole point: a propagating signal puts a
`2*pi*f*tau` ramp on the cross-spectrum, so averaging coherently over frequency
(V3, `icampsnfr.coherence.msc_matrix`) cancels it and scores a 10 s lag at 0.269
where the signals are in fact perfectly coherent. V3 is rejected. V1 (equal
weighting) stays reachable as `whiten=True` — a one-line pre-normalization that
reproduces it exactly — because every cmm result predating the refactor used it.

**The bias floor is 1/K, independent of bandwidth.** Under incoherent averaging
the null floor is set by the number of segments alone; adding frequency bins
reduces variance but not bias. Widening the band does not clean up a coherence
map — only more segments do. This must drive `nperseg`/`noverlap` choice, and it
means `debias` takes `n_eff = K`, not `K*F`.

## The existing code is known-broken — do not use it as a reference

Confirmed by measurement, not inspection:

| What | Measured |
|---|---|
| PSD vs `scipy.signal.welch`, `scaling="spectrum"` | 21.3x off |
| PSD `scaling="density"` | 0.5x off — the one-sided doubling is missing, so DC and Nyquist sit 2x high *relative* to interior bins |
| `project` -> `backproject` round trip | magnitude ratio 0.044, not 1 |
| Frequency axis when `x_in_coefs=True` | 0.1538 Hz spacing against a true 0.15625 |
| `compute_model_silhouette` | passes a similarity where sklearn wants a distance — sign-inverted |

**The one exception**: coherence itself is exact — 8.9e-16 against
`scipy.signal.coherence`, because the normalization error cancels in the ratio.
It is the only path in the current code that can be trusted.

Do not copy formulas out of `src/cmm/` into new code without checking them
against scipy first.

## Working rules

- **Measure, do not assume.** Every performance and numerical claim in this repo
  was measured before being written down, including the ones that overturned the
  expected answer. Keep it that way — "should be faster" and "should be
  equivalent" have both been wrong here.
- **Never mix a pure refactor with a change that moves numbers.** Separate
  commits, and say in the message what changes numerically and which stored
  results it invalidates. Old figures were computed with wrong code; that has to
  stay traceable.
- **`icampsnfr` is the only consumer** (`/groups/ahrens/home/ruttenv/code/icampsnfr`,
  imports `cmm.spectral` in `coherence.py`, `frc.py`, `spectrum.py`,
  `traces.py`). It is also the only end-to-end integration test either package
  has. Run its suite when changing the spectral core, not at the end.
- **Fail fast.** Raise with a message naming the offending value and the fix. No
  print-and-continue, no `exit()`, no silently dropping the imaginary part.
- Analysis scripts and I/O belong in `examples/` or the caller, not in the
  package. No pandas/tifffile/zarr in library code.

## Environment

- conda env `all`, Python 3.13, numpy 2.3.4, scipy 1.16.2, sklearn 1.7.2, dask 2025.9.0.
- **jax is not installed.** `cmm.cmm`, `cmm.cmm_funcs`, `cmm.power_iteration` and
  `cmm.ana_funcs` therefore fail to import — the `CMM` class is currently dead
  code. Do not add jax back; see the plan.
- Real data lives under `/nrs/ahrens/Virginia_nrs/` — **read only**, never write
  there and never accumulate data files in this repo.

## Benchmarking: always set the thread count explicitly

**This shell exports `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`.** `os.sched_getaffinity(0)` returns all 80 CPUs and there is
no cgroup quota — numpy/scipy BLAS is capped purely by those variables, while
dask's scheduler reads `cpu_count()` = 80 and ignores them.

This silently invalidated a whole round of benchmarking (1-thread numpy compared
against 80-thread dask, which reversed the conclusion). Measured BLAS scaling:
89.8 GFLOPS at 1 thread, 1535 at 32, degrading beyond 32 — so **32 threads is
the practical optimum, not 80**.

Rules: set threads explicitly (`threadpoolctl.threadpool_limits`, or env vars
before numpy is imported); record the thread count next to every timing; never
compare two libraries without checking both honor the same setting.

## Decided by measurement

- **`dask` is dropped** — blocked numpy over a `ThreadPoolExecutor` is 2.1–7.3x
  faster at 32 threads and uses less peak memory. See
  `logs/gate1_dependency_benchmark.md`. Caveat: the margin narrows at large K·F.
- **`jax` is dropped.**

- **Eigensolver: use `eigsh(S, which="LA")`**, where `S = D^-1/2 W D^-1/2` and
  `lambda_L = 1 - lambda_S`. On a real N=10000 coherence graph at 32 threads:
  dense 14.9 s vs eigsh 0.61 s (**24x**), `dLambda ~ 1e-15`, and **ARI = 1.0000
  in all 12 configurations** — identical clusters, not merely similar. Memory
  +5 MB vs +755 MB. Keep dense as `solver="dense"` for audits and eigengap.
  **Reject `lobpcg`** — eigenvalues agreed to 3e-5 but ARI was 0.39.
  A weak Laplacian gap does *not* imply slow Lanczos: a near-uniform affinity is
  near-rank-one, so S's whole non-trivial spectrum sits below an isolated
  `lambda_1 = 1`. The claim in `icampsnfr/spectral.py:21` is measured false.
- **Reject k-NN sparsification.** 7–110x faster but ARI vs dense peaked at 0.39
  and went *negative*; `max|dLambda|` 0.38–0.98. It is a different graph, not an
  approximation. The graph never disconnected — with median coherence 0.055
  against a 0.045 floor, top-k selection amplifies estimator noise.

## Open — the V1/V2 choice is not currently justified by evidence

Both documented arguments are **measured inert on this data**: no spectral tilt
(per-bin power varies 1.38x, `F_eff` = 14.1/15 — the 120 s high-pass already
removed the 1/f drift) and no dead bins (zero cells below 1e-2 of their voxel
median). Neither estimator's blind spot occurs.

**Yet they disagree materially** — cluster ARI 0.64 at k=7, 0.33 at k=15;
top-1% pair Jaccard 0.40. The choice changes the result while neither reason for
it applies. V2 stands as the decision; it is not yet an evidenced one.

Proposed discriminator, needing no ground truth: **label stability under segment
and voxel resampling** — prefer whichever estimator gives more reproducible
clusters. Not yet run.
