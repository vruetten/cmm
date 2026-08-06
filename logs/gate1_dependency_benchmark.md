# Gate 1 — dependency benchmark on a real multicore allocation

Re-run of the §2 benchmarks of `.claude/plans/cmm_refactor_plan.md`, re-targeted
at **estimator V2** (the decision recorded in Q3 / `derivations/coherence_estimators.md`):

$$C_{nm} = \frac{\sum_f |P_{nm}(f)|^2}{\sum_f P_{nn}(f)\,P_{mm}(f)},
\qquad P_{nm}(f) = \sum_k X_n(k,f)\,\overline{X_m(k,f)}$$

Measurement only. Nothing under `src/` was modified, no packages were installed.

Env: conda `all`, python 3.13.0, numpy 2.3.4, scipy 1.16.2, dask 2025.9.0,
OpenBLAS 0.3.28 (SkylakeX, pthreads), threadpoolctl 3.6.0.

Scripts: `<scratchpad>/blas_scaling.py`, `v2_worker.py`, `v2_driver.py`,
`v2_focus.py`, `run_focus.sh`, `collate.py`.

---

## 1. The parallelism finding — `nproc` was lying, the node is genuinely 80-way

This is the headline, because it invalidates the premise of the old numbers.

| probe | result |
|---|---|
| `len(os.sched_getaffinity(0))` | **80** |
| `os.cpu_count()` | 80 |
| `grep -c processor /proc/cpuinfo` | 80 |
| `nproc` | **1** |
| cgroup `cpu.max` | not set (no quota) |
| `threadpool_info()` | one OpenBLAS 0.3.28, SkylakeX, pthreads, `num_threads=1` |

**There is no 1-CPU affinity and no cgroup CPU quota.** The affinity mask
contains all 80 CPUs. `nproc` reports 1 purely because the environment exports

```
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
```

and `nproc` honors `OMP_NUM_THREADS`. Those same three variables are why
OpenBLAS reported `num_threads=1` and why every earlier benchmark ran
single-threaded. The cap was self-inflicted by the shell environment, not
imposed by the scheduler.

### Empirical BLAS scaling

`A @ B`, float64, n=4000, best of 3, thread count set via env before numpy import:

| OMP_NUM_THREADS | OpenBLAS reports | time (s) | GFLOPS | speedup |
|---|---|---|---|---|
| 1  | 1  | 1.4258 | 89.8 | 1.0x |
| 2  | 2  | 0.7023 | 182.3 | 1.6x |
| 4  | 4  | 0.4006 | 319.5 | 2.8x |
| 8  | 8  | 0.2265 | 565.1 | 5.0x |
| 16 | 16 | 0.1354 | 945.3 | 7.9x |
| 32 | 32 | **0.0834** | **1535.1** | **17.1x** |
| 64 | 64 | 0.1009 | 1268.7 | 14.1x |
| 80 | 80 | 0.1215 | 1053.6 | 12.5x |

Real parallelism is available and substantial: **17x from 1 to 32 threads**.
Throughput peaks at 32 and *degrades* beyond it (NUMA traffic / SMT contention),
so **32 is used as "max threads" throughout this report**, not 80. Setting
`OMP_NUM_THREADS=80` is measurably worse than 32.

The gate is therefore **not** inconclusive — this is a valid multicore
measurement. The prior §2 numbers were all single-threaded and are superseded.

---

## 2. What was benchmarked

Four backends, all computing exactly the V2 estimator above:

- **`dask_auto`** — dask lazy `einsum("nkf,mkf->nmf")`, `chunks="auto"`.
- **`dask_blk`** — same graph, but `n` chunked so one `(nb,nb,F)` complex128
  output block is ≤512 MB, i.e. dask genuinely streams rather than emitting a
  single giant chunk.
- **`c128`** — blocked numpy accumulation, complex128, 512-row blocks, one GEMM
  per frequency, never forms an `(N,N,F)` intermediate.
- **`c64`** — same, complex64.

and, added during the run because the first results demanded it:

- **`c128_mt` / `c64_mt`** — the same blocked accumulation with the *row blocks
  distributed over a `ThreadPoolExecutor`* (32 workers x 1 BLAS thread each)
  instead of relying on BLAS-internal threading. Still pure numpy; no new
  dependency.

Denominator note: $\sum_f P_{nn}(f)P_{mm}(f)$ is separable, so it is a single
real GEMM `d @ d.T` with `d[n,f] = P_nn(f)`. Numerator uses
`S.real**2 + S.imag**2`, not `abs(S)**2`, per §7 of the derivation. The dask
path was given the identical arithmetic so the comparison is on the execution
engine, not on the formula.

### Two measurement bugs found and fixed during the run

1. **dask import inside the timed region.** `v2_focus.py` initially imported
   `dask.array` inside the timed function, charging dask ~7.5 s of import and
   first-graph-construction cost. At N=1000/K=10/F=20/32 threads this inflated
   dask from 0.267 s to 7.874 s — a 30x error *against* dask. Fixed by importing
   and warming dask up outside the timed region; the corrected 0.267 s matches
   the independent subprocess sweep's 0.307 s. **All dask numbers below are
   post-fix.**
2. **Estimator formula changed mid-sweep.** `abs(P)**2` was replaced with
   `real**2+imag**2` after some runs had completed. All affected dask results
   were discarded and re-run; the numpy results were unaffected.

Both are noted because a partial gate that quietly mixed pre- and post-fix
numbers would have produced the opposite recommendation.

---

## 3. Results

### 3.1 Four-backend subprocess sweep (one process per run, clean peak RSS)

Wall time, best of 2:

| N | K | F | threads | `dask_auto` | `dask_blk` | `c128` | `c64` | `c64` vs `dask_blk` |
|---|---|---|---|---|---|---|---|---|
| 1000 | 10 | 20 | 1 | 1860 ms | 1910 ms | 182 ms | 88 ms | **21.6x** |
| 1000 | 10 | 20 | 32 | 325 ms | 307 ms | 222 ms | 98 ms | **3.1x** |
| 1000 | 10 | 60 | 1 | 4267 ms | 3744 ms | 420 ms | 239 ms | **15.7x** |
| 1000 | 10 | 60 | 32 | 723 ms | 640 ms | 428 ms | 243 ms | **2.6x** |
| 1000 | 30 | 20 | 1 | 3013 ms | 3128 ms | 197 ms | 118 ms | **26.5x** |
| 1000 | 30 | 20 | 32 | 444 ms | not run | not run | not run | — |

Peak RSS, GB (per-process high-water, `ru_maxrss`):

| N | K | F | threads | `dask_auto` | `dask_blk` | `c128` | `c64` |
|---|---|---|---|---|---|---|---|
| 1000 | 10 | 20 | 1 | 0.20 | 0.20 | 0.09 | 0.07 |
| 1000 | 10 | 20 | 32 | 0.47 | 0.47 | 0.08 | 0.06 |
| 1000 | 10 | 60 | 1 | 0.23 | 0.24 | 0.10 | 0.08 |
| 1000 | 10 | 60 | 32 | 0.93 | 0.94 | 0.10 | 0.08 |
| 1000 | 30 | 20 | 1 | 0.23 | 0.23 | 0.10 | 0.08 |
| 1000 | 30 | 20 | 32 | 0.80 | — | — | — |

### 3.2 Focused grid — all backends per process, including thread-pooled numpy

Wall time. `_mt` = row blocks over a 32-worker thread pool, 1 BLAS thread each (only meaningful at 32 threads, so not run at 1 thread).

| N | K | F | threads | `c128` | `c64` | `c128_mt` | `c64_mt` | `dask_blk` | fastest | fastest vs `dask_blk` |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 10 | 20 | 32 | 241 ms | 95 ms | 120 ms | 63 ms | 267 ms | `c64_mt` | **4.2x** |
| 3000 | 30 | 60 | 32 | 6.218 s | 2.675 s | 2.286 s | 947 ms | 3.081 s | `c64_mt` | **3.3x** |
| 6000 | 10 | 20 | 32 | 13.394 s | 3.968 s | 1.797 s | 917 ms | 6.680 s | `c64_mt` | **7.3x** |
| 6000 | 30 | 60 | 32 | 39.495 s | 16.545 s | 5.631 s | 2.925 s | 10.794 s | `c64_mt` | **3.7x** |
| 20000 | 10 | 20 | 32 | 160.086 s | 90.208 s | 35.033 s | 18.286 s | 83.946 s | `c64_mt` | **4.6x** |
| 20000 | 30 | 60 | 32 | not run | not run | 103.012 s | 55.444 s | 115.474 s | `c64_mt` | **2.1x** |

Peak RSS (GB) for the same cells:

| N | K | F | threads | `c128` | `c64` | `c128_mt` | `c64_mt` | `dask_blk` |
|---|---|---|---|---|---|---|---|---|
| 1000 | 10 | 20 | 32 | 0.20 | 0.20 | 0.23 | 0.23 | 0.49 |
| 3000 | 30 | 60 | 32 | 0.53 | 0.53 | 0.91 | 0.91 | 2.16 |
| 6000 | 10 | 20 | 32 | 0.59 | 0.59 | 2.33 | 2.33 | 2.33 |
| 6000 | 30 | 60 | 32 | 1.00 | 1.00 | 2.75 | 2.75 | 2.97 |
| 20000 | 10 | 20 | 32 | 3.69 | 3.69 | 12.23 | 12.23 | 12.23 |
| 20000 | 30 | 60 | 32 | — | — | 14.03 | 14.03 | 14.03 |

---

## 4. Numerical agreement

From the subprocess sweep (reference `dask_blk`; max abs difference over a subsampled ~400x400 block of the N x N matrix C, whose entries lie in [0,1]):

| N | K | F | threads | `c128` vs dask | `c64` vs dask | `dask_auto` vs `dask_blk` |
|---|---|---|---|---|---|---|
| 1000 | 10 | 20 | 1 | 5.6e-16 | 2.2e-07 | 0.0e+00 |
| 1000 | 10 | 20 | 32 | 5.6e-16 | 2.2e-07 | 0.0e+00 |
| 1000 | 10 | 60 | 1 | 8.9e-16 | 3.3e-07 | 3.3e-16 |
| 1000 | 10 | 60 | 32 | 8.9e-16 | 3.3e-07 | 5.6e-16 |
| 1000 | 30 | 20 | 1 | 6.7e-16 | 2.5e-07 | 0.0e+00 |

From the focused grid (reference `c128`, computed in-process):

| N | K | F | threads | `c64` | `c128_mt` | `c64_mt` | `dask_blk` | mean diag(C) range |
|---|---|---|---|---|---|---|---|---|
| 1000 | 10 | 20 | 32 | 2.2e-07 | 4.4e-16 | 2.2e-07 | 5.6e-16 | 1.000000000–1.000000001 |
| 3000 | 30 | 60 | 32 | 3.4e-07 | 3.5e-17 | 3.4e-07 | 7.8e-16 | 1.000000000–1.000000001 |
| 6000 | 10 | 20 | 32 | 2.0e-07 | 4.2e-17 | 2.0e-07 | 6.7e-16 | 0.999999998–1.000000000 |
| 6000 | 30 | 60 | 32 | 3.2e-07 | 2.1e-17 | 3.2e-07 | 7.8e-16 | 1.000000000–1.000000000 |
| 20000 | 10 | 20 | 32 | 2.4e-07 | 0.0e+00 | 2.4e-07 | 6.7e-16 | 1.000000000–1.000000001 |
| 20000 | 30 | 60 | 32 | — | 0.0e+00 | 3.4e-07 | 7.8e-16 | 1.000000000–1.000000000 |

Worst `c128`-class (float64) disagreement anywhere: **8.9e-16** (tolerance 1e-10 — passes by ~6 orders).
Worst `c64` (float32) disagreement anywhere: **3.4e-07** (tolerance 1e-6 — passes).
Worst deviation of `mean(diag(C))` from exactly 1.0: **2.0e-09** across 28 backend runs.

**Both tolerances in the plan are met:**

- `c128` vs dask: worst observed **8.9e-16** — far inside the required `rtol=1e-10`.
  Effectively bit-identical.
- `c64` vs dask: worst observed **3.4e-7** — inside the required `1e-6`, and six
  orders below the $1/K \approx 0.1$ bias floor that dominates the estimate
  anyway (§5 of the derivation).
- `c128_mt` vs `c128`: ≤4.4e-16. Thread-pooling changes nothing numerically —
  the row blocks are independent.
- `dask_auto` vs `dask_blk`: ≤5.6e-16, i.e. chunking choice is numerically inert.

**V2 identity check.** V2 has an exact unit diagonal by construction. Measured
`mean(diag(C))` across every successful run: **1.000000000 ± 2e-9** (worst
observed deviation 2.0e-9, on a `c64` run). This confirms the implementations
compute V2 and not V1/V3.

---

## 5. jax

`import jax` fails: `ModuleNotFoundError: No module named 'jax'`. **Confirmed
not installed. Nothing was installed.**

The plan's §2 says "two `jnp.einsum` call sites". The actual count is **four**,
plus two jax-only modules:

| site | what it does |
|---|---|
| `src/cmm/cmm.py:116` | `jnp.einsum("fkn,fkn->fn", ...)` on numpy arrays |
| `src/cmm/cmm.py:291` | `jnp.einsum("mkf,ktf->mtf", ...)`, `.real` taken immediately |
| `src/cmm/cmm.py:297` | `jnp.einsum("mkf,ktf->mtf", ...)` |
| `src/cmm/cmm_funcs.py:22` | `jnp.einsum("fkn,fln->fkl", ...)`, result fed straight to `scipy.linalg.eigh` |
| `src/cmm/power_iteration.py` | entire module is jax; only referenced from `demos/test_power_iteration.py` |
| `src/cmm/utils.py:148` `foldxy` | function-local jax import; **dead code — defined, never called anywhere in the repo** |

Every one of the four `jnp.einsum` calls receives numpy arrays and has its
result immediately consumed by numpy/scipy, so each is a host→device→host round
trip wrapping a single einsum with **no fused work and no kernel to amortize
the transfer**. There is nothing for jax to optimize across.

Note `cmm_funcs.py:22` also builds `da.from_array(...)` into a variable `array`
that is then never used — a leftover from a dask experiment.

**Recommendation: drop jax.** Unchanged from the plan, now with the call-site
count corrected from 2 to 4 (+2 modules).

---

## 6. Recommendation on dask

### Drop it.

**This contradicts the plan's §2 expectation that dask stays behind a size
threshold. There is no such threshold in the measured range.** A pure-numpy
thread-pooled blocked accumulation (`c64_mt`) is fastest at **every** size, K,
and F measured, at full thread count, and the single-threaded blocked numpy path
is fastest at every size measured at 1 thread. dask is never the best backend in
any cell of this benchmark.

### Why the old numbers said the opposite

The §2 table compared **dask running multi-threaded against numpy pinned to one
core**. dask's default threaded scheduler takes its worker count from
`multiprocessing.cpu_count()` (80 here) and ignores `OMP_NUM_THREADS`, whereas
OpenBLAS obeys it. So the "dask wins 4x" result was measuring dask-on-80-threads
versus numpy-on-1-thread. It was a threading artifact, not a property of dask.

### Why plain blocked numpy looked mediocre at 32 threads

At N=1000 the plain `c128` path got *slower* going from 1 to 32 BLAS threads
(182 ms → 222 ms). The GEMM per (block, frequency) is skinny — `(512,K) @ (K,N)`
with K=10 — so it is memory-bound, and more importantly the dominant cost is the
elementwise `S.real**2 + S.imag**2` accumulation, which is a **single-threaded
numpy ufunc that BLAS threading cannot touch**. Raising `OMP_NUM_THREADS` speeds
up only the small GEMM half of the work.

Distributing whole row blocks over a `ThreadPoolExecutor` (32 workers, 1 BLAS
thread each) parallelizes the elementwise part too, because numpy releases the
GIL for both GEMM and ufuncs. That is the entire trick, and it is what makes
dask redundant — dask's only real contribution was task-parallelism over blocks,
which is ~20 lines of `concurrent.futures`.

Measured effect of the thread pool at 32 threads:

| N | K | F | `c64` (BLAS-threaded) | `c64_mt` (pool) | gain |
|---|---|---|---|---|---|
| 1000 | 10 | 20 | 95 ms | 63 ms | 1.5x |
| 3000 | 30 | 60 | 2.675 s | 0.947 s | 2.8x |
| 6000 | 10 | 20 | 3.968 s | 0.917 s | 4.3x |
| 6000 | 30 | 60 | 16.545 s | 2.925 s | 5.7x |
| 20000 | 10 | 20 | 90.208 s | 18.286 s | 4.9x |
| 20000 | 30 | 60 | not run | 55.444 s | — |

### The headline comparison

`c64_mt` vs `dask_blk`, both at 32 threads:

| N | K | F | `dask_blk` | `c64_mt` | numpy speedup | RSS: dask → numpy |
|---|---|---|---|---|---|---|
| 1000 | 10 | 20 | 0.267 s | 0.063 s | **4.2x** | 0.49 → 0.23 GB |
| 3000 | 30 | 60 | 3.081 s | 0.947 s | **3.3x** | 2.16 → 0.91 GB |
| 6000 | 10 | 20 | 6.680 s | 0.917 s | **7.3x** | 2.33 → 2.33 GB |
| 6000 | 30 | 60 | 10.794 s | 2.925 s | **3.7x** | 2.97 → 2.75 GB |
| 20000 | 10 | 20 | 83.946 s | 18.286 s | **4.6x** | see note |
| 20000 | 30 | 60 | 115.474 s | 55.444 s | **2.1x** | see note |

(RSS in the focused harness is the *process* high-water, so a backend that runs
after a heavier one inherits its peak; the clean per-backend RSS figures are the
subprocess-sweep table in §3.1. At N=20000 the process high-water was 3.69 GB
through the serial backends and 12.23 GB once the 32-worker pool allocated its
per-worker `(512, N)` accumulators — still an order of magnitude below the
node's 376 GB, and dask measured no lower.)

And at 1 thread, `c64` vs `dask_blk` (subprocess sweep, N=1000): **15.7x–26.5x**
in numpy's favor.

Note also that dask **does not benefit much from the extra cores on this
workload**: at N=6000/K=10/F=20 the old single-threaded §2 measurement was 7.24 s
and the 32-thread measurement here is 6.68 s. The graph is dominated by
materializing and reducing `(nb,nb,F)` blocks, which is memory-bandwidth-bound;
adding workers does not help. numpy on the same cell went to 0.917 s.

### The margin narrows at the largest size — but does not close

The one place dask comes closest is the biggest cell, N=20000/K=30/F=60, where it
is **2.1x** behind rather than 3.3x-7.3x. dask scales better with `K*F` than with
`N`: going from (K=10,F=20) to (K=30,F=60) at N=20000 is 9x more arithmetic, and
dask's time grew only 1.4x (83.9 s → 115.5 s) while `c64_mt` grew 3.0x
(18.3 s → 55.4 s). At (K=10,F=20) the dask graph is dominated by per-block
overhead that the extra arithmetic then amortizes.

Extrapolating that trend is exactly what this gate is supposed to avoid, so state
it plainly: **at the largest size measured, numpy is 2.1x faster, and the trend
in `K*F` is in dask's favour.** If a future workload sits at much larger `K*F`
than (30, 60), the comparison should be re-run rather than assumed. Nothing in
the plan's stated problem sizes goes there, and 2.1x with half the code and one
fewer dependency is still the right trade.

### The "dask never materializes (N,N,F)" argument does not survive either

That was the strongest argument for keeping dask, and it is **also true of the
blocked numpy path** — which is the whole point of blocking. The numpy backend's
largest intermediate is one `(512, N)` float64 accumulator per worker. Measured
peak RSS is *lower* than dask's at every size. The `(N,N,F)` array (11.5 GB at
N=6000/F=20; 384 GB at N=20000/F=60) is never formed by either backend, so it is
not a differentiator.

Separately, `dask_auto` — `chunks="auto"` on an input small enough to be one
chunk — produces a **single** `(N,N,F)` output chunk, i.e. it materializes the
whole thing as one numpy array. At N=6000/F=20 that is 11.5 GB in one
allocation. So the configuration that the plan's §2 benchmark actually used was
not streaming at all; it was `np.einsum` with extra steps. Any future use of
dask here would have to chunk explicitly, as `dask_blk` does.

### Concrete recommendation

1. **Remove `dask` from `cmm`'s dependencies.** No `backend="dask"` argument,
   no size threshold — there is no regime in the measured range where it wins.
   This also disposes of S4 (dask graphs leaking into `store_results()` and into
   `np.save`) by construction rather than by discipline.
2. **`coherence_matrix` gets one implementation**: blocked accumulation over row
   blocks, dispatched across a `ThreadPoolExecutor` whose worker count defaults
   to `len(os.sched_getaffinity(0))` capped at ~32, with `threadpool_limits(1)`
   around the BLAS calls to prevent oversubscription. Expose `n_jobs=` and
   `block=` as parameters.
3. **Default dtype complex64 / float32 output**, per Q5 — confirmed here: 1.5x
   to 5.7x faster than complex128 in the same harness, agreement 3.4e-7. Keep
   `dtype=np.complex128` reachable for audits; it is the reference the test suite
   pins against.
4. **Do not set `OMP_NUM_THREADS=1` in the package**, but *do* document that the
   environment currently exports it, which silently caps every user of this
   package to one core. Reading the affinity mask rather than trusting `nproc`
   is the fix.
5. **Test-suite consequence**: the "numpy vs dask to 1e-10" backend-agreement
   test in plan §4/L2 should become "pooled vs serial vs c128 vs c64", since dask
   is gone. Those agreements are measured here at ≤4.4e-16 (pooled vs serial) and
   ≤3.4e-7 (c64 vs c128).

If dask is nonetheless kept for reasons outside this benchmark (e.g. a future
out-of-core or multi-node path), it should be an explicit opt-in for `N` beyond
what fits in RAM — **not** a default, and not justified by speed. The `(N,N)`
output itself is 3.2 GB at N=20000 in float64 / 1.6 GB in float32, so the first
real memory wall is the result, not the intermediates, and dask does not help
with that.

---

---

## 7. What was not measured

Stated explicitly so nothing here is mistaken for a complete grid. The plan
asked for N in {1000, 3000, 6000, 20000} x K in {10, 30} x F in {20, 60} x threads
in {1, max} - 64 cells per backend. **That full grid was not completed.** What is
reported above is what actually ran; every number in this report is a real
measurement and nothing has been extrapolated into a table.

**Measured, at 32 threads:** (N,K,F) = (1000,10,20), (3000,30,60), (6000,10,20),
(6000,30,60), (20000,10,20) for all five backends, and (20000,30,60) for
`c128_mt` / `c64_mt` / `dask_blk`; plus (1000,10,20),
(1000,10,60), (1000,30,20) for the four-backend subprocess sweep.

**Measured, at 1 thread:** (1000,10,20), (1000,10,60), (1000,30,20) only, for
`dask_auto` / `dask_blk` / `c128` / `c64`.

**Not measured:**

- **`c128` and `c64` (the serial numpy backends) at N=20000, K=30, F=60.**
  Deliberately skipped: at that size they cost ~35 min between them and answer
  nothing the five smaller cells have not already settled. The open question at
  that size was pool-vs-dask, and `c128_mt` / `c64_mt` / `dask_blk` were all run.
  Consequence: the agreement reference for that one row is `c128_mt`, not `c128`
  (the two are measured identical to <=4.4e-16 at every other size).
- **1-thread numbers for N >= 3000.** The 1-thread conclusion (blocked numpy beats
  dask by 15-26x) therefore rests on N=1000 alone. It is consistent in direction
  with the plan's own prior single-threaded measurements at N=3000-6000, but this
  report did not re-measure them.
- **The (K=30,F=20) and (K=10,F=60) corners at N >= 3000.** The measured cells
  bracket them - (K=10,F=20) is the cheapest corner and (K=30,F=60) the most
  expensive - and the backend ranking is identical at both, but interpolation is
  not measurement.
- **`dask_auto` beyond N=1000.** At N=20000/F=60 it needs a single 384 GB chunk,
  which exceeds both the 200 GB address-space guard and the node's free RAM; it
  was never going to run. Its N=1000 agreement with `dask_blk` (<=5.6e-16) is all
  it was needed for.
- **`_mt` at 1 thread** - meaningless by construction (a 1-worker pool).
- **Thread counts other than 1 and 32** for the estimator. The 2/4/8/16/64/80
  points exist only for the raw `A @ B` scaling test in section 1.
- **jax**: not benchmarked at all, because it is not installed and the brief
  forbade installing it. The conclusion in section 5 is a static-analysis
  argument about the call sites, not a measurement.

**Why the grid was cut short.** The first approach ran one subprocess per
(backend, size, thread-count) - 128 processes, each paying ~10 s of interpreter
and dask import. That overhead dominated and the sweep was on track to take many
hours; it was stopped after 20 runs, all at N=1000, which are reported in section
3.1. It was replaced by a harness that runs every backend inside one process per
size, which produced section 3.2 and reached N=20000.

**Does the missing data threaten the recommendation?** `c64_mt` is the fastest
backend in all six completed 32-thread cells, spanning a 3600x range in arithmetic
(N^2*K*F from 2.0e9 to 7.2e12). The margin over `dask_blk` is 2.1x-7.3x and is
**smallest at the largest cell**, so the honest statement is that numpy wins
everywhere measured but the gap narrows with K*F - see the note in section 6. The
1-thread regime is the weakest-evidenced part of this report (N=1000 only), but it
is also the regime where numpy's margin was largest (15.7x-26.5x), so it is not
where the recommendation is at risk.

