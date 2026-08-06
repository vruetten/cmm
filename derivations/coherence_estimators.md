# Band-averaged coherence: four estimators

There are four defensible ways to collapse a per-frequency coherence into a
single number per pair of signals, and `cmm` + `icampsnfr` between them use
three of the four — under overlapping names. This note fixes the notation,
states what each one normalizes by, and gives the measured behaviour that
distinguishes them.

All numbers below are measured, not asserted — but as of this revision the
generating code is **not** checked in, and the `scratchpad/bench*.py` scripts
compute V1 and V3, *not* the unwhitened V2 this note recommends. An independent
adversarial re-derivation (`logs/skeptic_plan_review.md`, finding 2) reproduced
the qualitative content of all three tables from scratch and confirmed the
underlying statistics, including that the 1/K null floor is the textbook
Beta(1, K-1) result. Until the generating scripts are checked in under
`derivations/scripts/`, treat the exact digits as unverified.

**§6 has been superseded — see the correction note at the end of this file.**

---

## 1. Setup

Let $X_n(k,f)$ be the STFT coefficient of signal $n$ in segment $k$ at
frequency $f$, with $k = 1 \dots K$ segments and $f \in \mathcal{B}$ a band of
$F$ bins. The segment-averaged cross- and auto-spectra are

$$
P_{nm}(f) \;=\; \sum_{k=1}^{K} X_n(k,f)\,\overline{X_m(k,f)},
\qquad
P_{nn}(f) \;=\; \sum_{k=1}^{K} |X_n(k,f)|^2 .
$$

The per-frequency **complex coherency** and **magnitude-squared coherence**
(MSC) are

$$
\gamma_{nm}(f) \;=\; \frac{P_{nm}(f)}{\sqrt{P_{nn}(f)\,P_{mm}(f)}},
\qquad
\gamma^2_{nm}(f) \;=\; |\gamma_{nm}(f)|^2 \in [0,1].
$$

Collapsing $\gamma^2(f)$ over the band forces two independent choices.

**Axis A — where the modulus goes.**
Take $|\cdot|$ *inside* the sum over $f$ (**incoherent** averaging: each
frequency contributes its own magnitude, phase relationships between
frequencies are discarded), or *outside* it (**coherent** averaging: the
complex cross-spectra are summed first, so a frequency-dependent phase can
cancel).

**Axis B — how frequencies are weighted.**
**Equal** weight per bin, or weight by power $P_{nn}(f)P_{mm}(f)$.

|                | equal weight | power weight |
|----------------|--------------|--------------|
| **incoherent** | **V1**       | **V2**       |
| **coherent**   | **V4**       | **V3**       |

---

## 2. The four estimators

### V1 — equal weight, incoherent  *(mean of per-frequency MSC)*

$$
\boxed{\;C^{(1)}_{nm} \;=\; \frac{1}{F}\sum_{f\in\mathcal{B}} \frac{|P_{nm}(f)|^{2}}{P_{nn}(f)\,P_{mm}(f)}\;}
$$

**Normalization.** Each term is normalized *separately*, by that bin's own
power product. A sum of $F$ numbers each already in $[0,1]$.

**Benefit.** Immune to spectral tilt — a $1/f$ background does not let the
lowest bin dominate. Directly interpretable as "average fraction of shared
variance per bin". Delay-invariant.

**Wrongness.** Divides by $P_{nn}(f)P_{mm}(f)$ at *every* bin, so one dead or
near-dead in-band bin either produces $0/0$ or amplifies pure noise to
$\gamma^2 \approx 1$ and drags the average up. And — see §5 — its null bias
floor is $1/K$ **regardless of $F$**: widening the band buys no bias
reduction whatsoever.

**In the code.** `cmm.spectral_funcs.compute_coherence` followed by `.mean(-1)`
in `CMM.allocate_data_to_clusters` (`cmm.py:220`). This is cmm's current
default and the estimator behind every existing cmm result.

---

### V2 — power weight, incoherent  *(ratio of sums)*

$$
\boxed{\;C^{(2)}_{nm} \;=\; \frac{\sum_{f} |P_{nm}(f)|^{2}}{\sum_{f} P_{nn}(f)\,P_{mm}(f)}\;}
$$

**Normalization.** One global division, by the summed power product. Equivalent
to a *weighted* mean of per-bin MSC,

$$
C^{(2)}_{nm} \;=\; \frac{\sum_f w_f\,\gamma^2_{nm}(f)}{\sum_f w_f},
\qquad w_f = P_{nn}(f)\,P_{mm}(f),
$$

which makes the relationship to V1 explicit: V1 is this with $w_f \equiv 1$.

**Benefit.** Numerically the most robust of the four — a single division by a
sum, so a dead bin contributes $\approx 0$ to numerator and denominator alike
instead of exploding. Diagonal is exactly $1$ (by construction
$\sum_f P_{nn}^2 / \sum_f P_{nn}^2$), so $1 - C$ is a valid zero-diagonal
dissimilarity usable directly as `metric="precomputed"`. Delay-invariant.
Bounded in $[0,1]$ by termwise Cauchy–Schwarz.

**Wrongness.** With a $1/f$-ish spectrum the weights span orders of magnitude
and the effective bandwidth collapses toward the lowest in-band bin. Two
signals that share only their strongest band score $\approx 1$ even when the
rest of the band is independent (measured: **1.000** against V1's 0.820, §4).
The nominal band is then not the band you are actually measuring.

**In the code.** `cmm.spectral_funcs.estimate_wiener_r`, as
`power_weighted_coherence` (`spectral_funcs.py:294`); and the `normalize=False`
branch of `CMM.compute_auto_coherence_from_coefs` (`cmm.py:193`) once summed
over $f$.

---

### V3 — power weight, coherent  *(MSC on the flattened $(k,f)$ axis)*

$$
\boxed{\;C^{(3)}_{nm} \;=\; \frac{\bigl|\sum_{k,f} X_n(k,f)\,\overline{X_m(k,f)}\bigr|^{2}}{\bigl(\sum_{k,f}|X_n(k,f)|^{2}\bigr)\bigl(\sum_{k,f}|X_m(k,f)|^{2}\bigr)}\;}
$$

**Normalization.** Product of *total* in-band powers. This is an ordinary MSC
in which $(k,f)$ is treated as a single flat sample axis of length $KF$.

**Benefit.** The largest effective sample count, $n_{\text{eff}} \approx K
F_{\text{eff}}$, hence by far the lowest bias floor (§5). Cheapest to compute:
one GEMM over the flattened $KF$ axis instead of $F$ separate ones — the source
of the ~24× in the benchmark. Diagonal exactly 1.

**Wrongness.** **Not delay-invariant.** A lag $\tau$ rotates the cross-spectrum
by $e^{i2\pi f\tau}$; summing coherently over $f$ then cancels once
$\tau \gtrsim 1/\Delta B$, where $\Delta B$ is the bandwidth. Measured on the
10–40 s band: **0.269** at a 10 s lag, **0.039** at 20 s, against V1's 0.930 and
0.755. It is legitimate only if you are willing to treat $(k,f)$ as exchangeable
samples of one stationary process — true across segments, *false* across
frequencies whenever there is propagation delay. **A travelling wave is scored
as incoherent and will be split into several clusters by lag.**

**In the code.** `icampsnfr.coherence.msc_matrix`.

---

### V4 — equal weight, coherent  *(squared modulus of the mean coherency)*

$$
\boxed{\;C^{(4)}_{nm} \;=\; \Bigl|\,\frac{1}{F}\sum_{f\in\mathcal{B}} \gamma_{nm}(f)\Bigr|^{2}\;}
$$

**Normalization.** Each bin whitened to unit coherency magnitude *first*, then
averaged coherently.

**Benefit.** The natural test for whether the *phase relationship is constant
across the band* — i.e. whether the pair is related by a pure time delay rather
than a frequency-dependent lag — with spectral tilt divided out. Lowest bias
floor of the four, cleanly $1/(KF)$ (§5).

**Wrongness.** Conflates two very different failures: a pair can have
$|\gamma(f)| = 1$ at every bin yet score $0$ if the phase rotates. As a
clustering similarity that is almost never what is wanted. Use as a
*diagnostic* beside V1/V2, not as the distance.

**In the code.** Not currently used in either repo. Listed for completeness —
it is the fourth cell, and it is the one that makes the structure below visible.

---

## 3. The four are one implementation

Whitening the coefficients per $(n,f)$,

$$
\hat{X}_n(k,f) \;=\; \frac{X_n(k,f)}{\sqrt{P_{nn}(f)}}
\quad\Longrightarrow\quad \hat{P}_{nn}(f) = 1 \;\;\forall f,
$$

sets every weight $w_f$ to 1 and therefore turns power weighting into equal
weighting *exactly*:

$$
C^{(2)}\bigl[\hat{X}\bigr] \;=\; C^{(1)}[X],
\qquad
C^{(3)}\bigl[\hat{X}\bigr] \;=\; C^{(4)}[X].
$$

Both verified numerically to `rtol=1e-12`. Unwhitened, V2 and V1 differ by up
to 0.045 on the same data.

So axis B is not a separate code path — it is a one-line pre-normalization.
The whole family is:

```python
coherence_matrix(coefs, whiten: bool, average: {"incoherent", "coherent"})
```

| `whiten` | `average`     | estimator |
|----------|---------------|-----------|
| `True`   | `incoherent`  | V1        |
| `False`  | `incoherent`  | V2        |
| `False`  | `coherent`    | V3        |
| `True`   | `coherent`    | V4        |

`CMM` already computes $\hat{X}$ as `coefs_xnkf_normalized` (`cmm.py:117`), so
the machinery exists.

---

## 4. Measured behaviour

**Delay sensitivity.** $y(t) = x(t-\tau)$, $f_s = 1$ Hz, band 10–40 s,
`nperseg=128`, 50 % overlap.

| $\tau$ | V1 eq/inc | V2 pw/inc | V3 pw/coh | V4 eq/coh |
|--------|-----------|-----------|-----------|-----------|
| 0 s    | 1.0000    | 1.0000    | 1.0000    | 1.0000    |
| 2 s    | 0.9970    | 0.9972    | 0.9516    | 0.9385    |
| 5 s    | 0.9817    | 0.9826    | 0.7302    | 0.6635    |
| 10 s   | 0.9301    | 0.9329    | 0.2685    | 0.1462    |
| 20 s   | 0.7549    | 0.7593    | 0.0391    | 0.0440    |
| 40 s   | 0.3281    | 0.3178    | 0.0055    | 0.0097    |

Axis A separates cleanly. (The residual decay in V1/V2 at 20–40 s is
segment-overlap loss — $\tau$ approaching `nperseg` — not phase cancellation.)

**Spectral tilt.** Strong shared high-frequency band plus an *independent* weak
low-frequency band, both inside the nominal band:

| V1 eq/inc | V2 pw/inc | V3 pw/coh | V4 eq/coh |
|-----------|-----------|-----------|-----------|
| 0.8202    | **1.0000**| 0.9947    | 0.6531    |

Axis B separates cleanly. The power-weighted pair report near-perfect coherence
for a pair that is independent over part of the band.

---

## 5. Bias floor — the part that decides most cases

For independent signals the estimators do not go to zero; they sit at a floor
set by the effective number of samples, $\mathbb{E}[\hat{C}] \approx
1/n_{\text{eff}}$. Monte Carlo, $K = 8$ segments, $1/f$ spectral tilt, 200 pairs:

| $F$ | V1 eq/inc | V2 pw/inc | V3 pw/coh | V4 eq/coh | $1/K$ | $1/(KF)$ |
|-----|-----------|-----------|-----------|-----------|-------|----------|
| 2   | 0.1204    | 0.1256    | 0.0692    | 0.0606    | 0.125 | 0.0625   |
| 5   | 0.1245    | 0.1156    | 0.0310    | 0.0233    | 0.125 | 0.0250   |
| 10  | 0.1265    | 0.1227    | 0.0200    | 0.0119    | 0.125 | 0.0125   |
| 20  | 0.1258    | 0.1221    | 0.0149    | 0.0061    | 0.125 | 0.0063   |
| 40  | 0.1248    | 0.1207    | 0.0104    | 0.0031    | 0.125 | 0.0031   |

> **Incoherent averaging gives $n_{\text{eff}} = K$, independent of bandwidth.**
> V1 and V2 sit at $1/K$ for every $F$. Adding frequency bins reduces their
> *variance* but not their *bias*. Coherent averaging gives
> $n_{\text{eff}} = K F_{\text{eff}}$ and the floor falls with the band.

At $K = 8$ that is a floor of **0.125 vs 0.003** — a factor of 40. With
overlapping segments the true $K_{\text{eff}}$ is smaller still, so V1/V2 floors
in practice are *worse* than 1/K.

V4 hits $1/(KF)$ exactly; V3 lands above it ($0.0104$ vs $0.0031$ at $F=40$)
because the $1/f$ weights reduce the participation ratio
$F_{\text{eff}} = (\sum_f w_f)^2 / \sum_f w_f^2$ to about 12 of the 40 bins.

**Debiasing trap.** `icampsnfr.coherence.debias_msc(C, n_eff)` is called with
`n_eff = n_seg * n_freqs`. That is correct for `msc_matrix`, which is V3. It is
**wrong by a factor of $F$ for V1 or V2**, which need `n_eff = K`. Applying the
V3 correction to a V1 matrix under-corrects the floor by 40× at $F = 40$.

---

## 6. Decision: V2 everywhere

**V2 is the estimator for both `cmm` and `icampsnfr`.** V3 is retired.

The reason is axis A, not axis B. Coherent averaging over frequency assumes the
phase relationship between two signals is constant across the band. A
propagating signal violates that by construction — the lag $\tau$ puts a
$2\pi f\tau$ ramp on the cross-spectrum, and summing coherently over $f$
cancels it. V3 scores a 10 s lag at 0.269 where the signals are in fact
perfectly coherent. For anything with spatial propagation it measures the
wrong thing.

Among the delay-invariant pair, V2 over V1 because it is numerically stabler
(one division by a sum, rather than $F$ per-bin divisions that blow up on a
dead bin) and has an exact unit diagonal.

V1 stays reachable as `whiten=True` — it costs one line (§3), and every cmm
result predating this note used it, so reproducing them needs it. V4 stays as
a diagnostic. Neither is a default.

**Consequences.**

1. **`icampsnfr` coherence output changes.** `msc_matrix` is V3; anything
   derived from it — cached matrices, dashboard configs, cluster labels,
   figures — is on the old estimator and needs regenerating.
2. **`debias_msc` must change.** It is called with
   `n_eff = n_seg * n_freqs`, correct for V3. V2 needs
   **`n_eff = n_seg`** — the bandwidth does not contribute (§5). Left as is, it
   under-corrects the floor by a factor of $F$ (40× at $F = 40$).
3. **Segment count is now the only bias lever.** Under V2 the floor is $1/K$
   and widening the band does not lower it. Reducing `nperseg` or increasing
   overlap to buy more segments is the only way down — and with overlapping
   segments $K_{\text{eff}} < K$, so the realized floor is worse than $1/K$.
   This should drive how `nperseg`/`noverlap` are chosen.
4. **Cost is acceptable.** V2 needs $F$ GEMMs where V3 needed one, but measured
   in the blocked formulation at $N=6000, K=10, F=20$: V3 1.80 s, V2 4.13 s
   (float32). 2.3× more than V3, but still 1.8× *faster* than the current dask
   path (7.40 s), and it avoids the 11.5 GB $(N,N,F)$ intermediate.
5. **Do not compare across estimators.** A V2 number and a V3 number from the
   same data are not the same quantity.

## 7. Implementation notes

- Compute $|P|^2$ as `(P * P.conj()).real`, not `np.abs(P)**2` — the latter
  takes a square root over an $(N,N,F)$ array only to square it again.
- V2 and V3 both have exact unit diagonals, so $1 - C$ is a valid dissimilarity
  with zero diagonal. V1's diagonal is also exactly 1.
- **Convention clash:** `icampsnfr.spectral.affinity_gaussian` takes a
  *similarity* $C$ and forms $1 - C$ internally. Passing it a distance
  double-inverts.
- **Name clash:** `cmm` calls V2 `power_weighted_coherence`;
  `icampsnfr.msc_matrix`'s docstring calls V3 "power-weighted". Both are
  power-weighted; they differ on axis A. Rename before either escapes into a
  published figure caption.

---

## 8. Correction to §6 (supersedes the V2-over-V1 argument)

§6 justified choosing V2 over V1 on the grounds that V2 is "numerically the
most robust of the four — a single division by a sum, so a dead bin contributes
≈ 0 to numerator and denominator alike instead of exploding."

**That claim is measured false.** Monte Carlo, two independent signals, K=10,
F=20, one in-band bin attenuated to relative power p, 200 trials:

| p | V1 | V2 |
|---|----|----|
| 1e+00 | 0.0994 | 0.0995 |
| 1e-02 | 0.0991 | 0.0987 |
| 1e-04 | 0.1000 | 0.1000 |
| 1e-08 | 0.1010 | 0.1008 |
| 1e-12 | 0.1003 | 0.0999 |

V1 and V2 are indistinguishable, both sitting at the 1/K = 0.1 null floor.
Attenuating a bin does not make V1 "amplify pure noise to γ² ≈ 1" — per-bin MSC
is *scale-invariant*, so shrinking a bin's power shrinks numerator and
denominator together and the ratio is unchanged. The same holds when the bin is
dead in only one of the two signals.

The sole real difference appears at **exactly** zero power: V1 returns `nan`,
V2 returns a finite value. That is a genuine advantage, but a narrow one — it is
fixed in V1 by dropping zero-power bins, which any implementation should do
anyway and which `msc_matrix`-style code already does at the voxel level.

**Against that, V2's failure mode is severe and was quantified in §4**: a pair
sharing only its strongest band scores **1.0000** under V2 where V1 gives 0.8202
(independently reproduced at V1=0.56 / V2=0.9999 with a different construction).
Power weighting means the result is dominated by whichever frequency carries the
most power. For 1/f-like spectra — which calcium and voltage imaging closely
are — that is always the lowest in-band bin, so a nominal "10–40 s band" analysis
silently collapses toward a 40 s analysis.

So on axis B the evidence, properly measured, favors **V1** — the opposite of
what §6 concluded. The §6 argument rested on an unmeasured claim about V1 that
turned out to be wrong, while V2's competing failure was measured and severe.

**Decision after this correction (unchanged default, informed):** V2 remains the
default, with V1 a first-class option via `whiten=True` that must be prominent
in the API and documented, not buried. The correction above stands as the record
of the tradeoff being accepted knowingly rather than by default: V2's power
weighting is a deliberate choice to let the frequencies carrying the most shared
power dominate. Implementations must document the consequence — see
`derivations/spectral_tilt_and_band_collapse.md` — and should report the
effective bin count $F_{\text{eff}}$ alongside any V2 matrix so the collapse is
visible rather than silent.

**What does not change:** axis A, which is the primary decision. V3 is still
rejected — coherent averaging over frequency assumes a phase relationship
constant across the band, which any propagating signal violates, and V3 scores a
10 s lag at 0.269 on perfectly coherent signals. Both V1 and V2 are
delay-invariant. The choice between them is the secondary axis, and it is the
one that needs revisiting.

**Open, and decisive:** which blind spot actually bites on this data. Both are
cheap to check on the real STFT coefficients: the spectral tilt (plot mean
in-band power across a few hundred real voxels — if it spans orders of
magnitude, V2 is compromised) and the dead-bin rate (fraction of (voxel,
frequency) cells at or near zero power — if ~0, V1's only weakness is moot).
Until that is measured, treat the V1-vs-V2 default as **undecided**.
