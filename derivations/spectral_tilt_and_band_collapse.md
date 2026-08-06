# Spectral tilt and band collapse under power-weighted coherence

Why a nominal "10–40 s band" analysis using estimator V2 silently becomes an
analysis of the slow edge of that band, how far it collapses, and what to do
about it.

Companion to `coherence_estimators.md`, which defines V1–V4. This note concerns
axis B (equal vs power weighting) only. Everything here applies to V2 and V3,
not to V1 or V4.

---

## 0. Notation

Repeated from `coherence_estimators.md` §1 so this note reads standalone.

$X_n(k,f)$ is the STFT coefficient of signal $n$ in segment $k$ at frequency
$f$, with $k = 1 \dots K$ segments and $f$ ranging over the $F$ bins inside the
band of interest. Subscripts $n$ and $m$ index the two signals being compared;
both run over all $N$ signals when the full $N \times N$ coherence matrix is
formed, so every expression below is evaluated once per pair. The
segment-summed cross- and auto-spectra are

$$
P_{nm}(f) \;=\; \sum_{k=1}^{K} X_n(k,f)\,\overline{X_m(k,f)},
\qquad
P_{nn}(f) \;=\; \sum_{k=1}^{K} \bigl|X_n(k,f)\bigr|^{2}.
$$

$P_{nn}(f)$ is real and non-negative — it is signal $n$'s **power** at frequency
$f$, and it is what "spectral tilt" refers to throughout this note.

The **magnitude-squared coherence** (MSC) at a single frequency is the
normalized squared cross-spectrum,

$$
\gamma^{2}_{nm}(f)
\;=\;
\frac{\bigl|P_{nm}(f)\bigr|^{2}}{P_{nn}(f)\,P_{mm}(f)}
\;\in\; [0,\,1],
$$

the fraction of signal $n$'s variance at frequency $f$ that is linearly
predictable from signal $m$ at that same frequency. It is $1$ when the two are
related by a fixed gain and phase at $f$ across all $K$ segments, and $0$ when
they are unrelated. Note it is **scale-invariant**: multiplying either signal's
amplitude at $f$ by any constant leaves $\gamma^2_{nm}(f)$ unchanged, because
the constant appears in numerator and denominator alike. That property is why
attenuating a bin does *not* destabilize V1, and it is used again in §7.

By Cauchy–Schwarz, $|P_{nm}(f)|^2 \le P_{nn}(f) P_{mm}(f)$, which is what bounds
$\gamma^2$ above by 1.

Estimators V1–V4 are defined in `coherence_estimators.md` §2. This note concerns
only V2 and V3 (see the header).

---

## 1. The mechanism

V2 is exactly a **weighted mean** of the per-frequency MSC defined above:

$$
C^{(2)}_{nm} \;=\; \frac{\sum_f |P_{nm}(f)|^2}{\sum_f P_{nn}(f)P_{mm}(f)}
\;=\; \frac{\sum_f w_f \, \gamma^2_{nm}(f)}{\sum_f w_f},
\qquad w_f \;=\; P_{nn}(f)\,P_{mm}(f).
$$

The middle step is just multiplying and dividing each numerator term by
$P_{nn}(f)P_{mm}(f)$: $|P_{nm}(f)|^2 = w_f \, \gamma^2_{nm}(f)$ by the definition
of $\gamma^2$ above.

The weight on each frequency is therefore the **product of the two signals'
powers there**. V1 is the same expression with $w_f \equiv 1$ — every bin
contributes its $\gamma^2$ equally, regardless of how much power carries it.

Now suppose both signals have a power-law spectrum, $P(f) \propto f^{-\alpha}$ —
the usual description of calcium and voltage imaging, where $\alpha \approx 1$
for physiological drift and can approach 2 where slow baseline dynamics
dominate. Then

$$
w_f \;\propto\; f^{-2\alpha}.
$$

**The exponent doubles.** A signal spectrum tilted as $1/f$ produces a weighting
tilted as $1/f^2$. Over a band spanning a frequency ratio
$r = f_{\text{hi}}/f_{\text{lo}}$, the lowest bin outweighs the highest by

$$
\frac{w_{\text{lo}}}{w_{\text{hi}}} \;=\; r^{2\alpha}.
$$

For the 10–40 s band, $r = 4$: a factor of **16** at $\alpha = 1$, and **256** at
$\alpha = 2$. The band is nominally two octaves wide; the weighting is not.

## 2. How far it collapses

Measured on the actual bin grid (fs = 1 Hz, `nperseg` = 128, band 10–40 s, which
lands on $F = 9$ bins spanning 0.0312–0.0938 Hz, i.e. periods 10.7–32.0 s):

| $\alpha$ | $w_{\text{lo}}/w_{\text{hi}}$ | $F_{\text{eff}}$ | $F_{\text{eff}}/F$ | weight in lowest bin | weighted-median period |
|---|---|---|---|---|---|
| 0.0 (white) | 1.0 | 9.00 | 1.00 | 11 % | 16.0 s |
| 0.5 | 3.0 | 7.91 | 0.88 | 20 % | 18.3 s |
| **1.0 (pink)** | **9.0** | **5.69** | **0.63** | **31 %** | **25.6 s** |
| 1.5 | 27.0 | 3.92 | 0.44 | 42 % | 25.6 s |
| **2.0 (brown)** | **81.0** | **2.86** | **0.32** | **53 %** | **32.0 s** |

where the **effective bin count** is the participation ratio of the weights,

$$
F_{\text{eff}} \;=\; \frac{\bigl(\sum_f w_f\bigr)^2}{\sum_f w_f^2}
\qquad (F_{\text{eff}} = F \text{ for equal weights, } 1 \text{ if one bin dominates}).
$$

Read the last column. At $\alpha = 2$ the weighted-median period is **32.0 s** —
the slowest bin in the band. Half the estimate's weight sits at or beyond the
band's slow edge. Nine bins are behaving like fewer than three.

For an arbitrary band, the continuous approximation with $\beta = 2\alpha$ on
$f \in [1, r]$ is

$$
\frac{F_{\text{eff}}}{F} \;=\;
\frac{\left(\int_1^r f^{-\beta}\,df\right)^2}{(r-1)\int_1^r f^{-2\beta}\,df},
$$

which lets you check any band and slope without simulating.

## 3. What this does to an actual measurement

The table above describes the weighting. This is the consequence.

Two signals made **coherent only in the upper (faster) half of the band** and
independent in the lower half. The true band-averaged MSC is 0.5 by
construction. 400 trials, K = 10:

| $\alpha$ | V1 reports | V2 reports |
|---|---|---|
| 0.0 | 0.499 | 0.522 |
| 1.0 | 0.502 | **0.283** |
| 2.0 | 0.500 | **0.154** |

V1 recovers 0.50 at every slope. V2 reports **0.15** at $\alpha = 2$ — it very
nearly misses a real, strong, half-band coherence, because the frequencies
carrying that coherence carry little power.

This is the mirror image of the failure recorded in `coherence_estimators.md`
§4, where a pair sharing only its *strongest* band scored 1.0000 under V2
against V1's 0.8202. Same mechanism, opposite sign:

> **V2 does not measure "how coherent are these two signals across the band". It
> measures "how coherent are they at the frequencies where they happen to have
> the most power". Under $1/f$, that means the slow edge.**

Whether that inflates or deflates the answer depends entirely on whether the
shared structure happens to coincide with the high-power frequencies. Both
directions are physically plausible, and neither is visible in the returned
number.

## 4. A second, quieter cost: variance

Averaging $F$ independent per-bin estimates reduces variance by $F$. Weighting
unevenly reduces it only by $F_{\text{eff}}$. Null distribution, 2000
independent pairs, K = 10:

| $\alpha$ | V1 sd | V2 sd | ratio |
|---|---|---|---|
| 0.0 | 0.0305 | 0.0331 | 1.09 |
| 1.0 | 0.0309 | 0.0396 | 1.29 |
| 2.0 | 0.0304 | 0.0555 | **1.82** |

Means agree (both at the $1/K \approx 0.1$ floor — the bias floor is unaffected;
see `coherence_estimators.md` §5). But V2's noise is **1.8× larger** at
$\alpha = 2$. In a clustering context that is a straightforwardly worse
similarity matrix: same signal, more scatter, so more spurious near-threshold
pairs.

## 5. Why it matters beyond a single number

**Cross-dataset comparability breaks.** Two recordings with different spectral
slopes have different effective bands, so their V2 coherence values are not the
same quantity. A fish with more baseline drift is being measured over a narrower,
slower band than one without — and nothing in the output says so.

**It varies *within* a dataset, across voxels.** Shot-noise-dominated voxels
(dim, deep, small) have flatter spectra than bright ones. Since $w_f$ depends on
*both* signals' spectra, different entries of the same coherence matrix are
effectively computed over different bands. The matrix is not one estimator
applied N² times; it is a family of estimators that varies smoothly with local
SNR. Any structure correlated with brightness or depth can therefore appear as
structure in the clustering.

**Narrowing the band helps less than expected.** The collapse depends on
$r^{2\alpha}$, so halving the band width only reduces the weight ratio by
$2^{2\alpha}$ — a factor of 4 at $\alpha = 1$, while also cutting $F$ and so
raising variance.

## 6. Diagnosing it

$F_{\text{eff}}$ is cheap — it needs only the per-voxel band powers already
computed for the denominator — and it is the single number that makes the
collapse visible:

```python
w = P_nn[:, None, :] * P_mm[None, :, :]          # (N, N, F), or per pair
F_eff = w.sum(-1) ** 2 / (w ** 2).sum(-1)         # (N, N)
```

Report `F_eff.median()` and `F_eff.min()` alongside any V2 matrix. Interpretation:

- $F_{\text{eff}}/F > 0.8$ — tilt is negligible, V2 ≈ V1, no action needed.
- $0.4 - 0.8$ — the band is meaningfully narrowed; state the weighted-median
  period in any figure caption rather than the nominal band.
- $< 0.4$ — the nominal band is not what is being measured. Either whiten, or
  redefine the band to what the weights actually select.

The per-voxel spectral slope and the weighted-median period are equally cheap and
worth logging with any stored coherence matrix.

## 7. Mitigations, in order of directness

1. **`whiten=True` (i.e. use V1).** Exact fix, one line, zero cost: dividing each
   $X_n(k,f)$ by $\sqrt{P_{nn}(f)}$ sets every $w_f$ to 1. See
   `coherence_estimators.md` §3 — the identity is exact to `rtol=1e-12`.
2. **Pre-whiten the traces** before the STFT (fit and divide out the $1/f^\alpha$
   trend). Flattens the weighting without changing the estimator, but adds a
   fitting step with its own choices.
3. **Split into log-spaced sub-bands**, compute V2 within each, and combine.
   Within a narrow sub-band $r \approx 1$ so tilt is negligible; combining across
   sub-bands then becomes an explicit, visible weighting decision instead of an
   implicit one.
4. **Narrow the band.** Weakest option — see §5.

## 8. What cmm does

V2 is the default, chosen knowingly (see `coherence_estimators.md` §8): letting
the frequencies with the most shared power dominate is a defensible scientific
position, and it is what the existing `icampsnfr` results assume.

The obligations that come with that choice:

- `whiten=True` must be a **prominent, documented** parameter, not a footnote.
  Flipping to V1 is the one-line escape hatch and users must know it exists.
- Any V2 coherence matrix should carry `F_eff` and the weighted-median period in
  its metadata, so the collapse is recorded rather than silent.
- Figure captions and methods text should state the **weighted-median period**,
  not only the nominal band, whenever $F_{\text{eff}}/F < 0.8$.
### Measured on this dataset: the collapse does **not** occur here

`logs/gate23_eigensolver_real_data.md`, N = 10 000 voxels from
`270705_..._18539/exp3`, fs = 0.596 Hz, band 0.025–0.1 Hz, K = 22, F = 15:

- **Per-bin mean power varies by only 1.38x across the band** — effectively flat,
  nothing like the $\alpha = 1$–$2$ tilts simulated in §2.
- **$F_{\text{eff}} = 14.1$ of 15 bins** ($F_{\text{eff}}/F = 0.94$), comfortably
  in the "tilt is negligible, V2 ≈ V1" regime of §6.

The reason is upstream: the pipeline's ~120 s high-pass detrend has already
removed the low-frequency drift that produces the $1/f$ tilt. **The band collapse
described in this note is real, but it is not currently biting on this data**,
because the preprocessing happens to defuse it.

Two consequences:

1. This note is a **guard against a regression**, not a description of a present
   problem. If the high-pass is weakened, the band widened toward DC, or the
   estimator applied to un-detrended traces, the collapse returns. The
   $F_{\text{eff}}$ diagnostic in §6 is what would catch that, which is the
   argument for computing it routinely rather than once.
2. The V2-over-V1 justification is now **inert in both directions** on this data:
   V2's tilt blind spot does not occur (measured above), and V1's dead-bin blind
   spot does not occur either (zero cells below even 1e-2 of their voxel median;
   minimum ratio 0.259). Neither documented argument discriminates.

**Yet the two estimators still disagree materially** on this dataset — top-1 %
pair Jaccard 0.40, top-100-neighbour overlap 0.54, cluster **ARI 0.64 at k = 7
and 0.33 at k = 15**, unchanged by matching `tau`. So the choice changes the
science while neither stated reason for it applies. See
`coherence_estimators.md` §8 for the standing decision (V2, knowingly) and
`logs/gate23_eigensolver_real_data.md` for the proposed discriminator that needs
no ground truth: label stability under segment and voxel resampling.

---

### Reproducing

The tables in §2, §3 and §4 were generated by a script that is **not yet checked
in** — see `coherence_estimators.md` §0 for the same caveat. It belongs in
`derivations/scripts/` keyed to each table before these numbers are cited
anywhere outside this repo.
