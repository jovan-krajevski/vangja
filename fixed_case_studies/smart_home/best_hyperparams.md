# Smart home — Bayesian hyperparameter selection: chosen values and their defence

Status: **frozen into `smart_home/config.py`** (deliberate freeze step of
HYPERPARAMETER_SELECTION.md §5; the selection scripts never edit `config.py`).
All evidence was computed on **training windows only** (the 91-day training
blocks of the `2016-04-01` and `2016-07-01` origins); the 260-day test
horizon was **never** used at any step (audit F-1).

Provenance of the evidence: `results_hyperparams/*_selection.md`,
`*_ts_cv.csv`, `*_averaging.md`, `*_robustness.md`,
`full_bayes_Fridge_[kW].md`, `smart_home_recommendations.json`
(commit `d7937e1`, vangja 0.2.4 / pymc 6.2.0 / arviz 1.3.0 / numpy 2.4.6).

---

## 1. Summary

| hyperparameter | a-priori frozen | selected | evidence basis | action |
|---|---|---|---|---|
| `intercept_sd` (FlatTrend level) | 0.5 | **0.1** | PSIS-LOO (ADVI + NUTS) > 1 dse; leave-future-out CRPS monotone | change |
| `beta_sd` (FourierSeasonality) | 1.5 | **0.25** | PSIS-LOO (ADVI + NUTS) > 1 dse; CRPS tie → parsimony; full-Bayes supports small | change (PPC caveat, §5) |
| `yearly_order` | 5 | **5** | NUTS verification: 3 vs 5 indistinguishable (Δelpd 0.04 ≪ dse 1.6); **transfer-dimension constraint** (§4.3) | keep |
| `weekly_order` | 3 | **2** | NUTS verification: 2 beats 3 by ~3 dse; CRPS; parsimony | change |

The selected model configuration is therefore:

```
FlatTrend(intercept_mean=0.5, intercept_sd=0.1, pool_type="individual")
  + UniformConstant(-1, 1) * FourierSeasonality(365.25, series_order=5, beta_sd=0.25,
                                                partial, shrinkage 1)
  + FourierSeasonality(7, series_order=2, beta_sd=0.25, partial, shrinkage 1)
```

## 2. Workflow recap (what the numbers mean)

Selection procedure (HYPERPARAMETER_SELECTION.md §2), all on training
windows:

1. **Prior-predictive calibration** of the sd's: fraction of prior-predictive
   draws inside the scaled `[-2, 2]` window. Target 30–60 % (flexible prior);
   < 5 % too wide, > 95 % too tight.
2. **Coordinate-wise PSIS-LOO** comparison of a small pre-registered grid,
   pointwise elpd pooled over the two training windows. Screening fits use
   ADVI; the **top-2 candidates are re-fitted with small NUTS runs**
   (`verify_top_k`) because ADVI is approximate (mean-field) and can rank
   candidates incorrectly.
3. **Conservative selection rule** (§2.2): keep the frozen value unless the
   best candidate beats it by more than one difference-standard-error (dse);
   among statistically indistinguishable candidates prefer the **smaller**
   value (parsimony / sampling speed).
4. **Cross-checks, reported but not vetoing** (§3): leave-future-out
   expanding-window CRPS (`ts_cv`), pseudo-BMA/stacking weights
   (`stacking`), ±1-step robustness sweeps (`robustness`), and a
   full-Bayes hyperprior demonstration (`full_bayes`) on one appliance.

## 3. `intercept_sd` (FlatTrend level): 0.5 → 0.1

Evidence (transfer arm; grid `[0.1, 0.25, 0.5, 1.0]`):

- **PSIS-LOO screening (ADVI):** 0.1 is best (elpd −5790.5); 1.0 −6078.3
  (Δ −287.8, dse 68.9); 0.5 −6431.2; 0.25 −7420.0. (ADVI ranking is noisy —
  0.25 is worst — which is exactly why the top-2 were re-checked with NUTS.)
- **NUTS verification (top-2: 0.1 vs 1.0):** 0.1 elpd 126.31 vs 1.0 123.26;
  Δelpd 3.05, dse 2.18 → 0.1 wins by ~1.4 dse (> 1 dse threshold).
- **Leave-future-out CRPS:** 0.1 0.841 < 0.25 0.923 < 0.5 0.979 < 1.0 1.048
  — monotone; 0.1 best.
- **Prior-predictive coverage:** 0.330–0.339 for *all* candidates (all
  "target"). Non-discriminating, as expected: FlatTrend is a constant level
  and coverage is dominated by the seasonal components. PPC is not the
  evidence for this parameter.
- **Model averaging:** 0.1 has the highest pseudo-BMA (0.39) and stacking
  (0.59) weight.
- **Robustness:** neighbourhood `[0.05, 0.1, 0.15]` flagged *sensitive* —
  elpd still improves below the grid (0.05: −5613.4). 0.1 is the smallest
  pre-registered value; this is a boundary disclosure, not a veto (§5).

**Defence.** The level of each appliance is estimated from 91 days; a mildly
informative prior `N(0.5, 0.1)` (the centre of the minmax-scaled `[0,1]`
range) regularises that estimate without constraining it. Every criterion
that can rank this parameter — PSIS-LOO on both ADVI and the gold-standard
NUTS fits, and leave-future-out CRPS — selects the smallest grid value 0.1,
and 0.1 beats the frozen 0.5 (and 1.0) by more than one dse. The full-Bayes
posterior (mean 0.68, 90 % ETI 0.17–1.68) is wide, so it is not the
discriminating evidence; LOO is.

## 4. `beta_sd` and the Fourier orders

### 4.1 `beta_sd` (seasonal coefficient prior sd): 1.5 → 0.25

Evidence (transfer arm; grid `[0.25, 0.5, 1.0, 1.5]`):

- **PSIS-LOO screening (ADVI):** 0.25 is best by a wide margin
  (elpd −5269.7); 0.5 −5721.3 (Δ −451.7, dse 66.4); 1.5 −6011.7;
  1.0 −6012.5. 0.25 wins by ~6.8 dse.
- **NUTS verification (top-2: 0.25 vs 0.5):** 0.25 elpd 124.90 vs 0.5 123.16;
  Δelpd 1.75, dse 1.08 → 0.25 wins by ~1.6 dse (> 1 dse).
- **Leave-future-out CRPS:** 0.25 (0.9868) and 1.5 (0.9865) are
  indistinguishable (Δ 0.0003); 0.25 has the **lowest MAE** (1.166 vs 1.172).
  The conservative tie-break then prefers the smaller value.
- **Prior-predictive coverage:** 0.25 → 0.906 ("too tight"), 0.5 → 0.737
  ("too tight"), 1.0 → 0.475 ("target"), 1.5 → 0.336 ("target"). This is the
  one dissenting check; see §5.
- **Full-Bayes hyperprior** (single appliance, HalfNormal(1) hyperpriors):
  `beta_sd_yearly` posterior mean 0.026 (90 % ETI 0.005–0.075),
  `beta_sd_weekly` mean 0.049 (0.012–0.117) — the data-supported scales are
  far below the Prophet default (10) and below the frozen 1.5.
- **Model averaging:** 0.25 highest (pseudo-BMA 0.37, stacking 0.66).

**Defence.** The primary criterion (PSIS-LOO) selects 0.25 on both the ADVI
screening and the NUTS verification, by more than one dse. Leave-future-out
CRPS cannot distinguish 0.25 from 1.5 and the selection rule then prefers
the smaller value (parsimony, and the lowest MAE). The full-Bayes
hyperprior posterior (≈ 0.03–0.05) shows the data actually support *even
stronger* shrinkage — 0.25 is a compromise between LOO and the PPC
heuristic, and an order of magnitude closer to the data-supported scale than
the Prophet default of 10. The old frozen 1.5 (and the Prophet default 10)
are weakly informative for a 91-day window and are the kind of diffuse prior
that the original study's overfitting was traced to.

### 4.2 `weekly_order`: 3 → 2

Evidence (no-transfer arm — weekly is not transferred; grid `[2, 3, 5]`):

- **PSIS-LOO screening (ADVI):** 3 best (elpd −8830.6), 2 second (−8937.9,
  Δ −107.4, dse 124.1) → **indistinguishable in ADVI**; 5 much worse
  (−10951.2).
- **NUTS verification (top-2: 2 vs 3):** 2 elpd 129.12 vs 3 125.10;
  Δelpd 4.01, dse 1.33 → 2 wins by ~3 dse. The ADVI tie is resolved by the
  gold-standard NUTS check.
- **Leave-future-out CRPS:** 2 (0.590) < 5 (0.604) < 3 (0.718) — 2 best.
- **Parsimony:** 2 uses 98 free parameters vs 110 for 3 (faster sampling,
  fewer coefficients to overfit on 91 days).
- **Robustness:** order 1 (outside the grid) scores even better in ADVI —
  boundary disclosure (§5).

**Defence.** Weekly seasonality is household-specific and is deliberately
**not** transferred (the temperature context has no weekly component), so
the order is a free choice. Both the NUTS verification (> 3 dse) and
leave-future-out CRPS prefer 2; ADVI alone cannot distinguish 2 from 3,
which is precisely why the workflow re-checks with NUTS. The smaller order
is also the parsimonious choice.

### 4.3 `yearly_order`: 5 kept (no change)

Evidence (no-transfer arm — order selection under transfer is invalid,
§below; grid `[3, 5, 8]`):

- **PSIS-LOO screening (ADVI):** 3 best (elpd −8315.3), 5 −8870.3
  (Δ −555.0, dse 102.3), 8 −9043.7.
- **NUTS verification (top-2: 3 vs 5):** 3 elpd 125.14 vs 5 125.10;
  Δelpd 0.035, dse 1.63 → **statistically indistinguishable**. The
  conservative rule therefore keeps the frozen value (no evidence to change).
- **Leave-future-out CRPS:** 3 (0.733) slightly better than 5 (0.763);
  8 much worse (0.940). Reported as a cross-check flag, not a veto.

**Decisive argument (transfer-dimension constraint).** The yearly
coefficients are transferred from the Boston temperature posterior with
`prior_from_idata`, and the temperature context model is fitted with
`series_order=5`. The transferred prior is a joint distribution over the
source's `2 × 5` yearly coefficients, so its dimension is **tied to the
source order**. The target's yearly order must equal the source order (5)
for the transfer to be dimensionally consistent; changing it would require
re-fitting the frozen context model. This is why the protocol runs order
selection on the **no-transfer arm** as a diagnostic only
(HYPERPARAMETER_SELECTION.md §2.4). Under the transfer design, 5 is the only
admissible value, and the NUTS verification confirms there is **no
statistical cost** to it (Δelpd 0.035 ≪ dse 1.63 versus the LOO-favoured 3).

## 5. Honest caveats (report these in the paper)

1. **PPC tension for `beta_sd=0.25`.** The prior-predictive coverage is
   0.91 ("too tight" relative to the 30–60 % flexibility target). This is
   the expected behaviour of a deliberately regularised prior: the model no
   longer treats seasonal swings of hundreds of percent as plausible. The
   30–60 % target was calibrated for *flexible* priors; for a short,
   transfer-regularised series, high coverage is the intended outcome. The
   dissenting check is disclosed rather than hidden, and the primary
   criterion (LOO) and the full-Bayes posterior both support the small value.
2. **Grid-boundary flags.** For `intercept_sd` and both orders, the
   ±1-step robustness sweep keeps improving at the *edge* of the
   pre-registered grid (0.05 for the level; order 6 yearly; order 1
   weekly). The pre-registered grids are frozen (§4 of candidates.py), so
   the recommended values are the grid-optimal choices; the boundary
   behaviour is reported as a transparency note, not used to extend the
   grid after the fact.
3. **ADVI vs NUTS disagreements.** ADVI screening and NUTS verification
   disagree on `weekly_order` (ADVI: 3 ≈ 2; NUTS: 2 > 3 by 3 dse) and
   `yearly_order` (ADVI: 3 > 5; NUTS: 3 ≈ 5). This is expected of a
   mean-field approximation; the workflow is designed so the final decision
   rests on the NUTS fits (and the conservative rule for `yearly_order`).
4. **CRPS cross-checks.** Leave-future-out CRPS prefers `beta_sd=1.5`
   (marginally, Δ 0.0003) and `yearly_order=3` (Δ 0.03) over the LOO picks.
   Both differences are far smaller than the CRPS differences that *do*
   matter (e.g. `weekly_order`: 0.59 vs 0.72) and are reported in the
   `*_selection.md` files as flags. The selection rule's primary criterion
   is LOO; CRPS is a cross-check.
5. **Full-Bayes is a single-appliance demonstration.** The hyperprior
   posterior (β_sd_yearly ≈ 0.03, β_sd_weekly ≈ 0.05) is from one appliance
   with HalfNormal(1) hyperpriors and no transfer; it is corroborating
   evidence for strong shrinkage, not a competing selection.

## 6. What was frozen into `config.py`

| field | value | where used |
|---|---|---|
| `intercept_sd` | 0.1 | `FlatTrend` (all configs) |
| `beta_sd` | 0.25 | yearly + weekly `FourierSeasonality` (all configs) |
| `yearly_order` | 5 | yearly `FourierSeasonality` (matches the temperature context model) |
| `weekly_order` | 2 | weekly `FourierSeasonality` |

These values are shared by **all** registry configs (`main` and every
ablation) so each ablation remains a single-perturbation experiment; only
the ablated feature differs (transfer, pooling, uniform constant,
regularisation, shrinkage — see `config.py`).

## 7. Notes for the paper section

- Method paragraph: Bayesian workflow on training windows only;
  prior-predictive calibration; coordinate-wise PSIS-LOO with ADVI
  screening and small-NUTS verification of the top-2; conservative
  selection rule (keep frozen unless beaten by > 1 dse; prefer smaller
  among indistinguishable); cross-checks (leave-future-out CRPS, model
  averaging, robustness, full-Bayes hyperpriors). No test-horizon data were
  used at any step.
- Result: the Prophet-inherited defaults were too diffuse for 91-day
  windows; the data select strong shrinkage (`beta_sd` 10 → 0.25,
  `intercept_sd` 5 → 0.1 in Prophet terms) and a parsimonious weekly order
  (2), while the yearly order is fixed at 5 by the transfer design (the
  temperature context model's order) with no measurable LOO cost.
- Disclose the caveats in §5 verbatim where relevant (especially the PPC
  tension for `beta_sd` and the grid-boundary behaviour).
- Emphasise the audit link: this procedure replaces the original
  2,592-configuration test-set grid (F-1) with a pre-registered,
  training-window-only workflow whose output is frozen deliberately.
