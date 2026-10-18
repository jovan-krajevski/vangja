---
name: forecasting-evaluation
description: Compute or report Vangja forecast metrics correctly. Use when scoring forecasts, aggregating Relative MAE, comparing arms on paired units, bootstrapping confidence intervals, reporting CRPS/coverage/sharpness or failure rates, or preparing any accuracy table. Contains the project's metric definitions, the epsilon exclusion rule, and the two-way block bootstrap.
---

# Forecasting evaluation

House rules: `AGENTS.md` § Non-negotiables (metric). The primary metric is
**Relative MAE against persistence**, aggregated by
**median** over target × origin units. Everything else is secondary. The full
definitions are in `references/metrics.md`; reference implementations are
`scripts/relative_mae.py` and `scripts/block_bootstrap.py`, and the canonical
project functions are in `src/vangja/utils.py` and
`fixed_case_studies/common.py`.

## When to use

- Any table, figure, or sentence containing MAE/RMSE/MAPE/Relative MAE/CRPS.
- Comparing arms (transfer vs no-transfer, model vs baseline).
- Deciding whether a unit counts (denominator rule).

## Rules

1. **Unit** = target series × forecast origin. Never pool raw errors across
   horizons.
2. **Denominator** = persistence (last training value carried forward,
   `src/vangja/utils.py:383`), computed in the **same scaled space** as the
   model fit. Baselines must use the matching scale mode
   (`fixed_case_studies/common.py:baseline_unit_metrics`, `common.py:254`):
   `maxabs_complete` for stocks, `minmax_individual` for smart home.
3. **ε rule.** If scaled persistence MAE `< 1e-3`
   (`REL_MAE_EPSILON`, `common.py:44`), the unit is excluded from the
   aggregate, **counted**, and its absolute MAE reported separately. Never clip
   the denominator; `relative_mae` returns `nan` in that case
   (`src/vangja/utils.py:325`).
4. **Aggregate = median** across units; also report mean, IQR, `prop_below_1`,
   the distribution, `n_units`, `n_excluded` (`common.aggregate_rel_mae`,
   `common.py:348`).
5. **Seeds.** Average each model's per-unit MAE across seeds before forming the
   ratio (`PROTOCOL.md` §4).
6. **Failures.** Report the failure rate next to every accuracy table; a
   success-only table is not comparable (`record_failure`, `common.py:558`).
   The same applies to unavailable series (`availability`, `runner.py:89`).
7. **Paired comparisons.** Compare on the intersection of units; report the
   negative-transfer rate (`paired_transfer_comparison`, `common.py:435`).
8. **Uncertainty.** CRPS (`fixed_case_studies/hyperparams/scoring.py:201`),
   empirical coverage and width per horizon/target, and sharpness. MAP
   residual intervals are a heuristic (WS-S-010) and must be labelled.
9. **Secondary metrics.** MAE, RMSE, MASE/RMSSE, safeguarded sMAPE/MAPE
   (`common.safeguarded_mape`, `common.py:167`). Raw MAPE is never primary: it
   explodes near zero (FINDINGS F-13).

## Block bootstrap (dependence-aware)

Origins overlap, stocks share market shocks. Resample **series** and
**origin blocks** with replacement; recompute the median paired difference;
report percentile CIs (`common.two_way_block_bootstrap`, `common.py:400`).
Half-year blocks by default (`add_origin_block`, `common.py:374`); origins
without dates become their own blocks. No pooled Diebold–Mariano, no
stock-only bootstrap.

```bash
python .pi/skills/forecasting-evaluation/scripts/relative_mae.py \
  --train train.csv --test test.csv --pred pred.csv --epsilon 1e-3 --out results/relmae
python .pi/skills/forecasting-evaluation/scripts/block_bootstrap.py \
  --paired paired.csv --diff-col diff_rel_mae --n-iter 2000 --seed 0 --out results/boot
```

Both write only into `--out` and print the documented JSON shape.

## Reporting template

```
| Arm | median Rel-MAE | mean | IQR | prop<1 | units | excluded | failed | failure rate |
| main | 0.91 | ... | ... | ... | 366 | 4 | 2 | 0.5 % |
```

Every number names the artifact (units CSV + manifest) and the config/seed.

## Do not

- Drop excluded units or failed fits before aggregating.
- Compute relative MAE in original units while the denominator is scaled.
- Compare arms with different exclusions; exclusion is per-unit, established
  before looking at model differences.
- Report a bootstrap CI over pooled errors.
- Crop the horizon, or silently change the horizon between arms.

See `scientific-integrity-and-anti-gaming` before publishing any table.
