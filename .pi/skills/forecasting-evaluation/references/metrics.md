# Metrics reference

## Primary: Relative MAE vs persistence

```
RelMAE(u) = MAE(model, u) / MAE(persistence, u)
persistence(u): ŷ_{t+h} = y_{t_last_train}
```

- Unit `u` = target × origin.
- `< 1` beats persistence, `= 1` ties, `> 1` worse.
- Aggregate (selection = reporting): **median** across units; also mean, Q1,
  Q3, proportion below 1, full distribution.
- Exclusions: scaled persistence MAE `< REL_MAE_EPSILON = 1e-3`
  (`fixed_case_studies/common.py:44`). Excluded units are counted and their
  absolute MAE reported separately.

Implementations:

| Where | What |
|---|---|
| `src/vangja/utils.py:325` | `relative_mae(y_true, y_pred, y_persistence, epsilon)` → `nan` when excluded |
| `src/vangja/utils.py:383` | `persistence_forecast(train_df, test_df)` → long `ds, series, yhat` |
| `fixed_case_studies/common.py:185` | `unit_metrics` — per-series metrics from a fitted model |
| `fixed_case_studies/common.py:254` | `baseline_unit_metrics` — same schema for model-free baselines |
| `fixed_case_studies/common.py:348` | `aggregate_rel_mae` — median/mean/IQR/prop<1/counts |
| `.pi/skills/forecasting-evaluation/scripts/relative_mae.py` | stdlib reference CLI (long-format CSVs) |

### Scale modes

| Study | Fit scaler | Baseline `scale_mode` |
|---|---|---|
| Stocks | `maxabs`, `complete` | `maxabs_complete` |
| Smart home | `minmax`, `individual` | `minmax_individual` |

A baseline scored in a different scale space is not comparable; the ε
denominator is defined on the scaled target by construction.

### Seed aggregation

For repeated seeds, average each model's per-unit MAE over seeds, then form the
ratio. Do not compute one ratio per seed and average ratios.

## Secondary metrics

| Metric | Where | Note |
|---|---|---|
| MAE, RMSE | `unit_metrics` output (`mae`, `rmse`) | original units |
| MASE / RMSSE | planned secondary | seasonal-naive denominator, report with seasonality choice |
| sMAPE / MAPE | `fixed_case_studies/common.py:167` | safeguarded; continuity only, never for selection |
| CRPS | `fixed_case_studies/hyperparams/scoring.py:201` | proper scoring rule on predictive ensembles |
| coverage / width / sharpness | calibration stage | per horizon and per target; MAP intervals are heuristic |
| negative-transfer rate | `common.py:435` | fraction of paired units where transfer is worse |

## Paired comparison

- Restrict both arms to the same unit set (inner join on `series × origin`).
- `diff = rel_mae_transfer − rel_mae_no_transfer`; negative favours transfer.
- Two-way block bootstrap (`common.py:400`): resample series and origin blocks
  with replacement, recompute the median difference per iteration, report the
  2.5/97.5 percentiles and `prop_diff_lt_0`.

```bash
python scripts/block_bootstrap.py --paired paired.csv \
  --series-col series --block-col origin_block \
  --diff-col diff_rel_mae --n-iter 2000 --seed 0
```

Output:

```json
{
  "n_paired_units": 366,
  "n_iter": 2000,
  "sample_median": -0.013,
  "median_diff": -0.012,
  "ci_2p5": -0.041,
  "ci_97p5": 0.017,
  "prop_diff_lt_0": 0.78,
  "negative_transfer_rate": 0.22
}
```

## Reporting checklist

- [ ] Same units for every arm; exclusions fixed before arm comparison.
- [ ] `n_units`, `n_excluded`, failed fits, unavailable series reported.
- [ ] Median, distribution, and at least one dependence-aware CI.
- [ ] Multiplicity: number of comparisons disclosed.
- [ ] Every number traceable to a units CSV + manifest.
