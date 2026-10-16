# Vangja datasets — loaders, scaling, and the traps

`src/vangja/datasets/**` is the only data-access layer (`AGENTS.md` § Map).
Nothing in `fixed_case_studies/**` may download or parse data inline; if a
loader is missing, add it here with a test. Hashing, revisions, and sidecars
are owned by the `data-provenance` skill.

## 1. Loader inventory (`src/vangja/datasets/loaders.py`)

| Loader | Returns | Provenance / units | Notes |
|---|---|---|---|
| `load_air_passengers()` `:17` | monthly `ds`,`y` 1949–1960 | Box–Jenkins | classic demo |
| `load_peyton_manning()` `:57` | daily page views 2007–2016 | Prophets docs | has an intentional gap subrange |
| `load_citi_bike_sales()` `:96` | daily rides 2013–2014 | UCI | transfer demo target |
| `load_nyc_temperature()` `:164` | daily series 2012–2017 | Tim Radtke `temperature.csv` | **docstring defect**: the body resamples `mean()` and does **no unit conversion**; the source is Kelvin, but the docstring promises "daily maximum … in Fahrenheit". Trust the body, not the docstring (audit WS-H-008). Fix the docstring deliberately; do not "fix" values by converting silently. |
| `load_kaggle_temperature(city, start_date, end_date, freq)` `:452` | hourly→`freq` series | Kaggle `historical-hourly-weather-data` | converts Kelvin → **Celsius**; needs `kagglehub` |
| `load_smart_home_readings(column, start_date, end_date, freq)` `:554` | one column (long format for a list) | Kaggle `HomeC.csv`, 2016, 1-min → kW | smart-home case study |
| `load_stock_data(tickers, split_date, window_size, horizon_size, cache_path=None, interpolate=False)` `:260` | `(train_df, test_df)` | Yahoo Finance via `yfinance` | trading days; typical price `(O+H+L+C)/4` |
| `get_sp500_tickers_at_date(date, cache_path=None)` `stocks.py:490` | sorted constituents | Wikipedia changes table (accurate ~1997+) | membership fixed at the origin; not conditioned on survival |
| `get_sp500_tickers_for_range(start, end, cache_path=None)` `stocks.py:521` | tickers present the whole range | same | excludes any ticker removed then re-added |

Synthetic: `generate_multi_store_data(seed=42)` and
`generate_hierarchical_products()` (`src/vangja/datasets/synthetic.py:34`,
`:162`) — pass the seed explicitly in tests.

## 2. Stocks: trading days only

- `interpolate=False` is the default and the study default. It fills nothing;
  rows remain trading days.
- If `interpolate=True`, interpolation is **split-first and within-split only**
  (`loaders.py:219 _interpolate_gaps_within_split`): no training value can
  depend on a test value, and no synthetic calendar row is created outside a
  series' own observed range. An interpolation that crosses the split is a
  data-leakage finding (audit F-9).
- A stock's 91-calendar-day training window has ~63 trading days;
  `MIN_TRAIN_OBS = 50` (`fixed_case_studies/stocks/config.py`) is the
  availability rule. Dropped series must be recorded, never silently removed.
- The context series is `^GSPC` (`CONTEXT_TICKER`) and is **never a target**;
  the negative control is `GC=F` (`fixed_case_studies/stocks/config.py`).

## 3. Scaling is part of the metric definition

`fit(..., scaler="maxabs"|"minmax", scale_mode="complete"|"individual")`
(`src/vangja/time_series.py:285`, `types.py`):

- `scale_mode="complete"`: one `(y_min, y_max)` for all rows; the stored
  `y_scale_params` has no per-series key, so downstream code detects
  individual scaling with `"scaler" not in self.y_scale_params`
  (`time_series.py:687`).
- `scale_mode="individual"`: per-series params; recover them with
  `fixed_case_studies/common.py:138 per_series_scale_params`, and un-scale
  with `common.py:158 scale_y`.

**Relative MAE must be computed on the same scaled target that was fitted.**
The ε-exclusion rule (`REL_MAE_EPSILON = 1e-3`,
`fixed_case_studies/common.py:44`) is defined on that scaled target, not on
prices. Mixing scales across arms is a metric finding; the `forecasting-evaluation`
skill owns the aggregate.

## 4. Provenance bookkeeping

- `record_sidecars(data_dir)` (`fixed_case_studies/common.py:121`) writes
  `data_sha256.json` next to the data. Data is never re-downloaded at run
  time; the fetch scripts (`01_fetch_data.py`) are the only network step.
- Availability/exclusions are manifest fields, not log lines
  (`record_artifact` `common.py:507`, `record_failure` `common.py:558`).
- Need network? Fetch once, hash, commit the hash — never the data dump
  (`.gitignore` covers `fixed_case_studies/**/data/` and `results*/`).
