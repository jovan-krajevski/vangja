# Datasets reference

## S&P 500 universe (stocks)

| Property | Value | Source |
|---|---|---|
| Frozen revision | `oldid=1306326561` (last revision before 2025-08-25 with both the constituents and the historical-changes table) | `src/vangja/datasets/stocks.py:23` |
| Fetch | `_fetch_sp500_wiki_tables(cache_path)` | `stocks.py:345` |
| Membership at a date | `get_sp500_tickers_at_date(date, cache_path)` | `stocks.py:490` |
| Range membership | `get_sp500_tickers_for_range(...)` | `stocks.py:521` |
| Cache | `fixed_case_studies/stocks/data/sp500_constituents/` | `fixed_case_studies/stocks/runner.py:35` |
| Universe rule | `members_at_origin`; `date_added` recorded where known | `fixed_case_studies/stocks/01_fetch_data.py` |
| Freeze artifact | `universe_<origin>.csv`, `availability.csv`, `data_sha256.json` | same |

- Accuracy of the historical changes table is ~1997 onward
  (`src/vangja/datasets/stocks.py:497`).
- The index mechanically contains the target stocks (FINDINGS F-10): this is
  overlap, not leakage; disclose it.
- Survivorship: universe membership is reconstructed; delisted/disappeared
  series are availability outcomes. No synthetic injection.

## Stock prices

| Property | Value |
|---|---|
| Loader | `load_stock_data(tickers, split_date, window_size, horizon_size, cache_path, interpolate=False)` (`src/vangja/datasets/loaders.py:260`) |
| Downloader | `_download_stock_data(tickers, cache_path)` (`src/vangja/datasets/stocks.py:83`) |
| Target variable | typical price `(O+H+L+C)/4` (PROTOCOL §2.3) |
| Time index | trading days only; no calendar interpolation |
| Windows | stocks train 91 d, horizon 365 d, context 1460 d (`stocks/config.py:32-34`) |
| Min observations | `MIN_TRAIN_OBS = 50` (`stocks/config.py:46`) |
| Context | `^GSPC`; negative control `GC=F` (`stocks/config.py:35-36`) |

## Smart home (appliances + temperature)

| Property | Value |
|---|---|
| Loaders | `load_smart_home_readings(column, freq="D")`, `load_kaggle_temperature(city, start_date, end_date, freq)` |
| Targets | `Furnace 1/2`, `Fridge`, `Wine cellar` [kW] (`smart_home/config.py:28-32`) |
| Temperature city | Boston; context from 2013-01-01 |
| Split | train 91 d, horizon 260 d; primary cutoff 2016-04-01; 2 rolling origins (`smart_home/config.py:39-45`) |
| Leakage guard | temperature frozen per training cutoff |
| Freeze | `fixed_case_studies/smart_home/data/` + `data_sha256.json` |

Only 4 target series: per-series effects and descriptive uncertainty, no
population-level inference (PROTOCOL §9).

## Sidecar formats

### `data_sha256.json` (project format)

```json
{
  "created_at": "2026-08-28T19:39:10.983709",
  "files": {
    "smart_home_daily.csv": "8cfd2a6106a5dc26ddf97a7aa116db4dabb9c799409132f80c9d30e432e74abd"
  }
}
```

Written by `common.record_sidecars` (`fixed_case_studies/common.py:121`) as
`data_dir.parent / "data_sha256.json"`. Paths are relative to the data dir.

### `sha256sum` format

```
8cfd2a6106a5dc26ddf97a7aa116db4dabb9c799409132f80c9d30e432e74abd  smart_home_daily.csv
```

`check_sidecars.py` accepts both; a bare hash line is also accepted (a single
file next to its sidecar, e.g. `CONFIRMATION_ORIGINS.sha256`).

## Licensing / redistribution

- Yahoo Finance via `yfinance`: provider terms apply; do not redistribute raw
  price data; results and hashes are fine.
- Wikipedia: CC BY-SA — attribute "List of S&P 500 companies", revision
  `1306326561`, and the retrieval route.
- Kaggle smart-home/temperature: dataset terms; keep raw files out of the
  repository.

## Verification recipe

```bash
python .pi/skills/data-provenance/scripts/check_sidecars.py \
  --root fixed_case_studies/smart_home/data \
  --sidecar fixed_case_studies/smart_home/data_sha256.json
```

Then confirm the same hash map appears in the run manifest and that the
manifest validator (`reproducible-experiments/scripts/check_manifest.py`)
recomputes it.
