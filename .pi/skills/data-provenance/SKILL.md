---
name: data-provenance
description: Fetch, freeze, hash, or verify the datasets behind a Vangja result. Use when downloading data, choosing a data revision, checking SHA-256 sidecars, reconstructing the S&P 500 universe, handling missing or delisted series, or when a result's data can no longer be identified. Contains the frozen Wikipedia revision, the availability accounting rules, and cache layout.
---

# Data provenance

House rules: `AGENTS.md` § Non-negotiables (data). A result without an
identifiable dataset revision is not reproducible. This
skill records where each Vangja dataset comes from, how it is frozen, and how
its absence is accounted for. Dataset details are in
`references/datasets.md`.

## When to use

- Fetching or refreshing any dataset (`fixed_case_studies/*/01_fetch_data.py`).
- Verifying sidecars before a run or after moving data.
- Reconstructing the S&P 500 universe or the smart-home context.
- Reporting missing, delisted, or unavailable series.

## Rules

1. **Frozen revisions, not live pages.** The S&P 500 universe is rebuilt from
   the pinned Wikipedia revision `oldid=1306326561`
   (`src/vangja/datasets/stocks.py:23`). The live page lost the historical
   changes table; a frozen revision also makes the universe reproducible.
2. **Hash everything.** `common.record_sidecars` (`fixed_case_studies/common.py:121`)
   writes `data_sha256.json` next to the data directory:
   `{"files": {"<relpath>": "<sha256>"}, "created_at": "..."}`.
   Verify with `scripts/check_sidecars.py` before trusting a run.
3. **No network at run time.** Fetching happens once in `01_fetch_data.py`;
   runs read the local cache and the manifest carries the hashes.
4. **Trading days only.** Stocks: `load_stock_data(..., interpolate=False)` is
   the default; never create weekend/holiday rows. Split first; interpolate only
   within a split, only if explicitly justified.
5. **No silent drops.** Missing/delisted series go to `availability.csv`
   (stocks) or the manifest `availability` block; failed downloads are
   recorded. `MIN_TRAIN_OBS = 50` is the only training-size floor
   (`fixed_case_studies/stocks/config.py:46`).
6. **Survivorship is disclosed.** Universe = `members_at_origin`; the same
   universe rule applies to all arms. Synthetic injection is not used. If
   historical membership cannot be reconstructed, say so as a limitation
   (absolute-optimistic, relative-more-robust).
7. **Context is not a target.** `^GSPC` (context) and `GC=F`
   (negative control) are fetched and modelled but excluded from all target
   aggregates (`fixed_case_studies/stocks/config.py:35-36`).
8. **No redistribution.** Raw data dumps stay out of git (`.gitignore` covers
   `data/` and `fixed_case_studies/**/data/`); commit manifests and hashes, not
   data. Respect the terms of Yahoo Finance/yfinance, Wikipedia (CC BY-SA,
   attribute the revision) and Kaggle.

## Fetch cycle

```bash
python fixed_case_studies/stocks/01_fetch_data.py       # network once
python fixed_case_studies/smart_home/01_fetch_data.py    # network once
```

Both write per-origin universes/availability, the cached series, and
`data_sha256.json`. Hashing helpers:

```bash
python .pi/skills/data-provenance/scripts/hash_tree.py \
  --root fixed_case_studies/stocks/data/tickers --out results/tickers_sha256.json
python .pi/skills/data-provenance/scripts/check_sidecars.py \
  --root fixed_case_studies/stocks/data/tickers --json
```

`hash_tree.py` writes a hash manifest (or `--sidecar FILE` for sha256sum
format); `check_sidecars.py` auto-discovers `data_sha256.json` and `*.sha256`
next to the root and fails on missing files, mismatches, and (unless
`--allow-extra`) unrecorded files.

## Cache layout

| Study | Path | Contents |
|---|---|---|
| Stocks tickers | `fixed_case_studies/stocks/data/tickers/` | per-ticker OHLCV CSVs |
| Stocks constituents | `fixed_case_studies/stocks/data/sp500_constituents/` | cached Wikipedia tables |
| Stocks universes | `fixed_case_studies/stocks/data/universe/universe_<origin>.csv` | `ticker,origin,universe_rule,date_added` |
| Stocks availability | `fixed_case_studies/stocks/data/availability.csv` | per-origin requested/available/missing |
| Smart home | `fixed_case_studies/smart_home/data/` | `smart_home_daily.csv`, `boston_temp_until_<cutoff>.csv` |

The smart-home temperature is frozen **per training cutoff**
(`smart_home/config.py:8-13`) so no future temperature can leak into the
transferred seasonality.

## Failure modes

- Live Wikipedia scrape returns a navbox instead of the changes table → use the
  frozen URL; if it ever fails, record the failure, do not silently fall back.
- A ticker renamed (`BRK-B` vs `BRK.B`) → membership reconstruction handles the
  dotted form; keep the raw `date_added` and record the mapping.
- Sidecar mismatch after a data refresh → the old results are invalidated;
  re-run, or pin the old data directory and keep its hash.
- `data_sha256.json` present but data dir moved → `check_sidecars.py --root`
  points at the data dir, sidecar discovery still finds the file via its
  `--sidecar` argument.
