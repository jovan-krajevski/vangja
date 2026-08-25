"""Fetch, provenance and freeze the stock universe and data.

Run once per machine / data refresh:

    python fixed_case_studies/stocks/01_fetch_data.py

What it does (PROTOCOL.md §13, PLAN P0-16):

1. For every origin (development 2013-01..2014-12; frozen confirmation
   2023-01..2024-12) reconstructs the S&P 500 membership **at the origin
   date** from the Wikipedia historical changes table
   (``get_sp500_tickers_at_date``). Membership is not conditioned on
   survival through the horizon; delisted/missing series are recorded as
   availability outcomes instead of being silently dropped.
2. Downloads the constituents, the S&P 500 index (``^GSPC``) and the
   negative-control context (gold futures ``GC=F``) through the package's
   cached yfinance loader. Only price history up to the data-provider
   cutoff is used; no confirmation *results* are produced or inspected
   here — only data availability is recorded.
3. Writes per-origin universe files (with membership dates where known),
   an availability record, and SHA-256 sidecars of every cached file.

Nothing in this script fits a model or looks at forecasts.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import pandas as pd

from fixed_case_studies import common
from fixed_case_studies.stocks import config as cfg
from fixed_case_studies.stocks.runner import CT_PATH, TICKERS_PATH
from vangja.datasets.stocks import (
    _download_stock_data,
    _fetch_sp500_wiki_tables,
    get_sp500_tickers_at_date,
)


def membership_dates(cache_path: Path) -> dict[str, str]:
    """date_added per ticker from the cached Wikipedia constituents table."""
    try:
        const_df, _ = _fetch_sp500_wiki_tables(cache_path)
        out = {}
        for _, row in const_df.iterrows():
            if pd.notna(row.get("date_added")):
                out[str(row["ticker"]).strip()] = str(row["date_added"])
        return out
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--origins",
        nargs="*",
        default=None,
        help="Subset of origins to fetch (default: all development + confirmation).",
    )
    args = ap.parse_args()

    confirmation = common.load_confirmation_origins(
        Path(__file__).resolve().parents[1] / "protocol"
    )
    all_origins = cfg.DEV_ORIGINS + [
        o.strftime("%Y-%m-%d") for o in confirmation["origin"]
    ]
    if args.origins:
        all_origins = [o for o in all_origins if o in args.origins]

    dates = membership_dates(CT_PATH)

    universe_dir = Path(__file__).resolve().parent / "data" / "universe"
    universe_dir.mkdir(parents=True, exist_ok=True)

    availability_rows = []
    for origin in all_origins:
        print(f"=== origin {origin} ===")
        universe = get_sp500_tickers_at_date(origin, cache_path=CT_PATH)
        universe_df = pd.DataFrame(
            {
                "ticker": universe,
                "origin": origin,
                "universe_rule": "members_at_origin",
                "date_added": [
                    dates.get(t.replace("-", "."), "") for t in universe
                ],
            }
        )
        universe_df.to_csv(universe_dir / f"universe_{origin}.csv", index=False)

        all_tickers = sorted(set(universe) | {cfg.CONTEXT_TICKER, cfg.NEGATIVE_CONTROL_TICKER})
        _download_stock_data(all_tickers, cache_path=TICKERS_PATH)

        # Availability: trading-day observations in the 91-day window before
        # the origin and presence in the 365-day horizon. Data *presence*
        # only — no model, no forecasts.
        availability = _availability(origin, universe)
        availability.update({"origin": origin})
        availability_rows.append(availability)
        print(f"  universe={len(universe)} available={availability['n_available_train_and_test']}")

    avail_df = pd.DataFrame(availability_rows)
    avail_path = Path(__file__).resolve().parent / "data" / "availability.csv"
    avail_df.to_csv(avail_path, index=False)

    hashes = common.record_sidecars(TICKERS_PATH)
    print(f"\nWrote {len(hashes)} data hash entries and availability record:")
    print(avail_path)


def _availability(origin, universe) -> dict:
    """Data-presence check without producing any result values."""
    from vangja.datasets import load_stock_data

    stocks_train, stocks_test = load_stock_data(
        universe,
        split_date=origin,
        window_size=cfg.TARGET_WINDOW_DAYS,
        horizon_size=cfg.HORIZON_DAYS,
        cache_path=TICKERS_PATH,
        interpolate=False,
    )
    counts = stocks_train.groupby("series").size()
    in_test = set(stocks_test["series"].unique())
    valid = sorted(
        [t for t in counts.index if counts[t] >= cfg.MIN_TRAIN_OBS and t in in_test]
    )
    return {
        "n_requested": len(set(universe)),
        "n_available_train_and_test": len(valid),
        "missing_from_train_or_test": sorted(set(universe) - set(valid)),
    }


if __name__ == "__main__":
    main()
