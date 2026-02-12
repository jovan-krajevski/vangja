"""Private helper functions for downloading historical stock data.

This module provides functions for downloading OHLCV data from Yahoo Finance,
computing typical prices, and determining S&P 500 index composition history.

The only public function is :func:`get_sp500_tickers_for_range`, which returns
tickers that were consistently part of the S&P 500 during a given time range.

All other functions are private helpers prefixed with ``_``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _compute_typical_price(df: pd.DataFrame) -> pd.Series:
    """Compute typical price as (Open + High + Low + Close) / 4.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``Open``, ``High``, ``Low``, ``Close``.

    Returns
    -------
    pd.Series
        Typical price series.
    """
    return (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4


def _safe_ticker_filename(ticker: str) -> str:
    """Convert a ticker symbol to a safe filename component.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., ``"^GSPC"``, ``"BRK.B"``).

    Returns
    -------
    str
        Filename-safe string.
    """
    return ticker.replace("^", "_").replace("/", "_").replace(".", "_")


def _download_stock_data(
    tickers: list[str],
    start: str | datetime,
    end: str | datetime,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Download historical daily OHLCV data for one or more tickers.

    Uses ``yfinance`` to batch-download data for efficiency when multiple
    tickers are requested. The ``end`` date is inclusive.

    Parameters
    ----------
    tickers : list[str]
        List of ticker symbols to download (e.g., ``["AAPL", "^GSPC"]``).
    start : str or datetime
        Start date for the data (inclusive).
    end : str or datetime
        End date for the data (inclusive).
    cache_path : Path or None, default None
        Directory path for caching downloaded data. If None, data is
        downloaded without caching. If provided, each ticker's data is
        stored as a CSV file in this directory. Parent directories are
        created if they do not exist. On subsequent calls, cached data
        is loaded instead of re-downloading.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``ds`` (datetime), ``ticker`` (str),
        ``Open``, ``High``, ``Low``, ``Close``, ``Volume`` (float),
        and ``typical_price`` (float).
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is required to download stock data. "
            "Install with: pip install vangja[datasets]"
        ) from e

    results: list[pd.DataFrame] = []
    tickers_to_download: list[str] = []

    # Check cache for each ticker
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
        for ticker in tickers:
            fname = cache_path / f"{_safe_ticker_filename(ticker)}.csv"
            if fname.exists():
                logger.info("Loading cached data for %s", ticker)
                df = pd.read_csv(fname, parse_dates=["ds"])
                mask = (df["ds"] >= pd.Timestamp(start)) & (
                    df["ds"] <= pd.Timestamp(end)
                )
                results.append(df[mask])
            else:
                tickers_to_download.append(ticker)
    else:
        tickers_to_download = list(tickers)

    if not tickers_to_download:
        if not results:
            return pd.DataFrame(
                columns=[
                    "ds",
                    "ticker",
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "typical_price",
                ]
            )
        return pd.concat(results, ignore_index=True)

    # Make end inclusive by adding one day (yfinance end is exclusive)
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start_str = start if isinstance(start, str) else start.strftime("%Y-%m-%d")

    # Download all missing tickers at once for speed
    if len(tickers_to_download) == 1:
        raw = yf.download(
            tickers_to_download[0],
            start=start_str,
            end=end_exclusive,
            auto_adjust=True,
            progress=False,
        )
    else:
        raw = yf.download(
            tickers_to_download,
            start=start_str,
            end=end_exclusive,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

    if raw.empty:
        logger.warning("No data returned from yfinance")
    else:
        for ticker in tickers_to_download:
            try:
                # Extract per-ticker data
                if len(tickers_to_download) == 1:
                    ticker_df = raw.copy()
                else:
                    ticker_df = raw[ticker].copy()

                # Flatten MultiIndex columns if present (newer yfinance)
                if isinstance(ticker_df.columns, pd.MultiIndex):
                    for idx, level in enumerate(ticker_df.columns.levels):
                        if "Open" in level:
                            ticker_df.columns = ticker_df.columns.get_level_values(idx)
                            break

                ticker_df = ticker_df.dropna(how="all")
                if ticker_df.empty:
                    logger.warning("No data for %s", ticker)
                    continue

                ticker_df = ticker_df.reset_index()

                # Normalize date column name
                for col in ("Date", "Datetime", "date"):
                    if col in ticker_df.columns:
                        ticker_df = ticker_df.rename(columns={col: "ds"})
                        break

                ticker_df["ds"] = pd.to_datetime(ticker_df["ds"]).dt.tz_localize(None)
                ticker_df["ticker"] = ticker
                ticker_df["typical_price"] = _compute_typical_price(ticker_df)

                if cache_path is not None:
                    fname = cache_path / f"{_safe_ticker_filename(ticker)}.csv"
                    ticker_df.to_csv(fname, index=False)

                results.append(ticker_df)
            except (KeyError, TypeError):
                logger.warning("No data for %s", ticker)

    if not results:
        return pd.DataFrame(
            columns=[
                "ds",
                "ticker",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "typical_price",
            ]
        )

    return pd.concat(results, ignore_index=True)


def _parse_constituents_table(table: pd.DataFrame) -> pd.DataFrame:
    """Parse the S&P 500 current constituents table from Wikipedia.

    Parameters
    ----------
    table : pd.DataFrame
        Raw DataFrame from ``pd.read_html`` for the first table on
        the Wikipedia S&P 500 companies page.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least columns ``ticker`` (str) and
        ``date_added`` (datetime).
    """
    df = table.copy()

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            str(c[-1]) if c[-1] and "Unnamed" not in str(c[-1]) else str(c[0])
            for c in df.columns
        ]

    # Rename to standard names based on content
    col_map: dict[str, str] = {}
    for col in df.columns:
        col_lower = str(col).lower()
        if "symbol" in col_lower or col_lower == "ticker":
            col_map[col] = "ticker"
        elif "date" in col_lower and "added" in col_lower:
            col_map[col] = "date_added"
        elif col_lower == "security":
            col_map[col] = "security"

    df = df.rename(columns=col_map)

    if "date_added" in df.columns:
        df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

    return df


def _parse_changes_table(table: pd.DataFrame) -> pd.DataFrame:
    """Parse the S&P 500 historical changes table from Wikipedia.

    Parameters
    ----------
    table : pd.DataFrame
        Raw DataFrame from ``pd.read_html`` for the second table on
        the Wikipedia S&P 500 companies page.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``date``, ``added_ticker``,
        ``added_name``, ``removed_ticker``, ``removed_name``.
    """
    df = table.copy()

    # Handle MultiIndex columns from merged header cells
    if isinstance(df.columns, pd.MultiIndex):
        new_cols: list[str] = []
        for i, col in enumerate(df.columns):
            level0 = str(col[0]).strip().lower()
            level1 = str(col[1]).strip().lower() if len(col) > 1 else ""

            if "date" in level0:
                new_cols.append("date")
            elif "added" in level0 and "ticker" in level1:
                new_cols.append("added_ticker")
            elif "added" in level0 and ("security" in level1 or "name" in level1):
                new_cols.append("added_name")
            elif "removed" in level0 and "ticker" in level1:
                new_cols.append("removed_ticker")
            elif "removed" in level0 and ("security" in level1 or "name" in level1):
                new_cols.append("removed_name")
            elif "reason" in level0:
                new_cols.append("reason")
            else:
                new_cols.append(f"col_{i}")
        df.columns = new_cols
    else:
        # Flat columns — rename by position
        cols = df.columns.tolist()
        if len(cols) >= 6:
            rename = {
                cols[0]: "date",
                cols[1]: "added_ticker",
                cols[2]: "added_name",
                cols[3]: "removed_ticker",
                cols[4]: "removed_name",
                cols[5]: "reason",
            }
            df = df.rename(columns=rename)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    keep = [
        c
        for c in [
            "date",
            "added_ticker",
            "added_name",
            "removed_ticker",
            "removed_name",
        ]
        if c in df.columns
    ]
    return df[keep]


def _fetch_sp500_wiki_tables(
    cache_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch S&P 500 constituent and changes tables from Wikipedia.

    Downloads the "List of S&P 500 companies" Wikipedia page and parses
    both the current constituents table and the historical changes table.

    Parameters
    ----------
    cache_path : Path or None, default None
        Directory for caching parsed tables as CSV files. If both
        ``sp500_constituents.csv`` and ``sp500_changes.csv`` exist in
        this directory, they are loaded instead of downloading.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(constituents_df, changes_df)``
    """
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
        const_file = cache_path / "sp500_constituents.csv"
        changes_file = cache_path / "sp500_changes.csv"

        if const_file.exists() and changes_file.exists():
            const_df = pd.read_csv(const_file)
            if "date_added" in const_df.columns:
                const_df["date_added"] = pd.to_datetime(const_df["date_added"])
            changes_df = pd.read_csv(changes_file)
            if "date" in changes_df.columns:
                changes_df["date"] = pd.to_datetime(changes_df["date"])
            return const_df, changes_df

    resp = requests.get(_SP500_WIKI_URL, headers=_HEADERS)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    const_df = _parse_constituents_table(tables[0])
    changes_df = _parse_changes_table(tables[1])

    if cache_path is not None:
        const_df.to_csv(const_file, index=False)
        changes_df.to_csv(changes_file, index=False)

    return const_df, changes_df


def _get_sp500_constituents_at_date(
    target_date: str | datetime | pd.Timestamp,
    const_df: pd.DataFrame | None = None,
    changes_df: pd.DataFrame | None = None,
    cache_path: Path | None = None,
) -> set[str]:
    """Determine which tickers were in the S&P 500 on a given date.

    Reconstructs the index composition by starting from the current
    constituents and reversing all historical changes that occurred
    after ``target_date``.

    Parameters
    ----------
    target_date : str, datetime, or pd.Timestamp
        The date to determine S&P 500 composition for.
    const_df : pd.DataFrame or None
        Pre-fetched constituents table. If None, fetched from Wikipedia.
    changes_df : pd.DataFrame or None
        Pre-fetched changes table. If None, fetched from Wikipedia.
    cache_path : Path or None
        Directory for caching Wikipedia data.

    Returns
    -------
    set[str]
        Set of ticker symbols in the S&P 500 on the target date.

    Notes
    -----
    Accuracy depends on Wikipedia's historical changes table, which has
    comprehensive data from approximately 1997 onwards. Earlier dates
    may be less accurate.
    """
    target = pd.Timestamp(target_date)

    if const_df is None or changes_df is None:
        const_df, changes_df = _fetch_sp500_wiki_tables(cache_path)

    # Start with current tickers
    current_tickers = set(const_df["ticker"].dropna().astype(str).str.strip())
    result = current_tickers.copy()

    # Sort changes newest first and undo those after target_date
    sorted_changes = changes_df.sort_values("date", ascending=False)

    for _, row in sorted_changes.iterrows():
        change_date = row["date"]
        if pd.isna(change_date) or change_date <= target:
            continue

        added = row.get("added_ticker")
        removed = row.get("removed_ticker")

        # Undo addition (wasn't there before this date)
        if pd.notna(added) and str(added).strip():
            result.discard(str(added).strip())

        # Undo removal (was still there before this date)
        if pd.notna(removed) and str(removed).strip():
            result.add(str(removed).strip())

    return result


def get_sp500_tickers_for_range(
    start_date: str | datetime | pd.Timestamp,
    end_date: str | datetime | pd.Timestamp,
    cache_path: Path | None = None,
) -> list[str]:
    """Get tickers consistently in the S&P 500 during a date range.

    Returns tickers that were part of the S&P 500 for the entire duration
    between ``start_date`` and ``end_date``. A ticker is excluded if it
    was removed at any point during the range, even if it was later
    re-added.

    Parameters
    ----------
    start_date : str, datetime, or pd.Timestamp
        Start of the date range (inclusive).
    end_date : str, datetime, or pd.Timestamp
        End of the date range (inclusive).
    cache_path : Path or None, default None
        Directory for caching Wikipedia data as CSV files. If None,
        data is fetched without caching. If provided, parent directories
        are created if they do not exist.

    Returns
    -------
    list[str]
        Sorted list of ticker symbols that were consistently in the
        S&P 500 during the entire date range.

    Raises
    ------
    ValueError
        If ``start_date`` is after ``end_date``.

    Notes
    -----
    Accuracy depends on Wikipedia's "List of S&P 500 companies"
    historical changes table, which has comprehensive data from
    approximately 1997 onwards. Results for earlier periods may be
    less accurate.

    Examples
    --------
    >>> from vangja.datasets.stocks import get_sp500_tickers_for_range
    >>> tickers = get_sp500_tickers_for_range(
    ...     "2020-01-01", "2020-12-31"
    ... )  # doctest: +SKIP
    >>> "AAPL" in tickers  # doctest: +SKIP
    True
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError(f"start_date ({start}) must be before end_date ({end})")

    # Fetch tables once, share across calls
    const_df, changes_df = _fetch_sp500_wiki_tables(cache_path)

    # Get constituents at start of range
    constituents_at_start = _get_sp500_constituents_at_date(
        start,
        const_df=const_df,
        changes_df=changes_df,
    )

    # Find tickers removed during the range
    range_changes = changes_df[
        (changes_df["date"] > start) & (changes_df["date"] <= end)
    ]

    removed_during: set[str] = set()
    for _, row in range_changes.iterrows():
        removed = row.get("removed_ticker")
        if pd.notna(removed) and str(removed).strip():
            removed_during.add(str(removed).strip())

    consistent = constituents_at_start - removed_during
    return sorted(consistent)
