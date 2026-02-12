"""Tests for vangja.datasets.stocks and the load_stock_data function.

All external calls (yfinance, pd.read_html) are mocked so that tests
run offline and deterministically.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from vangja.datasets.stocks import (
    _compute_typical_price,
    _download_stock_data,
    _fetch_sp500_wiki_tables,
    _get_sp500_constituents_at_date,
    _parse_changes_table,
    _parse_constituents_table,
    _safe_ticker_filename,
    get_sp500_tickers_for_range,
)
from vangja.datasets.loaders import load_stock_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_ohlcv():
    """Single-ticker OHLCV DataFrame mimicking yfinance output."""
    dates = pd.bdate_range("2020-01-02", periods=10)
    return pd.DataFrame(
        {
            "Open": np.arange(100, 110, dtype=float),
            "High": np.arange(102, 112, dtype=float),
            "Low": np.arange(98, 108, dtype=float),
            "Close": np.arange(101, 111, dtype=float),
            "Volume": [1_000_000] * 10,
        },
        index=pd.Index(dates, name="Date"),
    )


@pytest.fixture()
def multi_ohlcv():
    """Multi-ticker OHLCV DataFrame mimicking yfinance group_by='ticker'."""
    dates = pd.bdate_range("2020-01-02", periods=5)
    arrays = {
        ("AAPL", "Open"): np.arange(100, 105, dtype=float),
        ("AAPL", "High"): np.arange(102, 107, dtype=float),
        ("AAPL", "Low"): np.arange(98, 103, dtype=float),
        ("AAPL", "Close"): np.arange(101, 106, dtype=float),
        ("AAPL", "Volume"): [1_000_000] * 5,
        ("MSFT", "Open"): np.arange(200, 205, dtype=float),
        ("MSFT", "High"): np.arange(202, 207, dtype=float),
        ("MSFT", "Low"): np.arange(198, 203, dtype=float),
        ("MSFT", "Close"): np.arange(201, 206, dtype=float),
        ("MSFT", "Volume"): [2_000_000] * 5,
    }
    idx = pd.MultiIndex.from_tuples(arrays.keys(), names=["Ticker", "Price"])
    df = pd.DataFrame(
        np.column_stack(list(arrays.values())),
        index=pd.Index(dates, name="Date"),
        columns=idx,
    )
    return df


@pytest.fixture()
def mock_constituents_df():
    """Minimal current-constituents DataFrame."""
    return pd.DataFrame(
        {
            "ticker": [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META",
                "ABNB", "BX", "TSLA",
            ],
            "date_added": pd.to_datetime(
                [
                    "1982-11-30",
                    "1994-06-01",
                    "2005-11-18",
                    "2006-04-03",
                    "2013-12-23",
                    "2023-09-18",
                    "2023-09-18",
                    "2020-12-21",
                ]
            ),
            "security": [
                "Apple", "Microsoft", "Amazon", "Alphabet", "Meta",
                "Airbnb", "Blackstone", "Tesla",
            ],
        }
    )


@pytest.fixture()
def mock_changes_df():
    """Minimal historical-changes DataFrame."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-09-18", "2023-09-18", "2020-12-21"]
            ),
            "added_ticker": ["ABNB", "BX", "TSLA"],
            "added_name": ["Airbnb", "Blackstone", "Tesla"],
            "removed_ticker": ["NWL", "LNC", "AIV"],
            "removed_name": ["Newell", "Lincoln", "AIM"],
        }
    )


# ---------------------------------------------------------------------------
# _compute_typical_price
# ---------------------------------------------------------------------------


class TestComputeTypicalPrice:
    """Tests for the typical-price helper."""

    def test_basic(self):
        df = pd.DataFrame(
            {"Open": [10.0], "High": [12.0], "Low": [8.0], "Close": [11.0]}
        )
        result = _compute_typical_price(df)
        assert result.iloc[0] == pytest.approx(10.25)

    def test_multiple_rows(self):
        df = pd.DataFrame(
            {
                "Open": [10.0, 20.0],
                "High": [12.0, 24.0],
                "Low": [8.0, 16.0],
                "Close": [11.0, 22.0],
            }
        )
        result = _compute_typical_price(df)
        assert len(result) == 2
        assert result.iloc[0] == pytest.approx(10.25)
        assert result.iloc[1] == pytest.approx(20.5)

    def test_equal_prices(self):
        """When all prices are the same, typical price equals that value."""
        df = pd.DataFrame(
            {"Open": [50.0], "High": [50.0], "Low": [50.0], "Close": [50.0]}
        )
        result = _compute_typical_price(df)
        assert result.iloc[0] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# _safe_ticker_filename
# ---------------------------------------------------------------------------


class TestSafeTickerFilename:
    def test_caret(self):
        assert _safe_ticker_filename("^GSPC") == "_GSPC"

    def test_dot(self):
        assert _safe_ticker_filename("BRK.B") == "BRK_B"

    def test_slash(self):
        assert _safe_ticker_filename("BF/B") == "BF_B"

    def test_plain(self):
        assert _safe_ticker_filename("AAPL") == "AAPL"


# ---------------------------------------------------------------------------
# _download_stock_data
# ---------------------------------------------------------------------------


class TestDownloadStockData:

    @staticmethod
    def _mock_yf(return_value):
        """Create a mock yfinance module and inject it into sys.modules."""
        mock_yf = MagicMock()
        mock_yf.download.return_value = return_value
        return patch.dict("sys.modules", {"yfinance": mock_yf}), mock_yf

    def test_single_ticker(self, sample_ohlcv):
        ctx, mock_yf = self._mock_yf(sample_ohlcv)
        with ctx:
            result = _download_stock_data(
                ["AAPL"], "2020-01-01", "2020-01-15"
            )

        assert "ds" in result.columns
        assert "ticker" in result.columns
        assert "typical_price" in result.columns
        assert (result["ticker"] == "AAPL").all()
        assert len(result) == 10

    def test_typical_price_values(self, sample_ohlcv):
        ctx, _ = self._mock_yf(sample_ohlcv)
        with ctx:
            result = _download_stock_data(
                ["AAPL"], "2020-01-01", "2020-01-15"
            )

        expected = (
            sample_ohlcv["Open"]
            + sample_ohlcv["High"]
            + sample_ohlcv["Low"]
            + sample_ohlcv["Close"]
        ) / 4
        np.testing.assert_allclose(
            result["typical_price"].values, expected.values
        )

    def test_caching_creates_file(self, sample_ohlcv, tmp_path):
        cache = tmp_path / "cache"
        ctx, _ = self._mock_yf(sample_ohlcv)
        with ctx:
            _download_stock_data(
                ["AAPL"], "2020-01-01", "2020-01-15", cache_path=cache
            )

        assert (cache / "AAPL.csv").exists()

    def test_caching_avoids_redownload(self, sample_ohlcv, tmp_path):
        cache = tmp_path / "cache"
        ctx, mock_yf = self._mock_yf(sample_ohlcv)
        with ctx:
            _download_stock_data(
                ["AAPL"], "2020-01-01", "2020-01-15", cache_path=cache
            )
            assert mock_yf.download.call_count == 1

            _download_stock_data(
                ["AAPL"], "2020-01-01", "2020-01-15", cache_path=cache
            )
            # Should NOT have called download again
            assert mock_yf.download.call_count == 1

    def test_multi_ticker(self, multi_ohlcv):
        ctx, _ = self._mock_yf(multi_ohlcv)
        with ctx:
            result = _download_stock_data(
                ["AAPL", "MSFT"], "2020-01-01", "2020-01-10"
            )

        assert set(result["ticker"].unique()) == {"AAPL", "MSFT"}
        aapl = result[result["ticker"] == "AAPL"]
        msft = result[result["ticker"] == "MSFT"]
        assert len(aapl) == 5
        assert len(msft) == 5

    def test_empty_result(self):
        ctx, _ = self._mock_yf(pd.DataFrame())
        with ctx:
            result = _download_stock_data(
                ["INVALID"], "2020-01-01", "2020-01-10"
            )

        assert result.empty
        assert "ds" in result.columns
        assert "typical_price" in result.columns

    def test_cache_creates_parents(self, sample_ohlcv, tmp_path):
        cache = tmp_path / "deeply" / "nested" / "cache"
        ctx, _ = self._mock_yf(sample_ohlcv)
        with ctx:
            _download_stock_data(
                ["AAPL"], "2020-01-01", "2020-01-15", cache_path=cache
            )

        assert cache.exists()
        assert (cache / "AAPL.csv").exists()

    def test_no_cache(self, sample_ohlcv):
        """Without cache_path, data is returned but nothing written."""
        ctx, _ = self._mock_yf(sample_ohlcv)
        with ctx:
            result = _download_stock_data(
                ["AAPL"], "2020-01-01", "2020-01-15", cache_path=None
            )

        assert not result.empty

    def test_ds_column_is_datetime(self, sample_ohlcv):
        ctx, _ = self._mock_yf(sample_ohlcv)
        with ctx:
            result = _download_stock_data(
                ["AAPL"], "2020-01-01", "2020-01-15"
            )

        assert pd.api.types.is_datetime64_any_dtype(result["ds"])

    def test_ds_is_tz_naive(self, sample_ohlcv):
        tz_ohlcv = sample_ohlcv.copy()
        tz_ohlcv.index = tz_ohlcv.index.tz_localize("US/Eastern")
        ctx, _ = self._mock_yf(tz_ohlcv)
        with ctx:
            result = _download_stock_data(
                ["AAPL"], "2020-01-01", "2020-01-15"
            )

        assert result["ds"].dt.tz is None


# ---------------------------------------------------------------------------
# _parse_constituents_table
# ---------------------------------------------------------------------------


class TestParseConstituentsTable:

    def test_flat_columns(self):
        raw = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT"],
                "Security": ["Apple", "Microsoft"],
                "Date added": ["1982-11-30", "1994-06-01"],
            }
        )
        result = _parse_constituents_table(raw)
        assert "ticker" in result.columns
        assert "date_added" in result.columns
        assert list(result["ticker"]) == ["AAPL", "MSFT"]

    def test_date_added_is_datetime(self):
        raw = pd.DataFrame(
            {
                "Symbol": ["AAPL"],
                "Security": ["Apple"],
                "Date added": ["1982-11-30"],
            }
        )
        result = _parse_constituents_table(raw)
        assert pd.api.types.is_datetime64_any_dtype(result["date_added"])


# ---------------------------------------------------------------------------
# _parse_changes_table
# ---------------------------------------------------------------------------


class TestParseChangesTable:

    def test_flat_columns(self):
        raw = pd.DataFrame(
            {
                "Date": ["2023-09-18"],
                "Added_Ticker": ["ABNB"],
                "Added_Security": ["Airbnb"],
                "Removed_Ticker": ["NWL"],
                "Removed_Security": ["Newell"],
                "Reason": ["Market cap"],
            }
        )
        result = _parse_changes_table(raw)
        assert "date" in result.columns
        assert "added_ticker" in result.columns
        assert "removed_ticker" in result.columns

    def test_multiindex_columns(self):
        """Test parsing when pd.read_html returns MultiIndex headers."""
        arrays = [
            ["Date", "Added", "Added", "Removed", "Removed", "Reason"],
            ["Date", "Ticker", "Security", "Ticker", "Security", "Reason"],
        ]
        tuples = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples)
        data = [["2023-09-18", "ABNB", "Airbnb", "NWL", "Newell", "Cap"]]
        raw = pd.DataFrame(data, columns=index)

        result = _parse_changes_table(raw)
        assert "date" in result.columns
        assert "added_ticker" in result.columns
        assert "removed_ticker" in result.columns
        assert result["added_ticker"].iloc[0] == "ABNB"
        assert result["removed_ticker"].iloc[0] == "NWL"

    def test_nan_tickers(self):
        """Rows with only additions or only removals are handled."""
        raw = pd.DataFrame(
            {
                "Date": ["2023-01-01", "2023-02-01"],
                "Added_Ticker": ["NEWCO", None],
                "Added_Security": ["New Company", None],
                "Removed_Ticker": [None, "OLDCO"],
                "Removed_Security": [None, "Old Company"],
                "Reason": ["Spin-off", "Market cap"],
            }
        )
        result = _parse_changes_table(raw)
        assert len(result) == 2

    def test_invalid_dates_dropped(self):
        raw = pd.DataFrame(
            {
                "Date": ["2023-01-01", "not-a-date"],
                "Added_Ticker": ["A", "B"],
                "Added_Security": ["Aa", "Bb"],
                "Removed_Ticker": ["C", "D"],
                "Removed_Security": ["Cc", "Dd"],
                "Reason": ["x", "y"],
            }
        )
        result = _parse_changes_table(raw)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# _fetch_sp500_wiki_tables
# ---------------------------------------------------------------------------


class TestFetchSP500WikiTables:

    def test_returns_two_dataframes(
        self, mock_constituents_df, mock_changes_df
    ):
        raw_const = pd.DataFrame(
            {
                "Symbol": mock_constituents_df["ticker"],
                "Security": mock_constituents_df["security"],
                "Date added": mock_constituents_df["date_added"].astype(str),
            }
        )
        raw_changes = pd.DataFrame(
            {
                "Date": mock_changes_df["date"].astype(str),
                "Added_Ticker": mock_changes_df["added_ticker"],
                "Added_Security": mock_changes_df["added_name"],
                "Removed_Ticker": mock_changes_df["removed_ticker"],
                "Removed_Security": mock_changes_df["removed_name"],
                "Reason": ["Cap", "Cap", "Cap"],
            }
        )
        with patch(
            "vangja.datasets.stocks.pd.read_html",
            return_value=[raw_const, raw_changes],
        ):
            const, changes = _fetch_sp500_wiki_tables()

        assert isinstance(const, pd.DataFrame)
        assert isinstance(changes, pd.DataFrame)
        assert "ticker" in const.columns
        assert "date" in changes.columns

    def test_caching(self, tmp_path, mock_constituents_df, mock_changes_df):
        raw_const = pd.DataFrame(
            {
                "Symbol": mock_constituents_df["ticker"],
                "Security": mock_constituents_df["security"],
                "Date added": mock_constituents_df["date_added"].astype(str),
            }
        )
        raw_changes = pd.DataFrame(
            {
                "Date": mock_changes_df["date"].astype(str),
                "Added_Ticker": mock_changes_df["added_ticker"],
                "Added_Security": mock_changes_df["added_name"],
                "Removed_Ticker": mock_changes_df["removed_ticker"],
                "Removed_Security": mock_changes_df["removed_name"],
                "Reason": ["Cap", "Cap", "Cap"],
            }
        )
        cache = tmp_path / "wiki_cache"

        with patch(
            "vangja.datasets.stocks.pd.read_html",
            return_value=[raw_const, raw_changes],
        ) as mock_read:
            _fetch_sp500_wiki_tables(cache_path=cache)
            assert mock_read.call_count == 1

            # Second call should use cache
            _fetch_sp500_wiki_tables(cache_path=cache)
            assert mock_read.call_count == 1

        assert (cache / "sp500_constituents.csv").exists()
        assert (cache / "sp500_changes.csv").exists()


# ---------------------------------------------------------------------------
# _get_sp500_constituents_at_date
# ---------------------------------------------------------------------------


class TestGetSP500ConstituentsAtDate:

    def test_current_date_returns_current_set(
        self, mock_constituents_df, mock_changes_df
    ):
        """A future date should return the current constituents."""
        result = _get_sp500_constituents_at_date(
            "2030-01-01",
            const_df=mock_constituents_df,
            changes_df=mock_changes_df,
        )
        expected = {
            "AAPL", "MSFT", "AMZN", "GOOGL", "META",
            "ABNB", "BX", "TSLA",
        }
        assert result == expected

    def test_before_addition(
        self, mock_constituents_df, mock_changes_df
    ):
        """Tickers added after target_date should not appear."""
        # ABNB and BX were added 2023-09-18, TSLA on 2020-12-21
        result = _get_sp500_constituents_at_date(
            "2022-01-01",
            const_df=mock_constituents_df,
            changes_df=mock_changes_df,
        )
        # ABNB, BX added after 2022-01-01 → should NOT be in result
        assert "ABNB" not in result
        assert "BX" not in result
        # TSLA added 2020-12-21, before target → should be in result
        assert "TSLA" in result
        # NWL, LNC removed 2023-09-18 (after target) → restored
        assert "NWL" in result
        assert "LNC" in result
        # AIV removed 2020-12-21 (before target) → NOT restored
        assert "AIV" not in result
        # Original tickers still present
        assert "AAPL" in result
        assert "MSFT" in result

    def test_between_changes(
        self, mock_constituents_df, mock_changes_df
    ):
        """Date between changes should reflect the right set."""
        # After TSLA addition (2020-12-21) but before ABNB/BX (2023-09-18)
        result = _get_sp500_constituents_at_date(
            "2022-06-01",
            const_df=mock_constituents_df,
            changes_df=mock_changes_df,
        )
        assert "TSLA" in result  # Added 2020-12-21 (before target)
        assert "ABNB" not in result  # Added 2023-09-18 (after target)
        assert "NWL" in result  # Removed 2023-09-18 (still present)
        assert "AIV" not in result  # Removed 2020-12-21 (already gone)


# ---------------------------------------------------------------------------
# get_sp500_tickers_for_range
# ---------------------------------------------------------------------------


class TestGetSP500TickersForRange:

    def test_returns_sorted_list(
        self, mock_constituents_df, mock_changes_df
    ):
        with patch(
            "vangja.datasets.stocks._fetch_sp500_wiki_tables",
            return_value=(mock_constituents_df, mock_changes_df),
        ):
            result = get_sp500_tickers_for_range(
                "2024-01-01", "2024-12-31"
            )

        assert isinstance(result, list)
        assert result == sorted(result)

    def test_excludes_removed_during_range(
        self, mock_constituents_df, mock_changes_df
    ):
        """Tickers removed during the range should be excluded."""
        # ABNB, BX added and NWL, LNC removed on 2023-09-18
        with patch(
            "vangja.datasets.stocks._fetch_sp500_wiki_tables",
            return_value=(mock_constituents_df, mock_changes_df),
        ):
            result = get_sp500_tickers_for_range(
                "2023-01-01", "2023-12-31"
            )

        # NWL, LNC were removed during range → excluded
        assert "NWL" not in result
        assert "LNC" not in result
        # AAPL, MSFT should still be there
        assert "AAPL" in result
        assert "MSFT" in result

    def test_no_changes_during_range(
        self, mock_constituents_df, mock_changes_df
    ):
        """If no changes during range, start-set is returned intact."""
        with patch(
            "vangja.datasets.stocks._fetch_sp500_wiki_tables",
            return_value=(mock_constituents_df, mock_changes_df),
        ):
            result = get_sp500_tickers_for_range(
                "2024-06-01", "2024-06-30"
            )

        # No changes happened in June 2024 per mock data
        assert "AAPL" in result
        assert "MSFT" in result
        assert "META" in result

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="must be before"):
            get_sp500_tickers_for_range("2025-01-01", "2020-01-01")

    def test_all_items_are_strings(
        self, mock_constituents_df, mock_changes_df
    ):
        with patch(
            "vangja.datasets.stocks._fetch_sp500_wiki_tables",
            return_value=(mock_constituents_df, mock_changes_df),
        ):
            result = get_sp500_tickers_for_range(
                "2024-01-01", "2024-12-31"
            )

        assert all(isinstance(t, str) for t in result)


# ---------------------------------------------------------------------------
# load_stock_data
# ---------------------------------------------------------------------------


class TestLoadStockData:

    @pytest.fixture()
    def mock_download_data(self):
        """Sample data simulating _download_stock_data output."""
        dates = pd.bdate_range("2020-01-02", periods=30)
        return pd.DataFrame(
            {
                "ds": dates,
                "ticker": "AAPL",
                "Open": np.arange(100, 130, dtype=float),
                "High": np.arange(102, 132, dtype=float),
                "Low": np.arange(98, 128, dtype=float),
                "Close": np.arange(101, 131, dtype=float),
                "Volume": [1_000_000] * 30,
                "typical_price": (
                    np.arange(100, 130)
                    + np.arange(102, 132)
                    + np.arange(98, 128)
                    + np.arange(101, 131)
                )
                / 4.0,
            }
        )

    def test_train_test_split(self, mock_download_data):
        with patch(
            "vangja.datasets.stocks._download_stock_data",
            return_value=mock_download_data,
        ):
            train, test = load_stock_data(
                tickers=["AAPL"],
                split_date="2020-01-20",
                window_size=30,
                horizon_size=30,
            )

        assert all(train["ds"] < pd.Timestamp("2020-01-20"))
        assert all(test["ds"] >= pd.Timestamp("2020-01-20"))

    def test_output_columns(self, mock_download_data):
        with patch(
            "vangja.datasets.stocks._download_stock_data",
            return_value=mock_download_data,
        ):
            train, test = load_stock_data(
                tickers=["AAPL"],
                split_date="2020-01-20",
                window_size=30,
                horizon_size=30,
            )

        assert set(train.columns) == {"ds", "y", "series"}
        assert set(test.columns) == {"ds", "y", "series"}

    def test_series_column_has_ticker(self, mock_download_data):
        with patch(
            "vangja.datasets.stocks._download_stock_data",
            return_value=mock_download_data,
        ):
            train, _ = load_stock_data(
                tickers=["AAPL"],
                split_date="2020-01-20",
                window_size=30,
                horizon_size=30,
            )

        assert (train["series"] == "AAPL").all()

    def test_empty_data(self):
        empty = pd.DataFrame(
            columns=[
                "ds", "ticker", "Open", "High", "Low",
                "Close", "Volume", "typical_price",
            ]
        )
        with patch(
            "vangja.datasets.stocks._download_stock_data",
            return_value=empty,
        ):
            train, test = load_stock_data(
                tickers=["INVALID"],
                split_date="2020-01-20",
                window_size=30,
                horizon_size=30,
            )

        assert train.empty
        assert test.empty

    def test_interpolate_fills_gaps(self, mock_download_data):
        # Remove some rows to create gaps
        gapped = mock_download_data.drop(index=[2, 3, 4])

        with patch(
            "vangja.datasets.stocks._download_stock_data",
            return_value=gapped,
        ):
            train_no_interp, _ = load_stock_data(
                tickers=["AAPL"],
                split_date="2020-01-20",
                window_size=30,
                horizon_size=30,
                interpolate=False,
            )
        with patch(
            "vangja.datasets.stocks._download_stock_data",
            return_value=gapped,
        ):
            train_interp, _ = load_stock_data(
                tickers=["AAPL"],
                split_date="2020-01-20",
                window_size=30,
                horizon_size=30,
                interpolate=True,
            )

        # Interpolated should have more rows (weekends filled too)
        assert len(train_interp) >= len(train_no_interp)

    def test_ds_is_datetime(self, mock_download_data):
        with patch(
            "vangja.datasets.stocks._download_stock_data",
            return_value=mock_download_data,
        ):
            train, test = load_stock_data(
                tickers=["AAPL"],
                split_date="2020-01-20",
                window_size=30,
                horizon_size=30,
            )

        assert pd.api.types.is_datetime64_any_dtype(train["ds"])
        assert pd.api.types.is_datetime64_any_dtype(test["ds"])

    def test_y_is_numeric(self, mock_download_data):
        with patch(
            "vangja.datasets.stocks._download_stock_data",
            return_value=mock_download_data,
        ):
            train, test = load_stock_data(
                tickers=["AAPL"],
                split_date="2020-01-20",
                window_size=30,
                horizon_size=30,
            )

        assert pd.api.types.is_numeric_dtype(train["y"])
        assert pd.api.types.is_numeric_dtype(test["y"])

    def test_cache_path_forwarded(self, mock_download_data, tmp_path):
        cache = tmp_path / "stock_cache"
        with patch(
            "vangja.datasets.stocks._download_stock_data",
            return_value=mock_download_data,
        ) as mock_dl:
            load_stock_data(
                tickers=["AAPL"],
                split_date="2020-01-20",
                window_size=30,
                horizon_size=30,
                cache_path=cache,
            )

        # Verify cache_path was passed through
        _, kwargs = mock_dl.call_args
        assert kwargs["cache_path"] == cache

    def test_multi_ticker(self):
        dates = pd.bdate_range("2020-01-02", periods=20)
        data = pd.concat(
            [
                pd.DataFrame(
                    {
                        "ds": dates,
                        "ticker": ticker,
                        "Open": np.arange(100, 120, dtype=float),
                        "High": np.arange(102, 122, dtype=float),
                        "Low": np.arange(98, 118, dtype=float),
                        "Close": np.arange(101, 121, dtype=float),
                        "Volume": [1_000_000] * 20,
                        "typical_price": (
                            np.arange(100, 120)
                            + np.arange(102, 122)
                            + np.arange(98, 118)
                            + np.arange(101, 121)
                        )
                        / 4.0,
                    }
                )
                for ticker in ["AAPL", "MSFT"]
            ],
            ignore_index=True,
        )

        with patch(
            "vangja.datasets.stocks._download_stock_data",
            return_value=data,
        ):
            train, test = load_stock_data(
                tickers=["AAPL", "MSFT"],
                split_date="2020-01-15",
                window_size=30,
                horizon_size=30,
            )

        assert set(train["series"].unique()) == {"AAPL", "MSFT"}
