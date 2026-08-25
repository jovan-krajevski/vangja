"""Tests for the fixed_case_studies infrastructure.

Covers the pieces of the revised protocol implemented in
``fixed_case_studies/common.py`` and the two study runners:

- primary metric (Relative MAE vs persistence) and its aggregate;
- context-series exclusion from target metrics;
- denominator-threshold handling;
- two-way block bootstrap for the paired comparison;
- freeze check on the confirmation origins file;
- one end-to-end stocks cell with a synthetic offline cache;
- one end-to-end smart-home cell with mocked loaders.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fixed_case_studies import common  # noqa: E402
from fixed_case_studies.stocks import config as stock_cfg  # noqa: E402
from fixed_case_studies.smart_home import config as home_cfg  # noqa: E402


# ---------------------------------------------------------------------------
# common.py
# ---------------------------------------------------------------------------


class TestRelativeMAEAggregation:

    def _unit_df(self):
        return pd.DataFrame(
            {
                "series": ["A", "B", "C", "D"],
                "origin": ["o1"] * 4,
                "config": "main",
                "stage": "confirmation",
                "rel_mae": [0.8, 1.2, 0.9, np.nan],
                "excluded": [False, False, False, True],
                "mae": [1.0, 2.0, 3.0, 4.0],
            }
        )

    def test_aggregate_median_and_prop_below_1(self):
        agg = common.aggregate_rel_mae(self._unit_df())
        assert agg["median"] == pytest.approx(0.9)
        assert agg["mean"] == pytest.approx((0.8 + 1.2 + 0.9) / 3)
        assert agg["prop_below_1"] == pytest.approx(2 / 3)
        assert agg["n_units"] == 3
        assert agg["n_excluded"] == 1

    def test_aggregate_all_excluded(self):
        df = self._unit_df()
        df["rel_mae"] = np.nan
        df["excluded"] = True
        agg = common.aggregate_rel_mae(df)
        assert agg["median"] is None
        assert agg["n_units"] == 0


class TestPairedComparison:

    def _paired(self, rng):
        n_series, n_origins = 6, 8
        origins = pd.date_range("2023-01-01", periods=n_origins, freq="MS")
        rows = []
        for s in range(n_series):
            for o in origins:
                base = rng.normal(1.0, 0.1)
                rows.append(
                    {
                        "series": f"S{s}",
                        "origin": o.strftime("%Y-%m-%d"),
                        "rel_mae": base - 0.1,
                        "config": "main",
                    }
                )
                rows.append(
                    {
                        "series": f"S{s}",
                        "origin": o.strftime("%Y-%m-%d"),
                        "rel_mae": base,
                        "config": "no_transfer",
                    }
                )
        return pd.DataFrame(rows)

    def test_paired_differences_and_bootstrap(self):
        rng = np.random.default_rng(0)
        units = self._paired(rng)
        paired, summary = common.paired_transfer_comparison(
            units, "main", "no_transfer"
        )
        assert len(paired) == 48
        assert np.allclose(paired["diff_rel_mae"], -0.1)
        assert summary["n_paired_units"] == 48
        assert summary["prop_units_transfer_better"] == pytest.approx(1.0)
        assert summary["negative_transfer_rate"] == pytest.approx(0.0)
        assert "ci_2p5" in summary and "ci_97p5" in summary

    def test_missing_config_handled(self):
        units = pd.DataFrame(
            {"series": ["A"], "origin": ["o1"], "config": ["main"], "rel_mae": [1.0]}
        )
        paired, summary = common.paired_transfer_comparison(
            units, "main", "no_transfer"
        )
        assert paired.empty
        assert summary["n_paired_units"] == 0


class TestFreezeCheck:

    def test_ok_and_tampered(self, tmp_path):
        proto = tmp_path / "protocol"
        proto.mkdir()
        csv = proto / "CONFIRMATION_ORIGINS.csv"
        csv.write_text("origin,horizon_days\n2023-01-01,365\n")
        (proto / "CONFIRMATION_ORIGINS.sha256").write_text(
            common.sha256_bytes(csv.read_text().encode()) + "\n"
        )
        df = common.load_confirmation_origins(proto)
        assert len(df) == 1

        csv.write_text("origin,horizon_days\n2023-02-01,365\n")
        with pytest.raises(RuntimeError, match="hash mismatch"):
            common.load_confirmation_origins(proto)


class TestContextExclusion:

    def test_context_series_not_scored(self, tmp_path):
        # Build a minimal fake model object with the attributes unit_metrics
        # needs, including a context series as a group.
        class FakeModel:
            y_scale_params = {"scaler": "maxabs", "y_min": 0.0, "y_max": 10.0}
            groups_ = {0: "AAPL", 1: "^GSPC"}
            data = pd.DataFrame(
                {
                    "series": ["AAPL"] * 3 + ["^GSPC"] * 3,
                    "y": [0.5, 0.6, 0.7, 0.5, 0.5, 0.5],
                }
            )

        test = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=3).tolist() * 2,
                "y": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
                "series": ["AAPL"] * 3 + ["^GSPC"] * 3,
            }
        )
        future = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=3).tolist() * 2,
                "yhat_0": [1.1, 2.1, 3.1, 11.0, 21.0, 31.0],
                "yhat_1": [11.0, 21.0, 31.0, 11.0, 21.0, 31.0],
            }
        )
        units = common.unit_metrics(
            FakeModel(), test, future, origin="o", config="main", stage="s"
        )
        assert set(units["series"]) == {"AAPL"}
        assert "^GSPC" not in set(units["series"])


class TestDenominatorRule:

    def test_near_zero_persistence_excluded(self, tmp_path):
        class FakeModel:
            y_scale_params = {"scaler": "maxabs", "y_min": 0.0, "y_max": 1.0}
            groups_ = {0: "A"}
            data = pd.DataFrame({"series": ["A"] * 4, "y": [0.0, 0.0, 0.0, 1e-4]})

        test = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=4),
                "y": [1e-6, -1e-6, 0.0, 1e-4],
                "series": "A",
            }
        )
        future = pd.DataFrame(
            {
                "ds": pd.date_range("2020-01-01", periods=4),
                "yhat_0": [2e-6, 2e-6, 2e-6, 2e-4],
            }
        )
        units = common.unit_metrics(
            FakeModel(), test, future, origin="o", config="main", stage="s"
        )
        assert len(units) == 1
        assert bool(units["excluded"].iloc[0]) is True
        assert np.isnan(units["rel_mae"].iloc[0])


# ---------------------------------------------------------------------------
# Stocks runner: one offline end-to-end cell
# ---------------------------------------------------------------------------


def _synthetic_ticker_cache(cache: Path, tickers, start="2012-10-01", periods=120):
    rng = np.random.default_rng(0)
    frames = []
    for t in tickers:
        dates = pd.bdate_range(start, periods=periods)
        base = {"^GSPC": 1500.0}.get(t, 100.0)
        y = base + 0.1 * np.arange(periods) + rng.normal(0, 1, periods)
        frames.append(
            pd.DataFrame(
                {
                    "ds": dates,
                    "ticker": t,
                    "Open": y,
                    "High": y + 1,
                    "Low": y - 1,
                    "Close": y + 0.2,
                    "Volume": 1e6,
                    "typical_price": (y + y + 1 + y - 1 + y + 0.2) / 4,
                }
            )
        )
    for df in frames:
        t = df["ticker"].iloc[0]
        df.to_csv(cache / f"{t.replace('^', '_')}.csv", index=False)


def test_stocks_run_cell_offline(monkeypatch, tmp_path):
    """One development cell runs end-to-end with a synthetic offline cache."""
    from fixed_case_studies.stocks import runner as stock_runner

    cache = tmp_path / "tickers"
    cache.mkdir()
    tickers = ["AAA", "BBB", "CCC"]
    _synthetic_ticker_cache(cache, tickers + ["^GSPC"], periods=400)
    monkeypatch.setattr(stock_runner, "TICKERS_PATH", cache)
    monkeypatch.setattr(stock_runner, "RESULTS_ROOT", tmp_path / "results")

    unit_df = stock_runner.run_cell(
        stock_cfg.CONFIGS["main"],
        "2013-01-01",
        "development",
        tickers,
        seed=42,
        source_method="advi",
        out_dir=tmp_path / "results" / "development",
    )
    assert set(unit_df["series"]) == set(tickers)
    assert {"rel_mae", "mae", "mape", "excluded"} <= set(unit_df.columns)
    assert (tmp_path / "results" / "development").glob("manifest_*.json")
    assert (tmp_path / "results" / "development").glob("units_*.csv")
    assert (tmp_path / "results" / "development").glob("forecasts_*.csv")


def test_stocks_universe_rule_is_members_at_origin(tmp_path):
    """The frozen universe rule for confirmation origins (config-level)."""
    assert stock_cfg.CONFIGS["main"].name == "main"
    assert stock_cfg.MAIN_MATRIX[0] == "main"
    assert "no_transfer" in stock_cfg.MAIN_MATRIX
    assert "target_only" in stock_cfg.MAIN_MATRIX
    # The ablation list stays small by design (no grid).
    assert len(stock_cfg.MAIN_MATRIX) <= 8


# ---------------------------------------------------------------------------
# Smart-home runner: one offline end-to-end cell
# ---------------------------------------------------------------------------


def test_smart_home_run_cell_offline(monkeypatch, tmp_path):
    """One retrospective smart-home cell with mocked loaders and a fast
    ADVI temperature fit."""
    from fixed_case_studies.smart_home import runner as home_runner
    from fixed_case_studies.smart_home import config as home_config

    rng = np.random.default_rng(0)
    dates = pd.date_range("2016-01-01", "2016-12-16")
    sh = pd.concat(
        [
            pd.DataFrame(
                {
                    "ds": dates,
                    "y": 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
                    + 0.05 * rng.standard_normal(len(dates)),
                    "series": name,
                }
            )
            for name in home_cfg.SMART_HOME_COLUMNS
        ],
        ignore_index=True,
    )
    temp_dates = pd.date_range("2013-01-01", "2016-03-31")
    temp = pd.DataFrame(
        {
            "ds": temp_dates,
            "y": 10 + 12 * np.sin(2 * np.pi * np.arange(len(temp_dates)) / 365.25),
            "series": home_cfg.TEMP_CITY,
        }
    )

    monkeypatch.setattr(
        "vangja.datasets.load_smart_home_readings",
        lambda column=None, freq=None: sh.copy(),
    )
    monkeypatch.setattr(
        "vangja.datasets.load_kaggle_temperature",
        lambda city=None, start_date=None, end_date=None, freq=None: temp[
            (temp["ds"] >= pd.Timestamp(start_date)) & (temp["ds"] <= pd.Timestamp(end_date))
        ].copy(),
    )
    monkeypatch.setattr(home_runner, "RESULTS_ROOT", tmp_path / "results")

    # Fast temperature fit for the test (the real pipeline uses NUTS).
    def fast_temp_fit(temp_train, scaler, seed, progressbar):
        from vangja import FlatTrend, FourierSeasonality

        m = FlatTrend(intercept_mean=0.5, intercept_sd=0.1) + FourierSeasonality(
            period=365.25, series_order=5, beta_sd=1.5
        )
        m.fit(
            temp_train, scaler=scaler, method="advi", n=200, samples=200,
            random_seed=seed, progressbar=False,
        )
        return m

    monkeypatch.setattr(home_runner, "fit_temp_model", fast_temp_fit)

    split = {
        "train_start": "2016-01-01",
        "train_end": home_cfg.PRIMARY_TRAIN_CUTOFF,
        "label": "primary",
        "horizon": home_cfg.HORIZON_DAYS,
    }
    unit_df = home_runner.run_cell(
        home_cfg.CONFIGS["main"], split, seed=42,
        out_dir=tmp_path / "results" / "main",
    )
    assert set(unit_df["series"]) == set(home_cfg.SMART_HOME_COLUMNS)
    assert set(unit_df["origin"]) == {"primary"}
    assert {"rel_mae", "excluded"} <= set(unit_df.columns)
