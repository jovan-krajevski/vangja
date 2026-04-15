from pathlib import Path

import pandas as pd

from vangja import FourierSeasonality, LinearTrend
from vangja.datasets import get_sp500_tickers_for_range, load_stock_data
from vangja.utils import metrics


def rescale_dataset(
    smp_train: pd.DataFrame, stocks_train: pd.DataFrame, stocks_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    smp_min, smp_max = smp_train.iloc[-91:].y.min(), smp_train.iloc[-91:].y.max()
    for series in stocks_train["series"].unique():
        series_mask = stocks_train["series"] == series
        series_min, series_max = (
            stocks_train.loc[series_mask, "y"].min(),
            stocks_train.loc[series_mask, "y"].max(),
        )
        if series_max > series_min:
            stocks_train.loc[series_mask, "y"] = (
                stocks_train.loc[series_mask, "y"] - series_min
            ) / (series_max - series_min) * (smp_max - smp_min) + smp_min

            series_mask = stocks_test["series"] == series
            stocks_test.loc[series_mask, "y"] = (
                stocks_test.loc[series_mask, "y"] - series_min
            ) / (series_max - series_min) * (smp_max - smp_min) + smp_min

    return stocks_train, stocks_test


CT_PATH = Path(__file__).parent / "data/sp500_constituents"
TICKERS_PATH = Path(__file__).parent / "data/tickers"
RESULTS_FOLDER = Path(__file__).parent / "results" / "prophet"

RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


def run_prophet():
    # all months from 2013 to 2015
    start_dates = (
        pd.date_range(start="2013-01-01", end="2014-12-01", freq="MS")
        .strftime("%Y-%m-%d")
        .tolist()
    )

    tickers = get_sp500_tickers_for_range(
        "2012-09-01", "2015-01-01", cache_path=CT_PATH
    )

    for start_date in start_dates:
        print(f"\n=== Running Prophet baseline for {start_date} ===")

        results_file = RESULTS_FOLDER / f"results_{start_date}.csv"
        if results_file.exists():
            print(f"Skipping already computed date: {start_date}")
            continue

        stocks_train, stocks_test = load_stock_data(
            tickers,
            split_date=start_date,
            window_size=91,
            horizon_size=365,
            cache_path=TICKERS_PATH,
            interpolate=True,
        )

        smp_train, _ = load_stock_data(
            ["^GSPC"],
            split_date=start_date,
            window_size=91,
            horizon_size=365,
            cache_path=TICKERS_PATH,
            interpolate=True,
        )
        stocks_train, stocks_test = rescale_dataset(
            smp_train, stocks_train, stocks_test
        )

        result_rows = []

        for model_type in ["prophet_no_yearly", "prophet_yearly"]:
            print(f"Training {model_type}...")
            trend = LinearTrend(pool_type="individual")
            weekly = FourierSeasonality(
                period=7, series_order=3, beta_sd=1.5, pool_type="individual"
            )

            if model_type == "prophet_yearly":
                yearly = FourierSeasonality(
                    period=365.25, series_order=6, beta_sd=1.5, pool_type="individual"
                )
                model = trend + yearly + weekly
            else:
                model = trend + weekly

            model.fit(
                stocks_train,
                method="mapx",
                scale_mode="individual",
                sigma_pool_type="individual",
            )

            yhat = model.predict(horizon=365)
            model_metrics = metrics(stocks_test, yhat, pool_type="individual")

            for ts in model_metrics.index:
                result_rows.append(
                    {
                        "timeseries": ts,
                        "start_date": start_date,
                        "model": model_type,
                        **(model_metrics.loc[ts].to_dict()),
                    }
                )

        pd.DataFrame(result_rows).to_csv(results_file, index=False)


if __name__ == "__main__":
    run_prophet()
