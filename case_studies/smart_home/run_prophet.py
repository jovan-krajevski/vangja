from pathlib import Path

import pandas as pd

from vangja import FlatTrend, FourierSeasonality
from vangja.datasets import load_smart_home_readings
from vangja.utils import metrics

RESULTS_FOLDER = Path(__file__).parent / "results" / "prophet"
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_FOLDER / "metrics.csv"


def run_prophet():
    smart_home_df = load_smart_home_readings(
        column=["Furnace 1 [kW]", "Furnace 2 [kW]", "Fridge [kW]", "Wine cellar [kW]"],
        freq="D",
    )

    train_df = smart_home_df[smart_home_df["ds"] < "2016-04-01"]
    test_df = smart_home_df[smart_home_df["ds"] >= "2016-04-01"]

    result_rows = []

    for model_type in ["prophet_no_yearly", "prophet_yearly"]:
        print(f"Training {model_type}...")
        trend = FlatTrend(intercept_mean=0.5, intercept_sd=0.1, pool_type="individual")
        weekly = FourierSeasonality(
            period=7, series_order=3, beta_sd=1.5, pool_type="individual"
        )

        if model_type == "prophet_yearly":
            yearly = FourierSeasonality(
                period=365.25, series_order=5, beta_sd=1.5, pool_type="individual"
            )
            model = trend + yearly + weekly
        else:
            model = trend + weekly

        model.fit(
            train_df,
            method="mapx",
            scale_mode="individual",
            sigma_pool_type="individual",
        )

        yhat = model.predict(horizon=365)
        model_metrics = metrics(test_df, yhat, pool_type="individual")

        for ts in model_metrics.index:
            result_rows.append(
                {
                    "timeseries": ts,
                    "model": model_type,
                    **(model_metrics.loc[ts].to_dict()),
                }
            )

    df_results = pd.DataFrame(result_rows)
    df_results.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    run_prophet()
