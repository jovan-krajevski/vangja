from itertools import product
from pathlib import Path

import pandas as pd

from vangja import FlatTrend, FourierSeasonality, UniformConstant
from vangja.datasets import load_kaggle_temperature, load_smart_home_readings
from vangja.utils import metrics

RESULTS_FILE = Path(__file__).parent / "ts_metrics.csv"

# Define hyperparameter search space
uniform_constant = [True, False]
shrinkage_strength = [1, 10, 100, 1000, 10000]
intercept_sd = [0.1, 0.5, 1.0]
beta_sd = [0.5, 1.0, 1.5]
scalers = ["maxabs", "minmax"]

smart_home_df = load_smart_home_readings(
    column=["Furnace 1 [kW]", "Furnace 2 [kW]", "Fridge [kW]", "Wine cellar [kW]"],
    freq="D",
)
temp_dfs = {
    start_date: load_kaggle_temperature(
        city="Boston",
        start_date=start_date,
        end_date=smart_home_df["ds"].max().strftime("%Y-%m-%d"),
        freq="D",
    )
    for start_date in ["2015-01-01", "2014-01-01", "2013-01-01"]
}


def build_dataset(
    smart_home_df: pd.DataFrame | None = None, temp_df: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if smart_home_df is not None and temp_df is None:
        df = smart_home_df
    elif smart_home_df is None and temp_df is not None:
        df = temp_df
    elif smart_home_df is not None and temp_df is not None:
        df = pd.concat([smart_home_df, temp_df], ignore_index=True)
    else:
        raise ValueError(
            "Unexpected case: both smart_home_df and temp_df are not None."
        )

    train_df = df[df["ds"] < "2016-04-01"]
    test_df = df[df["ds"] >= "2016-04-01"]

    return train_df, test_df


##########################################################################
# Train the smart home model
##########################################################################
params = product(
    temp_dfs.keys(),
    uniform_constant,
    shrinkage_strength,
    intercept_sd,
    beta_sd,
    scalers,
)

train_tl_df, test_tl_df = build_dataset(smart_home_df=smart_home_df)
tl_h_dfs = {
    start_date: build_dataset(smart_home_df=smart_home_df, temp_df=temp_df)
    for start_date, temp_df in temp_dfs.items()
}

experiments: set[tuple[str, bool, int, float, float, str]] = set()

for start_date, uc, shs, isd, bsd, scaler in params:
    experiments.add((start_date, uc, shs, isd, bsd, scaler))

# Sort experiments to be deterministic
sorted_experiments = sorted(list(experiments))
processed_experiments = set()

if RESULTS_FILE.exists():
    existing_df = pd.read_csv(RESULTS_FILE)
    for _, row in existing_df.iterrows():
        processed_experiments.add(
            (
                row["start_date"],
                bool(row["uniform_constant"]),
                int(row["shrinkage_strength"]),
                float(row["intercept_sd"]),
                float(row["beta_sd"]),
                row["scaler"],
            )
        )

for i, experiment in enumerate(sorted_experiments):
    print(f"\n=== Running Experiment {i+1}/{len(sorted_experiments)} ===")
    if experiment in processed_experiments:
        print(f"Skipping already processed experiment: {experiment}")
        continue

    start_date, uc, shs, isd, bsd, scaler = experiment
    print(
        f"Training model with uniform_constant={uc}, "
        f"shrinkage_strength={shs}, "
        f"intercept_sd={isd}, "
        f"beta_sd={bsd}, "
        f"scaler={scaler}"
    )

    train_df, test_df = tl_h_dfs[start_date]

    trend = FlatTrend(intercept_mean=0.5, intercept_sd=isd, pool_type="individual")
    yearly = FourierSeasonality(
        period=365.25,
        series_order=5,
        beta_sd=bsd,
        pool_type="partial",
        shrinkage_strength=shs,
    )
    weekly = FourierSeasonality(
        period=7,
        series_order=3,
        beta_sd=bsd,
        pool_type="partial",
        shrinkage_strength=shs,
    )
    constant = UniformConstant(
        lower=-1, upper=1, pool_type="partial", shrinkage_strength=shs
    )

    model = (trend + constant * yearly + weekly) if uc else (trend + yearly + weekly)
    print(model)
    model.fit(
        train_df,
        scaler=scaler,
        method="mapx",
        scale_mode="individual",
        sigma_pool_type="individual",
    )
    yhat = model.predict(horizon=365)
    model_metrics = metrics(test_df, yhat, pool_type="individual")
    print(model_metrics)

    # Save metrics
    result_rows = []
    for ts in model_metrics.index:
        result_rows.append(
            {
                "timeseries": ts,
                "start_date": start_date,
                "uniform_constant": uc,
                "shrinkage_strength": shs,
                "intercept_sd": isd,
                "beta_sd": bsd,
                "scaler": scaler,
                **(model_metrics.loc[ts].to_dict()),
            }
        )

    result_df = pd.DataFrame(result_rows)
    write_header = not RESULTS_FILE.exists()
    result_df.to_csv(RESULTS_FILE, mode="a", header=write_header, index=False)

if RESULTS_FILE.exists():
    all_results = pd.read_csv(RESULTS_FILE)
    if not all_results.empty and "mape" in all_results.columns:
        tss = all_results["timeseries"].unique()
        for ts in tss:
            ts_results = all_results[all_results["timeseries"] == ts]
            best_ts_row = ts_results.loc[ts_results["mape"].idxmin()]
            print(f"\n=== Best Model Configuration for {ts} (Lowest MAPE) ===")
            print(best_ts_row)
