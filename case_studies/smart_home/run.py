from itertools import product
from pathlib import Path

import pandas as pd

from vangja import FlatTrend, FourierSeasonality, UniformConstant
from vangja.datasets import load_kaggle_temperature, load_smart_home_readings
from vangja.utils import metrics

RESULTS_FILE = Path(__file__).parent / "metrics.csv"

smart_home_df = load_smart_home_readings(
    column=["Furnace 1 [kW]", "Furnace 2 [kW]", "Fridge [kW]", "Wine cellar [kW]"],
    freq="D",
)
temp_df = load_kaggle_temperature(
    city="Boston",
    start_date="2016-01-01",
    end_date=smart_home_df["ds"].max().strftime("%Y-%m-%d"),
    freq="D",
)


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
# Train the temperature model
##########################################################################
train_temp_df, test_temp_df = build_dataset(smart_home_df=None, temp_df=temp_df)

temp_model = FlatTrend(intercept_sd=1) + FourierSeasonality(
    period=365.25, series_order=5, beta_sd=0.1
)
temp_model.fit(temp_df, scaler="minmax", method="nuts", samples=1000)

##########################################################################
# Train the smart home model
##########################################################################
uniform_constant = [True, False]
tune_method = ["parametric", "prior_from_idata"]
tune_loss_factor = [0, 1]
shrinkage_strength = [0, 1, 10, 100, 1000, 10000]

params = product(uniform_constant, tune_method, tune_loss_factor, shrinkage_strength)

train_tl_df, test_tl_df = build_dataset(smart_home_df=smart_home_df)
train_tl_h_df, test_tl_h_df = build_dataset(
    smart_home_df=smart_home_df, temp_df=temp_df
)

experiments: set[tuple[str, str, bool, str, int, int]] = set()

for uc, tm, tlf, shs in params:
    experiments.add(("without_temp_df", "individual", uc, tm, tlf, shs))
    if shs != 0:
        experiments.add(("with_temp_df", "partial", uc, tm, tlf, shs))

# Sort experiments to be deterministic
sorted_experiments = sorted(list(experiments))
processed_experiments = set()

if RESULTS_FILE.exists():
    existing_df = pd.read_csv(RESULTS_FILE)
    for _, row in existing_df.iterrows():
        processed_experiments.add(
            (
                row["use_temp_df"],
                row["hierarchical"],
                bool(row["uniform_constant"]),
                row["tune_method"],
                int(row["tune_loss_factor"]),
                int(row["shrinkage_strength"]),
            )
        )

for experiment in sorted_experiments:
    if experiment in processed_experiments:
        print(f"Skipping already processed experiment: {experiment}")
        continue

    use_temp_df, hierarchical, uc, tm, tlf, shs = experiment
    print(
        f"Training {hierarchical} model {use_temp_df} with uniform_constant={uc}, "
        f"tune_method={tm}, "
        f"tune_loss_factor={tlf}, "
        f"shrinkage_strength={shs}"
    )

    train_df, test_df = (
        (train_tl_df, test_tl_df) if not use_temp_df else (train_tl_h_df, test_tl_h_df)
    )

    trend = FlatTrend(intercept_sd=1)
    yearly = FourierSeasonality(
        period=365.25,
        series_order=5,
        pool_type=hierarchical,
        tune_method=tm,
        loss_factor_for_tune=tlf,
        shrinkage_strength=shs,
    )
    weekly = FourierSeasonality(
        period=7,
        series_order=3,
        beta_sd=1,
        pool_type=hierarchical,
        shrinkage_strength=shs,
    )
    constant = UniformConstant(lower=-1, upper=1, pool_type="partial")

    model = (trend + constant * yearly + weekly) if uc else (trend + yearly + weekly)
    print(model)
    model.fit(
        train_df,
        scaler="minmax",
        method="mapx",
        scale_mode="individual",
        sigma_pool_type="individual",
        t_scale_params=temp_model.t_scale_params,
        idata=temp_model.trace,
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
                "use_temp_df": use_temp_df,
                "hierarchical": hierarchical,
                "uniform_constant": uc,
                "tune_method": tm,
                "tune_loss_factor": tlf,
                "shrinkage_strength": shs,
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
