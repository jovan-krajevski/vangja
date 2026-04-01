from itertools import product
from pathlib import Path
from typing import Iterable, NamedTuple

import pandas as pd

from vangja import FourierSeasonality, LinearTrend, UniformConstant
from vangja.datasets import get_sp500_tickers_for_range, load_stock_data
from vangja.time_series import TimeSeriesModel
from vangja.utils import metrics

CT_PATH = Path(__file__).parent / "data/sp500_constituents"
TICKERS_PATH = Path(__file__).parent / "data/tickers"
RESULTS_FOLDER = Path(__file__).parent / "results" / "vangja"

RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)


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
        # Scale the train series to match the S&P 500 range
        stocks_train.loc[series_mask, "y"] = (
            stocks_train.loc[series_mask, "y"] - series_min
        ) / (series_max - series_min) * (smp_max - smp_min) + smp_min

        series_mask = stocks_test["series"] == series
        # Scale the test series to match the S&P 500 range
        stocks_test.loc[series_mask, "y"] = (
            stocks_test.loc[series_mask, "y"] - series_min
        ) / (series_max - series_min) * (smp_max - smp_min) + smp_min

    return stocks_train, stocks_test


# Define hyperparameter search space
window_size = [2 * 365, 3 * 365, 4 * 365]
uniform_constant = [True, False]
tune_method = ["parametric", "prior_from_idata"]
lt_tune_loss_factor = [0, 1]
fs_tune_loss_factor = [0, 1]
# shrinage_strength 0 is used to denote individual pooling, since division by 0 is not possible
lt_shrinkage_strength = [0, 1, 100, 10000]
fs_shrinkage_strength = [0, 1, 100, 10000]
slope_sd = [5.0]
intercept_sd = [5.0]
beta_sd = [1.0, 5.0, 10.0]
scalers = ["maxabs"]

# all months from 2013 to 2015
start_dates = (
    pd.date_range(start="2013-01-01", end="2014-12-01", freq="MS")
    .strftime("%Y-%m-%d")
    .tolist()
)

# Get the tickers for the S&P 500 for the specified date range
tickers = get_sp500_tickers_for_range("2012-09-01", "2015-01-01", cache_path=CT_PATH)


# Create a list of all combinations of hyperparameters
class Experiment(NamedTuple):
    use_smp500: str
    lt_hierarchical: str
    fs_hierarchical: str
    window_size: int
    uniform_constant: bool
    tune_method: str
    lt_tune_loss_factor: int
    fs_tune_loss_factor: int
    lt_shrinkage_strength: int
    fs_shrinkage_strength: int
    slope_sd: float
    intercept_sd: float
    beta_sd: float
    scaler: str


params = product(
    window_size,
    uniform_constant,
    tune_method,
    lt_tune_loss_factor,
    fs_tune_loss_factor,
    lt_shrinkage_strength,
    fs_shrinkage_strength,
    slope_sd,
    intercept_sd,
    beta_sd,
    scalers,
)

experiments: set[Experiment] = set()


for (
    ws,
    uc,
    tm,
    lt_tlf,
    fs_tlf,
    lt_ss,
    fs_ss,
    ssd,
    icd,
    bsd,
    scl,
) in params:
    if lt_ss == 0:
        if fs_ss == 0:
            experiments.add(
                Experiment(
                    use_smp500="without_smp500",
                    lt_hierarchical="lt_individual",
                    fs_hierarchical="fs_individual",
                    window_size=ws,
                    uniform_constant=uc,
                    tune_method=tm,
                    lt_tune_loss_factor=lt_tlf,
                    fs_tune_loss_factor=fs_tlf,
                    lt_shrinkage_strength=lt_ss,
                    fs_shrinkage_strength=fs_ss,
                    slope_sd=ssd,
                    intercept_sd=icd,
                    beta_sd=bsd,
                    scaler=scl,
                )
            )
        else:
            experiments.add(
                Experiment(
                    use_smp500="without_smp500",
                    lt_hierarchical="lt_individual",
                    fs_hierarchical="fs_partial",
                    window_size=ws,
                    uniform_constant=uc,
                    tune_method=tm,
                    lt_tune_loss_factor=lt_tlf,
                    fs_tune_loss_factor=fs_tlf,
                    lt_shrinkage_strength=lt_ss,
                    fs_shrinkage_strength=fs_ss,
                    slope_sd=ssd,
                    intercept_sd=icd,
                    beta_sd=bsd,
                    scaler=scl,
                )
            )
    else:
        if fs_ss == 0:
            experiments.add(
                Experiment(
                    use_smp500="without_smp500",
                    lt_hierarchical="lt_partial",
                    fs_hierarchical="fs_individual",
                    window_size=ws,
                    uniform_constant=uc,
                    tune_method=tm,
                    lt_tune_loss_factor=lt_tlf,
                    fs_tune_loss_factor=fs_tlf,
                    lt_shrinkage_strength=lt_ss,
                    fs_shrinkage_strength=fs_ss,
                    slope_sd=ssd,
                    intercept_sd=icd,
                    beta_sd=bsd,
                    scaler=scl,
                )
            )
        else:
            experiments.add(
                Experiment(
                    use_smp500="with_smp500",
                    lt_hierarchical="lt_partial",
                    fs_hierarchical="fs_partial",
                    window_size=ws,
                    uniform_constant=uc,
                    tune_method=tm,
                    lt_tune_loss_factor=lt_tlf,
                    fs_tune_loss_factor=fs_tlf,
                    lt_shrinkage_strength=lt_ss,
                    fs_shrinkage_strength=fs_ss,
                    slope_sd=ssd,
                    intercept_sd=icd,
                    beta_sd=bsd,
                    scaler=scl,
                )
            )

# Sort experiments to be deterministic
sorted_experiments = sorted(list(experiments))

for start_date in start_dates:
    processed_experiments: set[Experiment] = set()
    results_file = RESULTS_FOLDER / f"results_{start_date}.csv"
    mean_results_file = RESULTS_FOLDER / f"mean_results_{start_date}.csv"

    if mean_results_file.exists():
        existing_df = pd.read_csv(mean_results_file)
        for _, row in existing_df.iterrows():
            processed_experiments.add(
                Experiment(
                    use_smp500=row["use_smp500"],
                    lt_hierarchical=row["lt_hierarchical"],
                    fs_hierarchical=row["fs_hierarchical"],
                    window_size=int(row["window_size"]),
                    uniform_constant=bool(row["uniform_constant"]),
                    tune_method=row["tune_method"],
                    lt_tune_loss_factor=int(row["lt_tune_loss_factor"]),
                    fs_tune_loss_factor=int(row["fs_tune_loss_factor"]),
                    lt_shrinkage_strength=int(row["lt_shrinkage_strength"]),
                    fs_shrinkage_strength=int(row["fs_shrinkage_strength"]),
                    slope_sd=float(row["slope_sd"]),
                    intercept_sd=float(row["intercept_sd"]),
                    beta_sd=float(row["beta_sd"]),
                    scaler=row["scaler"],
                )
            )

    stocks_train, stocks_test = load_stock_data(
        tickers,
        split_date=start_date,
        window_size=91,
        horizon_size=365,
        cache_path=TICKERS_PATH,
        interpolate=True,
    )

    # window_size, slope_sd, intercept_sd, beta_sd, scl
    smp_models: dict[tuple[int, float, float, float, str], TimeSeriesModel] = {}

    for i, experiment in enumerate(sorted_experiments):
        print(f"\n=== Running Experiment {i+1}/{len(sorted_experiments)} ===")
        if experiment in processed_experiments:
            print(f"Skipping already processed experiment: {experiment}")
            continue

        print(
            f"Training LT {experiment.lt_hierarchical} FS {experiment.fs_hierarchical} model {experiment.use_smp500} with window_size={experiment.window_size}, "
            f"uniform_constant={experiment.uniform_constant}, "
            f"tune_method={experiment.tune_method}, "
            f"lt_tune_loss_factor={experiment.lt_tune_loss_factor}, "
            f"fs_tune_loss_factor={experiment.fs_tune_loss_factor}, "
            f"lt_shrinkage_strength={experiment.lt_shrinkage_strength}, "
            f"fs_shrinkage_strength={experiment.fs_shrinkage_strength}, "
            f"slope_sd={experiment.slope_sd}, "
            f"intercept_sd={experiment.intercept_sd}, "
            f"beta_sd={experiment.beta_sd}, "
            f"scaler={experiment.scaler}"
        )

        smp_train, smp_test = load_stock_data(
            ["^GSPC"],
            split_date=start_date,
            window_size=experiment.window_size,
            horizon_size=365,
            cache_path=TICKERS_PATH,
            interpolate=True,
        )
        train_df, test_df = rescale_dataset(smp_train, stocks_train, stocks_test)
        if experiment.use_smp500 == "with_smp500":
            train_df = pd.concat([train_df, smp_train], ignore_index=True)
            test_df = pd.concat([test_df, smp_test], ignore_index=True)

        smp_model = smp_models.get(
            (
                experiment.window_size,
                experiment.slope_sd,
                experiment.intercept_sd,
                experiment.beta_sd,
                experiment.scaler,
            ),
            None,
        )
        if smp_model is None:
            smp_model = LinearTrend(
                n_changepoints=25,
                slope_sd=experiment.slope_sd,
                intercept_sd=experiment.intercept_sd,
                delta_side="right",
                pool_type="complete",
                delta_pool_type="complete",
                tune_method=None,
                delta_tune_method=None,
            ) ** (
                FourierSeasonality(
                    period=365.25,
                    series_order=6,
                    beta_sd=experiment.beta_sd,
                    pool_type="complete",
                    tune_method=None,
                )
                + FourierSeasonality(
                    period=7,
                    series_order=3,
                    beta_sd=experiment.beta_sd,
                    pool_type="complete",
                    tune_method=None,
                )
            )

            smp_model.fit(
                data=smp_train,
                scaler=experiment.scaler,
                scale_mode="complete",
                sigma_pool_type="individual",
                method="advi",
            )
            smp_models[
                (
                    experiment.window_size,
                    experiment.slope_sd,
                    experiment.intercept_sd,
                    experiment.beta_sd,
                    experiment.scaler,
                )
            ] = smp_model
        else:
            smp_model = smp_models[
                (
                    experiment.window_size,
                    experiment.slope_sd,
                    experiment.intercept_sd,
                    experiment.beta_sd,
                    experiment.scaler,
                )
            ]

        trend = LinearTrend(
            n_changepoints=25 if experiment.use_smp500 == "with_smp500" else 0,
            slope_sd=experiment.slope_sd,
            intercept_sd=experiment.intercept_sd,
            delta_side="right",
            pool_type=(
                "partial"
                if experiment.lt_hierarchical == "lt_partial"
                else "individual"
            ),
            delta_pool_type="complete",
            tune_method=experiment.tune_method,
            delta_tune_method=None,
            loss_factor_for_tune=experiment.lt_tune_loss_factor,
            shrinkage_strength=lt_ss,
        )
        yearly = FourierSeasonality(
            period=365.25,
            series_order=6,
            beta_sd=experiment.beta_sd,
            pool_type=(
                "partial"
                if experiment.fs_hierarchical == "fs_partial"
                else "individual"
            ),
            tune_method=experiment.tune_method,
            loss_factor_for_tune=experiment.fs_tune_loss_factor,
            shrinkage_strength=fs_ss,
        )
        weekly = FourierSeasonality(
            period=7,
            series_order=3,
            beta_sd=experiment.beta_sd,
            pool_type="individual",
            tune_method=None,
        )
        constant = UniformConstant(
            lower=-1, upper=1, pool_type="individual", tune_method=None
        )
        model = (
            (trend ** (constant**yearly + weekly))
            if experiment.uniform_constant == "uniform_constant"
            else (trend ** (yearly + weekly))
        )
        print(model)

        model.fit(
            train_df,
            scaler=experiment.scaler,
            method="mapx",
            scale_mode="complete",
            sigma_pool_type="individual",
            t_scale_params=smp_model.t_scale_params,
            idata=smp_model.trace,
        )
        yhat = model.predict(horizon=365)
        model_metrics = metrics(test_df, yhat, pool_type="individual")
        print(model_metrics.mean()["mape"])

        # Save metrics
        result_rows = []
        for ts in model_metrics.index:
            result_rows.append(
                {
                    "timeseries": ts,
                    "start_date": start_date,
                    "use_smp500": experiment.use_smp500,
                    "lt_hierarchical": experiment.lt_hierarchical,
                    "fs_hierarchical": experiment.fs_hierarchical,
                    "window_size": experiment.window_size,
                    "uniform_constant": experiment.uniform_constant,
                    "tune_method": experiment.tune_method,
                    "lt_tune_loss_factor": experiment.lt_tune_loss_factor,
                    "fs_tune_loss_factor": experiment.fs_tune_loss_factor,
                    "lt_shrinkage_strength": lt_ss,
                    "fs_shrinkage_strength": fs_ss,
                    "slope_sd": experiment.slope_sd,
                    "intercept_sd": experiment.intercept_sd,
                    "beta_sd": experiment.beta_sd,
                    "scaler": experiment.scaler,
                    **model_metrics.loc[ts].to_dict(),
                }
            )

        result_df = pd.DataFrame(result_rows)
        write_header = not results_file.exists()
        result_df.to_csv(results_file, mode="a", header=write_header, index=False)

        mean_result_row = {
            "start_date": start_date,
            "use_smp500": experiment.use_smp500,
            "lt_hierarchical": experiment.lt_hierarchical,
            "fs_hierarchical": experiment.fs_hierarchical,
            "window_size": experiment.window_size,
            "uniform_constant": experiment.uniform_constant,
            "tune_method": experiment.tune_method,
            "lt_tune_loss_factor": experiment.lt_tune_loss_factor,
            "fs_tune_loss_factor": experiment.fs_tune_loss_factor,
            "lt_shrinkage_strength": lt_ss,
            "fs_shrinkage_strength": fs_ss,
            "slope_sd": experiment.slope_sd,
            "intercept_sd": experiment.intercept_sd,
            "beta_sd": experiment.beta_sd,
            "scaler": experiment.scaler,
            **model_metrics.mean().to_dict(),
        }
        mean_result_df = pd.DataFrame([mean_result_row])
        write_header = not mean_results_file.exists()
        mean_result_df.to_csv(
            mean_results_file, mode="a", header=write_header, index=False
        )

    if mean_results_file.exists():
        all_mean_results = pd.read_csv(mean_results_file)
        if not all_mean_results.empty and "mape" in all_mean_results.columns:
            best_row = all_mean_results.loc[all_mean_results["mape"].idxmin()]
            print(f"\nBest Hyperparameters for start_date {start_date}:")
            print(best_row)
