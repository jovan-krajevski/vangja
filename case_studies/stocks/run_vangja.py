from itertools import product
from pathlib import Path

import pandas as pd

from vangja import FourierSeasonality, LinearTrend, UniformConstant
from vangja.datasets import get_sp500_tickers_for_range, load_stock_data
from vangja.time_series import TimeSeriesModel
from vangja.utils import metrics

CT_PATH = Path(__file__).parent / "data/sp500_constituents"
TICKERS_PATH = Path(__file__).parent / "data/tickers"
RESULTS_FOLDER = Path(__file__).parent / "results" / "vangja"

RESULTS_FOLDER.mkdir(exist_ok=True)


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
tune_loss_factor = [0, 1]
lt_shrinkage_strength = [0, 1, 100, 10000]
fs_shrinkage_strength = [0, 1, 100, 10000]
slope_sd = [5.0]
intercept_sd = [5.0]
beta_sd = [1.0, 5.0, 10.0]
scalers = ["maxabs"]

# all months from 2013 to 2015
start_dates = (
    pd.date_range(start="2013-01-01", end="2015-12-01", freq="MS")
    .strftime("%Y-%m-%d")
    .tolist()
)

# Get the tickers for the S&P 500 for the specified date range
tickers = get_sp500_tickers_for_range("2012-09-01", "2015-01-01", cache_path=CT_PATH)

# Create a list of all combinations of hyperparameters
params = product(
    window_size,
    uniform_constant,
    tune_method,
    tune_loss_factor,
    lt_shrinkage_strength,
    fs_shrinkage_strength,
    slope_sd,
    intercept_sd,
    beta_sd,
    scalers,
)

experiments: set[
    tuple[str, str, str, int, bool, str, int, int, int, float, float, float, str]
] = set()

for (
    ws,
    uc,
    tm,
    tlf,
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
                (
                    "without_smp500",
                    "lt_individual",
                    "fs_individual",
                    ws,
                    uc,
                    tm,
                    tlf,
                    lt_ss,
                    fs_ss,
                    ssd,
                    icd,
                    bsd,
                    scl,
                )
            )
        else:
            experiments.add(
                (
                    "without_smp500",
                    "lt_individual",
                    "fs_partial",
                    ws,
                    uc,
                    tm,
                    tlf,
                    lt_ss,
                    fs_ss,
                    ssd,
                    icd,
                    bsd,
                    scl,
                )
            )
    else:
        if fs_ss == 0:
            experiments.add(
                (
                    "without_smp500",
                    "lt_partial",
                    "fs_individual",
                    ws,
                    uc,
                    tm,
                    tlf,
                    lt_ss,
                    fs_ss,
                    ssd,
                    icd,
                    bsd,
                    scl,
                )
            )
        else:
            experiments.add(
                (
                    "with_smp500",
                    "lt_partial",
                    "fs_partial",
                    ws,
                    uc,
                    tm,
                    tlf,
                    lt_ss,
                    fs_ss,
                    ssd,
                    icd,
                    bsd,
                    scl,
                )
            )

# Sort experiments to be deterministic
sorted_experiments = sorted(list(experiments))

for start_date in start_dates:
    processed_experiments: set[
        tuple[str, str, str, int, bool, str, int, int, int, float, float, float, str]
    ] = set()
    results_file = RESULTS_FOLDER / f"results_{start_date}.csv"
    mean_results_file = RESULTS_FOLDER / f"mean_results_{start_date}.csv"

    if mean_results_file.exists():
        existing_df = pd.read_csv(mean_results_file)
        for _, row in existing_df.iterrows():
            processed_experiments.add(
                (
                    row["use_smp500"],
                    row["lt_hierarchical"],
                    row["fs_hierarchical"],
                    int(row["window_size"]),
                    bool(row["uniform_constant"]),
                    row["tune_method"],
                    int(row["tune_loss_factor"]),
                    int(row["lt_shrinkage_strength"]),
                    int(row["fs_shrinkage_strength"]),
                    float(row["slope_sd"]),
                    float(row["intercept_sd"]),
                    float(row["beta_sd"]),
                    row["scaler"],
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

        (
            use_smp500,
            lt_hierarchical,
            fs_hierarchical,
            ws,
            uc,
            tm,
            tlf,
            lt_ss,
            fs_ss,
            ssd,
            icd,
            bsd,
            scl,
        ) = experiment
        print(
            f"Training LT {lt_hierarchical} FS {fs_hierarchical} model {use_smp500} with window_size={ws}, "
            f"uniform_constant={uc}, "
            f"tune_method={tm}, "
            f"tune_loss_factor={tlf}, "
            f"lt_shrinkage_strength={lt_ss}, "
            f"fs_shrinkage_strength={fs_ss}, "
            f"slope_sd={ssd}, "
            f"intercept_sd={icd}, "
            f"beta_sd={bsd}, "
            f"scaler={scl}"
        )

        smp_train, smp_test = load_stock_data(
            ["^GSPC"],
            split_date=start_date,
            window_size=ws,
            horizon_size=365,
            cache_path=TICKERS_PATH,
            interpolate=True,
        )
        train_df, test_df = rescale_dataset(smp_train, stocks_train, stocks_test)
        if use_smp500 == "with_smp500":
            train_df = pd.concat([train_df, smp_train], ignore_index=True)
            test_df = pd.concat([test_df, smp_test], ignore_index=True)

        smp_model = smp_models.get((ws, ssd, icd, bsd, scl), None)
        if smp_model is None:
            smp_model = LinearTrend(
                n_changepoints=25,
                changepoint_range=1,
                slope_sd=ssd,
                intercept_sd=icd,
                delta_side="right",
                pool_type="complete",
                delta_pool_type="complete",
                tune_method=None,
                delta_tune_method=None,
            ) ** (
                FourierSeasonality(
                    period=365.25,
                    series_order=6,
                    beta_sd=bsd,
                    pool_type="complete",
                    tune_method=None,
                )
                + FourierSeasonality(
                    period=7,
                    series_order=3,
                    beta_sd=bsd,
                    pool_type="complete",
                    tune_method=None,
                )
            )

            smp_model.fit(
                data=smp_train,
                scaler=scl,
                scale_mode="complete",
                sigma_pool_type="individual",
                method="advi",
            )
            smp_models[(ws, ssd, icd, bsd, scl)] = smp_model
        else:
            smp_model = smp_models[(ws, ssd, icd, bsd, scl)]

        trend = LinearTrend(
            n_changepoints=25,
            changepoint_range=1,
            slope_sd=ssd,
            intercept_sd=icd,
            delta_side="right",
            pool_type="partial" if lt_hierarchical == "lt_partial" else "individual",
            delta_pool_type="complete",
            tune_method=tm,
            delta_tune_method=None,
            shrinkage_strength=lt_ss,
        )
        yearly = FourierSeasonality(
            period=365.25,
            series_order=6,
            beta_sd=bsd,
            pool_type="partial" if fs_hierarchical == "fs_partial" else "individual",
            tune_method=tm,
            loss_factor_for_tune=tlf,
            shrinkage_strength=fs_ss,
        )
        weekly = FourierSeasonality(
            period=7,
            series_order=3,
            beta_sd=bsd,
            pool_type="individual",
            tune_method=None,
        )
        constant = UniformConstant(
            lower=-1, upper=1, pool_type="individual", tune_method=None
        )
        model = (
            (trend ** (constant**yearly + weekly))
            if uc
            else (trend ** (yearly + weekly))
        )
        print(model)

        model.fit(
            train_df,
            scaler=scl,
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
                    "use_smp500": use_smp500,
                    "lt_hierarchical": lt_hierarchical,
                    "fs_hierarchical": fs_hierarchical,
                    "window_size": ws,
                    "uniform_constant": uc,
                    "tune_method": tm,
                    "tune_loss_factor": tlf,
                    "lt_shrinkage_strength": lt_ss,
                    "fs_shrinkage_strength": fs_ss,
                    "slope_sd": ssd,
                    "intercept_sd": icd,
                    "beta_sd": bsd,
                    "scaler": scl,
                    **model_metrics.loc[ts].to_dict(),
                }
            )

        result_df = pd.DataFrame(result_rows)
        write_header = not results_file.exists()
        result_df.to_csv(results_file, mode="a", header=write_header, index=False)

        mean_result_row = {
            "start_date": start_date,
            "use_smp500": use_smp500,
            "lt_hierarchical": lt_hierarchical,
            "fs_hierarchical": fs_hierarchical,
            "window_size": ws,
            "uniform_constant": uc,
            "tune_method": tm,
            "tune_loss_factor": tlf,
            "lt_shrinkage_strength": lt_ss,
            "fs_shrinkage_strength": fs_ss,
            "slope_sd": ssd,
            "intercept_sd": icd,
            "beta_sd": bsd,
            "scaler": scl,
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
