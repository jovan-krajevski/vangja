import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vangja import FourierSeasonality, LinearTrend, UniformConstant
from vangja.datasets import get_sp500_tickers_for_range, load_stock_data
from vangja.utils import (
    metrics,
    plot_posterior_predictive,
    plot_prior_predictive,
    prior_predictive_coverage,
)

# read test_run from command line arguments, default to False if not provided
parser = argparse.ArgumentParser()
parser.add_argument("--test-run", action="store_true", help="Run in test mode")
args = parser.parse_args()

TEST_RUN = args.test_run

CT_PATH = Path("../data/sp500_constituents")
TICKERS_PATH = Path("../data/tickers")
RESULTS_FOLDER = Path(__file__).parent / "results" / "workflow"

RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

PLOT_IDX = 0


def plot_dfs(dfs: list[pd.DataFrame], titles: list[str], save_fns: list[str]):
    global PLOT_IDX
    fig, axes = plt.subplots(len(dfs), 1, figsize=(14, 4 * len(dfs)), sharex=False)

    if len(dfs) == 1:
        axes = [axes]

    for df, title, ax, save_fn in zip(dfs, titles, axes, save_fns):
        ax.plot(df["ds"], df["y"], linewidth=0.5, alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel("Typical Daily Price")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(RESULTS_FOLDER / f"{PLOT_IDX:02d}_{save_fn}.png")
    PLOT_IDX += 1


SPLIT_DATE = "2013-01-01"

smp_train, smp_test = load_stock_data(
    ["^GSPC"],
    split_date=SPLIT_DATE,
    window_size=365 if TEST_RUN else 4 * 365,
    horizon_size=365,
    cache_path=TICKERS_PATH,
    interpolate=True,
)

# We will only use a subset of the S&P 500 stocks to keep the model fitting time reasonable for this workflow example.
# In practice, you would likely want to use all available stocks.
tickers = get_sp500_tickers_for_range("2012-09-01", "2015-01-01", cache_path=CT_PATH)[
    ::100
]

stocks_train, stocks_test = load_stock_data(
    tickers,
    split_date=SPLIT_DATE,
    window_size=91,
    horizon_size=365,
    cache_path=TICKERS_PATH,
    interpolate=True,
)
print(f"Number of constituent stocks loaded: {len(stocks_train) / 91}")

smp_min, smp_max = smp_train.iloc[-91:].y.min(), smp_train.iloc[-91:].y.max()
for series in stocks_train["series"].unique():
    series_mask = stocks_train["series"] == series
    series_min, series_max = (
        stocks_train.loc[series_mask, "y"].min(),
        stocks_train.loc[series_mask, "y"].max(),
    )
    # Scale the stock series to match the S&P 500 range
    stocks_train.loc[series_mask, "y"] = (
        stocks_train.loc[series_mask, "y"] - series_min
    ) / (series_max - series_min) * (smp_max - smp_min) + smp_min

    series_mask = stocks_test["series"] == series
    # Scale the stock series to match the S&P 500 range
    stocks_test.loc[series_mask, "y"] = (
        stocks_test.loc[series_mask, "y"] - series_min
    ) / (series_max - series_min) * (smp_max - smp_min) + smp_min

# plot one stock to check the scaling
plot_dfs(
    [
        stocks_train[stocks_train["series"] == "AAPL"],
        smp_train[-91:],
        stocks_test[stocks_test["series"] == "AAPL"],
        smp_test,
    ],
    ["AAPL (scaled)", "S&P 500", "AAPL (scaled) - Test", "S&P 500 - Test"],
    ["scaled_stock_example"] * 4,
)

plot_dfs(
    [
        stocks_test[stocks_test["series"] == "AAPL"],
        stocks_test[stocks_test["series"] == "ADBE"],
        stocks_test[stocks_test["series"] == "MSFT"],
        smp_test,
    ],
    ["AAPL", "ADBE", "MSFT", "S&P 500"],
    ["aapl", "adbe", "msft", "smp"],
)


def create_smp_model():
    return LinearTrend(
        n_changepoints=25,
        changepoint_range=1,
        slope_sd=5.0,
        intercept_sd=5.0,
        delta_side="right",
        pool_type="complete",
        delta_pool_type="complete",
    ) ** (
        FourierSeasonality(
            period=365.25, series_order=6, beta_sd=5.0, pool_type="complete"
        )
        + FourierSeasonality(
            period=7, series_order=3, beta_sd=5.0, pool_type="complete"
        )
    )


smp_model = create_smp_model()
smp_model.fit(data=smp_train, scaler="maxabs", scale_mode="complete", method="mapx")
smp_prior_pred = smp_model.sample_prior_predictive()
smp_prior_pred_plot = plot_prior_predictive(
    smp_prior_pred, data=smp_model.data, show_hdi=True, show_ref_lines=True
)
smp_prior_pred_plot.figure.savefig(
    RESULTS_FOLDER / f"{PLOT_IDX:02d}_smp_prior_predictive.png"
)
PLOT_IDX += 1
print(f"Prior predictive coverage: {prior_predictive_coverage(smp_prior_pred)}")

smp_model = create_smp_model()
smp_model.fit(
    smp_train,
    scaler="maxabs",
    scale_mode="complete",
    method="nuts",
    samples=100 if TEST_RUN else 1000,
    tune=100 if TEST_RUN else 1000,
)
smp_summary = smp_model.convergence_summary()
smp_summary.to_csv(
    RESULTS_FOLDER / f"{PLOT_IDX:02d}_smp_convergence_summary.csv", index=False
)
PLOT_IDX += 1

smp_plot_trace = smp_model.plot_trace()
smp_plot_trace[0, 0].figure.tight_layout()
smp_plot_trace[0, 0].figure.savefig(
    RESULTS_FOLDER / f"{PLOT_IDX:02d}_smp_trace_plots.png"
)
PLOT_IDX += 1

smp_plot_energy = smp_model.plot_energy()
smp_plot_energy.figure.tight_layout()
smp_plot_energy.figure.savefig(RESULTS_FOLDER / f"{PLOT_IDX:02d}_smp_energy_plot.png")
PLOT_IDX += 1

smp_posterior_pred = smp_model.sample_posterior_predictive()
smp_posterior_pred_plot = plot_posterior_predictive(
    smp_posterior_pred, data=smp_model.data, show_hdi=True, show_ref_lines=True
)
smp_posterior_pred_plot.figure.savefig(
    RESULTS_FOLDER / f"{PLOT_IDX:02d}_smp_posterior_predictive.png"
)
PLOT_IDX += 1


def create_stocks_model():
    return LinearTrend(
        n_changepoints=25,
        # changepoint_range=1,
        slope_sd=5.0,
        intercept_sd=5.0,
        delta_side="right",
        pool_type="individual",
        delta_pool_type="complete",
        tune_method="parametric",
        shrinkage_strength=0,
    ) ** (
        UniformConstant(lower=-1, upper=1, pool_type="individual")
        ** FourierSeasonality(
            period=365.25,
            series_order=6,
            beta_sd=5.0,
            pool_type="individual",
            tune_method="parametric",
            shrinkage_strength=0,
            loss_factor_for_tune=1,
        )
        + FourierSeasonality(
            period=7, series_order=3, beta_sd=5.0, pool_type="individual"
        )
    )


stocks_model = create_stocks_model()
stocks_model.fit(
    stocks_train,
    scaler="maxabs",
    method="mapx",
    scale_mode="complete",
    sigma_pool_type="individual",
    t_scale_params=smp_model.t_scale_params,
    idata=smp_model.trace,
)
stocks_prior_pred = stocks_model.sample_prior_predictive()

for series_idx in range(stocks_model.n_groups):
    stocks_prior_pred_plot = plot_prior_predictive(
        stocks_prior_pred,
        data=stocks_model.data,
        series_idx=series_idx,
        group=stocks_model.group,
        show_hdi=True,
        show_ref_lines=True,
    )
    stocks_prior_pred_plot.figure.savefig(
        RESULTS_FOLDER / f"{PLOT_IDX:02d}_stocks_prior_predictive_{series_idx}.png"
    )
    PLOT_IDX += 1
    print(
        f"Prior predictive coverage (series {series_idx}): {prior_predictive_coverage(stocks_prior_pred, series_idx=series_idx, group=stocks_model.group)}"
    )

stocks_model = create_stocks_model()
stocks_model.fit(
    stocks_train,
    scaler="maxabs",
    method="nuts",
    samples=500 if TEST_RUN else 1000,
    tune=500 if TEST_RUN else 1000,
    scale_mode="complete",
    sigma_pool_type="individual",
    t_scale_params=smp_model.t_scale_params,
    idata=smp_model.trace,
)
future = stocks_model.predict_uncertainty(horizon=365)
stocks_model_metrics = metrics(stocks_test, future, pool_type="partial")
stocks_model_metrics.to_csv(
    RESULTS_FOLDER / f"{PLOT_IDX:02d}_stocks_model_metrics.csv", index=False
)
PLOT_IDX += 1

stocks_summary = stocks_model.convergence_summary()
stocks_summary.to_csv(
    RESULTS_FOLDER / f"{PLOT_IDX:02d}_stocks_convergence_summary.csv", index=False
)
PLOT_IDX += 1

stocks_trace_plot = stocks_model.plot_trace()
stocks_trace_plot[0, 0].figure.tight_layout()
stocks_trace_plot[0, 0].figure.savefig(
    RESULTS_FOLDER / f"{PLOT_IDX:02d}_stocks_trace_plots.png"
)
PLOT_IDX += 1

stocks_energy_plot = stocks_model.plot_energy()
stocks_energy_plot.figure.tight_layout()
stocks_energy_plot.figure.savefig(
    RESULTS_FOLDER / f"{PLOT_IDX:02d}_stocks_energy_plot.png"
)
PLOT_IDX += 1

stocks_posterior_pred = stocks_model.sample_posterior_predictive()
for series_idx in range(stocks_model.n_groups):
    plot_posterior_predictive(
        stocks_posterior_pred,
        data=stocks_model.data,
        series_idx=series_idx,
        group=stocks_model.group,
        show_hdi=True,
        show_ref_lines=True,
    ).figure.savefig(
        RESULTS_FOLDER / f"{PLOT_IDX:02d}_stocks_posterior_predictive_{series_idx}.png"
    )
    PLOT_IDX += 1

for series in stocks_model.groups_.values():
    stocks_model.plot(
        future,
        series=series,
        y_true=stocks_test[stocks_test["series"] == series],
        file_path=RESULTS_FOLDER / f"{PLOT_IDX:02d}_stocks_forecast_{series}.png",
    )
    PLOT_IDX += 1
