"""Classical baseline models for smart home energy forecasting.

Compares Vangja's transfer learning against classical time series methods:

- **ARIMA** (Auto-ARIMA via statsmodels)
- **Exponential Smoothing / Holt-Winters** (statsmodels)
- **Seasonal Naive** (repeat last observed seasonal cycle)

Each model is fit per-series on the same train split, then evaluated on
the same test set used by the ablation study.

Usage:
    python case_studies/smart_home/classical_baselines.py

Outputs are saved to ``case_studies/smart_home/results_classical/``.

Dependencies:
    pip install vangja[reproducibility]
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from vangja.datasets import load_smart_home_readings

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).parent / "results_classical"
OUTPUT_DIR.mkdir(exist_ok=True)

SMART_HOME_COLUMNS = [
    "Furnace 1 [kW]",
    "Furnace 2 [kW]",
    "Fridge [kW]",
    "Wine cellar [kW]",
]
TRAIN_CUTOFF = "2016-04-01"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data():
    """Load smart home data, split train/test."""
    print("Loading smart home data...")
    df = load_smart_home_readings(column=SMART_HOME_COLUMNS, freq="D")
    train = df[df["ds"] < TRAIN_CUTOFF].copy()
    test = df[df["ds"] >= TRAIN_CUTOFF].copy()
    print(f"  Train: {len(train)} rows, Test: {len(test)} rows")
    print(f"  Series: {train['series'].unique().tolist()}")
    return train, test


# ---------------------------------------------------------------------------
# Seasonal Naive
# ---------------------------------------------------------------------------
def seasonal_naive_forecast(
    train_series: pd.Series, horizon: int, season_length: int = 7
) -> np.ndarray:
    """Repeat the last ``season_length`` values cyclically.

    Parameters
    ----------
    train_series : pd.Series
        Training y values.
    horizon : int
        Number of steps to forecast.
    season_length : int
        Seasonal period (7 for weekly).

    Returns
    -------
    np.ndarray
    """
    last_season = train_series.values[-season_length:]
    reps = horizon // season_length + 1
    return np.tile(last_season, reps)[:horizon]


# ---------------------------------------------------------------------------
# Model fitting wrappers
# ---------------------------------------------------------------------------
def fit_arima(
    train_y: np.ndarray,
    horizon: int,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 0, 1, 7),
):
    """Fit ARIMA and return forecast + elapsed time.

    Parameters
    ----------
    train_y : np.ndarray
    horizon : int
    order : tuple
    seasonal_order : tuple

    Returns
    -------
    tuple : (forecast array, elapsed seconds)
    """
    t0 = time.time()
    try:
        model = ARIMA(
            train_y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(method_kwargs={"maxiter": 200})
        forecast = result.forecast(steps=horizon)
    except Exception:
        # Fallback to simpler order
        try:
            model = ARIMA(
                train_y,
                order=(1, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit()
            forecast = result.forecast(steps=horizon)
        except Exception:
            forecast = np.full(horizon, np.mean(train_y))
    return forecast, time.time() - t0


def fit_holt_winters(train_y: np.ndarray, horizon: int, seasonal_periods: int = 7):
    """Fit Holt-Winters Exponential Smoothing and return forecast.

    Parameters
    ----------
    train_y : np.ndarray
    horizon : int
    seasonal_periods : int

    Returns
    -------
    tuple : (forecast array, elapsed seconds)
    """
    t0 = time.time()
    try:
        model = ExponentialSmoothing(
            train_y,
            seasonal_periods=seasonal_periods,
            trend="add",
            seasonal="add",
            initialization_method="estimated",
        )
        result = model.fit(optimized=True)
        forecast = result.forecast(steps=horizon)
    except Exception:
        # Fallback without seasonality
        try:
            model = ExponentialSmoothing(
                train_y,
                trend="add",
                initialization_method="estimated",
            )
            result = model.fit(optimized=True)
            forecast = result.forecast(steps=horizon)
        except Exception:
            forecast = np.full(horizon, np.mean(train_y))
    return forecast, time.time() - t0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute metrics for a single series forecast.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray

    Returns
    -------
    dict
    """
    # Handle length mismatches
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all_forecasts(results: list[dict], train_df, test_df, save_dir):
    """Create a grid of prediction plots, one row per series, one col per model.

    Parameters
    ----------
    results : list[dict]
        Each dict has 'model', 'series', 'forecast', 'test_y', 'test_ds'.
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    save_dir : Path
    """
    series_list = sorted(set(r["series"] for r in results))
    models_list = sorted(set(r["model"] for r in results))

    fig, axes = plt.subplots(
        len(series_list),
        len(models_list),
        figsize=(6 * len(models_list), 4 * len(series_list)),
        squeeze=False,
    )

    for row, series_name in enumerate(series_list):
        for col, model_name in enumerate(models_list):
            ax = axes[row][col]
            match = [
                r
                for r in results
                if r["series"] == series_name and r["model"] == model_name
            ]
            if not match:
                ax.set_visible(False)
                continue
            r = match[0]
            series_train = train_df[train_df["series"] == series_name]
            ax.scatter(
                series_train["ds"],
                series_train["y"],
                s=2,
                color="C2",
                alpha=0.4,
                label="Train",
            )
            ax.scatter(
                r["test_ds"], r["test_y"], s=3, color="C1", label="Test", zorder=5
            )
            ax.plot(
                r["test_ds"],
                r["forecast"][: len(r["test_ds"])],
                color="C0",
                alpha=0.8,
                label="Pred",
            )
            if row == 0:
                ax.set_title(model_name, fontsize=10)
            if col == 0:
                ax.set_ylabel(series_name, fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "all_forecasts.png", dpi=120)
    plt.close(fig)


def plot_metrics_comparison(metrics_df, save_dir):
    """Bar chart comparing RMSE across models.

    Parameters
    ----------
    metrics_df : pd.DataFrame
    save_dir : Path
    """
    # Aggregate RMSE mean across series per model
    agg = metrics_df.groupby("model")["rmse"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(agg)), agg.values)
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(agg.index)
    ax.set_xlabel("Mean RMSE across series")
    ax.set_title("Classical Models — Mean RMSE")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(save_dir / "classical_rmse.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run classical baselines."""
    train_df, test_df = load_data()

    series_list = train_df["series"].unique()
    all_metrics: list[dict] = []
    all_results: list[dict] = []

    model_configs = {
        "Seasonal Naive (7)": "snaive7",
        "Seasonal Naive (30)": "snaive30",
        "ARIMA(1,1,1)(1,0,1,7)": "arima_7",
        "ARIMA(1,1,1)(0,0,0,0)": "arima_noseas",
        "Holt-Winters (add, 7)": "hw_7",
        "Holt-Winters (add, 30)": "hw_30",
    }

    for series_name in series_list:
        print(f"\n--- Series: {series_name} ---")
        s_train = train_df[train_df["series"] == series_name].sort_values("ds")
        s_test = test_df[test_df["series"] == series_name].sort_values("ds")

        train_y = s_train["y"].values
        test_y = s_test["y"].values
        test_ds = s_test["ds"].values
        horizon = len(test_y)

        for model_name, model_code in model_configs.items():
            print(f"  {model_name}...", end=" ")
            t0 = time.time()

            if model_code == "snaive7":
                forecast = seasonal_naive_forecast(s_train["y"], horizon, 7)
                elapsed = time.time() - t0
            elif model_code == "snaive30":
                forecast = seasonal_naive_forecast(s_train["y"], horizon, 30)
                elapsed = time.time() - t0
            elif model_code == "arima_7":
                forecast, elapsed = fit_arima(
                    train_y,
                    horizon,
                    order=(1, 1, 1),
                    seasonal_order=(1, 0, 1, 7),
                )
            elif model_code == "arima_noseas":
                forecast, elapsed = fit_arima(
                    train_y,
                    horizon,
                    order=(1, 1, 1),
                    seasonal_order=(0, 0, 0, 0),
                )
            elif model_code == "hw_7":
                forecast, elapsed = fit_holt_winters(train_y, horizon, 7)
            elif model_code == "hw_30":
                forecast, elapsed = fit_holt_winters(train_y, horizon, 30)
            else:
                continue

            m = evaluate_forecast(test_y, forecast)
            m["model"] = model_name
            m["series"] = series_name
            m["elapsed"] = elapsed
            all_metrics.append(m)

            all_results.append(
                {
                    "model": model_name,
                    "series": series_name,
                    "forecast": forecast,
                    "test_y": test_y,
                    "test_ds": test_ds,
                }
            )

            print(f"RMSE={m['rmse']:.4f}, time={elapsed:.2f}s")

    # Aggregate and save
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "classical_metrics.csv", index=False)

    # Summary table
    print("\n=== Per-Model Mean Metrics ===")
    summary = metrics_df.groupby("model")[["rmse", "mae", "mape"]].mean()
    summary = summary.sort_values("rmse")
    print(summary.to_string())
    summary.to_csv(OUTPUT_DIR / "classical_summary.csv")

    # Plots
    plot_all_forecasts(all_results, train_df, test_df, OUTPUT_DIR)
    plot_metrics_comparison(metrics_df, OUTPUT_DIR)
    print(f"\nResults saved to {OUTPUT_DIR}/")

    return metrics_df


if __name__ == "__main__":
    main()
