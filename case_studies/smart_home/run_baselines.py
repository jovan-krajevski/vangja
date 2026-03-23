from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from vangja.datasets import load_kaggle_temperature, load_smart_home_readings

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
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load smart home and temperature data, split by cutoff."""
    print("Loading data...")
    sh_df = load_smart_home_readings(column=SMART_HOME_COLUMNS, freq="D")
    train = sh_df[sh_df["ds"] < TRAIN_CUTOFF].copy()
    test = sh_df[sh_df["ds"] >= TRAIN_CUTOFF].copy()

    # Temperature for regression baselines
    temp_df = load_kaggle_temperature(
        city="Boston", start_date="2016-01-01", end_date="2017-01-01", freq="D"
    )

    print(f"  Smart home train: {len(train)} rows, test: {len(test)} rows")
    print(f"  Temperature: {len(temp_df)} rows")
    print(f"  Series: {train['series'].unique().tolist()}")
    return train, test, temp_df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, MAPE for a single series."""
    n = min(len(y_true), len(y_pred))
    yt, yp = y_true[:n], y_pred[:n]
    return {
        "rmse": root_mean_squared_error(yt, yp),
        "mae": mean_absolute_error(yt, yp),
        "mape": mean_absolute_percentage_error(yt, yp),
        "mse": mean_squared_error(yt, yp),
    }


# ---------------------------------------------------------------------------
# Model implementations
# ---------------------------------------------------------------------------


# 1. Seasonal Naive
def seasonal_naive(train_y: np.ndarray, horizon: int, period: int = 7) -> np.ndarray:
    """Repeat last seasonal cycle."""
    last = train_y[-period:]
    return np.tile(last, horizon // period + 1)[:horizon]


def seasonal_naive_mean(
    train_y: np.ndarray, horizon: int, period: int = 7
) -> np.ndarray:
    """Average of all observed seasonal cycles."""
    n = len(train_y)
    n_cycles = n // period
    if n_cycles < 1:
        return np.full(horizon, train_y.mean())
    mat = train_y[-(n_cycles * period) :].reshape(n_cycles, period)
    avg_cycle = mat.mean(axis=0)
    return np.tile(avg_cycle, horizon // period + 1)[:horizon]


# 2. Rolling window mean
def rolling_mean_forecast(
    train_y: np.ndarray, horizon: int, window: int = 7
) -> np.ndarray:
    """Constant forecast = mean of last `window` observations."""
    return np.full(horizon, train_y[-window:].mean())


# 3. ARIMA with order selection
def fit_arima_best(train_y: np.ndarray, horizon: int) -> tuple[np.ndarray, float, str]:
    """Try a set of ARIMA orders and pick the one with best AIC.

    Returns (forecast, elapsed_seconds, best_order_string).
    """
    orders_to_try = [
        # (p,d,q), (P,D,Q,s) — seasonal orders
        ((1, 0, 0), (0, 0, 0, 0)),  # AR(1)
        ((1, 1, 0), (0, 0, 0, 0)),  # ARIMA(1,1,0)
        ((1, 1, 1), (0, 0, 0, 0)),  # ARIMA(1,1,1)
        ((2, 1, 1), (0, 0, 0, 0)),  # ARIMA(2,1,1)
        ((1, 1, 1), (1, 0, 0, 7)),  # ARIMA with weekly AR
        ((1, 1, 1), (1, 0, 1, 7)),  # ARIMA with weekly ARMA
        ((2, 1, 2), (0, 0, 0, 0)),  # ARIMA(2,1,2)
        ((0, 1, 1), (0, 0, 0, 0)),  # IMA(1,1) random walk + MA
        ((1, 0, 1), (0, 0, 0, 0)),  # ARMA(1,1)
    ]

    best_aic = np.inf
    best_result = None
    best_label = "ARIMA(1,0,0)"

    t0 = time.time()
    for order, seasonal_order in orders_to_try:
        try:
            if seasonal_order == (0, 0, 0, 0):
                model = ARIMA(
                    train_y,
                    order=order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
            else:
                model = ARIMA(
                    train_y,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
            result = model.fit(method_kwargs={"maxiter": 300})
            if result.aic < best_aic:
                best_aic = result.aic
                best_result = result
                if seasonal_order == (0, 0, 0, 0):
                    best_label = f"ARIMA{order}"
                else:
                    best_label = f"SARIMA{order}x{seasonal_order}"
        except Exception:
            continue

    elapsed = time.time() - t0

    if best_result is not None:
        forecast = best_result.forecast(steps=horizon)
    else:
        forecast = np.full(horizon, train_y.mean())
        best_label = "fallback_mean"

    return forecast, elapsed, best_label


# 4. Fixed ARIMA orders (for comparison)
def fit_arima_fixed(
    train_y: np.ndarray,
    horizon: int,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (0, 0, 0, 0),
) -> tuple[np.ndarray, float]:
    """Fit ARIMA with fixed order."""
    t0 = time.time()
    try:
        if seasonal_order == (0, 0, 0, 0):
            model = ARIMA(
                train_y,
                order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        else:
            model = ARIMA(
                train_y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
        result = model.fit(method_kwargs={"maxiter": 300})
        forecast = result.forecast(steps=horizon)
    except Exception:
        forecast = np.full(horizon, train_y.mean())
    return forecast, time.time() - t0


# 5. Holt-Winters variants
def fit_holt_winters(
    train_y: np.ndarray,
    horizon: int,
    seasonal_periods: int = 7,
    trend: str = "add",
    seasonal: str = "add",
    damped_trend: bool = False,
) -> tuple[np.ndarray, float]:
    """Fit Exponential Smoothing with various configurations."""
    t0 = time.time()
    try:
        model = ExponentialSmoothing(
            train_y,
            seasonal_periods=seasonal_periods,
            trend=trend,
            seasonal=seasonal,
            damped_trend=damped_trend,
            initialization_method="estimated",
        )
        result = model.fit(optimized=True)
        forecast = result.forecast(steps=horizon)
    except Exception:
        try:
            # Fallback: no seasonality
            model = ExponentialSmoothing(
                train_y, trend="add", initialization_method="estimated"
            )
            result = model.fit(optimized=True)
            forecast = result.forecast(steps=horizon)
        except Exception:
            forecast = np.full(horizon, train_y.mean())
    return forecast, time.time() - t0


# 6. Temperature-informed linear regression
def fit_temp_regression(
    train_y: np.ndarray,
    train_dates: pd.Series,
    test_dates: pd.Series,
    temp_df: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, float]:
    """Linear regression: y ~ temperature + day_of_week + month.

    This is the classical analog of vangja's transfer learning approach:
    using external temperature data as a feature rather than transferring
    Bayesian priors.
    """
    t0 = time.time()
    try:
        # Build features
        def make_features(dates: pd.Series, temp_data: pd.DataFrame) -> pd.DataFrame:
            df = pd.DataFrame({"ds": pd.to_datetime(dates)})
            df = df.merge(temp_data.rename(columns={"y": "temp"}), on="ds", how="left")
            df["temp"] = df["temp"].ffill().bfill()
            df["dow"] = df["ds"].dt.dayofweek
            df["month"] = df["ds"].dt.month
            # One-hot encode dow and month
            dow_dummies = pd.get_dummies(df["dow"], prefix="dow", dtype=float)
            month_dummies = pd.get_dummies(df["month"], prefix="mon", dtype=float)
            features = pd.concat([df[["temp"]], dow_dummies, month_dummies], axis=1)
            return features

        X_train = make_features(train_dates, temp_df)
        # For test, generate date range from cutoff
        test_dates_range = pd.date_range(
            start=pd.Timestamp(TRAIN_CUTOFF), periods=horizon, freq="D"
        )
        X_test = make_features(pd.Series(test_dates_range), temp_df)

        # Align columns
        for col in X_train.columns:
            if col not in X_test.columns:
                X_test[col] = 0.0
        for col in X_test.columns:
            if col not in X_train.columns:
                X_train[col] = 0.0
        X_test = X_test[X_train.columns]

        reg = LinearRegression()
        reg.fit(X_train.values, train_y)
        forecast = reg.predict(X_test.values)
    except Exception:
        forecast = np.full(horizon, train_y.mean())
    return forecast, time.time() - t0


# 7. Temperature-only regression (simplest transfer analog)
def fit_temp_only_regression(
    train_y: np.ndarray,
    train_dates: pd.Series,
    test_dates: pd.Series,
    temp_df: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, float]:
    """Simple regression: y ~ temperature (linear relationship only)."""
    t0 = time.time()
    try:
        train_temp = (
            pd.DataFrame({"ds": pd.to_datetime(train_dates)})
            .merge(temp_df.rename(columns={"y": "temp"}), on="ds", how="left")["temp"]
            .ffill()
            .bfill()
            .values.reshape(-1, 1)
        )

        test_dates_range = pd.date_range(TRAIN_CUTOFF, periods=horizon, freq="D")
        test_temp = (
            pd.DataFrame({"ds": test_dates_range})
            .merge(temp_df.rename(columns={"y": "temp"}), on="ds", how="left")["temp"]
            .ffill()
            .bfill()
            .values.reshape(-1, 1)
        )

        reg = LinearRegression()
        reg.fit(train_temp, train_y)
        forecast = reg.predict(test_temp)
    except Exception:
        forecast = np.full(horizon, train_y.mean())
    return forecast, time.time() - t0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all_forecasts(results: list[dict], train_df, test_df, save_dir: Path):
    """Grid plot: one row per series, one column per model."""
    series_list = sorted(set(r["series"] for r in results))
    model_list = sorted(set(r["model"] for r in results))

    fig, axes = plt.subplots(
        len(series_list),
        len(model_list),
        figsize=(4 * len(model_list), 3.5 * len(series_list)),
        squeeze=False,
    )

    for row, sname in enumerate(series_list):
        for col, mname in enumerate(model_list):
            ax = axes[row][col]
            match = [r for r in results if r["series"] == sname and r["model"] == mname]
            if not match:
                ax.set_visible(False)
                continue
            r = match[0]
            s_train = train_df[train_df["series"] == sname]
            ax.scatter(
                s_train["ds"], s_train["y"], s=2, color="C2", alpha=0.4, label="Train"
            )
            ax.scatter(
                r["test_ds"], r["test_y"], s=2, color="C1", alpha=0.5, label="Test"
            )
            ax.plot(
                r["test_ds"],
                r["forecast"][: len(r["test_ds"])],
                color="C0",
                alpha=0.8,
                lw=1,
                label="Pred",
            )
            if row == 0:
                ax.set_title(mname, fontsize=7)
            if col == 0:
                ax.set_ylabel(sname.split("[")[0].strip(), fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)
    axes[0][0].legend(fontsize=5)
    plt.tight_layout()
    fig.savefig(save_dir / "all_forecasts.png", dpi=120)
    plt.close(fig)


def plot_metrics_comparison(metrics_df: pd.DataFrame, save_dir: Path):
    """Bar chart of mean RMSE per model."""
    agg = metrics_df.groupby("model")["rmse"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(agg))))
    ax.barh(range(len(agg)), agg.values, color="C0", alpha=0.7)
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(agg.index, fontsize=8)
    ax.set_xlabel("Mean RMSE across all series")
    ax.set_title("Classical Baselines — Mean RMSE")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(save_dir / "classical_rmse.png", dpi=120)
    plt.close(fig)


def plot_per_series_comparison(metrics_df: pd.DataFrame, save_dir: Path):
    """Grouped bar chart: RMSE per series per model."""
    series_list = sorted(metrics_df["series"].unique())
    model_list = sorted(metrics_df.groupby("model")["rmse"].mean().sort_values().index)

    fig, axes = plt.subplots(
        1, len(series_list), figsize=(5 * len(series_list), 5), squeeze=False
    )
    for i, sname in enumerate(series_list):
        ax = axes[0][i]
        sd = metrics_df[metrics_df["series"] == sname].set_index("model")
        sd = sd.reindex(model_list)
        ax.barh(range(len(sd)), sd["rmse"].values, color="C0", alpha=0.7)
        ax.set_yticks(range(len(sd)))
        ax.set_yticklabels(sd.index, fontsize=7)
        ax.set_xlabel("RMSE")
        ax.set_title(sname.split("[")[0].strip(), fontsize=9)
        ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(save_dir / "per_series_rmse.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> pd.DataFrame:
    """Run all classical baselines."""
    train_df, test_df, temp_df = load_data()
    series_list = train_df["series"].unique()

    all_metrics: list[dict] = []
    all_results: list[dict] = []

    # Define all models to run
    model_configs = [
        # Naive / simple
        ("Seasonal Naive (7d)", "snaive_7"),
        ("Seasonal Naive (30d)", "snaive_30"),
        ("Seasonal Naive Mean (7d)", "snaive_mean_7"),
        ("Rolling Mean (7d)", "rolling_7"),
        ("Rolling Mean (14d)", "rolling_14"),
        ("Rolling Mean (30d)", "rolling_30"),
        ("Global Mean", "global_mean"),
        # ARIMA
        ("ARIMA(1,1,1)", "arima_111"),
        ("ARIMA(2,1,1)", "arima_211"),
        ("SARIMA(1,1,1)(1,0,1,7)", "sarima_7"),
        ("Auto-ARIMA (best AIC)", "auto_arima"),
        # Exponential Smoothing
        ("HW (add, add, 7d)", "hw_aa_7"),
        ("HW (add, mul, 7d)", "hw_am_7"),
        ("HW (add_damped, add, 7d)", "hw_da_7"),
        ("HW (add, add, 30d)", "hw_aa_30"),
        # Temperature-informed
        ("Temp Regression (linear+dow+month)", "temp_reg"),
        ("Temp Only Regression", "temp_only_reg"),
    ]

    for series_name in series_list:
        print(f"\n--- {series_name} ---")
        s_train = train_df[train_df["series"] == series_name].sort_values("ds")
        s_test = test_df[test_df["series"] == series_name].sort_values("ds")

        train_y = s_train["y"].values
        test_y = s_test["y"].values
        test_ds = s_test["ds"].values
        horizon = len(test_y)

        for model_name, model_code in model_configs:
            print(f"  {model_name}...", end=" ")
            extra_info = ""

            if model_code == "snaive_7":
                forecast = seasonal_naive(train_y, horizon, 7)
                elapsed = 0.0
            elif model_code == "snaive_30":
                forecast = seasonal_naive(train_y, horizon, 30)
                elapsed = 0.0
            elif model_code == "snaive_mean_7":
                forecast = seasonal_naive_mean(train_y, horizon, 7)
                elapsed = 0.0
            elif model_code == "rolling_7":
                forecast = rolling_mean_forecast(train_y, horizon, 7)
                elapsed = 0.0
            elif model_code == "rolling_14":
                forecast = rolling_mean_forecast(train_y, horizon, 14)
                elapsed = 0.0
            elif model_code == "rolling_30":
                forecast = rolling_mean_forecast(train_y, horizon, 30)
                elapsed = 0.0
            elif model_code == "global_mean":
                forecast = np.full(horizon, train_y.mean())
                elapsed = 0.0
            elif model_code == "arima_111":
                forecast, elapsed = fit_arima_fixed(train_y, horizon, (1, 1, 1))
            elif model_code == "arima_211":
                forecast, elapsed = fit_arima_fixed(train_y, horizon, (2, 1, 1))
            elif model_code == "sarima_7":
                forecast, elapsed = fit_arima_fixed(
                    train_y, horizon, (1, 1, 1), (1, 0, 1, 7)
                )
            elif model_code == "auto_arima":
                forecast, elapsed, best_order = fit_arima_best(train_y, horizon)
                extra_info = f" (selected: {best_order})"
            elif model_code == "hw_aa_7":
                forecast, elapsed = fit_holt_winters(
                    train_y, horizon, 7, "add", "add", False
                )
            elif model_code == "hw_am_7":
                forecast, elapsed = fit_holt_winters(
                    train_y, horizon, 7, "add", "mul", False
                )
            elif model_code == "hw_da_7":
                forecast, elapsed = fit_holt_winters(
                    train_y, horizon, 7, "add", "add", True
                )
            elif model_code == "hw_aa_30":
                forecast, elapsed = fit_holt_winters(
                    train_y, horizon, 30, "add", "add", False
                )
            elif model_code == "temp_reg":
                forecast, elapsed = fit_temp_regression(
                    train_y, s_train["ds"], s_test["ds"], temp_df, horizon
                )
            elif model_code == "temp_only_reg":
                forecast, elapsed = fit_temp_only_regression(
                    train_y, s_train["ds"], s_test["ds"], temp_df, horizon
                )
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
                    "forecast": forecast[: len(test_ds)],
                    "test_y": test_y,
                    "test_ds": test_ds,
                }
            )

            print(f"RMSE={m['rmse']:.4f}{extra_info}, time={elapsed:.2f}s")

    # Aggregate and save
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT_DIR / "classical_metrics.csv", index=False)

    # Summary table
    print("\n" + "=" * 60)
    print("Per-Model Mean Metrics (sorted by RMSE)")
    print("=" * 60)
    summary = metrics_df.groupby("model")[["rmse", "mae", "mape", "elapsed"]].mean()
    summary = summary.sort_values("rmse")
    print(summary.to_string())
    summary.to_csv(OUTPUT_DIR / "classical_summary.csv")

    # Best model per series
    print(f"\n{'=' * 60}")
    print("Best Model per Series (by RMSE)")
    print("=" * 60)
    for sname in sorted(metrics_df["series"].unique()):
        sd = metrics_df[metrics_df["series"] == sname]
        best = sd.loc[sd["rmse"].idxmin()]
        print(
            f"  {sname}: {best['model']} "
            f"(RMSE={best['rmse']:.4f}, MAE={best['mae']:.4f})"
        )

    # Cross-comparison table: series×model
    print(f"\n{'=' * 60}")
    print("RMSE Matrix (series × model)")
    print("=" * 60)
    pivot = metrics_df.pivot_table(values="rmse", index="series", columns="model")
    pivot = pivot[summary.index]  # sort columns by mean RMSE
    print(pivot.round(4).to_string())
    pivot.to_csv(OUTPUT_DIR / "rmse_matrix.csv")

    # Plots
    plot_all_forecasts(all_results, train_df, test_df, OUTPUT_DIR)
    plot_metrics_comparison(metrics_df, OUTPUT_DIR)
    plot_per_series_comparison(metrics_df, OUTPUT_DIR)

    print(f"\nResults and plots saved to {OUTPUT_DIR}/")
    return metrics_df


if __name__ == "__main__":
    main()
