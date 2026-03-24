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

from vangja.datasets import get_sp500_tickers_for_range, load_stock_data

warnings.filterwarnings("ignore")

CT_PATH = Path(__file__).parent / "data/sp500_constituents"
TICKERS_PATH = Path(__file__).parent / "data/tickers"
OUTPUT_DIR = Path(__file__).parent / "results_classical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def rescale_dataset(
    smp_train: pd.DataFrame, stocks_train: pd.DataFrame, stocks_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rescale target stocks into S&P 500 equivalent bounds.
    Matches the logic used in run_vangja.py explicitly.
    """
    smp_min, smp_max = smp_train.iloc[-91:].y.min(), smp_train.iloc[-91:].y.max()
    for series in stocks_train["series"].unique():
        series_mask = stocks_train["series"] == series
        series_min, series_max = (
            stocks_train.loc[series_mask, "y"].min(),
            stocks_train.loc[series_mask, "y"].max(),
        )
        if series_max > series_min:
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

# 1. Naive (Last observation)
def naive_forecast(train_y: np.ndarray, horizon: int) -> np.ndarray:
    """Repeat last observation (Random Walk without drift)."""
    if len(train_y) == 0:
        return np.zeros(horizon)
    return np.full(horizon, train_y[-1])


def seasonal_naive(train_y: np.ndarray, horizon: int, period: int = 7) -> np.ndarray:
    """Repeat last seasonal cycle."""
    if len(train_y) < period:
        return naive_forecast(train_y, horizon)
    last = train_y[-period:]
    return np.tile(last, horizon // period + 1)[:horizon]


# 2. Rolling window mean
def rolling_mean_forecast(
    train_y: np.ndarray, horizon: int, window: int = 7
) -> np.ndarray:
    """Constant forecast = mean of last `window` observations."""
    if len(train_y) == 0:
        return np.zeros(horizon)
    return np.full(horizon, train_y[-min(len(train_y), window):].mean())


# 3. Fixed ARIMA orders
def fit_arima_fixed(
    train_y: np.ndarray,
    horizon: int,
    order: tuple = (1, 1, 0),
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
        result = model.fit(method_kwargs={"maxiter": 100})
        forecast = result.forecast(steps=horizon)
    except Exception:
        forecast = naive_forecast(train_y, horizon)
    return forecast, time.time() - t0


# 5. Holt-Winters variants
def fit_holt_winters(
    train_y: np.ndarray,
    horizon: int,
    seasonal_periods: int = 7,
    trend: str = "add",
    seasonal: str | None = "add",
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
            forecast = naive_forecast(train_y, horizon)
    return forecast, time.time() - t0


# 6. S&P-informed linear regression
def fit_smp_regression(
    train_y: np.ndarray,
    train_dates: pd.Series,
    test_dates: pd.Series,
    smp_df: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, float]:
    """Linear regression: y ~ S&P 500 index + day_of_week + month.

    This is the classical analog of vangja's transfer learning approach:
    using external market index data as a feature rather than transferring
    Bayesian priors.
    """
    t0 = time.time()
    try:
        def make_features(dates: pd.Series, smp_data: pd.DataFrame) -> pd.DataFrame:
            df = pd.DataFrame({"ds": pd.to_datetime(dates)})
            df = df.merge(smp_data.rename(columns={"y": "smp"}), on="ds", how="left")
            df["smp"] = df["smp"].ffill().bfill()
            df["dow"] = df["ds"].dt.dayofweek
            df["month"] = df["ds"].dt.month
            
            dow_dummies = pd.get_dummies(df["dow"], prefix="dow", dtype=float)
            month_dummies = pd.get_dummies(df["month"], prefix="mon", dtype=float)
            features = pd.concat([df[["smp"]], dow_dummies, month_dummies], axis=1)
            return features

        X_train = make_features(train_dates, smp_df).fillna(0)
        X_test = make_features(test_dates, smp_df).fillna(0)

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
        forecast = naive_forecast(train_y, horizon)
    return forecast, time.time() - t0


# 7. S&P-only regression
def fit_smp_only_regression(
    train_y: np.ndarray,
    train_dates: pd.Series,
    test_dates: pd.Series,
    smp_df: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, float]:
    """Simple regression: y ~ S&P 500 (linear relationship only)."""
    t0 = time.time()
    try:
        def make_features(dates: pd.Series, smp_data: pd.DataFrame) -> pd.DataFrame:
            df = pd.DataFrame({"ds": pd.to_datetime(dates)})
            df = df.merge(smp_data.rename(columns={"y": "smp"}), on="ds", how="left")
            df["smp"] = df["smp"].ffill().bfill()
            return df[["smp"]]

        X_train = make_features(train_dates, smp_df).fillna(0)
        X_test = make_features(test_dates, smp_df).fillna(0)
        
        reg = LinearRegression()
        reg.fit(X_train.values, train_y)
        forecast = reg.predict(X_test.values)
    except Exception:
        forecast = naive_forecast(train_y, horizon)
    return forecast, time.time() - t0


def plot_metrics_comparison(metrics_df: pd.DataFrame, save_dir: Path):
    """Bar chart of mean RMSE per model."""
    agg = metrics_df.groupby("model")["rmse"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(agg))))
    ax.barh(range(len(agg)), agg.values, color="C0", alpha=0.7)
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels(agg.index, fontsize=8)
    ax.set_xlabel("Mean RMSE across all series and start dates")
    ax.set_title("Classical Baselines — Mean RMSE")
    plt.tight_layout()
    fig.savefig(save_dir / "models_rmse_comparison.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> pd.DataFrame:
    start_dates = (
        pd.date_range(start="2013-01-01", end="2015-12-01", freq="MS")
        .strftime("%Y-%m-%d")
        .tolist()
    )
    
    tickers = get_sp500_tickers_for_range("2012-09-01", "2015-01-01", cache_path=CT_PATH)
    
    # Models to run
    # Note: Stock dataset loops through thousands of time series overall, 
    # so slower Auto-ARIMA baselines are omitted in favor of predefined fast approximations.
    model_configs = [
        ("Naive (Last Value)", "naive"),
        ("Seasonal Naive (7d)", "snaive_7"),
        ("Rolling Mean (7d)", "rolling_7"),
        ("Rolling Mean (30d)", "rolling_30"),
        ("ARIMA(1,1,0)", "arima_110"),
        ("ARIMA(1,1,1)", "arima_111"),
        ("ARIMA(2,1,1)", "arima_211"),
        ("HW (add, no seasonal)", "hw_a"),
        ("SMP Regression (linear+dow+month)", "smp_reg"),
        ("SMP Only Regression", "smp_only_reg"),
    ]
    
    all_metrics: list[dict] = []

    for start_date in start_dates:
        print(f"\n{'='*60}")
        print(f"Processing Start Date: {start_date}")
        print(f"{'='*60}")
        
        # Check if already processed
        results_file = OUTPUT_DIR / f"results_classical_{start_date}.csv"
        if results_file.exists():
            print(f"Skipping {start_date}, results file already exists.")
            existing_df = pd.read_csv(results_file)
            all_metrics.extend(existing_df.to_dict("records"))
            continue

        stocks_train, stocks_test = load_stock_data(
            tickers,
            split_date=start_date,
            window_size=91,
            horizon_size=365,
            cache_path=TICKERS_PATH,
            interpolate=True,
        )
        
        # 4 years window for SMP to match max 'ws' used in run_vangja.py
        smp_train, smp_test = load_stock_data(
            ["^GSPC"],
            split_date=start_date,
            window_size=4 * 365, 
            horizon_size=365,
            cache_path=TICKERS_PATH,
            interpolate=True,
        )
        
        # Rescale target dataset exactly like Vangja model pipeline does
        train_df, test_df = rescale_dataset(smp_train, stocks_train, stocks_test)
        smp_full = pd.concat([smp_train, smp_test], ignore_index=True)
        
        series_list = train_df["series"].unique()
        start_date_metrics = []
        
        for series_name in series_list:
            s_train = train_df[train_df["series"] == series_name].sort_values("ds")
            s_test = test_df[test_df["series"] == series_name].sort_values("ds")
            
            if s_test.empty or s_train.empty:
                continue
                
            train_y = s_train["y"].values
            test_y = s_test["y"].values
            horizon = len(test_y)
            
            for model_name, model_code in model_configs:
                t_start = time.time()
                
                if model_code == "naive":
                    forecast = naive_forecast(train_y, horizon)
                elif model_code == "snaive_7":
                    forecast = seasonal_naive(train_y, horizon, 7)
                elif model_code == "rolling_7":
                    forecast = rolling_mean_forecast(train_y, horizon, 7)
                elif model_code == "rolling_30":
                    forecast = rolling_mean_forecast(train_y, horizon, 30)
                elif model_code == "arima_110":
                    forecast, _ = fit_arima_fixed(train_y, horizon, (1, 1, 0))
                elif model_code == "arima_111":
                    forecast, _ = fit_arima_fixed(train_y, horizon, (1, 1, 1))
                elif model_code == "arima_211":
                    forecast, _ = fit_arima_fixed(train_y, horizon, (2, 1, 1))
                elif model_code == "hw_a":
                    # Exponential Smoothing without seasonality
                    forecast, _ = fit_holt_winters(train_y, horizon, trend="add", seasonal=None)
                elif model_code == "smp_reg":
                    forecast, _ = fit_smp_regression(
                        train_y, s_train["ds"], s_test["ds"], smp_full, horizon
                    )
                elif model_code == "smp_only_reg":
                    forecast, _ = fit_smp_only_regression(
                        train_y, s_train["ds"], s_test["ds"], smp_full, horizon
                    )
                else:
                    continue
                    
                elapsed = time.time() - t_start
                m = evaluate_forecast(test_y, forecast)
                m["model"] = model_name
                m["series"] = series_name
                m["start_date"] = start_date
                m["elapsed"] = elapsed
                start_date_metrics.append(m)
        
        # Save metrics for this start_date loop
        sd_metrics_df = pd.DataFrame(start_date_metrics)
        sd_metrics_df.to_csv(results_file, index=False)
        all_metrics.extend(start_date_metrics)
        print(f"  Processed {len(series_list)} timeseries matches.")

    metrics_df = pd.DataFrame(all_metrics)
    
    if not metrics_df.empty:
        # Summary table
        print("\n" + "=" * 60)
        print("Per-Model Mean Metrics (sorted by RMSE)")
        print("=" * 60)
        summary = metrics_df.groupby("model")[["rmse", "mae", "mape", "elapsed"]].mean()
        summary = summary.sort_values("rmse")
        print(summary.to_string())
        summary.to_csv(OUTPUT_DIR / "classical_summary_overall.csv")

        # Plots
        plot_metrics_comparison(metrics_df, OUTPUT_DIR)
        print(f"\nResults and plots saved to {OUTPUT_DIR}/")

    return metrics_df

if __name__ == "__main__":
    main()
