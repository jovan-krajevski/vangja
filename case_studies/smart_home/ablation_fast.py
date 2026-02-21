"""Fast ablation test for the smart home transfer learning pipeline.

This is a reduced-size version of the full ablation study, designed to
run quickly and validate that the entire pipeline works end-to-end.
Uses fewer hyperparameter combinations, shorter training, and two series.

Usage:
    python case_studies/smart_home/ablation_fast.py
"""

from __future__ import annotations

import itertools
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vangja import (
    BetaConstant,
    FlatTrend,
    FourierSeasonality,
    NormalConstant,
    UniformConstant,
)
from vangja.datasets import load_kaggle_temperature, load_smart_home_readings
from vangja.utils import metrics

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "results_fast"
OUTPUT_DIR.mkdir(exist_ok=True)

SMART_HOME_COLUMNS = ["Furnace 1 [kW]", "Fridge [kW]"]  # 2 series for speed
TRAIN_CUTOFF = "2016-04-01"
BASE_METHOD = "advi"  # fast VI instead of NUTS for testing
TARGET_METHOD = "map"  # fast MAP instead of mapx for testing
BASE_VI_N = 5000  # number of VI iterations (fast)
BASE_VI_SAMPLES = 200  # number of posterior samples

# Reduced hyperparameter grid for fast run
BASE_SERIES_ORDERS = [5]
BASE_BETA_SDS = [1]
SCALERS = ["minmax"]
TUNE_METHODS = ["parametric"]
POOL_TYPES = ["individual"]
LOSS_FACTORS = [0, 0.5]
SHRINKAGE_STRENGTHS = [10]
CONSTANT_TYPES = [None, "UniformConstant"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data():
    """Load temperature (base) and smart home (target) data."""
    print("Loading data...")
    temp_df = load_kaggle_temperature(
        city="Boston",
        start_date="2014-01-01 00:00:00",
        end_date="2016-01-01 00:00:00",
        freq="D",
    )
    smart_home_df = load_smart_home_readings(
        column=SMART_HOME_COLUMNS,
        freq="D",
    )
    train_df = smart_home_df[smart_home_df["ds"] < TRAIN_CUTOFF]
    test_df = smart_home_df[smart_home_df["ds"] >= TRAIN_CUTOFF]
    print(f"  Temperature data: {len(temp_df)} rows")
    print(f"  Smart home train: {len(train_df)} rows, test: {len(test_df)} rows")
    print(f"  Series: {train_df['series'].unique().tolist()}")
    return temp_df, train_df, test_df


# ---------------------------------------------------------------------------
# Constant component factory
# ---------------------------------------------------------------------------
def make_constant(name: str | None, pool_type: str):
    """Create a constant component by name.

    Parameters
    ----------
    name : str | None
        One of "UniformConstant", "BetaConstant", "NormalConstant", or None.
    pool_type : str
        Pool type for the constant.

    Returns
    -------
    Component or None
    """
    if name is None:
        return None
    if name == "UniformConstant":
        return UniformConstant(lower=-1, upper=1, pool_type=pool_type)
    elif name == "BetaConstant":
        return BetaConstant(lower=-1, upper=1, alpha=2, beta=2, pool_type=pool_type)
    elif name == "NormalConstant":
        return NormalConstant(mu=0, sd=1, pool_type=pool_type)
    else:
        raise ValueError(f"Unknown constant type: {name}")


# ---------------------------------------------------------------------------
# Base model training
# ---------------------------------------------------------------------------
def train_base_model(temp_df, series_order, beta_sd, scaler):
    """Train the base (temperature) model.

    Parameters
    ----------
    temp_df : pd.DataFrame
    series_order : int
    beta_sd : float
    scaler : str

    Returns
    -------
    tuple : (model, elapsed_time)
    """
    model = FlatTrend(intercept_sd=1) + FourierSeasonality(
        period=365.25, series_order=series_order, beta_sd=beta_sd
    )
    t0 = time.time()
    model.fit(
        temp_df, scaler=scaler, method=BASE_METHOD, n=BASE_VI_N, samples=BASE_VI_SAMPLES
    )
    elapsed = time.time() - t0
    return model, elapsed


# ---------------------------------------------------------------------------
# Target model training
# ---------------------------------------------------------------------------
def train_target_model(
    train_df,
    base_model,
    scaler,
    series_order,
    beta_sd,
    tune_method,
    pool_type,
    loss_factor,
    shrinkage,
    constant_type,
):
    """Train the target (smart home) model with transfer learning.

    Parameters
    ----------
    Multiple — see ablation grid.

    Returns
    -------
    tuple : (model, elapsed_time)
    """
    yearly_fs = FourierSeasonality(
        period=365.25,
        series_order=series_order,
        beta_sd=beta_sd,
        tune_method=tune_method,
        pool_type=pool_type,
        loss_factor_for_tune=loss_factor,
        shrinkage_strength=shrinkage,
    )
    weekly_fs = FourierSeasonality(
        period=7,
        series_order=3,
        beta_sd=1,
        pool_type=pool_type,
        shrinkage_strength=shrinkage,
    )
    trend = FlatTrend(intercept_sd=1, pool_type=pool_type, shrinkage_strength=shrinkage)

    constant = make_constant(constant_type, pool_type)
    if constant is not None:
        model = trend + constant * yearly_fs + weekly_fs
    else:
        model = trend + yearly_fs + weekly_fs

    t0 = time.time()
    fit_kwargs = dict(
        scaler=scaler,
        method=TARGET_METHOD,
        scale_mode="individual",
        sigma_pool_type=pool_type,
        t_scale_params=base_model.t_scale_params,
        idata=base_model.trace,
    )
    if pool_type == "partial":
        fit_kwargs["sigma_shrinkage_strength"] = shrinkage
    model.fit(train_df, **fit_kwargs)
    elapsed = time.time() - t0
    return model, elapsed


# ---------------------------------------------------------------------------
# Baseline model (no transfer learning)
# ---------------------------------------------------------------------------
def train_baseline(train_df, pool_type, shrinkage, include_yearly=True):
    """Train baseline without transfer learning.

    Parameters
    ----------
    train_df : pd.DataFrame
    pool_type : str
    shrinkage : float
    include_yearly : bool

    Returns
    -------
    tuple : (model, elapsed_time)
    """
    trend = FlatTrend(intercept_sd=1, pool_type=pool_type, shrinkage_strength=shrinkage)
    weekly_fs = FourierSeasonality(
        period=7,
        series_order=3,
        beta_sd=1,
        pool_type=pool_type,
        shrinkage_strength=shrinkage,
    )

    if include_yearly:
        yearly_fs = FourierSeasonality(
            period=365.25,
            series_order=5,
            beta_sd=1,
            pool_type=pool_type,
            shrinkage_strength=shrinkage,
        )
        model = trend + yearly_fs + weekly_fs
    else:
        model = trend + weekly_fs

    t0 = time.time()
    model.fit(
        train_df,
        scaler="minmax",
        method=TARGET_METHOD,
        scale_mode="individual",
        sigma_pool_type=pool_type,
    )
    elapsed = time.time() - t0
    return model, elapsed


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, test_df, pool_type):
    """Evaluate model on test set.

    Parameters
    ----------
    model : TimeSeriesModel
    test_df : pd.DataFrame
    pool_type : str

    Returns
    -------
    pd.DataFrame
    """
    first_series = test_df["series"].unique()[0]
    horizon = len(test_df[test_df["series"] == first_series])
    yhat = model.predict(horizon=horizon)
    return metrics(test_df, yhat, pool_type=pool_type)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_predictions(model, test_df, name, pool_type):
    """Plot predictions for each series.

    Parameters
    ----------
    model : TimeSeriesModel
    test_df : pd.DataFrame
    name : str
    pool_type : str
    """
    first_series = test_df["series"].unique()[0]
    horizon = len(test_df[test_df["series"] == first_series])
    yhat = model.predict(horizon=horizon)

    series_list = test_df["series"].unique()
    fig, axes = plt.subplots(len(series_list), 1, figsize=(14, 4 * len(series_list)))
    if len(series_list) == 1:
        axes = [axes]

    for idx, series_name in enumerate(series_list):
        ax = axes[idx]
        series_test = test_df[test_df["series"] == series_name]
        yhat_col = f"yhat_{idx}"
        if yhat_col in yhat.columns:
            ax.plot(yhat["ds"], yhat[yhat_col], label="Predicted", alpha=0.8)
        ax.scatter(
            series_test["ds"],
            series_test["y"],
            s=5,
            color="C1",
            label="Actual",
            zorder=5,
        )
        ax.set_title(f"{name} — {series_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = name.replace(" ", "_").replace("/", "_")[:60]
    fig.savefig(OUTPUT_DIR / f"pred_{safe_name}.png", dpi=100)
    plt.close(fig)


def plot_results_summary(results_df: pd.DataFrame):
    """Create a summary bar chart of RMSE across ablations.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have columns 'name' and 'rmse_mean'.
    """
    fig, ax = plt.subplots(figsize=(12, max(4, 0.4 * len(results_df))))
    sorted_df = results_df.sort_values("rmse_mean")
    ax.barh(range(len(sorted_df)), sorted_df["rmse_mean"])
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df["name"], fontsize=8)
    ax.set_xlabel("Mean RMSE across series")
    ax.set_title("Ablation Study — Fast Run")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "summary_rmse.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run the fast ablation study."""
    temp_df, train_df, test_df = load_data()

    all_results = []

    # ---- Baselines ----
    print("\n=== Baselines ===")
    for include_yearly in [True, False]:
        label = f"baseline_yearly={include_yearly}"
        print(f"  Training: {label}")
        try:
            bm, elapsed = train_baseline(
                train_df, "individual", 10, include_yearly=include_yearly
            )
            m = evaluate_model(bm, test_df, "individual")
            plot_predictions(bm, test_df, label, "individual")
            row = {
                "name": label,
                "type": "baseline",
                "include_yearly": include_yearly,
                "elapsed": elapsed,
                "rmse_mean": m["rmse"].mean(),
                "mae_mean": m["mae"].mean(),
                "mape_mean": m["mape"].mean(),
            }
            for series_name in m.index:
                row[f"rmse_{series_name}"] = m.loc[series_name, "rmse"]
                row[f"mae_{series_name}"] = m.loc[series_name, "mae"]
            all_results.append(row)
            print(f"    RMSE mean: {m['rmse'].mean():.4f}, time: {elapsed:.1f}s")
        except Exception as e:
            print(f"    FAILED: {e}")

    # ---- Transfer learning ablations ----
    print("\n=== Transfer Learning Ablations ===")
    grid = list(
        itertools.product(
            BASE_SERIES_ORDERS,
            BASE_BETA_SDS,
            SCALERS,
            TUNE_METHODS,
            POOL_TYPES,
            LOSS_FACTORS,
            SHRINKAGE_STRENGTHS,
            CONSTANT_TYPES,
        )
    )
    print(f"  Total configurations: {len(grid)}")

    # Cache base models to avoid re-training
    base_cache: dict[tuple, Any] = {}

    for i, (so, bsd, scaler, tm, pt, lf, ss, ct) in enumerate(grid):
        label = (
            f"so={so}_bsd={bsd}_sc={scaler}_tm={tm}_pt={pt}" f"_lf={lf}_ss={ss}_ct={ct}"
        )
        print(f"  [{i+1}/{len(grid)}] {label}")

        try:
            # Train or retrieve base model
            base_key = (so, bsd, scaler)
            if base_key not in base_cache:
                bm, bm_time = train_base_model(temp_df, so, bsd, scaler)
                base_cache[base_key] = (bm, bm_time)
            bm, bm_time = base_cache[base_key]

            # Train target model
            tm_model, tm_time = train_target_model(
                train_df, bm, scaler, so, bsd, tm, pt, lf, ss, ct
            )

            # Evaluate
            m = evaluate_model(tm_model, test_df, pt)
            plot_predictions(tm_model, test_df, label, pt)

            row = {
                "name": label,
                "type": "transfer",
                "series_order": so,
                "beta_sd": bsd,
                "scaler": scaler,
                "tune_method": tm,
                "pool_type": pt,
                "loss_factor": lf,
                "shrinkage": ss,
                "constant": ct,
                "base_time": bm_time,
                "target_time": tm_time,
                "elapsed": bm_time + tm_time,
                "rmse_mean": m["rmse"].mean(),
                "mae_mean": m["mae"].mean(),
                "mape_mean": m["mape"].mean(),
            }
            for series_name in m.index:
                row[f"rmse_{series_name}"] = m.loc[series_name, "rmse"]
                row[f"mae_{series_name}"] = m.loc[series_name, "mae"]
            all_results.append(row)
            print(
                f"    RMSE mean: {m['rmse'].mean():.4f}, "
                f"time: {bm_time + tm_time:.1f}s"
            )

        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback

            traceback.print_exc()

    # ---- Save results ----
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.csv'}")

    if len(results_df) > 0:
        plot_results_summary(results_df)
        print(f"Summary plot saved to {OUTPUT_DIR / 'summary_rmse.png'}")

    # Print summary
    print("\n=== Summary ===")
    print(
        results_df[["name", "rmse_mean", "mae_mean", "elapsed"]].to_string(index=False)
    )

    return results_df


if __name__ == "__main__":
    main()
