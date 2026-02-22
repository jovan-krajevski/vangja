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
import arviz as az
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
INCLUDE_SOURCE_IN_TARGET = [False, True]

# Checkpointing
BASE_MODEL_DIR = OUTPUT_DIR / "base_models"
BASE_MODEL_DIR.mkdir(exist_ok=True)
CHECKPOINT_CSV = OUTPUT_DIR / "results_checkpoint.csv"


# ---------------------------------------------------------------------------
# Checkpointing helpers
# ---------------------------------------------------------------------------
def _base_model_key_str(series_order: int, beta_sd: float, scaler: str) -> str:
    """Return a filesystem-safe string key for a base model configuration."""
    return f"base_so={series_order}_bsd={beta_sd}_sc={scaler}"


def save_base_model(model, series_order: int, beta_sd: float, scaler: str) -> None:
    """Persist a base model's trace and t_scale_params to disk (atomic)."""
    key = _base_model_key_str(series_order, beta_sd, scaler)
    nc_path = BASE_MODEL_DIR / f"{key}.nc"
    json_path = BASE_MODEL_DIR / f"{key}_tscale.json"

    tmp_nc = nc_path.with_suffix(".nc.tmp")
    model.trace.to_netcdf(str(tmp_nc))
    tmp_nc.rename(nc_path)

    tsp = {
        "ds_min": str(model.t_scale_params["ds_min"]),
        "ds_max": str(model.t_scale_params["ds_max"]),
    }
    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(tsp, indent=2))
    tmp_json.rename(json_path)


def load_base_model_cache(series_order: int, beta_sd: float, scaler: str):
    """Load a cached base model trace and t_scale_params from disk.

    Returns
    -------
    tuple of (az.InferenceData, dict) or None
    """
    key = _base_model_key_str(series_order, beta_sd, scaler)
    nc_path = BASE_MODEL_DIR / f"{key}.nc"
    json_path = BASE_MODEL_DIR / f"{key}_tscale.json"
    if not nc_path.exists() or not json_path.exists():
        return None
    trace = az.from_netcdf(str(nc_path))
    tsp_raw = json.loads(json_path.read_text())
    t_scale_params = {
        "ds_min": pd.Timestamp(tsp_raw["ds_min"]),
        "ds_max": pd.Timestamp(tsp_raw["ds_max"]),
    }
    return trace, t_scale_params


def load_checkpoint() -> set[str]:
    """Load the set of already-completed experiment names from checkpoint."""
    if not CHECKPOINT_CSV.exists():
        return set()
    try:
        df = pd.read_csv(CHECKPOINT_CSV)
        return set(df["name"].tolist())
    except Exception:
        return set()


def append_checkpoint(row: dict) -> None:
    """Append one result row to the checkpoint CSV (atomic)."""
    df_new = pd.DataFrame([row])
    if CHECKPOINT_CSV.exists():
        df_old = pd.read_csv(CHECKPOINT_CSV)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    tmp = CHECKPOINT_CSV.with_suffix(".csv.tmp")
    df_all.to_csv(tmp, index=False)
    tmp.rename(CHECKPOINT_CSV)


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

    Checks for a cached trace on disk first.  If found, loads it instead
    of re-running VI.  On a fresh run, persists the trace for crash recovery.

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
    cached = load_base_model_cache(series_order, beta_sd, scaler)
    if cached is not None:
        print(f"    [cache hit] base model so={series_order} bsd={beta_sd} sc={scaler}")
        trace, t_scale_params = cached
        model = FlatTrend(intercept_sd=1) + FourierSeasonality(
            period=365.25, series_order=series_order, beta_sd=beta_sd
        )
        model.fit(temp_df, scaler=scaler, method="map")
        model.trace = trace
        model.t_scale_params = t_scale_params
        return model, 0.0

    model = FlatTrend(intercept_sd=1) + FourierSeasonality(
        period=365.25, series_order=series_order, beta_sd=beta_sd
    )
    t0 = time.time()
    model.fit(
        temp_df, scaler=scaler, method=BASE_METHOD, n=BASE_VI_N, samples=BASE_VI_SAMPLES
    )
    elapsed = time.time() - t0

    # Persist to disk for crash recovery
    save_base_model(model, series_order, beta_sd, scaler)
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
    include_source=False,
    temp_df=None,
):
    """Train the target (smart home) model with transfer learning.

    Parameters
    ----------
    Multiple — see ablation grid.
    include_source : bool
        If True, include the source (temperature) series in the target
        training data for joint hierarchical fitting.
    temp_df : pd.DataFrame, optional
        Temperature data. Required when ``include_source=True``.

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
    # Optionally include source time series in the target training data
    fit_df = train_df
    if include_source and temp_df is not None:
        source_df = temp_df.copy()
        source_df["series"] = "temperature_source"
        fit_df = pd.concat([train_df, source_df], ignore_index=True)

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
    model.fit(fit_df, **fit_kwargs)
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
def _compute_horizon(model, test_df):
    """Compute the horizon needed to cover the entire test set.

    When transfer learning shifts ``t_scale_params`` to a base model whose
    ``ds_max`` is earlier than the target training data, the number of
    forecast days must span the gap between ``ds_max`` and the last test
    date — not just the number of test data points.

    Parameters
    ----------
    model : TimeSeriesModel
    test_df : pd.DataFrame

    Returns
    -------
    int
    """
    last_test_date = pd.Timestamp(test_df["ds"].max())
    ds_max = pd.Timestamp(model.t_scale_params["ds_max"])
    return (last_test_date - ds_max).days


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
    horizon = _compute_horizon(model, test_df)
    yhat = model.predict(horizon=horizon)
    return metrics(test_df, yhat, pool_type=pool_type)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_predictions(model, test_df, name, pool_type, train_df=None, save_dir=None):
    """Plot predictions for each series, clipped to target date range.

    Parameters
    ----------
    model : TimeSeriesModel
    test_df : pd.DataFrame
    name : str
    pool_type : str
    train_df : pd.DataFrame, optional
        Training data. If given, train points are shown on the plot.
    save_dir : Path, optional
        Directory to save plots. Defaults to OUTPUT_DIR.
    """
    if save_dir is None:
        save_dir = OUTPUT_DIR
    horizon = _compute_horizon(model, test_df)
    yhat = model.predict(horizon=horizon)

    series_list = test_df["series"].unique()
    group_map = {v: k for k, v in model.groups_.items()}
    fig, axes = plt.subplots(len(series_list), 1, figsize=(14, 4 * len(series_list)))
    if len(series_list) == 1:
        axes = [axes]

    for idx, series_name in enumerate(series_list):
        ax = axes[idx]
        series_test = test_df[test_df["series"] == series_name]
        group_code = group_map.get(series_name, idx)
        yhat_col = f"yhat_{group_code}"

        # Determine the date range to show (target series only)
        if train_df is not None:
            series_train = train_df[train_df["series"] == series_name]
            date_min = series_train["ds"].min()
        else:
            date_min = series_test["ds"].min()
        date_max = series_test["ds"].max()

        # Clip predictions to target date range
        yhat_clipped = yhat[(yhat["ds"] >= date_min) & (yhat["ds"] <= date_max)]

        if yhat_col in yhat_clipped.columns:
            ax.plot(
                yhat_clipped["ds"],
                yhat_clipped[yhat_col],
                label="Predicted",
                alpha=0.8,
            )
        if train_df is not None:
            ax.scatter(
                series_train["ds"],
                series_train["y"],
                s=3,
                color="C2",
                label="Train",
                alpha=0.5,
                zorder=4,
            )
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
    fig.savefig(save_dir / f"pred_{safe_name}.png", dpi=100)
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


def plot_best_per_series(
    results_df: pd.DataFrame,
    model_cache: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: Path | None = None,
):
    """Plot a dedicated prediction plot for the best model per series.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results with per-series RMSE columns.
    model_cache : dict
        Mapping from config name to (model, pool_type).
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    save_dir : Path, optional
    """
    if save_dir is None:
        save_dir = OUTPUT_DIR

    rmse_cols = [
        c for c in results_df.columns if c.startswith("rmse_") and c != "rmse_mean"
    ]
    if not rmse_cols:
        return

    for col in sorted(rmse_cols):
        series_name = col[len("rmse_") :]
        valid = results_df.dropna(subset=[col])
        if valid.empty:
            continue
        best_idx = valid[col].idxmin()
        best_row = valid.loc[best_idx]
        best_name = best_row["name"]

        if best_name not in model_cache:
            continue

        model, pool_type = model_cache[best_name]
        group_map = {v: k for k, v in model.groups_.items()}
        group_code = group_map.get(series_name)
        if group_code is None:
            continue

        horizon = _compute_horizon(model, test_df)
        try:
            yhat = model.predict_uncertainty(horizon=horizon)
            has_uncertainty = True
        except Exception:
            yhat = model.predict(horizon=horizon)
            has_uncertainty = False
        yhat_col = f"yhat_{group_code}"
        lower_col = f"yhat_lower_{group_code}"
        upper_col = f"yhat_upper_{group_code}"

        series_train = train_df[train_df["series"] == series_name]
        series_test = test_df[test_df["series"] == series_name]

        date_min = series_train["ds"].min()
        date_max = series_test["ds"].max()
        yhat_clipped = yhat[(yhat["ds"] >= date_min) & (yhat["ds"] <= date_max)]

        fig, ax = plt.subplots(figsize=(14, 5))
        if yhat_col in yhat_clipped.columns:
            ax.plot(
                yhat_clipped["ds"],
                yhat_clipped[yhat_col],
                label="Predicted",
                alpha=0.8,
                color="C0",
                lw=1.5,
            )
        if has_uncertainty and lower_col in yhat_clipped.columns:
            ax.fill_between(
                yhat_clipped["ds"],
                yhat_clipped[lower_col],
                yhat_clipped[upper_col],
                alpha=0.2,
                color="C0",
                label="95% interval",
            )
        ax.scatter(
            series_train["ds"],
            series_train["y"],
            s=5,
            color="C2",
            label="Train",
            alpha=0.5,
            zorder=4,
        )
        ax.scatter(
            series_test["ds"],
            series_test["y"],
            s=8,
            color="C1",
            label="Test",
            zorder=5,
        )
        rmse_val = best_row[col]
        ax.set_title(
            f"Best model for {series_name} (RMSE={rmse_val:.4f})\n{best_name}",
            fontsize=10,
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        safe_series = series_name.replace(" ", "_").replace("/", "_")[:40]
        fig.savefig(save_dir / f"best_{safe_series}.png", dpi=120)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run the fast ablation study.

    Supports crash recovery via checkpointing.  Completed experiments
    are loaded from ``results_checkpoint.csv`` and skipped on restart.
    Base model traces are cached in ``results_fast/base_models/``.
    """
    np.random.seed(42)
    temp_df, train_df, test_df = load_data()

    # Load checkpoint to skip already-completed experiments
    completed = load_checkpoint()
    if completed:
        print(f"Checkpoint loaded: {len(completed)} experiments already completed.")

    all_results = []
    if CHECKPOINT_CSV.exists():
        prev_df = pd.read_csv(CHECKPOINT_CSV)
        all_results = prev_df.to_dict("records")

    model_cache: dict[str, Any] = {}  # name -> (model, pool_type)

    # ---- Baselines ----
    print("\n=== Baselines ===")
    for include_yearly in [True, False]:
        label = f"baseline_yearly={include_yearly}"
        if label in completed:
            print(f"  [skip] {label} (already completed)")
            continue
        print(f"  Training: {label}")
        try:
            bm, elapsed = train_baseline(
                train_df, "individual", 10, include_yearly=include_yearly
            )
            m = evaluate_model(bm, test_df, "individual")
            plot_predictions(bm, test_df, label, "individual", train_df=train_df)
            model_cache[label] = (bm, "individual")
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
                row[f"mape_{series_name}"] = m.loc[series_name, "mape"]
            all_results.append(row)
            append_checkpoint(row)
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
            INCLUDE_SOURCE_IN_TARGET,
        )
    )
    # Filter: include_source only makes sense with partial pooling
    grid = [g for g in grid if g[4] == "partial" or g[8] is False]
    print(f"  Total configurations: {len(grid)}")

    # Cache base models to avoid re-training
    base_cache: dict[tuple, Any] = {}

    for i, (so, bsd, scaler, tm, pt, lf, ss, ct, inc_src) in enumerate(grid):
        label = (
            f"so={so}_bsd={bsd}_sc={scaler}_tm={tm}_pt={pt}"
            f"_lf={lf}_ss={ss}_ct={ct}_src={inc_src}"
        )
        if label in completed:
            print(f"  [{i+1}/{len(grid)}] [skip] {label}")
            continue
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
                train_df,
                bm,
                scaler,
                so,
                bsd,
                tm,
                pt,
                lf,
                ss,
                ct,
                include_source=inc_src,
                temp_df=temp_df,
            )

            # Evaluate
            m = evaluate_model(tm_model, test_df, pt)
            plot_predictions(tm_model, test_df, label, pt, train_df=train_df)
            model_cache[label] = (tm_model, pt)

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
                "include_source": inc_src,
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
                row[f"mape_{series_name}"] = m.loc[series_name, "mape"]
            all_results.append(row)
            append_checkpoint(row)
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

        # Plot best model per series
        plot_best_per_series(results_df, model_cache, train_df, test_df)
        print(f"Best-per-series plots saved to {OUTPUT_DIR}/")

    # Print summary
    print("\n=== Summary ===")
    print(
        results_df[["name", "rmse_mean", "mae_mean", "mape_mean", "elapsed"]].to_string(
            index=False
        )
    )

    # Best model per series
    if len(results_df) > 0:
        rmse_cols = [
            c for c in results_df.columns if c.startswith("rmse_") and c != "rmse_mean"
        ]
        print("\n=== Best Model per Series (by RMSE) ===")
        for col in sorted(rmse_cols):
            series_name = col[len("rmse_") :]
            valid = results_df.dropna(subset=[col])
            if valid.empty:
                continue
            best_idx = valid[col].idxmin()
            best = valid.loc[best_idx]
            mae_col = f"mae_{series_name}"
            mape_col = f"mape_{series_name}"
            mae_val = best[mae_col] if mae_col in best.index else float("nan")
            mape_val = best[mape_col] if mape_col in best.index else float("nan")
            print(
                f"  {series_name}: {best['name']}  "
                f"(RMSE={best[col]:.4f}, MAE={mae_val:.4f}, "
                f"MAPE={mape_val:.4f})"
            )

    return results_df


if __name__ == "__main__":
    main()
