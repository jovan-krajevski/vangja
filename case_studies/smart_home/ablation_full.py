"""Full ablation study for transfer learning on the smart home dataset.

This script evaluates the impact of each hyperparameter on the transfer
learning pipeline by sweeping over a comprehensive grid. Each run trains
a base model on temperature data and transfers seasonality knowledge to
the smart home energy dataset.

Usage:
    python case_studies/smart_home/ablation_full.py [--dry-run]

Outputs are saved to ``case_studies/smart_home/results/``:

- ``results.csv``  — one row per configuration with all metrics.
- ``summary_rmse.png`` — bar chart ranking all configurations by RMSE.
- ``hparam_*.png`` — one plot per hyperparameter showing its effect.
- ``pred_*.png`` — prediction plots for selected configurations.

See ``ablations.md`` for the full experiment design.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

SMART_HOME_COLUMNS = [
    "Furnace 1 [kW]",
    "Furnace 2 [kW]",
    "Fridge [kW]",
    "Wine cellar [kW]",
]
TRAIN_CUTOFF = "2016-04-01"
BASE_METHOD = "nuts"
BASE_NUTS_SAMPLES = 1000
BASE_NUTS_CHAINS = 4
TARGET_METHOD = "mapx"

# Full hyperparameter grid
BASE_SERIES_ORDERS = [5, 10]
BASE_BETA_SDS = [0.1, 1, 10]
SCALERS = ["minmax", "standard"]
TUNE_METHODS = ["parametric", "prior_from_idata"]
POOL_TYPES = ["individual", "partial"]
LOSS_FACTORS = [-1, -0.5, 0, 0.5, 1]
SHRINKAGE_STRENGTHS = [1, 10, 100, 1000]
CONSTANT_TYPES = [None, "UniformConstant", "BetaConstant", "NormalConstant"]
EXTRA_SEASONALITIES = [None, "monthly", "quarterly", "monthly+quarterly"]
INCLUDE_SOURCE_IN_TARGET = [False, True]

# Prediction plot series (subset)
PLOT_SERIES = ["Furnace 1 [kW]", "Fridge [kW]"]

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
    """Persist a base model's trace and t_scale_params to disk.

    The ArviZ InferenceData is stored as a NetCDF file and the
    t_scale_params as a JSON sidecar.  Writes are atomic (write to
    a temporary file, then rename) so a crash mid-write cannot corrupt
    the cache.
    """
    key = _base_model_key_str(series_order, beta_sd, scaler)
    nc_path = BASE_MODEL_DIR / f"{key}.nc"
    json_path = BASE_MODEL_DIR / f"{key}_tscale.json"

    # Atomic write for NetCDF
    tmp_nc = nc_path.with_suffix(".nc.tmp")
    model.trace.to_netcdf(str(tmp_nc))
    tmp_nc.rename(nc_path)

    # Atomic write for t_scale_params
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
        The trace and t_scale_params, or None if no cache exists.
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
    """Load the set of already-completed experiment names from the checkpoint CSV."""
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
# Component factories
# ---------------------------------------------------------------------------
def make_constant(name: str | None, pool_type: str):
    """Create a constant component by name."""
    if name is None:
        return None
    factories = {
        "UniformConstant": lambda: UniformConstant(
            lower=-1, upper=1, pool_type=pool_type
        ),
        "BetaConstant": lambda: BetaConstant(
            lower=-1, upper=1, alpha=2, beta=2, pool_type=pool_type
        ),
        "NormalConstant": lambda: NormalConstant(mu=0, sd=1, pool_type=pool_type),
    }
    if name not in factories:
        raise ValueError(f"Unknown constant type: {name}")
    return factories[name]()


def make_extra_seasonalities(spec: str | None, pool_type: str, shrinkage: float):
    """Create extra seasonality components (monthly/quarterly).

    Parameters
    ----------
    spec : str or None
        One of None, "monthly", "quarterly", "monthly+quarterly".
    pool_type : str
    shrinkage : float

    Returns
    -------
    list of FourierSeasonality
    """
    components = []
    if spec is None:
        return components
    if "monthly" in spec:
        components.append(
            FourierSeasonality(
                period=30.4375,
                series_order=3,
                beta_sd=1,
                pool_type=pool_type,
                shrinkage_strength=shrinkage,
            )
        )
    if "quarterly" in spec:
        components.append(
            FourierSeasonality(
                period=91.3125,
                series_order=3,
                beta_sd=1,
                pool_type=pool_type,
                shrinkage_strength=shrinkage,
            )
        )
    return components


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
def train_base_model(temp_df, series_order, beta_sd, scaler):
    """Train the base (temperature) model using NUTS.

    Checks for a cached trace on disk first.  If found, creates the model,
    fits it, and replaces its trace with the cached version — avoiding the
    expensive NUTS sampling.  When training from scratch, persists the trace
    to disk for future runs.
    """
    cached = load_base_model_cache(series_order, beta_sd, scaler)
    if cached is not None:
        print(f"    [cache hit] base model so={series_order} bsd={beta_sd} sc={scaler}")
        trace, t_scale_params = cached
        # Still need a fitted model object for the target to extract
        # t_scale_params and trace from.  We do a fast MAP fit and then
        # overwrite the trace.
        model = FlatTrend(intercept_sd=1) + FourierSeasonality(
            period=365.25,
            series_order=series_order,
            beta_sd=beta_sd,
        )
        model.fit(temp_df, scaler=scaler, method="map")
        model.trace = trace
        model.t_scale_params = t_scale_params
        return model, 0.0  # elapsed = 0 (from cache)

    model = FlatTrend(intercept_sd=1) + FourierSeasonality(
        period=365.25,
        series_order=series_order,
        beta_sd=beta_sd,
    )
    t0 = time.time()
    model.fit(
        temp_df,
        scaler=scaler,
        method=BASE_METHOD,
        samples=BASE_NUTS_SAMPLES,
        chains=BASE_NUTS_CHAINS,
    )
    elapsed = time.time() - t0

    # Persist to disk for crash recovery
    save_base_model(model, series_order, beta_sd, scaler)
    return model, elapsed


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
    extra_seas,
    include_source=False,
    temp_df=None,
):
    """Train the target (smart home) model with transfer learning.

    Parameters
    ----------
    include_source : bool
        If True, include the source (temperature) series in the target
        training data for joint hierarchical fitting.
    temp_df : pd.DataFrame, optional
        Temperature data. Required when ``include_source=True``.
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
    trend = FlatTrend(
        intercept_sd=1,
        pool_type=pool_type,
        shrinkage_strength=shrinkage,
    )

    constant = make_constant(constant_type, pool_type)
    if constant is not None:
        model = trend + constant * yearly_fs + weekly_fs
    else:
        model = trend + yearly_fs + weekly_fs

    # Append extra seasonalities
    extras = make_extra_seasonalities(extra_seas, pool_type, shrinkage)
    for extra in extras:
        model = model + extra

    t0 = time.time()
    # Optionally include source time series in the target training data
    fit_df = train_df
    if include_source and temp_df is not None:
        source_df = temp_df.copy()
        source_df["series"] = "temperature_source"
        fit_df = pd.concat([train_df, source_df], ignore_index=True)

    fit_kwargs: dict[str, Any] = dict(
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
    return model, time.time() - t0


def train_baseline(train_df, pool_type, shrinkage, include_yearly=True):
    """Train baseline without transfer learning."""
    trend = FlatTrend(
        intercept_sd=1,
        pool_type=pool_type,
        shrinkage_strength=shrinkage,
    )
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
    fit_kwargs: dict[str, Any] = dict(
        scaler="minmax",
        method=TARGET_METHOD,
        scale_mode="individual",
        sigma_pool_type=pool_type,
    )
    if pool_type == "partial":
        fit_kwargs["sigma_shrinkage_strength"] = shrinkage
    model.fit(train_df, **fit_kwargs)
    return model, time.time() - t0


# ---------------------------------------------------------------------------
# Evaluation helpers
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
    """Evaluate model on test set."""
    horizon = _compute_horizon(model, test_df)
    yhat = model.predict(horizon=horizon)
    return metrics(test_df, yhat, pool_type=pool_type)


def extract_model_params(model, is_base=False):
    """Extract learned parameters from a fitted model for reporting.

    Parameters
    ----------
    model : TimeSeriesModel
    is_base : bool

    Returns
    -------
    dict
    """
    params = {}
    if is_base and model.trace is not None:
        for var in model.trace.posterior.data_vars:
            vals = model.trace.posterior[var].values
            params[f"base_{var}_mean"] = float(np.mean(vals))
            params[f"base_{var}_std"] = float(np.std(vals))
    elif not is_base and model.map_approx is not None:
        for var, val in model.map_approx.items():
            if isinstance(val, np.ndarray):
                params[f"target_{var}"] = val.tolist()
            else:
                params[f"target_{var}"] = float(val)
    return params


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_predictions(model, test_df, train_df, name, pool_type, save_dir):
    """Plot predictions for selected series, clipped to target date range."""
    horizon = _compute_horizon(model, test_df)
    yhat = model.predict(horizon=horizon)

    series_to_plot = [s for s in PLOT_SERIES if s in test_df["series"].unique()]
    if not series_to_plot:
        return

    fig, axes = plt.subplots(
        len(series_to_plot), 1, figsize=(14, 4 * len(series_to_plot))
    )
    if len(series_to_plot) == 1:
        axes = [axes]

    group_map = {v: k for k, v in model.groups_.items()}
    for idx, series_name in enumerate(series_to_plot):
        ax = axes[idx]
        series_test = test_df[test_df["series"] == series_name]
        series_train = train_df[train_df["series"] == series_name]

        group_code = group_map.get(series_name, idx)
        yhat_col = f"yhat_{group_code}"

        # Clip predictions to target series date range
        date_min = series_train["ds"].min()
        date_max = series_test["ds"].max()
        yhat_clipped = yhat[(yhat["ds"] >= date_min) & (yhat["ds"] <= date_max)]

        if yhat_col in yhat_clipped.columns:
            ax.plot(
                yhat_clipped["ds"],
                yhat_clipped[yhat_col],
                label="Predicted",
                alpha=0.8,
                color="C0",
            )
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
            series_test["ds"], series_test["y"], s=5, color="C1", label="Test", zorder=5
        )
        ax.set_title(f"{name} — {series_name}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = name.replace(" ", "_").replace("/", "_")[:60]
    fig.savefig(save_dir / f"pred_{safe_name}.png", dpi=100)
    plt.close(fig)


def plot_results_summary(results_df, save_dir):
    """Create summary bar chart ranking all configurations by RMSE."""
    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * len(results_df))))
    sorted_df = results_df.sort_values("rmse_mean")
    colors = ["C3" if t == "baseline" else "C0" for t in sorted_df["type"]]
    ax.barh(range(len(sorted_df)), sorted_df["rmse_mean"], color=colors)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df["name"], fontsize=7)
    ax.set_xlabel("Mean RMSE across series")
    ax.set_title("Ablation Study — All Configurations (red = baseline)")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(save_dir / "summary_rmse.png", dpi=150)
    plt.close(fig)


def plot_hparam_effects(results_df, save_dir):
    """Create per-hyperparameter effect plots.

    For each hyperparameter, show a box/strip plot of RMSE values grouped
    by the hyperparameter level, averaged across all other settings.
    """
    transfer_df = results_df[results_df["type"] == "transfer"].copy()
    if transfer_df.empty:
        return

    hparams = [
        "series_order",
        "beta_sd",
        "scaler",
        "tune_method",
        "pool_type",
        "loss_factor",
        "shrinkage",
        "constant",
        "extra_seas",
        "include_source",
    ]
    hparams = [h for h in hparams if h in transfer_df.columns]

    for hp in hparams:
        fig, ax = plt.subplots(figsize=(10, 5))
        # Convert to string for categorical plotting
        transfer_df[hp] = transfer_df[hp].astype(str)
        try:
            sns.stripplot(
                data=transfer_df,
                x=hp,
                y="rmse_mean",
                ax=ax,
                alpha=0.5,
                jitter=True,
                size=4,
            )
            sns.boxplot(
                data=transfer_df,
                x=hp,
                y="rmse_mean",
                ax=ax,
                showfliers=False,
                boxprops=dict(alpha=0.3),
            )
        except Exception:
            ax.bar(
                range(transfer_df[hp].nunique()),
                transfer_df.groupby(hp)["rmse_mean"].mean().values,
            )
        ax.set_title(f"Effect of {hp} on RMSE")
        ax.set_xlabel(hp)
        ax.set_ylabel("Mean RMSE")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        fig.savefig(save_dir / f"hparam_{hp}.png", dpi=100)
        plt.close(fig)


def plot_best_per_series(
    results_df: pd.DataFrame,
    model_cache: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_dir: Path,
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
    save_dir : Path
    """
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
def main(dry_run=False):
    """Run the full ablation study.

    Supports crash recovery via checkpointing.  Completed experiments
    are loaded from ``results_checkpoint.csv`` and skipped on restart.
    Base model traces are cached in ``results/base_models/``.

    Parameters
    ----------
    dry_run : bool
        If True, only print the grid size without running anything.
    """
    np.random.seed(42)
    temp_df, train_df, test_df = load_data()

    # Build grid
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
            EXTRA_SEASONALITIES,
            INCLUDE_SOURCE_IN_TARGET,
        )
    )
    # Filter: shrinkage only matters for partial pooling
    grid = [g for g in grid if g[4] == "partial" or g[6] == SHRINKAGE_STRENGTHS[0]]
    # Filter: include_source only makes sense with partial pooling
    grid = [g for g in grid if g[4] == "partial" or g[9] is False]

    print(f"\nTotal configurations: {len(grid)} + baselines")
    if dry_run:
        print("Dry run — exiting.")
        return

    # Load checkpoint to skip already-completed experiments
    completed = load_checkpoint()
    if completed:
        print(f"Checkpoint loaded: {len(completed)} experiments already completed.")

    all_results: list[dict] = []
    # Reload previous results so final CSV is complete
    if CHECKPOINT_CSV.exists():
        prev_df = pd.read_csv(CHECKPOINT_CSV)
        all_results = prev_df.to_dict("records")

    base_cache: dict[tuple, Any] = {}
    model_cache: dict[str, Any] = {}  # name -> (model, pool_type)

    # ---- Baselines ----
    print("\n=== Baselines ===")
    for pt in POOL_TYPES:
        for shrinkage in [10] if pt == "partial" else [1]:
            for include_yearly in [True, False]:
                label = f"baseline_pt={pt}_ss={shrinkage}" f"_yearly={include_yearly}"
                if label in completed:
                    print(f"  [skip] {label} (already completed)")
                    continue
                print(f"  Training: {label}")
                try:
                    bm, elapsed = train_baseline(
                        train_df, pt, shrinkage, include_yearly
                    )
                    m = evaluate_model(bm, test_df, pt)
                    plot_predictions(bm, test_df, train_df, label, pt, OUTPUT_DIR)
                    model_cache[label] = (bm, pt)
                    row = {
                        "name": label,
                        "type": "baseline",
                        "pool_type": pt,
                        "shrinkage": shrinkage,
                        "include_yearly": include_yearly,
                        "elapsed": elapsed,
                        "rmse_mean": m["rmse"].mean(),
                        "mae_mean": m["mae"].mean(),
                        "mape_mean": m["mape"].mean(),
                    }
                    for sn in m.index:
                        row[f"rmse_{sn}"] = m.loc[sn, "rmse"]
                        row[f"mae_{sn}"] = m.loc[sn, "mae"]
                        row[f"mape_{sn}"] = m.loc[sn, "mape"]
                    all_results.append(row)
                    append_checkpoint(row)
                    print(f"    RMSE: {m['rmse'].mean():.4f}, time: {elapsed:.1f}s")
                except Exception as e:
                    print(f"    FAILED: {e}")

    # ---- Transfer learning ablations ----
    print(f"\n=== Transfer Learning Ablations ({len(grid)} configs) ===")
    for i, (so, bsd, scaler, tm, pt, lf, ss, ct, es, inc_src) in enumerate(grid):
        label = (
            f"so={so}_bsd={bsd}_sc={scaler}_tm={tm}_pt={pt}"
            f"_lf={lf}_ss={ss}_ct={ct}_es={es}_src={inc_src}"
        )
        if label in completed:
            print(f"  [{i+1}/{len(grid)}] [skip] {label}")
            continue
        print(f"  [{i+1}/{len(grid)}] {label}")

        try:
            # Base model (cached in-memory and on disk)
            base_key = (so, bsd, scaler)
            if base_key not in base_cache:
                bm, bm_time = train_base_model(temp_df, so, bsd, scaler)
                base_cache[base_key] = (bm, bm_time)
            bm, bm_time = base_cache[base_key]

            # Target model
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
                es,
                include_source=inc_src,
                temp_df=temp_df,
            )

            # Evaluate
            m = evaluate_model(tm_model, test_df, pt)

            # Plot predictions for selected configs
            if i % max(1, len(grid) // 20) == 0:
                plot_predictions(tm_model, test_df, train_df, label, pt, OUTPUT_DIR)
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
                "extra_seas": es,
                "include_source": inc_src,
                "base_time": bm_time,
                "target_time": tm_time,
                "elapsed": bm_time + tm_time,
                "rmse_mean": m["rmse"].mean(),
                "mae_mean": m["mae"].mean(),
                "mape_mean": m["mape"].mean(),
            }
            for sn in m.index:
                row[f"rmse_{sn}"] = m.loc[sn, "rmse"]
                row[f"mae_{sn}"] = m.loc[sn, "mae"]
                row[f"mape_{sn}"] = m.loc[sn, "mape"]
            all_results.append(row)
            append_checkpoint(row)
            print(
                f"    RMSE: {m['rmse'].mean():.4f}, " f"time: {bm_time + tm_time:.1f}s"
            )

        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback

            traceback.print_exc()

    # ---- Save & plot ----
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.csv'}")

    if len(results_df) > 0:
        plot_results_summary(results_df, OUTPUT_DIR)
        plot_hparam_effects(results_df, OUTPUT_DIR)
        plot_best_per_series(results_df, model_cache, train_df, test_df, OUTPUT_DIR)
        print(f"Plots saved to {OUTPUT_DIR}/")

    # Top 10 summary
    print("\n=== Top 10 Configurations by RMSE ===")
    top10 = results_df.nsmallest(10, "rmse_mean")
    print(
        top10[["name", "rmse_mean", "mae_mean", "mape_mean", "elapsed"]].to_string(
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
    parser = argparse.ArgumentParser(description="Full ablation study")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print grid size and exit"
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
