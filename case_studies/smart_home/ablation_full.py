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

# Prediction plot series (subset)
PLOT_SERIES = ["Furnace 1 [kW]", "Fridge [kW]"]


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
    """Train the base (temperature) model using NUTS."""
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
    return model, time.time() - t0


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
):
    """Train the target (smart home) model with transfer learning."""
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
    model.fit(train_df, **fit_kwargs)
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
def evaluate_model(model, test_df, pool_type):
    """Evaluate model on test set."""
    first_series = test_df["series"].unique()[0]
    horizon = len(test_df[test_df["series"] == first_series])
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
    """Plot predictions for selected series."""
    first_series = test_df["series"].unique()[0]
    horizon = len(test_df[test_df["series"] == first_series])
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

        if yhat_col in yhat.columns:
            ax.plot(
                yhat["ds"], yhat[yhat_col], label="Predicted", alpha=0.8, color="C0"
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(dry_run=False):
    """Run the full ablation study.

    Parameters
    ----------
    dry_run : bool
        If True, only print the grid size without running anything.
    """
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
        )
    )
    # Filter: shrinkage only matters for partial pooling
    grid = [g for g in grid if g[4] == "partial" or g[6] == SHRINKAGE_STRENGTHS[0]]

    print(f"\nTotal configurations: {len(grid)} + baselines")
    if dry_run:
        print("Dry run — exiting.")
        return

    all_results: list[dict] = []
    base_cache: dict[tuple, Any] = {}

    # ---- Baselines ----
    print("\n=== Baselines ===")
    for pt in POOL_TYPES:
        for shrinkage in [10] if pt == "partial" else [1]:
            for include_yearly in [True, False]:
                label = f"baseline_pt={pt}_ss={shrinkage}" f"_yearly={include_yearly}"
                print(f"  Training: {label}")
                try:
                    bm, elapsed = train_baseline(
                        train_df, pt, shrinkage, include_yearly
                    )
                    m = evaluate_model(bm, test_df, pt)
                    plot_predictions(bm, test_df, train_df, label, pt, OUTPUT_DIR)
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
                    all_results.append(row)
                    print(f"    RMSE: {m['rmse'].mean():.4f}, time: {elapsed:.1f}s")
                except Exception as e:
                    print(f"    FAILED: {e}")

    # ---- Transfer learning ablations ----
    print(f"\n=== Transfer Learning Ablations ({len(grid)} configs) ===")
    for i, (so, bsd, scaler, tm, pt, lf, ss, ct, es) in enumerate(grid):
        label = (
            f"so={so}_bsd={bsd}_sc={scaler}_tm={tm}_pt={pt}"
            f"_lf={lf}_ss={ss}_ct={ct}_es={es}"
        )
        print(f"  [{i+1}/{len(grid)}] {label}")

        try:
            # Base model (cached by series_order, beta_sd, scaler)
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
            )

            # Evaluate
            m = evaluate_model(tm_model, test_df, pt)

            # Plot predictions for selected configs
            if i % max(1, len(grid) // 20) == 0:
                plot_predictions(tm_model, test_df, train_df, label, pt, OUTPUT_DIR)

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
            all_results.append(row)
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
        print(f"Plots saved to {OUTPUT_DIR}/")

    # Top 10 summary
    print("\n=== Top 10 Configurations by RMSE ===")
    top10 = results_df.nsmallest(10, "rmse_mean")
    print(top10[["name", "rmse_mean", "mae_mean", "elapsed"]].to_string(index=False))

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full ablation study")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print grid size and exit"
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
