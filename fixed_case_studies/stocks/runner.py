"""Shared fitting/evaluation pipeline for the stocks case study.

The pipeline is deliberately small: one frozen configuration per arm, no
grid search, no test-horizon selection.  Everything is seeded and every
artifact records its provenance (see ``fixed_case_studies/common.py``).
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from vangja import FourierSeasonality, LinearTrend
from vangja.datasets import load_stock_data

from fixed_case_studies import common
from fixed_case_studies.stocks.config import (
    CONTEXT_TICKER,
    CONTEXT_WINDOW_DAYS,
    HORIZON_DAYS,
    MIN_TRAIN_OBS,
    NEGATIVE_CONTROL_TICKER,
    TARGET_WINDOW_DAYS,
    StockConfig,
)

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
TICKERS_PATH = Path(__file__).resolve().parent / "data" / "tickers"
CT_PATH = Path(__file__).resolve().parent / "data" / "sp500_constituents"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_origin_data(
    origin: str,
    tickers: list[str],
    *,
    negative_control: bool = False,
    max_stocks: int | None = None,
    seed: int = common.BASE_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load context + targets for one origin on trading days (no
    interpolation; review F-9/P0-7).

    Returns ``(smp_train, smp_test, stocks_train, stocks_test, availability)``.
    ``availability`` records the universe rule outcome for provenance.
    """
    context_ticker = NEGATIVE_CONTROL_TICKER if negative_control else CONTEXT_TICKER

    smp_train, smp_test = load_stock_data(
        [context_ticker],
        split_date=origin,
        window_size=CONTEXT_WINDOW_DAYS,
        horizon_size=HORIZON_DAYS,
        cache_path=TICKERS_PATH,
        interpolate=False,
    )

    stocks_train, stocks_test = load_stock_data(
        tickers,
        split_date=origin,
        window_size=TARGET_WINDOW_DAYS,
        horizon_size=HORIZON_DAYS,
        cache_path=TICKERS_PATH,
        interpolate=False,
    )

    counts = stocks_train.groupby("series").size()
    in_test = set(stocks_test["series"].unique())
    valid = sorted(
        [t for t in counts.index if counts[t] >= MIN_TRAIN_OBS and t in in_test]
    )
    requested = len(set(tickers))
    availability = {
        "universe_rule": "members_at_origin",
        "n_requested": requested,
        "n_available_train_and_test": len(valid),
        "missing_from_train_or_test": sorted(set(tickers) - set(valid)),
        "min_train_obs": MIN_TRAIN_OBS,
    }

    if max_stocks is not None and len(valid) > max_stocks:
        rng = np.random.default_rng(seed)
        valid = sorted(rng.choice(valid, size=max_stocks, replace=False))
        availability["n_used"] = len(valid)
        availability["subset_seed"] = seed
    else:
        availability["n_used"] = len(valid)

    stocks_train = stocks_train[stocks_train["series"].isin(valid)].reset_index(drop=True)
    stocks_test = stocks_test[stocks_test["series"].isin(valid)].reset_index(drop=True)
    return smp_train, smp_test, stocks_train, stocks_test, availability


def rescale_dataset(
    smp_train: pd.DataFrame,
    stocks_train: pd.DataFrame,
    stocks_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rescale each target series into the S&P 500 training range.

    This preprocessing (inherited from the original study) makes the target
    scales comparable before the global maxabs scaling of the joint model.
    The rescaling uses **training data only**.
    """
    smp_min, smp_max = smp_train["y"].min(), smp_train["y"].max()
    for series in stocks_train["series"].unique():
        mask = stocks_train["series"] == series
        s_min, s_max = (
            stocks_train.loc[mask, "y"].min(),
            stocks_train.loc[mask, "y"].max(),
        )
        if s_max <= s_min:
            continue
        stocks_train.loc[mask, "y"] = (
            (stocks_train.loc[mask, "y"] - s_min) / (s_max - s_min)
        ) * (smp_max - smp_min) + smp_min

        mask = stocks_test["series"] == series
        stocks_test.loc[mask, "y"] = (
            (stocks_test.loc[mask, "y"] - s_min) / (s_max - s_min)
        ) * (smp_max - smp_min) + smp_min
    return stocks_train, stocks_test


# ---------------------------------------------------------------------------
# Source (context) model
# ---------------------------------------------------------------------------


def build_source_model():
    """The S&P 500 context model: piecewise trend + yearly + weekly
    seasonality (multiplicative), complete pooling, no transfer."""
    trend = LinearTrend(
        n_changepoints=25,
        slope_sd=5.0,
        intercept_sd=5.0,
        delta_side="right",
        pool_type="complete",
        delta_pool_type="complete",
        tune_method=None,
        delta_tune_method=None,
    )
    yearly = FourierSeasonality(
        period=365.25, series_order=6, beta_sd=5.0,
        pool_type="complete", tune_method=None,
    )
    weekly = FourierSeasonality(
        period=7, series_order=3, beta_sd=5.0,
        pool_type="complete", tune_method=None,
    )
    return trend ** (yearly + weekly)


def fit_source(
    smp_train: pd.DataFrame,
    *,
    method: str,
    seed: int,
    target_accept: float = 0.9,
    progressbar: bool = False,
):
    """Fit the context model.

    ``method="advi"`` is used for development screening only; confirmation
    runs use ``method="nuts"`` so that the transferred posterior carries
    real covariance (review F-8 / PLAN P0-17).
    """
    model = build_source_model()
    if method == "advi":
        model.fit(
            smp_train,
            scaler="maxabs",
            scale_mode="complete",
            sigma_pool_type="individual",
            method="advi",
            n=20000,
            samples=2000,
            random_seed=seed,
            progressbar=progressbar,
        )
    else:
        model.fit(
            smp_train,
            scaler="maxabs",
            scale_mode="complete",
            sigma_pool_type="individual",
            method="nuts",
            samples=500,
            tune=500,
            chains=2,
            cores=2,
            nuts_sampler="nutpie",
            target_accept=target_accept,
            random_seed=seed,
            progressbar=progressbar,
        )
    return model


# ---------------------------------------------------------------------------
# Target model
# ---------------------------------------------------------------------------


def build_target_model(cfg: StockConfig):
    trend = LinearTrend(
        n_changepoints=25,
        slope_sd=5.0,
        intercept_sd=5.0,
        delta_side="right",
        pool_type=cfg.lt_pool,
        delta_pool_type="complete",
        tune_method=cfg.tune_method,
        delta_tune_method=None,
        loss_factor_for_tune=cfg.lt_loss_factor,
        shrinkage_strength=cfg.lt_shrinkage,
    )
    yearly = FourierSeasonality(
        period=365.25,
        series_order=6,
        beta_sd=5.0,
        pool_type=cfg.fs_pool,
        tune_method=cfg.tune_method,
        loss_factor_for_tune=cfg.fs_loss_factor,
        shrinkage_strength=cfg.fs_shrinkage,
    )
    weekly = FourierSeasonality(
        period=7, series_order=3, beta_sd=5.0,
        pool_type="individual", tune_method=None,
    )
    return trend ** (yearly + weekly)


def fit_target(
    cfg: StockConfig,
    train_df: pd.DataFrame,
    source_model,
    source_train_df: pd.DataFrame,
    *,
    seed: int,
    method: str = "mapx",
    progressbar: bool = False,
):
    """Fit one target configuration and return the fitted model."""
    model = build_target_model(cfg)
    kwargs = dict(
        scaler="maxabs",
        method=method,
        scale_mode="complete",
        sigma_pool_type="individual",
        t_scale_params=source_model.t_scale_params,
        random_seed=seed,
        progressbar=progressbar,
    )
    if cfg.tune_method is not None:
        kwargs["idata"] = source_model.trace
    if cfg.include_context_group:
        kwargs["include_source_in_target"] = True
        kwargs["source_data"] = source_train_df
    model.fit(train_df, **kwargs)
    return model


# ---------------------------------------------------------------------------
# One origin x config cell
# ---------------------------------------------------------------------------


def run_cell(
    cfg: StockConfig,
    origin: str,
    stage: str,
    tickers: list[str],
    *,
    seed: int,
    source_method: str,
    out_dir: Path,
    max_stocks: int | None = None,
    progressbar: bool = False,
) -> pd.DataFrame:
    """Fit source + target for one (origin, config) cell and record the
    full artifact bundle. Returns the per-unit metric rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.name}__{origin}__seed{seed}".replace("/", "_")
    if (out_dir / f"manifest_{tag}.json").exists():
        units_path = out_dir / f"units_{tag}.csv"
        if units_path.exists():
            return pd.read_csv(units_path)

    smp_train, smp_test, stocks_train, stocks_test, availability = load_origin_data(
        origin,
        tickers,
        negative_control=cfg.negative_control,
        max_stocks=max_stocks,
        seed=seed,
    )
    stocks_train, stocks_test = rescale_dataset(smp_train, stocks_train, stocks_test)

    try:
        source_model = load_or_fit_source(
            smp_train,
            origin=origin,
            method=source_method,
            seed=source_seed_for(origin),
            cache_dir=out_dir / "source_cache",
            progressbar=progressbar,
        )
        model = fit_target(
            cfg, stocks_train, source_model, smp_train, seed=seed,
            progressbar=progressbar,
        )
        yhat = model.predict(horizon=HORIZON_DAYS)
        unit_df = common.unit_metrics(
            model, stocks_test, yhat, origin=origin, config=cfg.name, stage=stage
        )
        unit_df["seed"] = seed
        common.record_artifact(
            out_dir,
            study="stocks",
            stage=stage,
            origin=origin,
            config=cfg.name,
            model=model,
            unit_df=unit_df,
            future_df=yhat,
            extra={
                "availability": availability,
                "source_method": source_method,
                "context": (
                    NEGATIVE_CONTROL_TICKER
                    if cfg.negative_control
                    else CONTEXT_TICKER
                ),
            },
        )
        return unit_df
    except Exception as err:  # failed fit -> recorded, unit excluded
        common.record_failure(
            out_dir,
            study="stocks",
            stage=stage,
            origin=origin,
            config=cfg.name,
            seed=seed,
            error=err,
            traceback_text=traceback.format_exc(),
        )
        raise


def get_universe_for_origin(origin: str, require_frozen: bool = False) -> list[str]:
    """Load the frozen universe for an origin (membership at origin date).

    Reads the per-origin universe CSV written by ``01_fetch_data.py``.
    With ``require_frozen=True`` (used by the confirmation runs) a missing
    frozen universe is an error — confirmation must never silently fall
    back to a live scrape.
    """
    universe_dir = Path(__file__).resolve().parent / "data" / "universe"
    path = universe_dir / f"universe_{origin}.csv"
    if path.exists():
        return pd.read_csv(path)["ticker"].tolist()
    if require_frozen:
        raise FileNotFoundError(
            f"Frozen universe missing for {origin}; run 01_fetch_data.py first "
            f"(expected {path})."
        )
    # Fallback: reconstruct live (still deterministic per origin).
    from vangja.datasets import get_sp500_tickers_at_date

    return get_sp500_tickers_at_date(origin, cache_path=CT_PATH)


def source_seed_for(origin: str, base: int = common.BASE_SEED) -> int:
    """Seed for the context fit, constant across configs of one origin so
    every configuration of an origin transfers the identical source posterior
    (config differences are then purely target-model changes)."""
    ts = pd.Timestamp(origin)
    return base + 1000 * (ts.year - 2013) * 12 + (ts.month - 1)


def seed_for(origin_index: int, config_index: int, base: int = common.BASE_SEED) -> int:
    """Deterministic per-cell seed (recorded in every artifact)."""
    return base + origin_index * 100 + config_index


# ---------------------------------------------------------------------------
# Source-fit caching (the NUTS context fit is expensive; refit per origin
# only, keyed by origin x method x seed)
# ---------------------------------------------------------------------------


def _dump_t_scale(params: dict) -> str:
    """Serialize t_scale_params: Timestamps become ISO strings."""
    out = dict(params)
    for k in ("ds_min", "ds_max"):
        if k in out and hasattr(out[k], "isoformat"):
            out[k] = out[k].isoformat()
    return json.dumps(out)


def _load_t_scale(text: str) -> dict:
    """Deserialize t_scale_params: ISO strings become Timestamps."""
    out = json.loads(text)
    for k in ("ds_min", "ds_max"):
        if k in out:
            out[k] = pd.Timestamp(out[k])
    return out


@dataclass
class SourceRef:
    """Lightweight handle to a cached source fit."""

    trace: az.InferenceData
    t_scale_params: dict


def load_or_fit_source(
    smp_train: pd.DataFrame,
    *,
    origin: str,
    method: str,
    seed: int,
    cache_dir: Path,
    progressbar: bool = False,
) -> SourceRef:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"source__{origin}__{method}__seed{seed}".replace("/", "_")
    zarr_path = cache_dir / f"{tag}.zarr"
    json_path = cache_dir / f"{tag}.json"
    if zarr_path.exists() and json_path.exists():
        return SourceRef(
            trace=az.from_zarr(zarr_path),
            t_scale_params=_load_t_scale(json_path.read_text()),
        )
    model = fit_source(
        smp_train, method=method, seed=seed, progressbar=progressbar
    )
    model.trace.to_zarr(zarr_path)
    json_path.write_text(_dump_t_scale(model.t_scale_params))
    return SourceRef(trace=model.trace, t_scale_params=model.t_scale_params)
