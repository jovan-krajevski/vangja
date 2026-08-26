"""Shared fitting/evaluation pipeline for the smart-home case study.

The study is **retrospective** (PROTOCOL.md §3): the 2016 period was already
evaluated in the original study, so every output is labelled
"retrospective" and no confirmatory claims are made from it.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import pandas as pd

from vangja import FlatTrend, FourierSeasonality, UniformConstant

from fixed_case_studies import common
from fixed_case_studies.smart_home.config import (
    HORIZON_DAYS,
    SmartHomeConfig,
    train_test_for_split,
)

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_temp_model():
    """Boston temperature context model (flat trend + yearly seasonality)."""
    return FlatTrend(intercept_mean=0.5, intercept_sd=0.1) + FourierSeasonality(
        period=365.25, series_order=5, beta_sd=1.5
    )


def fit_temp_model(temp_train: pd.DataFrame, scaler: str, seed: int, progressbar: bool):
    model = build_temp_model()
    model.fit(
        temp_train,
        scaler=scaler,
        method="nuts",
        samples=1000,
        tune=1000,
        chains=2,
        cores=2,
        nuts_sampler="nutpie",
        target_accept=0.9,
        random_seed=seed,
        progressbar=progressbar,
    )
    return model


def load_or_fit_temp_model(
    temp_train: pd.DataFrame,
    *,
    split_label: str,
    scaler: str,
    seed: int,
    cache_dir: Path,
    progressbar: bool = False,
):
    """Cached temperature fit (NUTS), keyed by split x seed.

    All configs of one split transfer the identical temperature posterior.
    """
    import json
    import pickle

    def _dump(params: dict) -> str:
        out = dict(params)
        for k in ("ds_min", "ds_max"):
            if k in out and hasattr(out[k], "isoformat"):
                out[k] = out[k].isoformat()
        return json.dumps(out)

    def _load(text: str) -> dict:
        out = json.loads(text)
        for k in ("ds_min", "ds_max"):
            if k in out:
                out[k] = pd.Timestamp(out[k])
        return out

    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"temp__{split_label}__seed{seed}"
    pkl_path = cache_dir / f"{tag}.pkl"
    json_path = cache_dir / f"{tag}.json"
    if pkl_path.exists() and json_path.exists():
        model = build_temp_model()
        # Rehydrate only the attributes the target fit consumes.
        with open(pkl_path, "rb") as fh:
            model.trace = pickle.load(fh)
        model.t_scale_params = _load(json_path.read_text())
        return model
    model = fit_temp_model(
        temp_train, scaler=scaler, seed=seed, progressbar=progressbar
    )
    with open(pkl_path, "wb") as fh:
        pickle.dump(model.trace, fh)
    json_path.write_text(_dump(model.t_scale_params))
    return model


def build_target_model(cfg: SmartHomeConfig):
    trend = FlatTrend(
        intercept_mean=0.5, intercept_sd=cfg.intercept_sd, pool_type="individual"
    )
    yearly = FourierSeasonality(
        period=365.25,
        series_order=5,
        beta_sd=cfg.beta_sd,
        pool_type=cfg.yearly_pool,
        tune_method=cfg.tune_method,
        loss_factor_for_tune=cfg.loss_factor,
        shrinkage_strength=cfg.shrinkage,
    )
    weekly = FourierSeasonality(
        period=7,
        series_order=3,
        beta_sd=cfg.beta_sd,
        pool_type=cfg.weekly_pool,
        shrinkage_strength=cfg.shrinkage,
    )
    constant = UniformConstant(
        lower=-1, upper=1,
        pool_type=cfg.yearly_pool,
        shrinkage_strength=cfg.shrinkage,
    )
    if cfg.uniform_constant:
        return trend + constant * yearly + weekly
    return trend + yearly + weekly


def fit_target(
    cfg: SmartHomeConfig,
    train_df: pd.DataFrame,
    temp_model,
    temp_train_df: pd.DataFrame,
    *,
    seed: int,
    progressbar: bool = False,
):
    model = build_target_model(cfg)
    kwargs = dict(
        scaler="minmax",
        method="mapx",
        scale_mode="individual",
        sigma_pool_type="individual",
        t_scale_params=temp_model.t_scale_params,
        random_seed=seed,
        progressbar=progressbar,
    )
    if cfg.tune_method is not None:
        kwargs["idata"] = temp_model.trace
    if cfg.include_context_group:
        kwargs["include_source_in_target"] = True
        kwargs["source_data"] = temp_train_df
    model.fit(train_df, **kwargs)
    return model


# ---------------------------------------------------------------------------
# One origin x config cell
# ---------------------------------------------------------------------------


def run_cell(
    cfg: SmartHomeConfig,
    split: dict,
    *,
    seed: int,
    out_dir: Path,
    progressbar: bool = False,
) -> pd.DataFrame:
    """Fit temperature + target for one (origin, config) cell and record
    the full artifact bundle. Returns the per-unit metric rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{cfg.name}__{split['label']}__seed{seed}"
    if (out_dir / f"manifest_{tag}.json").exists():
        units_path = out_dir / f"units_{tag}.csv"
        if units_path.exists():
            return pd.read_csv(units_path)

    train, test, temp_train = train_test_for_split(split["train_end"])

    try:
        temp_model = load_or_fit_temp_model(
            temp_train,
            split_label=split["label"],
            scaler="minmax",
            seed=common.BASE_SEED,
            cache_dir=out_dir / "temp_cache",
            progressbar=progressbar,
        )
        model = fit_target(
            cfg, train, temp_model, temp_train, seed=seed,
            progressbar=progressbar,
        )
        horizon = split.get("horizon", HORIZON_DAYS)
        yhat = model.predict(horizon=horizon)
        unit_df = common.unit_metrics(
            model, test, yhat, origin=split["label"],
            config=cfg.name, stage="retrospective",
        )
        unit_df["seed"] = seed
        common.record_artifact(
            out_dir,
            study="smart_home",
            stage="retrospective",
            origin=split["label"],
            config=cfg.name,
            model=model,
            unit_df=unit_df,
            future_df=yhat,
            extra={
                "split": split,
                "temp_context_end": temp_train["ds"].max().strftime("%Y-%m-%d"),
                "horizon": horizon,
            },
        )
        return unit_df
    except Exception as err:
        common.record_failure(
            out_dir,
            study="smart_home",
            stage="retrospective",
            origin=split["label"],
            config=cfg.name,
            seed=seed,
            error=err,
            traceback_text=traceback.format_exc(),
        )
        raise


def seed_for(origin_index: int, config_index: int, base: int = common.BASE_SEED) -> int:
    return base + origin_index * 100 + config_index
