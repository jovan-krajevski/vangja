"""Strategy B — leave-future-out time-series CV with CRPS.

HYPERPARAMETER_SELECTION.md §3.1.  PSIS-LOO is optimistic for autocorrelated
time series; this strategy evaluates candidates by **expanding-window
forecasts inside the training window**: train on the first ``k`` days,
predict the next ``h`` days, score the full posterior-predictive ensemble
with the proper scoring rule CRPS, extend the window, repeat.  The test
horizon is never used: the held-out blocks are segments of the training
window itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fixed_case_studies.hyperparams.scoring import crps_ensemble


def expanding_folds(
    train: pd.DataFrame, n_initial: int, step: int, horizon: int
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window (train_block, held_out_block) folds, split by date.

    Splits on the **unique dates** of ``train`` (multi-series data has one
    row per series per date), so every fold contains all series for the
    training/held-out dates.

    Parameters
    ----------
    train : pd.DataFrame
        Training data with a ``ds`` column.
    n_initial : int
        Number of **dates** in the first training block.
    step : int
        Number of dates added between folds.
    horizon : int
        Number of held-out **dates** scored per fold.

    Returns
    -------
    list[tuple[pd.DataFrame, pd.DataFrame]]
        ``(train_block, held_out_block)`` pairs; the last block is dropped if
        it would be shorter than ``horizon``.
    """
    dates = pd.to_datetime(train["ds"]).sort_values().unique()
    folds = []
    k = n_initial
    while k + horizon <= len(dates):
        tr_dates = dates[:k]
        hold_dates = dates[k : k + horizon]
        folds.append(
            (
                train[pd.to_datetime(train["ds"]).isin(tr_dates)],
                train[pd.to_datetime(train["ds"]).isin(hold_dates)],
            )
        )
        k += step
    return folds


def predictive_draws(model, future: pd.DataFrame, n_samples: int = 200, seed: int = 42):
    """Propagate posterior draws through the model to a future frame.

    Mirrors the draw propagation inside ``TimeSeriesModel.predict_uncertainty``
    but returns the full ensemble ``(n_samples, n_groups, n_timesteps)``
    instead of quantiles, so any proper scoring rule can be applied.

    Parameters
    ----------
    model : TimeSeriesModel
        A fitted model with ``model.trace`` (MCMC or VI).
    future : pd.DataFrame
        Future frame with ``ds`` and ``t`` columns.
    n_samples : int
        Number of posterior draws to propagate.
    seed : int
        Seed for the draw subsampling.

    Returns
    -------
    np.ndarray
        ``(n_samples, n_groups, n_timesteps)`` predictive ensemble.
    """
    posterior = model.trace.posterior
    n_chains = posterior.sizes["chain"]
    n_draws = posterior.sizes["draw"]
    rng = np.random.default_rng(seed)
    flat = rng.choice(n_chains * n_draws, size=n_samples, replace=False)
    outs = []
    for f in flat:
        ci, di = int(f % n_chains), int(f // n_chains)
        draw = {var: posterior[var].values[ci, di] for var in posterior.data_vars}
        outs.append(model._predict(future[["ds", "t"]], "mapx", draw, None))
    return np.stack(outs)


def fold_future(model, held_out: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Future frame for a held-out block on the fitted model's time scale.

    ``t`` is computed on ``model.t_scale_params`` (which for the stocks study
    is the *context* time scale — the same scale the study forecasts on), so
    the exact held-out ``ds`` values are mapped to the model's ``t`` without
    positional slicing.
    """
    ds_min = pd.Timestamp(model.t_scale_params["ds_min"])
    ds_max = pd.Timestamp(model.t_scale_params["ds_max"])
    t = (pd.to_datetime(held_out["ds"]) - ds_min) / (ds_max - ds_min)
    return pd.DataFrame({"ds": pd.to_datetime(held_out["ds"]).values, "t": t.values})


def score_fold(
    model,
    held_out: pd.DataFrame,
    *,
    n_samples: int = 200,
    seed: int = 42,
    freq: str = "D",
) -> dict:
    """Score one fold: mean CRPS + MAE of the predictive mean (scaled space).

    Returns
    -------
    dict with keys ``crps``, ``mae``, ``n_groups``, ``n_points``.
    """
    future = fold_future(model, held_out, freq=freq)
    # Align the ensemble to the observed held-out points (match by ds).
    future = future.set_index("ds")
    held = held_out.set_index("ds")
    common_idx = future.index.intersection(held.index)
    if len(common_idx) == 0:
        return {"crps": float("nan"), "mae": float("nan"), "n_groups": 0, "n_points": 0}
    ens = predictive_draws(
        model, future.loc[common_idx].reset_index(), n_samples=n_samples, seed=seed
    )
    y = held.loc[common_idx, "y"].to_numpy()

    crps_vals, mae_vals = [], []
    for g in range(ens.shape[1]):
        for j in range(ens.shape[2]):
            crps_vals.append(crps_ensemble(ens[:, g, j], float(y[j])))
            mae_vals.append(abs(float(np.mean(ens[:, g, j])) - float(y[j])))
    if not crps_vals:
        return {
            "crps": float("nan"),
            "mae": float("nan"),
            "n_groups": ens.shape[1],
            "n_points": 0,
        }
    return {
        "crps": float(np.mean(crps_vals)),
        "mae": float(np.mean(mae_vals)),
        "n_groups": int(ens.shape[1]),
        "n_points": int(len(crps_vals)),
    }


def run_ts_cv(
    model_factory,
    train: pd.DataFrame,
    param: str,
    candidates: list[float],
    *,
    fit_kwargs: dict,
    n_initial_frac: float = 0.6,
    step: int = 7,
    horizon: int = 14,
    n_samples: int = 150,
    seed: int = 42,
    progressbar: bool = False,
    name_fn=None,
) -> pd.DataFrame:
    """Leave-future-out CRPS comparison of candidate values.

    Parameters
    ----------
    model_factory : callable
        ``model_factory(**{param: value}) -> TimeSeriesModel``.
    train : pd.DataFrame
        Training window only (columns ``ds``, ``y``, optionally ``series``).
    param : str
        Hyperparameter name.
    candidates : list[float]
        Candidate values.
    fit_kwargs : dict
        Forwarded to ``model.fit``; must include ``method`` (ADVI for speed).
    n_initial_frac : float
        Fraction of the training window in the first fold's training block.
    step : int
        Observations added per fold.
    horizon : int
        Held-out horizon per fold (in observations).
    n_samples : int
        Posterior draws propagated per fold.

    Returns
    -------
    pd.DataFrame
        One row per candidate: ``[param, crps, mae, n_folds, wall_time_s]``.
    """
    name_fn = name_fn or (lambda p, v: f"{p}={v}")
    n_initial = max(10, int(pd.to_datetime(train["ds"]).nunique() * n_initial_frac))
    folds = expanding_folds(train, n_initial, step, horizon)
    if not folds:
        raise ValueError(
            f"training window of {len(train)} obs is too short for folds "
            f"(n_initial={n_initial}, horizon={horizon})"
        )
    freq = "B" if pd.Series(train["ds"]).diff().dropna().dt.days.median() < 7 else "D"

    rows = []
    for value in candidates:
        name = name_fn(param, value)
        crps_list, mae_list = [], []
        import time

        t0 = time.perf_counter()
        for train_block, held_out in folds:
            model = model_factory(**{param: value})
            kwargs = dict(fit_kwargs)
            kwargs.setdefault("random_seed", seed)
            kwargs.setdefault("progressbar", progressbar)
            model.fit(train_block, **kwargs)
            score = score_fold(
                model, held_out, n_samples=n_samples, seed=seed, freq=freq
            )
            if np.isfinite(score["crps"]):
                crps_list.append(score["crps"])
                mae_list.append(score["mae"])
        wall = time.perf_counter() - t0
        rows.append(
            {
                "name": name,
                param: value,
                "crps": float(np.mean(crps_list)) if crps_list else float("nan"),
                "mae": float(np.mean(mae_list)) if mae_list else float("nan"),
                "n_folds": len(folds),
                "wall_time_s": wall,
            }
        )
    return pd.DataFrame(rows).sort_values("crps").reset_index(drop=True)
