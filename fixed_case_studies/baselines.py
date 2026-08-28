"""Classical forecasting baselines for the fixed case studies.

Implements the model-free / classical-statistics baselines of the protocol
(PROTOCOL.md §5, README §3.5): **persistence** (the Relative-MAE
denominator), **drift**, **seasonal naive**, **rolling / global mean**,
**ARIMA** (fixed orders and an AIC-selected search) and **Holt-Winters**.
Per-origin evaluation is done by :func:`common.baseline_unit_metrics`, so
the baseline unit rows have the identical schema as the vangja arms and the
report scripts aggregate them together.

Conventions shared with the rest of the suite:

- forecasts are evaluated on the **frozen train/test splits only** (trading
  days for stocks — no calendar interpolation, review F-9);
- Relative MAE vs persistence is computed in the same scaled space the
  vangja models use, with the same denominator rule (epsilon = 1e-3);
- every origin is seeded and checkpointed, and every artifact bundle
  records provenance (commit, versions, per-baseline wall times);
- the temperature-/S&P-informed *regressions* of the legacy scripts are
  deliberately **not** included: their test-period features would require
  context values beyond the training cutoff — exactly the future leak the
  protocol forbids.

Requires ``statsmodels`` (``uv sync --extra reproducibility``).
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from fixed_case_studies import common

try:  # optional dependency (reproducibility extra)
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ImportError:  # pragma: no cover - clear message instead of a raw ImportError
    ARIMA = None
    ExponentialSmoothing = None

    def _raise_missing() -> None:
        raise RuntimeError(
            "statsmodels is required for the classical baselines. "
            "Install it with:  uv sync --extra reproducibility"
        )


# ---------------------------------------------------------------------------
# Simple model-free baselines
# ---------------------------------------------------------------------------


def persistence(train_y: np.ndarray, horizon: int) -> np.ndarray:
    """Random-walk persistence: last training value carried forward."""
    if len(train_y) == 0:
        return np.full(horizon, np.nan)
    return np.full(horizon, train_y[-1])


def drift(train_y: np.ndarray, horizon: int) -> np.ndarray:
    """Random walk with linear drift: last + slope * h.

    The slope is the per-step average change of the training window,
    ``(y_T - y_1) / (T - 1)`` (the standard drift baseline).
    """
    n = len(train_y)
    if n < 2:
        return persistence(train_y, horizon)
    slope = (train_y[-1] - train_y[0]) / (n - 1)
    return train_y[-1] + slope * np.arange(1, horizon + 1, dtype=float)


def seasonal_naive(train_y: np.ndarray, horizon: int, period: int) -> np.ndarray:
    """Seasonal naive: repeat the last observed seasonal cycle."""
    if len(train_y) < period:
        return persistence(train_y, horizon)
    last = train_y[-period:]
    return np.tile(last, horizon // period + 1)[:horizon]


def seasonal_naive_mean(train_y: np.ndarray, horizon: int, period: int) -> np.ndarray:
    """Seasonal naive with the cycle averaged over all observed cycles."""
    n = len(train_y)
    n_cycles = n // period
    if n_cycles < 1:
        return np.full(horizon, np.nanmean(train_y) if n else np.nan)
    mat = train_y[-(n_cycles * period):].reshape(n_cycles, period)
    avg_cycle = mat.mean(axis=0)
    return np.tile(avg_cycle, horizon // period + 1)[:horizon]


def rolling_mean(train_y: np.ndarray, horizon: int, window: int) -> np.ndarray:
    """Constant forecast = mean of the last ``window`` training values."""
    if len(train_y) == 0:
        return np.full(horizon, np.nan)
    w = min(len(train_y), window)
    return np.full(horizon, train_y[-w:].mean())


def global_mean(train_y: np.ndarray, horizon: int) -> np.ndarray:
    """Constant forecast = mean of the whole training window."""
    return np.full(horizon, np.nanmean(train_y) if len(train_y) else np.nan)


# ---------------------------------------------------------------------------
# ARIMA (statsmodels)
# ---------------------------------------------------------------------------

# (p, d, q) non-seasonal candidates and the seasonal (P, D, Q, s) templates.
_ARIMA_NONSEASONAL = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 1), (2, 1, 2),
                      (0, 1, 1), (1, 0, 1)]
_ARIMA_SEASONAL_TEMPLATES = [(1, 1, 1, 1, 0, 0), (1, 1, 1, 1, 0, 1)]


def fit_arima(
    train_y: np.ndarray,
    horizon: int,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (0, 0, 0, 0),
) -> np.ndarray:
    """Fit a (S)ARIMA model and return the ``horizon``-step forecast.

    Falls back to persistence on any fitting error (single series must
    always produce a finite forecast; failures are per-series, never fatal).
    """
    if ARIMA is None:  # pragma: no cover
        _raise_missing()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(
                train_y,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(method_kwargs={"maxiter": 300})
            forecast = np.asarray(result.forecast(steps=horizon), dtype=float)
    except Exception:
        return persistence(train_y, horizon)
    return forecast


def fit_arima_best(
    train_y: np.ndarray,
    horizon: int,
    seasonal_period: int = 7,
) -> np.ndarray:
    """AIC-selected (S)ARIMA over a small frozen candidate set.

    ``seasonal_period`` is the seasonal cycle length of the data (7 for the
    daily smart-home series, 5 for the trading-day stock series).
    """
    if ARIMA is None:  # pragma: no cover
        _raise_missing()
    best_aic = np.inf
    best_forecast = persistence(train_y, horizon)
    candidates = [(o, (0, 0, 0, 0)) for o in _ARIMA_NONSEASONAL]
    candidates += [
        (tuple(t[:3]), tuple(t[3:]) + (seasonal_period,))
        for t in _ARIMA_SEASONAL_TEMPLATES
    ]
    for order, seasonal_order in candidates:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = ARIMA(
                    train_y,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(method_kwargs={"maxiter": 300})
                if result.aic < best_aic:
                    best_aic = float(result.aic)
                    best_forecast = np.asarray(
                        result.forecast(steps=horizon), dtype=float
                    )
            except Exception:
                continue
    if not np.all(np.isfinite(best_forecast)):
        return persistence(train_y, horizon)
    return best_forecast


# ---------------------------------------------------------------------------
# Holt-Winters (statsmodels)
# ---------------------------------------------------------------------------


def fit_holt_winters(
    train_y: np.ndarray,
    horizon: int,
    seasonal_periods: int = 7,
    trend: str = "add",
    seasonal: str | None = "add",
    damped_trend: bool = False,
) -> np.ndarray:
    """Exponential smoothing forecast; falls back to persistence on error."""
    if ExponentialSmoothing is None:  # pragma: no cover
        _raise_missing()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                train_y,
                seasonal_periods=seasonal_periods,
                trend=trend,
                seasonal=seasonal,
                damped_trend=damped_trend,
                initialization_method="estimated",
            )
            result = model.fit(optimized=True)
            forecast = np.asarray(result.forecast(steps=horizon), dtype=float)
    except Exception:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    train_y, trend="add", initialization_method="estimated"
                )
                result = model.fit(optimized=True)
                forecast = np.asarray(result.forecast(steps=horizon), dtype=float)
        except Exception:
            return persistence(train_y, horizon)
    return forecast


# ---------------------------------------------------------------------------
# Registry (codes -> human-readable descriptions, used by the report)
# ---------------------------------------------------------------------------

BASELINE_DESCRIPTIONS: dict[str, str] = {
    # Generic codes (fallback descriptions).
    "persistence": "Random-walk persistence (last training value; the Relative-MAE denominator)",
    "drift": "Random walk with linear drift estimated from the training window",
    "snaive": "Seasonal naive (repeat the last observed cycle)",
    "snaive_mean": "Seasonal naive (cycle averaged over the training window)",
    "rolling": "Constant forecast = mean of the last k training values",
    "global_mean": "Constant forecast = mean of the whole training window",
    "arima": "Fixed-order ARIMA",
    "arima_best": "ARIMA/SARIMA selected by AIC over a small frozen candidate set",
    "hw": "Holt-Winters exponential smoothing",
    # Smart home (daily, period 7).
    "snaive_7": "Seasonal naive, 7-day cycle",
    "snaive_mean_7": "Seasonal naive mean, 7-day cycle",
    "rolling_7": "Rolling mean (7d)",
    "rolling_30": "Rolling mean (30d)",
    "arima_111": "ARIMA(1,1,1)",
    "arima_211": "ARIMA(2,1,1)",
    "hw_aa_7": "Holt-Winters (A,A,7d)",
    "hw_am_7": "Holt-Winters (A,M,7d)",
    "hw_da_7": "Holt-Winters damped trend (A,A,7d)",
    # Stocks (trading days, period 5).
    "snaive_5": "Seasonal naive, 5 trading days",
    "snaive_mean_5": "Seasonal naive mean, 5 trading days",
    "rolling_5": "Rolling mean (5 trading days)",
    "rolling_21": "Rolling mean (21 trading days)",
    "hw_aa_5": "Holt-Winters (A,A,5d)",
    "hw_am_5": "Holt-Winters (A,M,5d)",
    "hw_da_5": "Holt-Winters damped trend (A,A,5d)",
}


# ---------------------------------------------------------------------------
# Per-origin evaluation + artifact recording
# ---------------------------------------------------------------------------


def evaluate_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    specs: list[tuple[str, str, Callable]],
    origin: str,
    stage: str,
    scale_mode: str,
    *,
    seed: int = common.BASE_SEED,
    exclude_series: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Run every baseline in ``specs`` on one origin.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        Frozen train/test splits (long format ``ds``/``y``/``series``).
    specs : list of (code, human_name, func)
        ``func(train_y, horizon) -> np.ndarray`` forecast. The per-series
        horizon is the length of that series' test window, so series with
        different test lengths (e.g. delisted stocks) are handled.
    origin, stage : str
        Labels recorded in every unit row.
    scale_mode : str
        ``"minmax_individual"`` (smart home) or ``"maxabs_complete"``
        (stocks) — see :func:`common.baseline_unit_metrics`.

    Returns
    -------
    (unit_df, forecasts_df, elapsed)
        ``unit_df`` schema-identical to ``common.unit_metrics`` (one row per
        series x baseline); ``forecasts_df`` long (``ds``, ``series``,
        ``baseline``, ``yhat``); ``elapsed`` per-baseline wall seconds.
    """
    unit_frames: list[pd.DataFrame] = []
    forecast_frames: list[pd.DataFrame] = []
    elapsed: dict[str, float] = {}
    series_names = sorted(train_df["series"].unique())

    for code, _name, func in specs:
        t0 = time.perf_counter()
        fh_parts: list[pd.DataFrame] = []
        for sname in series_names:
            tr = train_df[train_df["series"] == sname].sort_values("ds")
            te = test_df[test_df["series"] == sname].sort_values("ds")
            if tr.empty or te.empty:
                continue
            train_y = tr["y"].values
            horizon = int(len(te))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    yhat = np.asarray(func(train_y, horizon), dtype=float)
                except Exception:
                    yhat = persistence(train_y, horizon)
            if yhat.shape != (horizon,) or not np.all(np.isfinite(yhat)):
                yhat = persistence(train_y, horizon)
            fh_parts.append(
                pd.DataFrame({"ds": te["ds"].values, "series": sname, "yhat": yhat})
            )
        elapsed[code] = time.perf_counter() - t0
        forecasts = (
            pd.concat(fh_parts, ignore_index=True)
            if fh_parts
            else pd.DataFrame(columns=["ds", "series", "yhat"])
        )
        forecast_frames.append(forecasts.assign(baseline=code))
        unit_frames.append(
            common.baseline_unit_metrics(
                train_df,
                test_df,
                forecasts,
                origin=origin,
                baseline=code,
                stage=stage,
                scale_mode=scale_mode,
                seed=seed,
                exclude_series=exclude_series,
            )
        )

    unit_df = (
        pd.concat(unit_frames, ignore_index=True)
        if unit_frames
        else pd.DataFrame()
    )
    forecasts_df = (
        pd.concat(forecast_frames, ignore_index=True)
        if forecast_frames
        else pd.DataFrame(columns=["ds", "series", "baseline", "yhat"])
    )
    return unit_df, forecasts_df, elapsed


def run_baselines_origin(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    specs: list[tuple[str, str, Callable]],
    *,
    origin: str,
    stage: str,
    scale_mode: str,
    study: str,
    out_dir: Path,
    seed: int = common.BASE_SEED,
    exclude_series: set[str] | None = None,
) -> pd.DataFrame:
    """Evaluate all baselines for one origin and record the artifact bundle.

    Checkpointed: an existing ``manifest_*.json`` + ``units_*.csv`` is
    loaded and returned instead of re-running. Every bundle records
    provenance (commit, versions, per-baseline wall times).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"baselines__{origin}__seed{seed}".replace("/", "_")
    manifest_path = out_dir / f"manifest_{tag}.json"
    units_path = out_dir / f"units_{tag}.csv"
    if manifest_path.exists() and units_path.exists():
        return pd.read_csv(units_path)

    unit_df, forecasts_df, elapsed = evaluate_baselines(
        train_df,
        test_df,
        specs,
        origin=origin,
        stage=stage,
        scale_mode=scale_mode,
        seed=seed,
        exclude_series=exclude_series,
    )
    forecasts_df.to_csv(out_dir / f"forecasts_{tag}.csv", index=False)
    unit_df.to_csv(units_path, index=False)
    common.save_json(
        manifest_path,
        {
            "artifact_schema": "vangja-fixed-case-studies-1.0",
            "study": study,
            "stage": stage,
            "origin": origin,
            "config": "baselines",
            "seed": seed,
            "commit": common.commit_sha(),
            "created_at": datetime.now().isoformat(),
            "environment": common.environment_info(),
            "baselines": {
                code: {"name": name, "elapsed_s": round(elapsed.get(code, 0.0), 3)}
                for code, name, _ in specs
            },
            "n_units": int(len(unit_df)),
            "files": {
                "forecasts": f"forecasts_{tag}.csv",
                "units": units_path.name,
            },
        },
    )
    return unit_df
