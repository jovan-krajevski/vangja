"""Strategy E — full-Bayes hyperpriors on the prior scales.

HYPERPARAMETER_SELECTION.md §3.4.  Instead of *selecting* ``intercept_sd`` /
``beta_sd``, put a prior on them and sample them jointly with the model
parameters; the posterior of the sd's *is* the data-supported value, with
uncertainty.  Implemented as a demonstration on the single-series
flat-trend + seasonality structure (the smart-home building block) with
NUTS.  Discrete hyperparameters (``n_changepoints``, ``series_order``) are
not covered here — for those, strategy C (BMA over candidates) is the
practical equivalent.

The model is built directly in PyMC (not through vangja components) because
the components require *fixed* prior sd's; the scaling replicates the
smart-home study's per-series min-max scaling, so the sd posteriors are on
the study's scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

EPOCH = pd.Timestamp("1970-01-01")


def _fourier(df: pd.DataFrame, period: float, order: int) -> np.ndarray:
    """Fourier features on the study's daily time index (like the package)."""
    t = (df["ds"] - EPOCH).dt.days.values.astype(float)
    x_T = t * np.pi * 2
    out = np.empty((len(df), 2 * order))
    for i in range(order):
        c = x_T * (i + 1) / period
        out[:, 2 * i] = np.sin(c)
        out[:, 2 * i + 1] = np.cos(c)
    return out


def fit_full_bayes(
    df: pd.DataFrame,
    *,
    yearly_order: int = 5,
    weekly_order: int = 3,
    hyperprior_sd: float = 1.0,
    samples: int = 600,
    tune: int = 600,
    chains: int = 2,
    seed: int = 42,
    progressbar: bool = False,
):
    """Fit the flat-trend + seasonality model with hyperpriors on the sd's.

    Parameters
    ----------
    df : pd.DataFrame
        Single-series training data (columns ``ds``, ``y``).
    yearly_order, weekly_order : int
        Fourier orders (fixed — they are selected by the LOO workflow).
    hyperprior_sd : float
        Scale of the ``HalfNormal`` hyperpriors on the sd's.
    samples, tune, chains : int
        NUTS settings (small — this is a small model).
    seed : int
        Seeded for reproducibility.

    Returns
    -------
    tuple[pm.Model, az.InferenceData, dict]
        ``(model, trace, scale)`` where ``scale = {"y_min", "y_max"}`` is the
        per-series min-max scaling applied to ``y``.
    """
    df = df.sort_values("ds").reset_index(drop=True)
    y_min, y_max = float(df["y"].min()), float(df["y"].max())
    y = (df["y"] - y_min) / (y_max - y_min)
    t = (df["ds"] - df["ds"].min()) / (df["ds"].max() - df["ds"].min())

    x_yearly = _fourier(df, 365.25, yearly_order)
    x_weekly = _fourier(df, 7, weekly_order)

    with pm.Model() as model:
        # Hyperpriors on the Prophet-inherited prior scales (HalfNormal).
        intercept_sd = pm.HalfNormal("intercept_sd", sigma=hyperprior_sd)
        beta_sd_yearly = pm.HalfNormal("beta_sd_yearly", sigma=hyperprior_sd)
        beta_sd_weekly = pm.HalfNormal("beta_sd_weekly", sigma=hyperprior_sd)

        intercept = pm.Normal("intercept", mu=0.5, sigma=intercept_sd)
        beta_y = pm.Normal(
            "beta_yearly", mu=0.0, sigma=beta_sd_yearly, shape=2 * yearly_order
        )
        beta_w = pm.Normal(
            "beta_weekly", mu=0.0, sigma=beta_sd_weekly, shape=2 * weekly_order
        )
        sigma = pm.HalfNormal("sigma", sigma=0.5)

        mu = (
            intercept
            + pt.dot(x_yearly, beta_y)
            + pt.dot(x_weekly, beta_w)
        )
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            samples,
            tune=tune,
            chains=chains,
            random_seed=seed,
            progressbar=progressbar,
        )

    scale = {"y_min": y_min, "y_max": y_max}
    return model, trace, scale


def report_summary(model, trace, out_dir: Path, label: str) -> Path:
    """Write the hyperprior posterior summary (mean + HDI) as markdown."""
    import arviz as az

    out_dir.mkdir(parents=True, exist_ok=True)
    var_names = ["intercept_sd", "beta_sd_yearly", "beta_sd_weekly"]
    try:
        summary = az.summary(trace, var_names=var_names, hdi_prob=0.9)
    except TypeError:  # arviz >= 1.x renamed hdi_prob -> ci_prob
        summary = az.summary(trace, var_names=var_names, ci_prob=0.9)
    # Locate the interval columns regardless of the arviz generation
    # (hdi_5%/hdi_95%, ci_5%/ci_95%, or eti90_lb/eti90_ub in arviz 1.x).
    lo = next(
        (
            c
            for c in summary.columns
            if c.startswith(("hdi", "ci")) and "5%" in c or c.endswith("_lb")
        ),
        None,
    )
    hi = next(
        (
            c
            for c in summary.columns
            if c.startswith(("hdi", "ci")) and "95%" in c or c.endswith("_ub")
        ),
        None,
    )
    if lo is None or hi is None:
        lo, hi = summary.columns[-2], summary.columns[-1]
    path = out_dir / f"full_bayes_{label}.md"
    lines = [
        f"# Full-Bayes hyperpriors: posterior of the prior scales ({label})",
        "",
        "The sd's are sampled jointly with the model parameters "
        "(HalfNormal(1) hyperpriors); the posterior is the data-supported "
        "regularisation scale on the min-max scaled target.",
        "",
        f"| param | mean | sd | {lo} | {hi} |",
        "|---|---|---|---|---|",
    ]
    for name in var_names:
        row = summary.loc[name]
        lines.append(
            f"| {name} | {row['mean']:.3f} | {row['sd']:.3f} | "
            f"{row[lo]:.3f} | {row[hi]:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))
    return path
