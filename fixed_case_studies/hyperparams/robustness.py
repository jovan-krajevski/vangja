"""Strategy D — robustness / flatness sweep around the chosen values.

HYPERPARAMETER_SELECTION.md §3.3.  The paper's stated design goal is models
"resilient to changes in these parameters".  This module perturbs each
chosen hyperparameter by ±50 % (continuous) or ±1 step (discrete), refits,
and reports the range of PSIS-LOO elpd over the perturbation.  If the
chosen value sits in a flat region (range within ~1 dse, or small relative
to the effect of the next hyperparameter), the exact number is not
over-interpreted — a neighbouring value behaves identically, which
neutralises "you tuned on the development set".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fixed_case_studies.hyperparams import bayesian
from fixed_case_studies.hyperparams.candidates import (
    CONTINUOUS_PERTURBATION,
    DISCRETE_PERTURBATION,
    DISCRETE_PARAMS,
)


def perturb_values(value, *, continuous: bool) -> list[float]:
    """Values around ``value``: {value*(1-p), value, value*(1+p)} or
    {value-1, value, value+1} for discrete parameters (floored at >=1)."""
    if not continuous:
        return [max(1, int(value) - DISCRETE_PERTURBATION), int(value), int(value) + DISCRETE_PERTURBATION]
    lo = max(1e-3, value * (1.0 - CONTINUOUS_PERTURBATION))
    hi = value * (1.0 + CONTINUOUS_PERTURBATION)
    return [round(lo, 4), value, round(hi, 4)]


def robustness_sweep(
    model_factory,
    frames: list[tuple[str, pd.DataFrame]],
    param: str,
    chosen: float,
    *,
    fit_kwargs: dict | None = None,
    fit_kwargs_fn=None,
    seed: int = 42,
    cache_dir: Path | None = None,
    cache_tag: str = "",
    progressbar: bool = False,
) -> tuple[pd.DataFrame, str]:
    """PSIS-LOO over a perturbation of ``chosen``; returns (table, verdict).

    Verdict:
    - "flat" if the elpd range over the perturbation is within 1 dse of the
      best perturbed value — the choice is robust;
    - "sensitive" otherwise (report the best neighbour; consider it).
    """
    continuous = param not in DISCRETE_PARAMS
    values = perturb_values(chosen, continuous=continuous)
    table, evidence = bayesian.run_loo_selection(
        model_factory,
        frames,
        param,
        values,
        fit_kwargs=fit_kwargs,
        fit_kwargs_fn=fit_kwargs_fn,
        seed=seed,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
        progressbar=progressbar,
    )
    if table.empty:
        return table, "no fits succeeded"
    best = table.loc[table["elpd"].idxmax()]
    worst = table.loc[table["elpd"].idxmin()]
    span = float(best["elpd"]) - float(worst["elpd"])
    dse = float(worst["dse"]) if len(table) > 1 else float(best["se"])
    verdict = (
        f"flat (elpd range {span:.2f} <= {dse:.2f} dse): the choice of "
        f"{param}={chosen} is robust to ±{CONTINUOUS_PERTURBATION if continuous else DISCRETE_PERTURBATION} perturbation"
        if span <= dse
        else f"sensitive (elpd range {span:.2f} > {dse:.2f} dse): reconsider {param}={chosen}"
    )
    return table, verdict


def write_robustness_report(
    out_dir: Path, study: str, param: str, table: pd.DataFrame, verdict: str, chosen: float
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{study}_{param}_robustness.md"
    text = "\n".join(
        [
            f"# {study}: robustness of `{param}={chosen}`",
            "",
            f"**Verdict:** {verdict}.",
            "",
            bayesian._table_md(table) if table is not None and not table.empty else "_no fits_",
            "",
        ]
    )
    path.write_text(text)
    return path
