"""Strategy C — model averaging instead of hard selection.

HYPERPARAMETER_SELECTION.md §3.2.  Selecting one hyperparameter value
discards hyperparameter uncertainty.  This module computes **pseudo-BMA
weights with a Bayesian bootstrap** and **log-score stacking** from the
pointwise elpd of the candidate fits (Vehtari, Gelman & Gabry 2017; Yao et
al. 2018), and combines per-candidate forecasts into a model-averaged
forecast.  If the weights concentrate on one value, selection and averaging
coincide; if they are diffuse, the paper can report the averaged model and
say the exact choice is immaterial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fixed_case_studies.hyperparams.scoring import pseudo_bma_weights, stacking_weights


def weight_table(pointwise: dict[str, np.ndarray], n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Candidate weights (pseudo-BMA with Bayesian bootstrap + stacking).

    Parameters
    ----------
    pointwise : dict[str, np.ndarray]
        Candidate name -> pointwise elpd (concatenated over units/origins).
    n_boot : int
        Bayesian-bootstrap resamples for the pseudo-BMA weights.
    seed : int
        Seed for the bootstrap.

    Returns
    -------
    pd.DataFrame
        Columns ``[name, elpd, pseudo_bma_weight, stacking_weight]`` sorted
        by elpd.
    """
    names = list(pointwise)
    if not names:
        return pd.DataFrame(columns=["name", "elpd", "pseudo_bma_weight", "stacking_weight"])
    # Drop candidates with no pointwise elpd (failed fits) before stacking.
    pointwise = {n: np.asarray(v, dtype=float) for n, v in pointwise.items() if np.asarray(v).size > 0}
    names = list(pointwise)
    if not names:
        return pd.DataFrame(columns=["name", "elpd", "pseudo_bma_weight", "stacking_weight"])
    pbm = pseudo_bma_weights(pointwise, n_boot=n_boot, seed=seed)
    stk = stacking_weights(pointwise)
    table = pd.DataFrame(
        {
            "name": names,
            "elpd": [float(np.sum(pointwise[n])) for n in names],
            "pseudo_bma_weight": pbm,
            "stacking_weight": stk,
        }
    )
    return table.sort_values("elpd", ascending=False).reset_index(drop=True)


def average_forecasts(
    forecasts: dict[str, np.ndarray], weights: np.ndarray, names: list[str]
) -> np.ndarray:
    """Weighted average of per-candidate forecast arrays.

    Parameters
    ----------
    forecasts : dict[str, np.ndarray]
        Candidate name -> forecast array (any consistent shape, e.g.
        ``(n_groups, n_timesteps)`` per candidate).
    weights : np.ndarray
        Candidate weights (same order as ``names``).
    names : list[str]
        Candidate names (same order as ``weights``).

    Returns
    -------
    np.ndarray
        Weighted average forecast.
    """
    total = None
    for w, name in zip(weights, names):
        arr = np.asarray(forecasts[name], dtype=float)
        total = arr * w if total is None else total + arr * w
    return total


def averaging_report(
    out_dir, study: str, param: str, weight_df: pd.DataFrame
) -> None:
    """Write the model-averaging markdown fragment (weights + verdict)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# {study}: model averaging for `{param}`", ""]
    if weight_df.empty:
        lines.append("_no candidate fits succeeded_")
    else:
        lines.append("| name | elpd | pseudo-BMA weight | stacking weight |")
        lines.append("|---|---|---|---|")
        for _, row in weight_df.iterrows():
            lines.append(
                f"| {row['name']} | {row['elpd']:.2f} | {row['pseudo_bma_weight']:.3f} "
                f"| {row['stacking_weight']:.3f} |"
            )
        best = weight_df.iloc[0]
        w = float(best["pseudo_bma_weight"])
        verdict = (
            "weights are concentrated: selection and averaging coincide"
            if w > 0.8
            else "weights are diffuse: the data cannot distinguish the "
            "candidates; report the averaged model"
        )
        lines.append("")
        lines.append(f"**Verdict:** {verdict}.")
    lines.append("")
    (out_dir / f"{study}_{param}_averaging.md").write_text("\n".join(lines))
