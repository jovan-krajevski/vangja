"""Pre-registered candidate values for the Prophet-inherited hyperparameters.

These grids are **frozen before any selection results are inspected**
(HYPERPARAMETER_SELECTION.md §4).  Amending them is a protocol change:
update the strategy document first, exactly like amending the frozen
configurations in ``config.py``.

Design rules:
- every grid contains the **current frozen value** (the baseline);
- values are few and a priori plausible (no large grids — the audit's F-1
  was about 2,592-configuration test-set selection);
- sd candidates skew **smaller** than the Prophet defaults, because the
  workflow exists to decide whether the narrower priors are supported;
- ``CURRENT`` mirrors ``config.py`` so the report can state the baseline
  and the recommended delta explicitly.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Smart home (FlatTrend intercept_sd, FourierSeasonality beta_sd / order)
# ---------------------------------------------------------------------------

SMART_HOME_CURRENT: dict[str, float] = {
    "intercept_sd": 0.1,
    "beta_sd": 0.25,
    "yearly_order": 5,
    "weekly_order": 2,
}

SMART_HOME: dict[str, list[float]] = {
    "intercept_sd": [0.1, 0.25, 0.5, 1.0],
    "beta_sd": [0.25, 0.5, 1.0, 1.5],
    "yearly_order": [3, 5, 8],
    "weekly_order": [2, 3, 5],
}

# ---------------------------------------------------------------------------
# Stocks (LinearTrend n_changepoints / slope_sd / intercept_sd,
# FourierSeasonality beta_sd / order)
# ---------------------------------------------------------------------------

STOCKS_CURRENT: dict[str, float] = {
    "n_changepoints": 25,
    "slope_sd": 5.0,
    "intercept_sd": 5.0,
    "beta_sd": 5.0,
    "yearly_order": 6,
    "weekly_order": 3,
}

STOCKS: dict[str, list[float]] = {
    "n_changepoints": [3, 5, 10, 25],
    "slope_sd": [0.5, 1.0, 2.5, 5.0],
    "intercept_sd": [0.5, 1.0, 2.5, 5.0],
    "beta_sd": [1.0, 2.5, 5.0, 10.0],
    "yearly_order": [4, 6, 10],
    "weekly_order": [2, 3, 5],
}

# Parameters whose candidates are prior-scale sd's (Step-1 prior-predictive
# calibration applies) vs. structural counts (LOO-only).
PRIOR_SCALE_PARAMS = {"intercept_sd", "slope_sd", "beta_sd"}
DISCRETE_PARAMS = {"n_changepoints", "yearly_order", "weekly_order"}

# Robustness perturbation around a chosen value (strategy D): +/-50 % for
# continuous parameters, +/-1 step for discrete ones.
CONTINUOUS_PERTURBATION = 0.5
DISCRETE_PERTURBATION = 1

# Default representative subset for the stocks screening (documented device;
# the full study then runs the selected values on the full universe).
STOCKS_SELECTION_ORIGINS = ["2013-01-01", "2013-07-01", "2014-01-01"]
STOCKS_SELECTION_MAX_STOCKS = 10


def assert_current_in_candidates(study: str) -> None:
    """Sanity-check that the frozen baseline is inside every candidate grid.

    Raises
    ------
    ValueError
        If a frozen value is missing from its candidate list.
    """
    grids, current = {
        "smart_home": (SMART_HOME, SMART_HOME_CURRENT),
        "stocks": (STOCKS, STOCKS_CURRENT),
    }[study]
    for param, values in grids.items():
        if current[param] not in values:
            raise ValueError(
                f"{study}: current value {current[param]!r} of '{param}' is not "
                f"in the pre-registered candidate grid {values}. The grid must "
                "contain the frozen baseline."
            )
