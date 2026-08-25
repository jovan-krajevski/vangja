"""Frozen configuration registry for the smart-home case study.

Same design principle as the stocks study: **no grid search, no test-set
selection**.  One a priori configuration plus a small frozen ablation list.

The smart-home study has no independent confirmation data (PROTOCOL.md
§3): every number it produces is labelled **retrospective development**.
The primary split is train = 91 days (2016-01-01 .. 2016-03-31), horizon =
260 days (2016-04-01 .. 2016-12-16); two additional rolling origins with
shorter horizons are run as sensitivities.

An important protocol correction vs. the original runner: the Boston
temperature context is now cut off at the **end of the target training
window** of each origin (the original runner loaded it through the end of
the test period, which leaks the future into the transferred seasonality).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Data constants
# ---------------------------------------------------------------------------

SMART_HOME_COLUMNS = [
    "Furnace 1 [kW]",
    "Furnace 2 [kW]",
    "Fridge [kW]",
    "Wine cellar [kW]",
]
TEMP_CITY = "Boston"
TEMP_START = "2013-01-01"  # maximum available context (stable physics)
DATA_END = "2016-12-16"    # last available smart-home date

# Primary split: first 91 days train, remaining 260 days test.
PRIMARY_TRAIN_CUTOFF = "2016-04-01"
# Rolling-origin sensitivities: 91-day training windows, shorter horizons.
ROLLING_ORIGINS = [
    {"train_start": "2016-01-01", "train_end": "2016-04-01", "label": "origin_apr"},
    {"train_start": "2016-04-01", "train_end": "2016-07-01", "label": "origin_jul"},
]
HORIZON_DAYS = 260  # primary horizon (Apr 1 .. Dec 16)

# ---------------------------------------------------------------------------
# Configuration registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmartHomeConfig:
    name: str
    description: str
    tune_method: str | None      # None disables transfer from temperature
    loss_factor: float           # seasonal amplitude-cap factor
    yearly_pool: str             # "partial" or "individual"
    weekly_pool: str
    shrinkage: int
    uniform_constant: bool       # sign-flipping factor on the yearly seasonality
    intercept_sd: float
    beta_sd: float
    include_context_group: bool = False


CONFIGS: dict[str, SmartHomeConfig] = {
    "main": SmartHomeConfig(
        name="main",
        description=(
            "Transfer of the yearly seasonality from the Boston temperature "
            "posterior (prior_from_idata) + partial pooling + sign-flipping "
            "uniform constant; no regularisation potential."
        ),
        tune_method="prior_from_idata",
        loss_factor=0.0,
        yearly_pool="partial",
        weekly_pool="partial",
        shrinkage=1,
        uniform_constant=True,
        intercept_sd=0.5,
        beta_sd=1.5,
    ),
    # --- Reduced transfer x hierarchy design (PROTOCOL.md §7) ---------------
    "no_transfer": SmartHomeConfig(
        name="no_transfer",
        description=(
            "Arm 2: hierarchical pooling only, no transfer (honest relabel of "
            "the former 'TimeSeers' arm)."
        ),
        tune_method=None,
        loss_factor=0.0,
        yearly_pool="partial",
        weekly_pool="partial",
        shrinkage=1,
        uniform_constant=True,
        intercept_sd=0.5,
        beta_sd=1.5,
    ),
    "target_only": SmartHomeConfig(
        name="target_only",
        description=(
            "Arm 1: individual pooling, no transfer, no context group "
            "(isolated per-appliance Prophet-like Vangja; honest relabel of "
            "the former 'Prophet' arm)."
        ),
        tune_method=None,
        loss_factor=0.0,
        yearly_pool="individual",
        weekly_pool="individual",
        shrinkage=1,
        uniform_constant=False,
        intercept_sd=0.5,
        beta_sd=1.5,
    ),
    "context_group": SmartHomeConfig(
        name="context_group",
        description=(
            "Arm 3: temperature included as an additional hierarchical group "
            "(co-learning) but no transfer prior."
        ),
        tune_method=None,
        loss_factor=0.0,
        yearly_pool="partial",
        weekly_pool="partial",
        shrinkage=1,
        uniform_constant=True,
        intercept_sd=0.5,
        beta_sd=1.5,
        include_context_group=True,
    ),
    # --- Single-perturbation ablations of the main configuration ------------
    "regularised": SmartHomeConfig(
        name="regularised",
        description=(
            "Main configuration with the seasonal amplitude-cap potential "
            "enabled (phi=1). Active here: the 91-day window is shorter than "
            "half the yearly period."
        ),
        tune_method="prior_from_idata",
        loss_factor=1.0,
        yearly_pool="partial",
        weekly_pool="partial",
        shrinkage=1,
        uniform_constant=True,
        intercept_sd=0.5,
        beta_sd=1.5,
    ),
    "uniform_constant_off": SmartHomeConfig(
        name="uniform_constant_off",
        description="Main configuration without the sign-flipping uniform constant.",
        tune_method="prior_from_idata",
        loss_factor=0.0,
        yearly_pool="partial",
        weekly_pool="partial",
        shrinkage=1,
        uniform_constant=False,
        intercept_sd=0.5,
        beta_sd=1.5,
    ),
    "shrinkage_10": SmartHomeConfig(
        name="shrinkage_10",
        description="Main configuration with shrinkage strength 10 instead of 1.",
        tune_method="prior_from_idata",
        loss_factor=0.0,
        yearly_pool="partial",
        weekly_pool="partial",
        shrinkage=10,
        uniform_constant=True,
        intercept_sd=0.5,
        beta_sd=1.5,
    ),
}

MAIN_MATRIX = list(CONFIGS.keys())
FINALIST_PAIRS = [("main", "no_transfer")]

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def train_test_for_split(train_end: str, include_temp: bool = False):
    """Load smart-home (and optionally temperature) data cut at ``train_end``.

    The temperature context always ends at ``train_end`` (no future leak).
    """
    from vangja.datasets import load_kaggle_temperature, load_smart_home_readings

    sh_df = load_smart_home_readings(column=SMART_HOME_COLUMNS, freq="D")
    train = sh_df[sh_df["ds"] < train_end].copy()
    test = sh_df[
        (sh_df["ds"] >= train_end) & (sh_df["ds"] <= DATA_END)
    ].copy()

    temp_train = load_kaggle_temperature(
        city=TEMP_CITY,
        start_date=TEMP_START,
        end_date=pd.Timestamp(train_end) - pd.Timedelta(days=1),
        freq="D",
    )
    return train, test, temp_train
