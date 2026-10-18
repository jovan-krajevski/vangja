"""Frozen configuration registry for the stocks case study.

Design principle (the core change vs. the original study):

    There is **no hyperparameter grid and no test-horizon selection**.
    Exactly one configuration — chosen *a priori* on modelling grounds
    (and informed by the retrospective study, disclosed as such) — plus a
    small, frozen list of single-perturbation ablations are run.  No
    configuration is ever re-selected based on confirmation results.

The choice of each frozen value is justified in `fixed_case_studies/README.md`
(§"Frozen configurations and why").  This file is part of the pre-registered
protocol: do not change values here without amending the README and
re-hashing the confirmation origins.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data constants
# ---------------------------------------------------------------------------

DEV_ORIGINS = list(
    __import__("pandas").date_range("2013-01-01", "2014-12-01", freq="MS")
    .strftime("%Y-%m-%d")
    .tolist()
)
"""The original 24 origins. Retrospective development/diagnostic data only."""

TARGET_WINDOW_DAYS = 91   # target training window (calendar days)
HORIZON_DAYS = 365        # forecast horizon (calendar days)
CONTEXT_WINDOW_DAYS = 1460  # S&P 500 context window (4 years, calendar days)
CONTEXT_TICKER = "^GSPC"
NEGATIVE_CONTROL_TICKER = "GC=F"  # gold futures: unrelated context (P1-9)

# High-index-weight mega caps for the targeted S&P-overlap sensitivity.
# Frozen list; weights vary over time, so this is an approximation, disclosed
# in the README.
HIGH_WEIGHT_STOCKS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META",
    "BRK-B", "TSLA", "AVGO", "LLY", "JPM", "XOM",
]

MIN_TRAIN_OBS = 50  # required trading-day observations in the 91-day window

# ---------------------------------------------------------------------------
# Configuration registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StockConfig:
    name: str
    description: str
    # Transfer (from the context posterior). None disables transfer.
    tune_method: str | None
    # Trend regularization factor (negative-quadratic potential).
    lt_loss_factor: float
    # Seasonal regularization factor (one-sided amplitude cap).
    fs_loss_factor: float
    # Pooling: "partial" (hierarchical) or "individual".
    lt_pool: str
    fs_pool: str
    # Shrinkage strengths for the hierarchical components.
    lt_shrinkage: int
    fs_shrinkage: int
    # Context series concatenated as an additional hierarchical group.
    include_context_group: bool = False
    # Use the negative-control context instead of the S&P 500.
    negative_control: bool = False
    # Restrict to the high-weight subset (targeted sensitivity only).
    high_weight_subset: bool = False
    # Number of trend changepoints for the *target* model. The source model
    # keeps the Prophet default of 25 (it has ~1000 observations); on a
    # ~63-observation target window fewer changepoints are defensible.
    n_changepoints: int = 25


# The frozen main configuration. Every value is justified in the README.
CONFIGS: dict[str, StockConfig] = {
    "main": StockConfig(
        name="main",
        description=(
            "Transfer from S&P 500 posterior (prior_from_idata) + partial "
            "pooling; trend regularised (phi=1); seasonal amplitude cap off."
        ),
        tune_method="prior_from_idata",
        lt_loss_factor=1.0,
        fs_loss_factor=0.0,
        lt_pool="partial",
        fs_pool="partial",
        lt_shrinkage=100,
        fs_shrinkage=100,
    ),
    # --- The reduced transfer x hierarchy design (PROTOCOL.md §7) ----------
    "transfer_only": StockConfig(
        name="transfer_only",
        description=(
            "Arm 2 (protocol): transfer prior from the S&P 500 posterior "
            "+ individual pooling; no hierarchical pooling, no context "
            "group. Isolates the pure effect of the informed prior."
        ),
        tune_method="prior_from_idata",
        lt_loss_factor=1.0,
        fs_loss_factor=0.0,
        lt_pool="individual",
        fs_pool="individual",
        lt_shrinkage=100,
        fs_shrinkage=100,
    ),
    "no_transfer": StockConfig(
        name="no_transfer",
        description=(
            "Arm 3: hierarchical pooling only, no transfer. Honest relabel of "
            "the former 'TimeSeers' arm (it is a Vangja configuration, not "
            "the TimeSeers package)."
        ),
        tune_method=None,
        lt_loss_factor=1.0,
        fs_loss_factor=0.0,
        lt_pool="partial",
        fs_pool="partial",
        lt_shrinkage=100,
        fs_shrinkage=100,
    ),
    "target_only": StockConfig(
        name="target_only",
        description=(
            "Arm 1: no transfer, individual pooling, no context group. "
            "Isolated per-stock Prophet-like Vangja (honest relabel of the "
            "former 'Prophet' arm)."
        ),
        tune_method=None,
        lt_loss_factor=0.0,
        fs_loss_factor=0.0,
        lt_pool="individual",
        fs_pool="individual",
        lt_shrinkage=100,
        fs_shrinkage=100,
    ),
    "context_group": StockConfig(
        name="context_group",
        description=(
            "Arm 3: S&P 500 included as an additional hierarchical group "
            "(co-learning) but no transfer prior."
        ),
        tune_method=None,
        lt_loss_factor=1.0,
        fs_loss_factor=0.0,
        lt_pool="partial",
        fs_pool="partial",
        lt_shrinkage=100,
        fs_shrinkage=100,
        include_context_group=True,
    ),
    "combined": StockConfig(
        name="combined",
        description=(
            "Arm 4: transfer prior AND S&P 500 as a hierarchical group "
            "(confounds the two contributions; only run to reproduce the "
            "former headline, not to attribute gains)."
        ),
        tune_method="prior_from_idata",
        lt_loss_factor=1.0,
        fs_loss_factor=0.0,
        lt_pool="partial",
        fs_pool="partial",
        lt_shrinkage=100,
        fs_shrinkage=100,
        include_context_group=True,
    ),
    # --- Single-perturbation ablations of the main configuration -----------
    "trend_reg_off": StockConfig(
        name="trend_reg_off",
        description="Main configuration with the trend regularisation potential disabled.",
        tune_method="prior_from_idata",
        lt_loss_factor=0.0,
        fs_loss_factor=0.0,
        lt_pool="partial",
        fs_pool="partial",
        lt_shrinkage=100,
        fs_shrinkage=100,
    ),
    "seasonal_reg_on": StockConfig(
        name="seasonal_reg_on",
        description=(
            "Main configuration with the seasonal amplitude cap enabled "
            "(active here: the 91-day target window is shorter than half the "
            "yearly period)."
        ),
        tune_method="prior_from_idata",
        lt_loss_factor=1.0,
        fs_loss_factor=1.0,
        lt_pool="partial",
        fs_pool="partial",
        lt_shrinkage=100,
        fs_shrinkage=100,
    ),
    "tight_fs_shrinkage": StockConfig(
        name="tight_fs_shrinkage",
        description=(
            "Main configuration with very tight seasonal shrinkage (the "
            "retrospective paper's setting)."
        ),
        tune_method="prior_from_idata",
        lt_loss_factor=1.0,
        fs_loss_factor=0.0,
        lt_pool="partial",
        fs_pool="partial",
        lt_shrinkage=100,
        fs_shrinkage=10000,
    ),
    # --- Targeted sensitivity (P1-9), high-weight subset only --------------
    "transfer_gold": StockConfig(
        name="transfer_gold",
        description=(
            "Negative-control context: transfer from gold futures (GC=F) "
            "instead of the S&P 500, on the high-weight subset only."
        ),
        tune_method="prior_from_idata",
        lt_loss_factor=1.0,
        fs_loss_factor=0.0,
        lt_pool="partial",
        fs_pool="partial",
        lt_shrinkage=100,
        fs_shrinkage=100,
        negative_control=True,
        high_weight_subset=True,
    ),
    "transfer_gold_smp": StockConfig(
        name="transfer_gold_smp",
        description=(
            "Matched arm for the negative control: S&P 500 transfer on the "
            "same high-weight subset."
        ),
        tune_method="prior_from_idata",
        lt_loss_factor=1.0,
        fs_loss_factor=0.0,
        lt_pool="partial",
        fs_pool="partial",
        lt_shrinkage=100,
        fs_shrinkage=100,
        high_weight_subset=True,
    ),
}

# The frozen finalists used for the paired dependence-aware comparison
# (PROTOCOL.md §9): transfer vs no-transfer on the same pooling structure.
# `seasonal_reg_on` is the SELECTED configuration (see RUN_NOTES: chosen on
# development evidence only, before any confirmation analysis).
SELECTED_CONFIG = "seasonal_reg_on"
FINALIST_PAIRS = [
    ("seasonal_reg_on", "no_transfer"),
    ("transfer_only", "target_only"),
]

# Configs run on every origin of the main matrix (development + confirmation).
MAIN_MATRIX = [
    "main",
    "transfer_only",
    "no_transfer",
    "target_only",
    "context_group",
    "trend_reg_off",
    "seasonal_reg_on",
    "tight_fs_shrinkage",
]

# Additional configs only used in targeted analyses.
TARGETED = ["combined", "transfer_gold", "transfer_gold_smp"]
