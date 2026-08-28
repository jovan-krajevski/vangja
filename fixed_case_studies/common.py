"""Shared infrastructure for the fixed case studies.

This module implements the pieces of the revised protocol
(`PROTOCOL.md` in the review) that are shared by both case studies:

- **Primary metric.** Relative MAE against persistence per
  (target x origin) unit, with a denominator threshold ``EPSILON`` applied on
  the *scaled* target (the scale used for fitting). Aggregates: median
  (the headline), mean, IQR, proportion below 1, exclusions and failure
  counts. MAPE is kept only as a safeguarded secondary metric.
- **Persistence baseline.** Last observed value carried forward, computed in
  scaled units so the denominator rule is well defined.
- **Provenance.** Every artifact records the producing commit SHA, package
  versions, seeds, configuration, and data hashes.
- **Dependence-aware paired comparison.** Two-way block resampling over
  stocks and forecast-origin blocks (no pooled Diebold-Mariano, no
  stock-only bootstrap) for the transfer vs no-transfer comparison.
- **Freeze checks.** The confirmation-origin file is hashed and verified
  before any confirmation scoring.

Nothing in this module downloads data, fits models, or looks at test
horizons. All randomness is seeded explicitly.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ---------------------------------------------------------------------------
# Frozen protocol constants
# ---------------------------------------------------------------------------

# Denominator rule (PROTOCOL.md §4): Relative-MAE units whose persistence MAE
# on the *scaled* target is strictly below this threshold are excluded from
# the relative-MAE aggregate (their absolute MAE is reported separately).
REL_MAE_EPSILON = 1e-3

# Global seed policy (PROTOCOL.md §10). Every fit gets a seed derived from
# these constants; they are recorded in every artifact.
BASE_SEED = 42
FINALIST_SEEDS = (42, 20240816, 7)  # repeated-seed robustness for finalists

# Series names that are context, never targets (PROTOCOL.md §7).
CONTEXT_SERIES = {"source", "^GSPC"}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def commit_sha(repo_root: Path | None = None) -> str:
    """Return the producing commit SHA (or 'unknown' if not in a git repo)."""
    root = repo_root or Path(__file__).resolve().parents[1]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def environment_info() -> dict:
    """Versions of the packages that produce artifacts (PROTOCOL.md §13)."""
    import importlib.metadata as im

    info = {}
    for pkg in ("vangja", "pymc", "arviz", "numpy", "pandas", "pytensor",
                "scikit-learn"):
        try:
            info[pkg] = im.version(pkg)
        except im.PackageNotFoundError:
            info[pkg] = None
    return info


def sha256_file(path: Path) -> str:
    """SHA-256 of a file (data provenance sidecars, PROTOCOL.md §13)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str, sort_keys=True)


def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def record_sidecars(data_dir: Path) -> dict[str, str]:
    """Write a SHA-256 manifest for every file under ``data_dir``."""
    hashes = {}
    if data_dir.exists():
        for f in sorted(data_dir.rglob("*")):
            if f.is_file():
                hashes[str(f.relative_to(data_dir))] = sha256_file(f)
    manifest = data_dir.parent / "data_sha256.json"
    save_json(manifest, {"files": hashes, "created_at": datetime.now().isoformat()})
    return hashes


# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------


def per_series_scale_params(model) -> dict[str, tuple[float, float]]:
    """(y_min, y_max) used for fitting, per series name.

    Handles both complete scaling (a single shared dict with a 'scaler' key)
    and individual scaling (a per-group dict of dicts).
    """
    if isinstance(model.y_scale_params, dict) and "scaler" in model.y_scale_params:
        shared = model.y_scale_params
        return {
            name: (shared["y_min"], shared["y_max"])
            for name in model.groups_.values()
        }
    return {
        name: (
            model.y_scale_params[code]["y_min"],
            model.y_scale_params[code]["y_max"],
        )
        for code, name in model.groups_.items()
    }


def scale_y(y, y_min: float, y_max: float) -> np.ndarray:
    """Apply the model's y scaling: (y - y_min) / (y_max - y_min).

    This reproduces both the maxabs (y_min=0) and minmax scalers used by
    ``TimeSeriesModel._process_data``.
    """
    return (np.asarray(y, dtype=float) - y_min) / (y_max - y_min)


def safeguarded_mape(y, yhat, floor: float = 1e-6) -> float:
    """MAPE with a floor on |y| so near-zero targets cannot explode.

    MAPE is only a secondary, continuity metric (PROTOCOL.md §4); the
    primary metric is Relative MAE.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = np.where(np.abs(y) < floor, floor, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.mean(np.abs((y - yhat) / denom)))


# ---------------------------------------------------------------------------
# Unit metrics (per target x origin)
# ---------------------------------------------------------------------------


def unit_metrics(
    model,
    test_df: pd.DataFrame,
    future_df: pd.DataFrame,
    origin: str,
    config: str,
    stage: str,
    exclude_series: set[str] | None = None,
    epsilon: float = REL_MAE_EPSILON,
) -> pd.DataFrame:
    """Compute per-series metrics for one fitted model on one origin.

    Relative MAE is computed in the *scaled* space used for fitting, and the
    denominator rule uses the persistence MAE in that space.  Absolute MAE /
    RMSE / safeguarded MAPE are reported in original units as secondaries.

    Context series (``exclude_series``, plus the frozen ``CONTEXT_SERIES``)
    never appear in the output (PROTOCOL.md §7).
    """
    exclude = set(CONTEXT_SERIES)
    if exclude_series:
        exclude |= set(exclude_series)

    scale_params = per_series_scale_params(model)
    last_scaled = model.data.groupby("series")["y"].last()

    rows = []
    for code, name in model.groups_.items():
        if name in exclude or name not in last_scaled.index:
            continue
        merged = (
            test_df[test_df["series"] == name][["ds", "y"]]
            .merge(future_df[["ds", f"yhat_{code}"]], on="ds", how="inner")
            .dropna()
        )
        if merged.empty:
            continue
        y = merged["y"].values
        yhat = merged[f"yhat_{code}"].values
        y_min, y_max = scale_params[name]

        y_s = scale_y(y, y_min, y_max)
        yhat_s = scale_y(yhat, y_min, y_max)
        pers_s = np.full(len(y), float(last_scaled.loc[name]))

        mae_s = mean_absolute_error(y_s, yhat_s)
        pers_mae_s = mean_absolute_error(y_s, pers_s)
        excluded = bool(pers_mae_s < epsilon)
        rel_mae = float("nan") if excluded else float(mae_s / pers_mae_s)

        rows.append(
            {
                "series": name,
                "origin": origin,
                "config": config,
                "stage": stage,
                "n": int(len(y)),
                "mae": float(mean_absolute_error(y, yhat)),
                "rmse": float(root_mean_squared_error(y, yhat)),
                "mape": safeguarded_mape(y, yhat),
                "mae_scaled": float(mae_s),
                "persistence_mae_scaled": float(pers_mae_s),
                "rel_mae": rel_mae,
                "excluded": excluded,
            }
        )
    return pd.DataFrame(rows)


def baseline_unit_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    yhat_df: pd.DataFrame,
    origin: str,
    baseline: str,
    stage: str,
    scale_mode: str = "minmax_individual",
    epsilon: float = REL_MAE_EPSILON,
    exclude_series: set[str] | None = None,
    seed: int = BASE_SEED,
) -> pd.DataFrame:
    """Per-series metrics for one model-free baseline forecast.

    Schema-identical to :func:`unit_metrics` (``config`` = baseline code) so
    the report scripts can aggregate the classical baselines together with
    the vangja arms. ``yhat_df`` is long format (``ds``, ``series``,
    ``yhat``) for a single baseline.

    Relative MAE is computed in the **same scaled space the vangja models
    use**, so the denominators are directly comparable:

    - ``minmax_individual`` (smart home): per-series min-max on the training
      window (matches ``scaler="minmax", scale_mode="individual"``);
    - ``maxabs_complete`` (stocks): one global scale on the (rescaled)
      training data, ``y_min = 0``, ``y_max = max|y|`` (matches
      ``scaler="maxabs", scale_mode="complete"``).

    Persistence = the last training value in that space (the denominator,
    PROTOCOL.md §4). Context series are never scored.
    """
    exclude = set(CONTEXT_SERIES)
    if exclude_series:
        exclude |= set(exclude_series)

    if scale_mode == "maxabs_complete":
        y_max_global = float(np.abs(train_df["y"]).max()) if len(train_df) else 1.0
        if not np.isfinite(y_max_global) or y_max_global == 0:
            y_max_global = 1.0

    rows = []
    for name in sorted(train_df["series"].unique()):
        if name in exclude:
            continue
        tr = train_df[train_df["series"] == name].sort_values("ds")
        te = test_df[test_df["series"] == name].sort_values("ds")
        fh = yhat_df[yhat_df["series"] == name].sort_values("ds")
        if tr.empty or te.empty or fh.empty:
            continue
        merged = te[["ds", "y"]].merge(fh[["ds", "yhat"]], on="ds", how="inner").dropna()
        if merged.empty:
            continue
        y = merged["y"].values
        yhat = merged["yhat"].values
        if scale_mode == "maxabs_complete":
            y_min, y_max = 0.0, y_max_global
        else:
            y_min = float(tr["y"].min())
            y_max = float(tr["y"].max())
            if y_max <= y_min:
                y_max = y_min + 1.0  # flat series: avoid division by zero

        y_s = scale_y(y, y_min, y_max)
        yhat_s = scale_y(yhat, y_min, y_max)
        last_s = scale_y(float(tr["y"].iloc[-1]), y_min, y_max)
        pers_s = np.full(len(y), last_s)

        mae_s = mean_absolute_error(y_s, yhat_s)
        pers_mae_s = mean_absolute_error(y_s, pers_s)
        excluded = bool(pers_mae_s < epsilon)
        rel_mae = float("nan") if excluded else float(mae_s / pers_mae_s)

        rows.append(
            {
                "series": name,
                "origin": origin,
                "config": baseline,
                "stage": stage,
                "n": int(len(y)),
                "mae": float(mean_absolute_error(y, yhat)),
                "rmse": float(root_mean_squared_error(y, yhat)),
                "mape": safeguarded_mape(y, yhat),
                "mae_scaled": float(mae_s),
                "persistence_mae_scaled": float(pers_mae_s),
                "rel_mae": rel_mae,
                "excluded": excluded,
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def aggregate_rel_mae(unit_df: pd.DataFrame) -> dict:
    """Primary aggregate: median Relative MAE across target x origin units."""
    vals = unit_df["rel_mae"].dropna()
    if vals.empty:
        return {
            "median": None,
            "mean": None,
            "q1": None,
            "q3": None,
            "prop_below_1": None,
            "n_units": 0,
            "n_excluded": int(unit_df["excluded"].sum()) if "excluded" in unit_df else 0,
        }
    return {
        "median": float(vals.median()),
        "mean": float(vals.mean()),
        "q1": float(vals.quantile(0.25)),
        "q3": float(vals.quantile(0.75)),
        "prop_below_1": float((vals < 1).mean()),
        "n_units": int(len(vals)),
        "n_excluded": int(unit_df["excluded"].sum()) if "excluded" in unit_df else 0,
    }


def add_origin_block(unit_df: pd.DataFrame, block_by: str = "half_year") -> pd.DataFrame:
    """Add an ``origin_block`` column grouping overlapping origins.

    Origins within the same block have heavily overlapping horizons, so the
    two-way block bootstrap resamples whole blocks (PROTOCOL.md §9).
    """
    out = unit_df.copy()
    if not out["origin"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").all():
        # Non-date origin labels (e.g. the smart-home splits): each origin
        # is its own block.
        out["origin_block"] = out["origin"].astype(str)
        return out
    origins = pd.to_datetime(out["origin"])
    if block_by == "half_year":
        out["origin_block"] = (
            origins.dt.year.astype(str) + "-H" + ((origins.dt.month >= 7) + 1).astype(str)
        )
    else:  # year blocks
        out["origin_block"] = origins.dt.year.astype(str)
    return out


def two_way_block_bootstrap(
    paired: pd.DataFrame,
    diff_col: str = "diff_rel_mae",
    series_col: str = "series",
    block_col: str = "origin_block",
    n_iter: int = 2000,
    seed: int = 0,
) -> dict:
    """Two-way block resampling over series and origin blocks (PROTOCOL §9).

    Resamples target series with replacement and forecast-origin *blocks*
    with replacement (so overlapping horizons stay grouped), recomputes the
    median paired difference each iteration, and returns percentile CIs.
    """
    rng = np.random.default_rng(seed)
    series_choices = paired[series_col].unique()
    block_choices = paired[block_col].unique()
    medians = []
    for _ in range(n_iter):
        s = rng.choice(series_choices, size=len(series_choices), replace=True)
        b = rng.choice(block_choices, size=len(block_choices), replace=True)
        sub = paired[paired[series_col].isin(s) & paired[block_col].isin(b)]
        if sub.empty:
            continue
        medians.append(float(sub[diff_col].median()))
    medians = np.asarray(medians)
    return {
        "n_iter": int(len(medians)),
        "median_diff": float(np.median(medians)),
        "ci_2p5": float(np.percentile(medians, 2.5)),
        "ci_97p5": float(np.percentile(medians, 97.5)),
        "prop_diff_lt_0": float((medians < 0).mean()),
    }


def paired_transfer_comparison(
    unit_df: pd.DataFrame,
    transfer_config: str,
    no_transfer_config: str,
    stage_filter: str | None = None,
    n_iter: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Paired (per target x origin) Relative-MAE differences.

    Returns the paired differences and the two-way block-bootstrap summary
    for the transfer vs no-transfer comparison (frozen finalists only).
    """
    df = unit_df.copy()
    if stage_filter is not None:
        df = df[df["stage"] == stage_filter]
    if transfer_config not in set(df["config"]) or no_transfer_config not in set(
        df["config"]
    ):
        empty = pd.DataFrame(
            columns=["series", "origin", "transfer_rel_mae",
                     "no_transfer_rel_mae", "diff_rel_mae", "origin_block"]
        )
        return empty, {
            "n_paired_units": 0,
            "prop_units_transfer_better": None,
            "median_paired_diff": None,
            "negative_transfer_rate": None,
            "note": "one of the configs is missing; run both first",
        }
    a = (
        df[df["config"] == transfer_config]
        .set_index(["series", "origin"])["rel_mae"]
        .rename(transfer_config)
    )
    b = (
        df[df["config"] == no_transfer_config]
        .set_index(["series", "origin"])["rel_mae"]
        .rename(no_transfer_config)
    )
    joined = (
        pd.concat([a, b], axis=1, join="inner")
        .dropna()
        .reset_index()
        .rename(columns={transfer_config: "transfer_rel_mae",
                         no_transfer_config: "no_transfer_rel_mae"})
    )
    joined["diff_rel_mae"] = (
        joined["transfer_rel_mae"] - joined["no_transfer_rel_mae"]
    )
    joined = add_origin_block(joined)
    summary = two_way_block_bootstrap(joined, n_iter=n_iter, seed=seed)
    summary["n_paired_units"] = int(len(joined))
    summary["prop_units_transfer_better"] = float(
        (joined["diff_rel_mae"] < 0).mean()
    )
    summary["median_paired_diff"] = float(joined["diff_rel_mae"].median())
    summary["negative_transfer_rate"] = float(
        (joined["diff_rel_mae"] > 0).mean()
    )
    return joined, summary


# ---------------------------------------------------------------------------
# Artifact recording
# ---------------------------------------------------------------------------


def record_artifact(
    out_dir: Path,
    *,
    study: str,
    stage: str,
    origin: str,
    config: str,
    model,
    unit_df: pd.DataFrame,
    future_df: pd.DataFrame,
    extra: dict | None = None,
) -> dict:
    """Write the per-run artifact bundle (PROTOCOL.md §11).

    Saves the aligned forecasts, the per-unit metrics, and a JSON manifest
    with provenance (commit, versions, seeds, fit diagnostics, aggregate).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{config}__{origin}__seed{model.random_seed}".replace("/", "_")

    future_path = out_dir / f"forecasts_{tag}.csv"
    future_df.to_csv(future_path, index=False)

    unit_path = out_dir / f"units_{tag}.csv"
    unit_df.to_csv(unit_path, index=False)

    manifest = {
        "artifact_schema": "vangja-fixed-case-studies-1.0",
        "study": study,
        "stage": stage,
        "origin": origin,
        "config": config,
        "seed": model.random_seed,
        "commit": commit_sha(),
        "created_at": datetime.now().isoformat(),
        "environment": environment_info(),
        "fit_info": getattr(model, "fit_info", {}),
        "aggregate": aggregate_rel_mae(unit_df),
        "n_units": int(len(unit_df)),
        "files": {
            "forecasts": future_path.name,
            "units": unit_path.name,
        },
    }
    if extra:
        manifest["extra"] = extra
    manifest_path = out_dir / f"manifest_{tag}.json"
    save_json(manifest_path, manifest)
    return manifest


def record_failure(
    out_dir: Path,
    *,
    study: str,
    stage: str,
    origin: str,
    config: str,
    seed,
    error: Exception,
    traceback_text: str,
) -> None:
    """Record a failed fit so failure rates can be reported (PROTOCOL §4)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "failures.csv"
    row = pd.DataFrame(
        [
            {
                "study": study,
                "stage": stage,
                "origin": origin,
                "config": config,
                "seed": seed,
                "error": type(error).__name__,
                "message": str(error)[:500],
                "traceback": traceback_text[:2000],
                "at": datetime.now().isoformat(),
            }
        ]
    )
    header = not path.exists()
    row.to_csv(path, mode="a", header=header, index=False)


# ---------------------------------------------------------------------------
# Frozen confirmation origins
# ---------------------------------------------------------------------------


def load_confirmation_origins(protocol_dir: Path) -> pd.DataFrame:
    """Load the frozen confirmation origins and verify their hash.

    The hash of ``CONFIRMATION_ORIGINS.csv`` is recorded in the README before
    any confirmation results were produced; scoring must refuse to run if
    the file has been modified (PROTOCOL.md §2.2).
    """
    path = protocol_dir / "CONFIRMATION_ORIGINS.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen confirmation origins: {path}")
    text = path.read_text()
    digest = sha256_bytes(text.encode())
    stored = (protocol_dir / "CONFIRMATION_ORIGINS.sha256").read_text().strip()
    # Accept both the bare hash and the two-column output of sha256sum.
    expected = stored.split()[0] if stored else ""
    if digest != expected:
        raise RuntimeError(
            "CONFIRMATION_ORIGINS.csv hash mismatch: the frozen origins were "
            "modified after being committed. Refusing to score confirmation."
        )
    return pd.read_csv(path, parse_dates=["origin"])
