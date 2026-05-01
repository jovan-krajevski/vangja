"""Generate LaTeX tables and ablation studies for the Vangja paper.

Reads results from ``case_studies/<dataset>/results/<model>/`` and writes
LaTeX tables to ``case_studies/tables/``.

Datasets
--------
* ``smart_home`` -- four daily energy time series. Reported per series.
* ``stocks``     -- 366 daily stock series + S&P 500 (excluded). Reported as
                    averages over all series (except ``^GSPC``) and all
                    24 monthly rolling start dates.

Models
------
* ``baselines`` -- classical baselines (ARIMA, Holt-Winters, etc.).
* ``prophet``   -- Facebook Prophet (with / without yearly seasonality).
* ``timeseers`` -- Hierarchical Bayesian Prophet (Timeseers package).
* ``vangja``    -- This work.

The script is robust to partial result files: it averages over whatever is
available so it can be re-run as more results come in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
TABLES_DIR = ROOT / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT / "tables" / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Identifiers                                                                  #
# --------------------------------------------------------------------------- #
SMART_HOME_SERIES = [
    "Fridge [kW]",
    "Furnace 1 [kW]",
    "Furnace 2 [kW]",
    "Wine cellar [kW]",
]
SMART_HOME_SERIES_SHORT = {
    "Fridge [kW]": "Fridge",
    "Furnace 1 [kW]": "Furnace 1",
    "Furnace 2 [kW]": "Furnace 2",
    "Wine cellar [kW]": "Wine cellar",
}

# Vangja experiment-defining columns (everything that varies in the grid)
SH_VANGJA_PARAMS = [
    "use_temp_df",
    "hierarchical",
    "uniform_constant",
    "tune_method",
    "tune_loss_factor",
    "shrinkage_strength",
    "intercept_sd",
    "beta_sd",
    "scaler",
    "start_date",
]
SH_TIMESEERS_PARAMS = [
    "uniform_constant",
    "shrinkage_strength",
    "intercept_sd",
    "beta_sd",
    "scaler",
    "start_date",
]
ST_VANGJA_PARAMS = [
    "use_smp500",
    "lt_hierarchical",
    "fs_hierarchical",
    "window_size",
    "uniform_constant",
    "tune_method",
    "lt_tune_loss_factor",
    "fs_tune_loss_factor",
    "lt_shrinkage_strength",
    "fs_shrinkage_strength",
    "slope_sd",
    "intercept_sd",
    "beta_sd",
    "scaler",
]
ST_TIMESEERS_PARAMS = [
    "window_size",
    "uniform_constant",
    "lt_shrinkage_strength",
    "fs_shrinkage_strength",
    "slope_sd",
    "intercept_sd",
    "beta_sd",
    "scaler",
    "delta_side",
]

METRIC_COLS = ["rmse", "mae", "mape"]


# --------------------------------------------------------------------------- #
# Loaders                                                                      #
# --------------------------------------------------------------------------- #
def _read_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --- Smart home --------------------------------------------------------------
def load_sh_vangja() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "smart_home/results/vangja/metrics.csv")
    return df[df["timeseries"].isin(SMART_HOME_SERIES)].copy()


def load_sh_timeseers() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "smart_home/results/timeseers/metrics.csv")
    return df[df["timeseries"].isin(SMART_HOME_SERIES)].copy()


def load_sh_prophet() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "smart_home/results/prophet/metrics.csv")
    return df[df["timeseries"].isin(SMART_HOME_SERIES)].copy()


def load_sh_baselines() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "smart_home/results/baselines/classical_metrics.csv")
    df = df.rename(columns={"series": "timeseries"})
    return df[df["timeseries"].isin(SMART_HOME_SERIES)].copy()


# --- Stocks (per-experiment averages excluding ^GSPC, then averaged across months)
def _aggregate_stocks_file(path: Path, group_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["timeseries"] != "^GSPC"]
    df["_n"] = 1
    agg = (
        df.groupby(group_cols, dropna=False)[METRIC_COLS + ["_n"]].mean().reset_index()
    )
    agg["start_date"] = path.stem.split("_")[-1]
    return agg


def _load_stocks(model: str, params: list[str], pattern: str) -> pd.DataFrame:
    """Aggregate per-series stocks files into per-experiment, per-month means
    (excluding ^GSPC), then return a frame ready for averaging across months.

    Cached because the vangja per-series files are large.
    """
    cache_path = CACHE_DIR / f"stocks_{model}.pkl"
    folder = ROOT / f"stocks/results/{model}"
    files = sorted(folder.glob(pattern))
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        cached_dates = set(cached["start_date"].unique())
        new_files = [f for f in files if f.stem.split("_")[-1] not in cached_dates]
        if not new_files:
            return cached
        new_part = pd.concat(
            [_aggregate_stocks_file(f, params) for f in new_files], ignore_index=True
        )
        out = pd.concat([cached, new_part], ignore_index=True)
    else:
        if not files:
            return pd.DataFrame()
        out = pd.concat(
            [_aggregate_stocks_file(f, params) for f in files], ignore_index=True
        )
    out.to_pickle(cache_path)
    return out


def load_st_vangja() -> pd.DataFrame:
    return _load_stocks("vangja", ST_VANGJA_PARAMS, "results_*.csv")


def load_st_timeseers() -> pd.DataFrame:
    return _load_stocks("timeseers", ST_TIMESEERS_PARAMS, "results_*.csv")


def load_st_prophet() -> pd.DataFrame:
    files = sorted((ROOT / "stocks/results/prophet").glob("results_*.csv"))
    df = _read_csvs(files)
    return df  # no ^GSPC here


def load_st_baselines() -> pd.DataFrame:
    files = sorted((ROOT / "stocks/results/baselines").glob("results_classical_*.csv"))
    df = _read_csvs(files)
    if "series" in df.columns:
        df = df.rename(columns={"series": "timeseries"})
    return df


# --------------------------------------------------------------------------- #
# Aggregation helpers                                                          #
# --------------------------------------------------------------------------- #
def average_across_months(df: pd.DataFrame, params: list[str]) -> pd.DataFrame:
    """Average per-experiment metrics across all available months."""
    if df.empty:
        return df
    return df.groupby(params, dropna=False)[METRIC_COLS].mean().reset_index()


def best_by_mape(df: pd.DataFrame, params: list[str]) -> pd.Series:
    """Return the row with the smallest MAPE."""
    agg = average_across_months(df, params) if "start_date" in df.columns else df
    return agg.loc[agg["mape"].idxmin()].copy()


def per_series_best(
    df: pd.DataFrame, group_col: str, params: list[str]
) -> pd.DataFrame:
    """For each series, return the experiment with the smallest MAPE."""
    rows = []
    for series, sub in df.groupby(group_col):
        agg = sub.groupby(params, dropna=False)[METRIC_COLS].mean().reset_index()
        best = agg.loc[agg["mape"].idxmin()].copy()
        best[group_col] = series
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# LaTeX writers                                                                #
# --------------------------------------------------------------------------- #
def _fmt(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.{digits}f}"


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "--"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.1f}\\%"


def write_table(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# Best-model tables                                                            #
# --------------------------------------------------------------------------- #
def best_models_smart_home() -> pd.DataFrame:
    """Return a long DataFrame of best per-series model in each category."""
    out_rows = []

    sh_v = load_sh_vangja()
    sh_t = load_sh_timeseers()
    sh_p = load_sh_prophet()
    sh_b = load_sh_baselines()
    # Only consider delta_side==left for timeseers (smart home has none, ok)

    for series in SMART_HOME_SERIES:
        # Vangja
        sub = sh_v[sh_v["timeseries"] == series]
        agg = (
            sub.groupby(SH_VANGJA_PARAMS, dropna=False)[METRIC_COLS]
            .mean()
            .reset_index()
        )
        b = agg.loc[agg["mape"].idxmin()]
        out_rows.append(("Vangja", series, b))

        # Timeseers
        sub = sh_t[sh_t["timeseries"] == series]
        agg = (
            sub.groupby(SH_TIMESEERS_PARAMS, dropna=False)[METRIC_COLS]
            .mean()
            .reset_index()
        )
        b = agg.loc[agg["mape"].idxmin()]
        out_rows.append(("Timeseers", series, b))

        # Prophet -- pick best variant
        sub = sh_p[sh_p["timeseries"] == series]
        b = sub.loc[sub["mape"].idxmin()]
        out_rows.append(("Prophet", series, b))

        # Baselines
        sub = sh_b[sh_b["timeseries"] == series]
        b = sub.loc[sub["mape"].idxmin()]
        out_rows.append(("Baseline", series, b))

    rows = []
    for cat, series, b in out_rows:
        rows.append(
            {
                "category": cat,
                "series": series,
                "rmse": b["rmse"],
                "mae": b["mae"],
                "mape": b["mape"],
                "model": b.get("model", ""),
            }
        )
    return pd.DataFrame(rows)


def best_models_stocks() -> pd.DataFrame:
    rows = []
    # Vangja
    v = load_st_vangja()
    if not v.empty:
        b = best_by_mape(v, ST_VANGJA_PARAMS)
        rows.append(
            {"category": "Vangja", **{k: b[k] for k in METRIC_COLS}, "model": ""}
        )

    # Timeseers (only delta_side == left)
    t = load_st_timeseers()
    if not t.empty:
        t_left = t[t["delta_side"] == "left"]
        b = best_by_mape(t_left, ST_TIMESEERS_PARAMS)
        rows.append(
            {"category": "Timeseers", **{k: b[k] for k in METRIC_COLS}, "model": ""}
        )

    # Prophet -- average across series + months per model variant
    p = load_st_prophet()
    if not p.empty:
        agg = p.groupby("model")[METRIC_COLS].mean().reset_index()
        b = agg.loc[agg["mape"].idxmin()]
        rows.append(
            {
                "category": "Prophet",
                **{k: b[k] for k in METRIC_COLS},
                "model": b["model"],
            }
        )

    # Baselines
    bs = load_st_baselines()
    if not bs.empty:
        agg = bs.groupby("model")[METRIC_COLS].mean().reset_index()
        b = agg.loc[agg["mape"].idxmin()]
        rows.append(
            {
                "category": "Baseline",
                **{k: b[k] for k in METRIC_COLS},
                "model": b["model"],
            }
        )

    return pd.DataFrame(rows)


def write_best_smart_home_table(df: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Series & Model & RMSE & MAE & MAPE \\",
        r"\hline",
    ]
    for series in SMART_HOME_SERIES:
        sub = df[df["series"] == series]
        # find best mape across categories for bolding
        best_cat = sub.loc[sub["mape"].idxmin(), "category"] if not sub.empty else None
        first = True
        for cat in ["Baseline", "Prophet", "Timeseers", "Vangja"]:
            row = sub[sub["category"] == cat]
            if row.empty:
                continue
            r = row.iloc[0]
            mape_s = _fmt(r["mape"])
            if cat == best_cat:
                mape_s = f"\\textbf{{{mape_s}}}"
            series_label = SMART_HOME_SERIES_SHORT[series] if first else ""
            lines.append(
                f"{series_label} & {cat} & {_fmt(r['rmse'])} & {_fmt(r['mae'])} & {mape_s} \\\\"
            )
            first = False
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    write_table(TABLES_DIR / "smart_home_best.tex", "\n".join(lines))


def write_best_stocks_table(df: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Model & RMSE & MAE & MAPE \\",
        r"\hline",
    ]
    if not df.empty:
        best_cat = df.loc[df["mape"].idxmin(), "category"]
        for cat in ["Baseline", "Prophet", "Timeseers", "Vangja"]:
            row = df[df["category"] == cat]
            if row.empty:
                continue
            r = row.iloc[0]
            mape_s = _fmt(r["mape"])
            if cat == best_cat:
                mape_s = f"\\textbf{{{mape_s}}}"
            lines.append(
                f"{cat} & {_fmt(r['rmse'], 2)} & {_fmt(r['mae'], 2)} & {mape_s} \\\\"
            )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    write_table(TABLES_DIR / "stocks_best.tex", "\n".join(lines))


# --------------------------------------------------------------------------- #
# Ablation helpers                                                             #
# --------------------------------------------------------------------------- #
def _filter(df: pd.DataFrame, fixed: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for k, v in fixed.items():
        mask &= df[k] == v
    return df[mask]


def ablate_param(
    df: pd.DataFrame,
    group_params: list[str],
    fixed: dict,
    vary: str,
    baseline_value,
    series_col: str | None = None,
) -> pd.DataFrame:
    """Vary `vary` while keeping all other parameters fixed; return a frame
    with one row per (series, value).  MAPE delta is relative to baseline_value.
    """
    sub = _filter(df, fixed)
    if sub.empty:
        return pd.DataFrame()

    if series_col:
        agg = (
            sub.groupby([series_col, vary], dropna=False)[METRIC_COLS]
            .mean()
            .reset_index()
        )
    else:
        agg = sub.groupby([vary], dropna=False)[METRIC_COLS].mean().reset_index()

    # compute delta
    rows = []
    grp_cols = [series_col] if series_col else []
    for keys, g in agg.groupby(grp_cols, dropna=False) if grp_cols else [((), agg)]:
        baseline = g[g[vary] == baseline_value]
        base_mape = baseline["mape"].iloc[0] if not baseline.empty else np.nan
        for _, r in g.iterrows():
            rec = {vary: r[vary], "mape": r["mape"]}
            if grp_cols:
                rec[series_col] = keys if not isinstance(keys, tuple) else keys[0]
            rec["delta_pct"] = (
                100.0 * (r["mape"] - base_mape) / base_mape if base_mape else np.nan
            )
            rows.append(rec)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Smart-home ablations                                                         #
# --------------------------------------------------------------------------- #
def _sh_ablation_baseline(sh_v: pd.DataFrame, fixed: dict) -> pd.Series:
    """Return per-series baseline MAPE for the best Vangja config."""
    base = sh_v.copy()
    for k, v in fixed.items():
        base = base[base[k] == v]
    return base.groupby("timeseries")["mape"].mean()


def _sh_variant(sh_v: pd.DataFrame, fixed: dict, overrides: dict) -> pd.Series:
    """Apply overrides to the fixed dict, drop None-valued keys, return per-series MAPE."""
    new_fixed = {**fixed, **overrides}
    new_fixed = {k: v for k, v in new_fixed.items() if v is not None}
    sub = sh_v.copy()
    for k, v in new_fixed.items():
        sub = sub[sub[k] == v]
    return sub.groupby("timeseries")["mape"].mean()


def smart_home_ablations() -> None:
    sh_v = load_sh_vangja()
    sh_t = load_sh_timeseers()

    overall = (
        sh_v.groupby(SH_VANGJA_PARAMS, dropna=False)[METRIC_COLS].mean().reset_index()
    )
    best = overall.loc[overall["mape"].idxmin()]
    print("\n[Smart home] Best Vangja config (avg across series):")
    print(best)

    fixed = {p: best[p] for p in SH_VANGJA_PARAMS}
    baseline = _sh_ablation_baseline(sh_v, fixed)

    # Each ablation is (label, list of (variant_name, overrides_dict))
    # overrides=None means "disabled" (use Timeseers analogue)
    ablations: list[tuple[str, list[tuple[str, dict]]]] = []

    other = not best["uniform_constant"]
    ablations.append(
        (
            "Disable uniform constant",
            [(str(other), {"uniform_constant": other})],
        )
    )

    other = 1 - int(best["tune_loss_factor"])
    ablations.append(
        (
            (
                "Disable regularization (loss factor)"
                if best["tune_loss_factor"]
                else "Enable regularization (loss factor)"
            ),
            [(str(other), {"tune_loss_factor": other})],
        )
    )

    other_tm = (
        "prior_from_idata" if best["tune_method"] == "parametric" else "parametric"
    )
    ablations.append(
        (
            "Switch tune method",
            [(other_tm, {"tune_method": other_tm})],
        )
    )

    # Disable partial pooling: in the smart-home grid, individual pooling implies
    # without_temp_df and shrinkage_strength=0.  We expose this as a single ablation.
    ablations.append(
        (
            "Disable transfer + partial pooling",
            [
                (
                    "individual / no temp",
                    {
                        "hierarchical": "individual",
                        "use_temp_df": "without_temp_df",
                        "shrinkage_strength": 0,
                        "tune_method": None,
                        "tune_loss_factor": None,
                    },
                )
            ],
        )
    )

    # Shrinkage strength sweep (skip 0 -- that's the individual-pooling marker)
    shrinkage_values = sorted(x for x in sh_v["shrinkage_strength"].unique() if x != 0)
    ablations.append(
        (
            "Shrinkage strength",
            [(str(s), {"shrinkage_strength": s}) for s in shrinkage_values],
        )
    )

    # No tune method (Timeseers): same uniform_constant, shrinkage_strength,
    # intercept_sd, beta_sd, scaler, start_date.
    ts_query = (
        (sh_t["uniform_constant"] == best["uniform_constant"])
        & (sh_t["shrinkage_strength"] == best["shrinkage_strength"])
        & (sh_t["intercept_sd"] == best["intercept_sd"])
        & (sh_t["beta_sd"] == best["beta_sd"])
        & (sh_t["scaler"] == best["scaler"])
        & (sh_t["start_date"] == best["start_date"])
    )
    ts_match = sh_t[ts_query]
    if ts_match.empty:
        # If exact match missing, take the best Timeseers config (averaged across series)
        ts_best = (
            sh_t.groupby(SH_TIMESEERS_PARAMS, dropna=False)[METRIC_COLS]
            .mean()
            .reset_index()
        )
        ts_best = ts_best.loc[ts_best["mape"].idxmin()]
        ts_match = sh_t.copy()
        for p in SH_TIMESEERS_PARAMS:
            ts_match = ts_match[ts_match[p] == ts_best[p]]
    ts_per_series = ts_match.groupby("timeseries")["mape"].mean()

    # Base series size (start_date)
    other_dates = [d for d in sh_v["start_date"].unique() if d != best["start_date"]]
    ablations.append(
        (
            "Base series size (start date)",
            [(d, {"start_date": d}) for d in sorted(other_dates)],
        )
    )

    # Build rows for the LaTeX table
    table_rows: list[tuple[str, str, dict[str, float], dict[str, float]]] = []
    for label, variants in ablations:
        for variant_name, overrides in variants:
            mape_per_series = _sh_variant(sh_v, fixed, overrides)
            mapes = {s: mape_per_series.get(s, np.nan) for s in SMART_HOME_SERIES}
            deltas = {
                s: (
                    100.0 * (mapes[s] - baseline[s]) / baseline[s]
                    if not pd.isna(mapes[s]) and baseline.get(s, np.nan)
                    else np.nan
                )
                for s in SMART_HOME_SERIES
            }
            table_rows.append((label, variant_name, mapes, deltas))

    # Add Timeseers analogue row
    ts_mapes = {s: ts_per_series.get(s, np.nan) for s in SMART_HOME_SERIES}
    ts_deltas = {
        s: (
            100.0 * (ts_mapes[s] - baseline[s]) / baseline[s]
            if not pd.isna(ts_mapes[s]) and baseline.get(s, np.nan)
            else np.nan
        )
        for s in SMART_HOME_SERIES
    }
    table_rows.append(
        (
            "No tune method (Timeseers analogue)",
            "--",
            ts_mapes,
            ts_deltas,
        )
    )

    write_smart_home_ablation_table(table_rows, baseline)


def write_smart_home_ablation_table(
    rows: list[tuple[str, str, dict, dict]],
    baseline: pd.Series,
) -> None:
    series_order = SMART_HOME_SERIES
    short_names = [SMART_HOME_SERIES_SHORT[s] for s in series_order]

    lines = []
    lines.append(r"\begin{tabular}{ll" + "rr" * len(series_order) + "}")
    lines.append(r"\hline")
    header = "Ablation & Variant"
    for s in short_names:
        header += f" & MAPE ({s}) & $\\Delta$"
    header += r" \\"
    lines.append(header)
    lines.append(r"\hline")

    base_row = r"\textbf{Best Vangja} & --"
    for s in series_order:
        v = baseline.get(s, np.nan)
        base_row += f" & \\textbf{{{_fmt(v)}}} & --"
    base_row += r" \\"
    lines.append(base_row)
    lines.append(r"\hline")

    last_label = None
    for label, variant, mapes, deltas in rows:
        # Skip rows that have no data at all
        if all(pd.isna(mapes[s]) for s in series_order):
            continue
        label_cell = label if label != last_label else ""
        if label_cell:
            if last_label is not None:
                lines.append(r"\hline")
            last_label = label
        line = f"{label_cell} & {variant}"
        for s in series_order:
            line += f" & {_fmt(mapes[s])} & {_fmt_pct(deltas[s])}"
        line += r" \\"
        lines.append(line)
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    write_table(TABLES_DIR / "smart_home_ablation.tex", "\n".join(lines))

    # Impact summary: mean |delta| across series & variants per ablation
    impact: dict[str, list[float]] = {}
    for label, variant, mapes, deltas in rows:
        vals = [abs(deltas[s]) for s in series_order if not pd.isna(deltas[s])]
        if not vals:
            continue
        impact.setdefault(label, []).extend(vals)
    impact_pairs = sorted(
        ((label, float(np.mean(v))) for label, v in impact.items()),
        key=lambda x: -x[1],
    )
    lines = [
        r"\begin{tabular}{lr}",
        r"\hline",
        r"Ablation & Mean $|\Delta\text{MAPE}|$ \\",
        r"\hline",
    ]
    for label, val in impact_pairs:
        lines.append(f"{label} & {_fmt_pct(val)} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    write_table(TABLES_DIR / "smart_home_ablation_impact.tex", "\n".join(lines))


# --------------------------------------------------------------------------- #
# Stocks ablations                                                             #
# --------------------------------------------------------------------------- #
def _st_ablation_mape(v_avg: pd.DataFrame, fixed: dict, overrides: dict) -> float:
    new_fixed = {**fixed, **overrides}
    new_fixed = {k: val for k, val in new_fixed.items() if val is not None}
    sub = v_avg.copy()
    for k, val in new_fixed.items():
        sub = sub[sub[k] == val]
    if sub.empty:
        return np.nan
    # If multiple rows match (because we relaxed constraints), take the best one
    return float(sub["mape"].min())


def stocks_ablations() -> None:
    v = load_st_vangja()
    t = load_st_timeseers()
    if v.empty:
        print("[Stocks] No vangja data available.")
        return

    v_avg = average_across_months(v, ST_VANGJA_PARAMS)
    best = v_avg.loc[v_avg["mape"].idxmin()]
    print("\n[Stocks] Best Vangja config (avg across series & months):")
    print(best)

    fixed = {p: best[p] for p in ST_VANGJA_PARAMS}
    base_mape = float(best["mape"])

    # Each ablation is (label, list of (variant_name, overrides_dict))
    ablations: list[tuple[str, list[tuple[str, dict]]]] = []

    other = not best["uniform_constant"]
    ablations.append(
        ("Disable uniform constant", [(str(other), {"uniform_constant": other})])
    )

    other = 1 - int(best["lt_tune_loss_factor"])
    ablations.append(
        (
            "Trend regularization (loss factor)",
            [(str(other), {"lt_tune_loss_factor": other})],
        )
    )

    other = 1 - int(best["fs_tune_loss_factor"])
    ablations.append(
        (
            "Seasonality regularization (loss factor)",
            [(str(other), {"fs_tune_loss_factor": other})],
        )
    )

    other_tm = (
        "prior_from_idata" if best["tune_method"] == "parametric" else "parametric"
    )
    ablations.append(("Switch tune method", [(other_tm, {"tune_method": other_tm})]))

    # Disable lt partial pooling -> lt_ss=0, lt_individual, forces use_smp500=without_smp500
    ablations.append(
        (
            "Disable trend partial pooling",
            [
                (
                    "lt_individual",
                    {
                        "lt_hierarchical": "lt_individual",
                        "lt_shrinkage_strength": 0,
                        "use_smp500": "without_smp500",
                    },
                )
            ],
        )
    )
    # Disable fs partial pooling
    ablations.append(
        (
            "Disable seasonality partial pooling",
            [
                (
                    "fs_individual",
                    {
                        "fs_hierarchical": "fs_individual",
                        "fs_shrinkage_strength": 0,
                        "use_smp500": "without_smp500",
                    },
                )
            ],
        )
    )

    # Exclude S&P 500: any of the without_smp500 configs (relax lt/fs constraints)
    ablations.append(
        (
            "Exclude S\\&P 500 series",
            [
                (
                    "without_smp500",
                    {
                        "use_smp500": "without_smp500",
                        "lt_hierarchical": None,
                        "fs_hierarchical": None,
                        "lt_shrinkage_strength": None,
                        "fs_shrinkage_strength": None,
                    },
                )
            ],
        )
    )

    # Trend shrinkage strength sweep (only when lt_partial)
    lt_values = sorted(x for x in v["lt_shrinkage_strength"].unique() if x != 0)
    ablations.append(
        (
            "Trend shrinkage strength",
            [(str(int(s)), {"lt_shrinkage_strength": s}) for s in lt_values],
        )
    )
    fs_values = sorted(x for x in v["fs_shrinkage_strength"].unique() if x != 0)
    ablations.append(
        (
            "Seasonality shrinkage strength",
            [(str(int(s)), {"fs_shrinkage_strength": s}) for s in fs_values],
        )
    )

    beta_values = sorted(v["beta_sd"].unique())
    ablations.append(
        (
            r"$\beta$ prior std",
            [(str(s), {"beta_sd": s}) for s in beta_values],
        )
    )

    win_values = sorted(v["window_size"].unique())
    ablations.append(
        (
            "Base series window size (days)",
            [(str(int(s)), {"window_size": s}) for s in win_values],
        )
    )

    # No tune method (Timeseers, delta_side=left/right)
    ts_rows: list[tuple[str, float]] = []
    if not t.empty:
        t_avg = average_across_months(t, ST_TIMESEERS_PARAMS)
        # Match params best we can
        match_cols = [
            "window_size",
            "uniform_constant",
            "slope_sd",
            "intercept_sd",
            "beta_sd",
            "scaler",
        ]
        # Shrinkage match: if best is high, match closest existing value (timeseers lacks 0)
        m = t_avg.copy()
        for c in match_cols:
            m = m[m[c] == best[c]]
        # Match shrinkage if exact value exists; otherwise drop the constraint
        if not m.empty:
            for col in ["lt_shrinkage_strength", "fs_shrinkage_strength"]:
                if best[col] in m[col].unique():
                    m = m[m[col] == best[col]]
        if m.empty:
            print(
                "[Stocks] No matching Timeseers config for tune-method ablation; "
                "reporting best Timeseers result instead."
            )
            m = t_avg
        for side in ["left", "right"]:
            ms = m[m["delta_side"] == side]
            if ms.empty:
                ts_rows.append((f"delta side = {side}", np.nan))
            else:
                ts_rows.append((f"delta side = {side}", float(ms["mape"].min())))
    if ts_rows:
        ablations.append(
            (
                "No tune method (Timeseers)",
                [(name, {"__ts_value__": val}) for name, val in ts_rows],
            )
        )

    # Build table rows
    table_rows: list[tuple[str, str, float, float]] = []
    for label, variants in ablations:
        for variant_name, overrides in variants:
            if "__ts_value__" in overrides:
                m = overrides["__ts_value__"]
            else:
                m = _st_ablation_mape(v_avg, fixed, overrides)
            d = 100.0 * (m - base_mape) / base_mape if not pd.isna(m) else np.nan
            table_rows.append((label, variant_name, m, d))

    write_stocks_ablation_table(table_rows, base_mape)


def write_stocks_ablation_table(
    rows: list[tuple[str, str, float, float]],
    base_mape: float,
) -> None:
    lines = [
        r"\begin{tabular}{llrr}",
        r"\hline",
        r"Ablation & Variant & MAPE & $\Delta$ \\",
        r"\hline",
        f"\\textbf{{Best Vangja}} & -- & \\textbf{{{_fmt(base_mape)}}} & -- \\\\",
        r"\hline",
    ]
    last_label = None
    for label, variant, mape, delta in rows:
        if pd.isna(mape):
            continue
        label_cell = label if label != last_label else ""
        if label_cell:
            if last_label is not None:
                lines.append(r"\hline")
            last_label = label
        lines.append(
            f"{label_cell} & {variant} & {_fmt(mape)} & {_fmt_pct(delta)} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    write_table(TABLES_DIR / "stocks_ablation.tex", "\n".join(lines))

    # Impact: mean |delta| per ablation across variants
    impact: dict[str, list[float]] = {}
    for label, variant, mape, delta in rows:
        if pd.isna(delta):
            continue
        impact.setdefault(label, []).append(abs(delta))
    impact_pairs = sorted(
        ((label, float(np.mean(v))) for label, v in impact.items()),
        key=lambda x: -x[1],
    )
    lines = [
        r"\begin{tabular}{lr}",
        r"\hline",
        r"Ablation & Mean $|\Delta\text{MAPE}|$ \\",
        r"\hline",
    ]
    for label, val in impact_pairs:
        lines.append(f"{label} & {_fmt_pct(val)} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    write_table(TABLES_DIR / "stocks_ablation_impact.tex", "\n".join(lines))


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=== Smart home: best models ===")
    sh_best = best_models_smart_home()
    print(sh_best.to_string(index=False))
    write_best_smart_home_table(sh_best)

    print("\n=== Stocks: best models ===")
    st_best = best_models_stocks()
    print(st_best.to_string(index=False))
    write_best_stocks_table(st_best)

    print("\n=== Smart home ablations ===")
    smart_home_ablations()

    print("\n=== Stocks ablations ===")
    stocks_ablations()

    print("\nAll tables written to", TABLES_DIR)


if __name__ == "__main__":
    main()
