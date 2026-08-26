"""Report and paired-comparison analysis for the stocks case study.

    python fixed_case_studies/stocks/07_report.py

Aggregates the artifact bundles produced by the development, confirmation
and targeted runs and produces:

- ``results/report/aggregates.csv`` — per config x stage: median / mean /
  IQR / proportion-below-1 of Relative MAE (the primary metric), unit and
  exclusion counts, failure counts, and secondary MAE/RMSE/safeguarded-MAPE
  means. Context series are never present in any aggregate (verified).
- ``results/report/paired_transfer_vs_no_transfer.csv`` +
  ``...bootstrap.json`` — the frozen transfer vs no-transfer comparison on
  the confirmation stage with two-way block resampling over stocks and
  half-year origin blocks (PROTOCOL.md §9); includes the negative-transfer
  rate.
- ``results/report/tables/*.tex`` — LaTeX table snippets.

No configuration is selected here; the report only summarises the frozen
matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import pandas as pd

from fixed_case_studies import common
from fixed_case_studies.stocks import config as cfg
from fixed_case_studies.stocks.runner import RESULTS_ROOT


def collect_units(results_root: Path) -> pd.DataFrame:
    frames = []
    for pattern in ("development", "confirmation", "confirmation_sensitivity",
                    "covariance_transfer"):
        base = results_root / pattern
        if not base.exists():
            continue
        for f in base.rglob("units_*.csv"):
            df = pd.read_csv(f)
            if "stage" not in df.columns:
                df["stage"] = "confirmation" if pattern != "development" else "development"
            frames.append(df)
    if not frames:
        raise SystemExit("No unit CSVs found; run the pipeline first.")
    units = pd.concat(frames, ignore_index=True)

    # Seed aggregation (PROTOCOL.md §4): average per-unit MAEs over seeds
    # before forming Relative MAE, then recompute aggregates on the
    # averaged units.
    if "seed" in units.columns and units["seed"].nunique() > 1:
        keep = ["series", "origin", "config", "stage", "excluded"]
        mean_cols = ["mae", "rmse", "mape", "mae_scaled", "persistence_mae_scaled"]
        grouped = units.groupby(keep, as_index=False, dropna=False)[mean_cols].mean()
        grouped["rel_mae"] = grouped["mae_scaled"] / grouped["persistence_mae_scaled"]
        grouped.loc[grouped["excluded"], "rel_mae"] = float("nan")
        units = grouped
    return units


def collect_failures(results_root: Path) -> pd.DataFrame:
    frames = []
    for f in results_root.rglob("failures.csv"):
        frames.append(pd.read_csv(f))
    if not frames:
        return pd.DataFrame(columns=["stage", "config"])
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    args = ap.parse_args()

    out_dir = RESULTS_ROOT / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    units = collect_units(RESULTS_ROOT)
    failures = collect_failures(RESULTS_ROOT)

    # Freeze check (P0-12): no context series in any target aggregate.
    context_leaks = set(units["series"]) & common.CONTEXT_SERIES
    if context_leaks:
        raise SystemExit(
            f"Context series found in target aggregates: {context_leaks}"
        )

    rows = []
    for (stage, config_name), g in units.groupby(["stage", "config"]):
        agg = common.aggregate_rel_mae(g)
        n_fail = int(((failures["stage"] == stage) & (failures["config"] == config_name)).sum())
        rows.append(
            {
                "stage": stage,
                "config": config_name,
                **agg,
                "mean_mae": float(g["mae"].mean()),
                "mean_rmse": float(g["rmse"].mean()),
                "mean_mape": float(g["mape"].mean()),
                "n_failures": n_fail,
                "description": cfg.CONFIGS.get(config_name, cfg.StockConfig(
                    name=config_name, description="",
                    tune_method=None, lt_loss_factor=0, fs_loss_factor=0,
                    lt_pool="", fs_pool="", lt_shrinkage=0, fs_shrinkage=0,
                )).description,
            }
        )
    aggregates = pd.DataFrame(rows).sort_values(["stage", "config"])
    aggregates.to_csv(out_dir / "aggregates.csv", index=False)

    # Frozen paired comparisons on confirmation: transfer vs no-transfer on
    # the same pooling structure (PROTOCOL §9).
    for a_name, b_name in cfg.FINALIST_PAIRS:
        paired, summary = common.paired_transfer_comparison(
            units, a_name, b_name, stage_filter="confirmation",
            n_iter=args.n_bootstrap,
        )
        paired.to_csv(
            out_dir / f"paired_{a_name}_vs_{b_name}.csv", index=False
        )
        common.save_json(out_dir / f"paired_{a_name}_vs_{b_name}.json", summary)

    # LaTeX table snippets.
    tex_dir = out_dir / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    _write_latex(aggregates, tex_dir)

    print("=== Aggregates (primary metric: Relative MAE vs persistence) ===\n"
          f"(selected configuration: {cfg.SELECTED_CONFIG})")
    print(
        aggregates[
            ["stage", "config", "median", "mean", "q1", "q3",
             "prop_below_1", "n_units", "n_excluded", "n_failures", "mean_mape"]
        ].to_string(index=False)
    )
    print("\n=== Paired comparisons (confirmation, block bootstrap) ===")
    for a_name, b_name in cfg.FINALIST_PAIRS:
        summary = common.load_json(out_dir / f"paired_{a_name}_vs_{b_name}.json")
        print(f"--- {a_name} vs {b_name} ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")
    print(f"\nReport written to {out_dir}")


def _write_latex(aggregates: pd.DataFrame, tex_dir: Path) -> None:
    stage_names = {"development": "development (retrospective)",
                   "confirmation": "confirmation (held-out origins)"}
    for stage, g in aggregates.groupby("stage"):
        if stage not in stage_names:
            continue
        g = g[g["config"].isin(cfg.MAIN_MATRIX)]
        rows = []
        for _, r in g.iterrows():
            med = f"{r['median']:.3f}" if pd.notna(r["median"]) else "---"
            p1 = f"{100*r['prop_below_1']:.1f}" if pd.notna(r["prop_below_1"]) else "---"
            rows.append(
                f"    {r['config']} & {med} & {r['mean_mae']:.4f} "
                f"& {r['mean_mape']:.3f} & {p1}\\% \\\\"
            )
        table = (
            "\\begin{tabular}{lcccc}\n"
            "\\toprule\n"
            "Configuration & Median Rel.MAE & Mean MAE & Mean MAPE & "
            "\\% better than persistence \\\\\n"
            "\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )
        (tex_dir / f"stocks_{stage}_table.tex").write_text(table)


if __name__ == "__main__":
    main()
