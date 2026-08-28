"""Report for the smart-home case study (retrospective).

    python fixed_case_studies/smart_home/05_report.py

Aggregates the artifact bundles: per config x split, the primary aggregate
(median Relative MAE vs persistence) plus mean/IQR/proportion-below-1,
secondary MAE/RMSE/safeguarded MAPE, exclusion and failure counts.
Both the vangja arms (``03_run_main.py``) and the classical baselines
(``04_run_baselines.py``) are aggregated — they share the same per-unit
schema — and the best classical baseline is compared with the main
configuration.
For the frozen finalists (main vs no_transfer) on the primary split it also
reports per-appliance effects and the negative-transfer rate.

The four appliance series support descriptive per-series effects only —
no population-level inference is implied (PROTOCOL.md §9).  Context series
(the temperature) never appear in any aggregate (verified).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from fixed_case_studies import baselines, common
from fixed_case_studies.smart_home import config as cfg
from fixed_case_studies.smart_home.runner import RESULTS_ROOT


def config_description(name: str) -> str:
    """Human-readable description for vangja configs and baselines."""
    if name in cfg.CONFIGS:
        return cfg.CONFIGS[name].description
    return baselines.BASELINE_DESCRIPTIONS.get(name, "")


def collect_units(results_root: Path) -> pd.DataFrame:
    frames = []
    for f in results_root.rglob("units_*.csv"):
        frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit("No unit CSVs found; run the pipeline first.")
    units = pd.concat(frames, ignore_index=True)
    if "seed" in units.columns and units["seed"].nunique() > 1:
        keep = ["series", "origin", "config", "stage", "excluded"]
        mean_cols = ["mae", "rmse", "mape", "mae_scaled", "persistence_mae_scaled"]
        grouped = units.groupby(keep, as_index=False, dropna=False)[mean_cols].mean()
        grouped["rel_mae"] = grouped["mae_scaled"] / grouped["persistence_mae_scaled"]
        grouped.loc[grouped["excluded"], "rel_mae"] = float("nan")
        units = grouped
    return units


def main() -> None:
    out_dir = RESULTS_ROOT / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    units = collect_units(RESULTS_ROOT)
    failures = []
    for f in RESULTS_ROOT.rglob("failures.csv"):
        failures.append(pd.read_csv(f))
    failures = (
        pd.concat(failures, ignore_index=True)
        if failures
        else pd.DataFrame(columns=["origin", "config"])
    )

    context_leaks = set(units["series"]) & common.CONTEXT_SERIES
    if context_leaks:
        raise SystemExit(f"Context series found in target aggregates: {context_leaks}")

    rows = []
    for (origin, config_name), g in units.groupby(["origin", "config"]):
        agg = common.aggregate_rel_mae(g)
        n_fail = int(
            ((failures["origin"] == origin) & (failures["config"] == config_name)).sum()
        )
        rows.append(
            {
                "origin": origin,
                "config": config_name,
                **agg,
                "mean_mae": float(g["mae"].mean()),
                "mean_rmse": float(g["rmse"].mean()),
                "mean_mape": float(g["mape"].mean()),
                "n_failures": n_fail,
                "description": config_description(config_name),
            }
        )
    aggregates = pd.DataFrame(rows).sort_values(["origin", "config"])
    aggregates.to_csv(out_dir / "aggregates.csv", index=False)

    # Baseline comparison on the primary split: main vs the classical
    # baselines (primary metric: median Relative MAE).
    primary = units[units["origin"] == "primary"]
    # Baselines = any config that is not a vangja registry config.
    baseline_codes = sorted(set(primary["config"].unique()) - set(cfg.CONFIGS))
    main_med = primary.loc[primary["config"] == "main", "rel_mae"].median()
    base_med = (
        primary[primary["config"].isin(baseline_codes)]
        .groupby("config")["rel_mae"]
        .median()
        .dropna()
        .sort_values()
    )
    baseline_cmp = pd.DataFrame(
        {"baseline": base_med.index, "median_rel_mae": base_med.values}
    )
    baseline_cmp["main_median_rel_mae"] = main_med
    baseline_cmp["beats_main"] = baseline_cmp["median_rel_mae"] < main_med
    baseline_cmp.to_csv(out_dir / "baseline_comparison_primary.csv", index=False)

    # Per-appliance finalist comparison on the primary split.
    primary = units[
        (units["origin"] == "primary") & (units["config"].isin(["main", "no_transfer"]))
    ]
    per_app = primary.pivot_table(
        index="series", columns="config", values="rel_mae"
    ).reset_index()
    per_app["transfer_wins"] = per_app["main"] < per_app["no_transfer"]
    per_app.to_csv(out_dir / "per_appliance.csv", index=False)

    paired, summary = common.paired_transfer_comparison(
        units, "main", "no_transfer", stage_filter=None
    )
    paired.to_csv(out_dir / "paired_main_vs_no_transfer.csv", index=False)
    common.save_json(out_dir / "paired_summary.json", summary)

    print("=== Smart-home aggregates (retrospective; Relative MAE) ===")
    print(
        aggregates[
            [
                "origin",
                "config",
                "median",
                "mean",
                "q1",
                "q3",
                "prop_below_1",
                "n_units",
                "n_excluded",
                "n_failures",
                "mean_mape",
            ]
        ].to_string(index=False)
    )
    print("\n=== Per-appliance, primary split ===")
    print(per_app.to_string(index=False))
    print("\n=== Best classical baseline vs main (primary split) ===")
    if baseline_cmp.empty:
        print("  no baseline units found; run 04_run_baselines.py first")
    else:
        best = baseline_cmp.iloc[0]
        print(
            f"  main (median Rel.MAE): {main_med:.3f}; "
            f"best baseline: {best['baseline']} ({best['median_rel_mae']:.3f}); "
            f"{int(baseline_cmp['beats_main'].sum())} of "
            f"{len(baseline_cmp)} baselines beat main."
        )
        print(baseline_cmp.to_string(index=False))
    print("\n=== Paired main vs no_transfer (descriptive) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nReport written to {out_dir}")


if __name__ == "__main__":
    main()
