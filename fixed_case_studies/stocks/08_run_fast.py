"""Fast iteration harness for the stocks case study.

    python fixed_case_studies/stocks/08_run_fast.py --origins 2013-01-01 \
        --configs main transfer_only --max-stocks 30 --horizon 90

Purpose: rapid diagnostic screening while iterating on model choices.
Three accelerations vs the frozen study:

1. **Fewer stocks** — a seeded subset of the universe (default 30; use
   ``--tickers highweight`` for the frozen 12 mega caps).
2. **ADVI source** — the context model is fitted with ADVI instead of NUTS
   (explicitly allowed for development screening; never used as covariance
   evidence — the frozen confirmation runs keep NUTS).
3. **Shorter horizon** — ``--horizon`` (default 90 calendar days ≈ 63
   trading days) instead of 365.

Everything else (trading-day data, Relative MAE vs persistence, seeding,
artifact recording) is identical to the frozen pipeline, and results land
in ``results/fast/`` so they can never be confused with frozen-study
artifacts.  Nothing here changes the frozen configuration: any tweak that
survives fast iteration must be re-run through the frozen pipeline.

Examples
--------
    # one config, one origin, 30 stocks, 90-day horizon (seconds–minutes)
    python fixed_case_studies/stocks/08_run_fast.py --origins 2013-01-01 \\
        --configs main

    # full main matrix on all dev origins, 50 stocks (fast dev sweep)
    python fixed_case_studies/stocks/08_run_fast.py --max-stocks 50

    # try a custom tweak (defensible single change, documented in RUN_NOTES)
    python fixed_case_studies/stocks/08_run_fast.py --configs main \\
        --experiment lt_loss 0.0
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import numpy as np
import pandas as pd

from fixed_case_studies import common
from fixed_case_studies.stocks import config as cfg
from fixed_case_studies.stocks.runner import (
    RESULTS_ROOT,
    get_universe_for_origin,
    run_cell,
    seed_for,
)


def build_experiment_config(name: str, overrides: dict) -> cfg.StockConfig:
    """Build a StockConfig from the frozen main + per-field overrides."""
    base = cfg.CONFIGS[name]
    return cfg.StockConfig(
        name=f"{name}__" + "_".join(f"{k}{v}" for k, v in sorted(overrides.items())),
        description=(
            f"Fast-iteration tweak of {name}: "
            + ", ".join(f"{k}={v}" for k, v in sorted(overrides.items()))
        ),
        tune_method=overrides.get("tune_method", base.tune_method),
        lt_loss_factor=overrides.get("lt_loss_factor", base.lt_loss_factor),
        fs_loss_factor=overrides.get("fs_loss_factor", base.fs_loss_factor),
        lt_pool=overrides.get("lt_pool", base.lt_pool),
        fs_pool=overrides.get("fs_pool", base.fs_pool),
        lt_shrinkage=overrides.get("lt_shrinkage", base.lt_shrinkage),
        fs_shrinkage=overrides.get("fs_shrinkage", base.fs_shrinkage),
        include_context_group=base.include_context_group,
        negative_control=base.negative_control,
        high_weight_subset=base.high_weight_subset,
        n_changepoints=int(overrides.get("n_changepoints", base.n_changepoints)),
    )


def subset_tickers(tickers: list[str], origin: str, max_stocks: int | None,
                   mode: str) -> list[str]:
    if mode == "highweight":
        return [t for t in tickers if t in cfg.HIGH_WEIGHT_STOCKS]
    if max_stocks is None or len(tickers) <= max_stocks:
        return tickers
    rng = np.random.default_rng(common.BASE_SEED + hash(origin) % 2**31)
    return sorted(rng.choice(sorted(tickers), size=max_stocks, replace=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", nargs="*", default=["2013-01-01", "2013-07-01", "2014-01-01"])
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--max-stocks", type=int, default=30)
    ap.add_argument("--tickers", choices=["subset", "highweight"], default="subset")
    ap.add_argument("--horizon", type=int, default=90)
    ap.add_argument(
        "--experiment",
        nargs=2,
        action="append",
        metavar=("FIELD", "VALUE"),
        default=[],
        help="Override one field of the selected config(s); repeatable.",
    )
    ap.add_argument("--source-method", choices=["advi", "nuts"], default="advi")
    args = ap.parse_args()

    configs = cfg.MAIN_MATRIX if args.configs is None else args.configs
    cells: dict[str, cfg.StockConfig] = {}
    for name in configs:
        if name not in cfg.CONFIGS:
            raise KeyError(f"Unknown config {name}")
        cells[name] = cfg.CONFIGS[name]
    if args.experiment:
        overrides = {}
        for field, value in args.experiment:
            try:
                value = float(value)
            except ValueError:
                pass
            overrides[field] = value
        cells = {
            name: build_experiment_config(name, overrides) for name in cells
        }

    out_dir = RESULTS_ROOT / "fast"
    for oi, origin in enumerate(args.origins):
        tickers = get_universe_for_origin(origin)
        tickers = subset_tickers(
            tickers, origin, args.max_stocks, args.tickers
        )
        print(f"[fast] origin={origin} n_stocks={len(tickers)} "
              f"horizon={args.horizon} source={args.source_method}", flush=True)
        for ci, (name, cell_cfg) in enumerate(cells.items()):
            print(f"[fast] {origin} :: {name}", flush=True)
            run_cell(
                cell_cfg, origin, "fast", tickers,
                seed=seed_for(oi, ci),
                source_method=args.source_method,
                out_dir=out_dir,
                max_stocks=None,  # already subset
                horizon_days=args.horizon,
            )

    # Quick summary of the fast results.
    frames = [pd.read_csv(f) for f in out_dir.glob("units_*.csv")]
    if frames:
        units = pd.concat(frames, ignore_index=True)
        print("\n=== Fast-iteration summary (median Relative MAE vs persistence) ===")
        for cfg_name, g in units.groupby("config"):
            agg = common.aggregate_rel_mae(g)
            print(f"{cfg_name:40s} median={agg['median']:.3f} "
                  f"mean={agg['mean']:.3f} p<1={agg['prop_below_1']:.2f} "
                  f"n={agg['n_units']}")
    print("\nFast results in", out_dir, "(kept separate from frozen-study results)")


if __name__ == "__main__":
    main()
