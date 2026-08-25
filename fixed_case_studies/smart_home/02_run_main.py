"""Main (retrospective) run for the smart-home case study.

    python fixed_case_studies/smart_home/02_run_main.py

Runs the frozen configuration matrix on the primary split (91-day train,
260-day horizon) and the two rolling-origin sensitivities.  The 2016
period was already evaluated in the original study, so **all outputs are
labelled retrospective development data** (PROTOCOL.md §3) and no
confirmatory claims may be built on them.

The Boston temperature context is cut off at the end of each training
window (the original runner leaked the future into the context fit — a
protocol correction documented in the README).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import pandas as pd

from fixed_case_studies import common
from fixed_case_studies.smart_home import config as cfg
from fixed_case_studies.smart_home.config import PRIMARY_TRAIN_CUTOFF, ROLLING_ORIGINS
from fixed_case_studies.smart_home.runner import RESULTS_ROOT, run_cell, seed_for


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--progressbar", action="store_true")
    args = ap.parse_args()

    configs = cfg.MAIN_MATRIX if args.configs is None else args.configs
    out_dir = RESULTS_ROOT / "main"

    data_end = pd.Timestamp(cfg.DATA_END)
    splits = [
        {"train_start": "2016-01-01", "train_end": PRIMARY_TRAIN_CUTOFF,
         "label": "primary", "horizon": cfg.HORIZON_DAYS},
    ] + [
        {
            **r,
            "horizon": int((data_end - pd.Timestamp(r["train_end"])).days) + 1,
        }
        for r in ROLLING_ORIGINS
    ]

    for oi, split in enumerate(splits):
        for ci, name in enumerate(configs):
            cell_cfg = cfg.CONFIGS[name]
            # Repeated-seed robustness for the frozen finalists (P1-10).
            seeds = (
                list(common.FINALIST_SEEDS)
                if name in {"main", "no_transfer"}
                else [seed_for(oi, ci)]
            )
            for seed in seeds:
                print(f"[retrospective] {split['label']} :: {name} :: seed {seed}", flush=True)
                run_cell(
                    cell_cfg, split, seed=seed, out_dir=out_dir,
                    progressbar=args.progressbar,
                )
    print("Smart-home main run finished. Results in", out_dir)


if __name__ == "__main__":
    main()
