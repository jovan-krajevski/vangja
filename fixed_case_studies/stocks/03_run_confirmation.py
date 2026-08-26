"""Confirmation run for the stocks case study.

    python fixed_case_studies/stocks/03_run_confirmation.py

Runs the frozen configuration matrix on the **new** confirmation origins
(2023-01-01 .. 2024-12-01).  These origins were never evaluated in the
original study and their results have not been inspected.

Protocol guarantees implemented here (PROTOCOL.md §2.2, §5, §10):

- ``CONFIRMATION_ORIGINS.csv`` is hash-verified before any scoring; the run
  refuses to proceed if the frozen file was modified.
- One global configuration per case study (frozen a priori in
  ``stocks/config.py``); no per-target or per-origin selection.
- The source (S&P 500) model is refitted with **NUTS** (nutpie,
  ``target_accept=0.9``) at every origin, so the transferred posterior
  carries genuine covariance (review F-8, PLAN P0-17).
- The main configuration and the no-transfer arm are refitted across
  multiple seeds (``FINALIST_SEEDS``) for seed-robustness reporting.
- Every target is scored exactly once per (config, seed); failures are
  recorded and reported, never silently dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

from fixed_case_studies import common
from fixed_case_studies.stocks import config as cfg
from fixed_case_studies.stocks.runner import (
    RESULTS_ROOT,
    get_universe_for_origin,
    run_cell,
    seed_for,
)

PROTOCOL_DIR = Path(__file__).resolve().parents[1] / "protocol"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", nargs="*", default=None)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument(
        "--seed-mode",
        choices=["single", "finalists"],
        default="finalists",
        help="single seed everywhere, or repeated seeds for finalists.",
    )
    ap.add_argument("--progressbar", action="store_true")
    args = ap.parse_args()

    # Freeze check before any confirmation scoring.
    confirmation = common.load_confirmation_origins(PROTOCOL_DIR)
    origins = (
        [o.strftime("%Y-%m-%d") for o in confirmation["origin"]]
        if args.origins is None
        else args.origins
    )
    configs = cfg.MAIN_MATRIX if args.configs is None else args.configs
    out_dir = RESULTS_ROOT / "confirmation"

    for oi, origin in enumerate(origins):
        tickers = get_universe_for_origin(origin, require_frozen=True)
        for ci, name in enumerate(configs):
            if name not in cfg.CONFIGS:
                raise KeyError(f"Unknown config {name}")
            cell_cfg = cfg.CONFIGS[name]
            seeds = (
                list(common.FINALIST_SEEDS)
                if args.seed_mode == "finalists"
                and name in {"main", "no_transfer", "transfer_only", "target_only"}
                else [seed_for(oi, ci)]
            )
            for seed in seeds:
                print(
                    f"[confirmation] {origin} :: {name} :: seed {seed}", flush=True
                )
                run_cell(
                    cell_cfg,
                    origin,
                    "confirmation",
                    tickers,
                    seed=seed,
                    source_method="nuts",
                    out_dir=out_dir,
                    progressbar=args.progressbar,
                )
    print("Confirmation run finished. Results in", out_dir)


if __name__ == "__main__":
    main()
