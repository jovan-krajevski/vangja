"""Targeted analyses beyond the main matrix.

    python fixed_case_studies/stocks/04_run_ablations.py

Runs only the configs that are analysed separately from the main matrix:

- ``combined`` (transfer + context-as-group) on the **development** origins —
  reproduces the former headline design for retrospective comparability.
- ``transfer_gold`` / ``transfer_gold_smp`` on the **confirmation** origins,
  restricted to the frozen high-weight mega-cap subset — the targeted
  S&P-overlap sensitivity with an unrelated negative-control context
  (PLAN P1-9, PROTOCOL.md §2.4).

Everything is frozen a priori; nothing is selected from these results.
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
    ap.add_argument("--progressbar", action="store_true")
    args = ap.parse_args()

    # 1. Reproduce the former headline design on development origins only.
    for oi, origin in enumerate(cfg.DEV_ORIGINS):
        tickers = get_universe_for_origin(origin)
        print(f"[development] {origin} :: combined", flush=True)
        run_cell(
            cfg.CONFIGS["combined"],
            origin,
            "development",
            tickers,
            seed=seed_for(oi, 20),
            source_method="advi",
            out_dir=RESULTS_ROOT / "development",
            progressbar=args.progressbar,
        )

    # 2. Negative-control sensitivity on confirmation origins (high-weight
    #    subset only, both contexts matched).
    confirmation = common.load_confirmation_origins(PROTOCOL_DIR)
    origins = [o.strftime("%Y-%m-%d") for o in confirmation["origin"]]
    for oi, origin in enumerate(origins):
        tickers = get_universe_for_origin(origin, require_frozen=True)
        # Restrict to the frozen high-weight mega caps present in the
        # universe (data availability may drop a few).
        subset = [t for t in tickers if t in cfg.HIGH_WEIGHT_STOCKS]
        for ci, name in enumerate(["transfer_gold", "transfer_gold_smp"]):
            print(f"[confirmation-sensitivity] {origin} :: {name}", flush=True)
            run_cell(
                cfg.CONFIGS[name],
                origin,
                "confirmation",
                subset,
                seed=seed_for(oi, 30 + ci),
                source_method="nuts",
                out_dir=RESULTS_ROOT / "confirmation_sensitivity",
                progressbar=args.progressbar,
            )

    print("Targeted analyses finished.")


if __name__ == "__main__":
    main()
