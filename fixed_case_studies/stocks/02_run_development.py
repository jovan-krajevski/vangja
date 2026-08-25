"""Development (retrospective) run for the stocks case study.

    python fixed_case_studies/stocks/02_run_development.py

Runs the frozen configuration matrix (main + 6 a priori ablations) on the
original 24 origins (Jan 2013 – Dec 2014).  These origins have already been
inspected in the original study, so **every number produced here is
retrospective development/diagnostic data** — it is never presented as
confirmation (PROTOCOL.md §2.1).

The source (S&P 500) model is fitted with ADVI here: efficient development
screening is explicitly allowed, but ADVI mean-field posteriors are never
used as evidence of covariance preservation (PLAN P0-17).  Confirmation runs
(`03_run_confirmation.py`) refit the source with NUTS.

Checkpointing: completed cells are skipped on restart.  Use
``--origins 2013-01-01 2013-02-01`` or ``--configs main no_transfer`` to
limit the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

from fixed_case_studies.stocks import config as cfg
from fixed_case_studies.stocks.runner import (
    RESULTS_ROOT,
    get_universe_for_origin,
    run_cell,
    seed_for,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", nargs="*", default=None)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--progressbar", action="store_true")
    args = ap.parse_args()

    origins = cfg.DEV_ORIGINS if args.origins is None else args.origins
    configs = cfg.MAIN_MATRIX if args.configs is None else args.configs
    out_dir = RESULTS_ROOT / "development"

    for oi, origin in enumerate(origins):
        tickers = get_universe_for_origin(origin)
        for ci, name in enumerate(configs):
            if name not in cfg.CONFIGS:
                raise KeyError(f"Unknown config {name}")
            cell_cfg = cfg.CONFIGS[name]
            print(f"[development] {origin} :: {name}", flush=True)
            run_cell(
                cell_cfg,
                origin,
                "development",
                tickers,
                seed=seed_for(oi, ci),
                source_method="advi",
                out_dir=out_dir,
                progressbar=args.progressbar,
            )
    print("Development run finished. Results in", out_dir)


if __name__ == "__main__":
    main()
