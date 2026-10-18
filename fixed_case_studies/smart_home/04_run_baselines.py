"""Classical baselines for the smart-home case study (retrospective).

    python fixed_case_studies/smart_home/04_run_baselines.py

Runs the frozen model-free / classical baselines (persistence, drift,
seasonal naive, rolling / global mean, ARIMA, Holt-Winters) on the
**identical frozen splits** used by ``03_run_main.py`` — the primary split
(91-day train, 260-day horizon) and the two rolling origins. Only the
training window is ever used for fitting; the test horizon is never touched
by any baseline (audit F-1). All data come from the frozen local CSVs
written by ``01_fetch_data.py`` (SHA-256 sidecars).

Outputs land in ``results/baselines/`` with the same per-unit schema as the
vangja arms (``common.baseline_unit_metrics``), so ``05_report.py``
aggregates models and baselines together. Stage label: ``retrospective``
(the 2016 period is development data only, PROTOCOL.md §3).

Requires ``statsmodels`` (``uv sync --extra reproducibility``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import pandas as pd

from fixed_case_studies import baselines, common
from fixed_case_studies.smart_home import config as cfg
from fixed_case_studies.smart_home.runner import RESULTS_ROOT

DATA_DIR = Path(__file__).resolve().parent / "data"

# Daily smart-home series: weekly seasonality has period 7.
PERIOD = 7


def load_frozen_split(train_end: str):
    """Frozen data cut at ``train_end`` (identical to the study splits)."""
    sh = pd.read_csv(DATA_DIR / "smart_home_daily.csv", parse_dates=["ds"])
    cutoff = pd.Timestamp(train_end)
    train = sh[sh["ds"] < cutoff].copy()
    test = sh[(sh["ds"] >= cutoff) & (sh["ds"] <= pd.Timestamp(cfg.DATA_END))].copy()
    return train, test


def baseline_specs() -> list[tuple[str, str, object]]:
    """(code, human name, func(train_y, horizon) -> forecast)."""
    f = baselines
    p = PERIOD
    return [
        ("persistence", "Persistence (random walk)", f.persistence),
        ("drift", "Drift", f.drift),
        ("snaive_7", "Seasonal naive (7d)", lambda y, h: f.seasonal_naive(y, h, p)),
        (
            "snaive_mean_7",
            "Seasonal naive mean (7d)",
            lambda y, h: f.seasonal_naive_mean(y, h, p),
        ),
        ("rolling_7", "Rolling mean (7d)", lambda y, h: f.rolling_mean(y, h, p)),
        ("rolling_30", "Rolling mean (30d)", lambda y, h: f.rolling_mean(y, h, 30)),
        ("global_mean", "Global mean", f.global_mean),
        ("arima_111", "ARIMA(1,1,1)", lambda y, h: f.fit_arima(y, h, order=(1, 1, 1))),
        ("arima_211", "ARIMA(2,1,1)", lambda y, h: f.fit_arima(y, h, order=(2, 1, 1))),
        (
            "arima_best",
            "ARIMA (AIC)",
            lambda y, h: f.fit_arima_best(y, h, seasonal_period=p),
        ),
        ("hw_aa_7", "Holt-Winters (A,A,7d)", lambda y, h: f.fit_holt_winters(y, h, p)),
        (
            "hw_am_7",
            "Holt-Winters (A,M,7d)",
            lambda y, h: f.fit_holt_winters(y, h, p, seasonal="mul"),
        ),
        (
            "hw_da_7",
            "Holt-Winters damped (A,A,7d)",
            lambda y, h: f.fit_holt_winters(y, h, p, damped_trend=True),
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=common.BASE_SEED)
    ap.add_argument(
        "--progressbar", action="store_true", help="unused; kept for CLI parity"
    )
    args = ap.parse_args()
    seed = args.seed

    # Identical splits to 03_run_main.py (primary + rolling origins).
    data_end = pd.Timestamp(cfg.DATA_END)
    splits = [
        {
            "train_start": "2016-01-01",
            "train_end": cfg.PRIMARY_TRAIN_CUTOFF,
            "label": "primary",
            "horizon": cfg.HORIZON_DAYS,
        },
    ] + [
        {
            **r,
            "horizon": int((data_end - pd.Timestamp(r["train_end"])).days) + 1,
        }
        for r in cfg.ROLLING_ORIGINS
    ]

    specs = baseline_specs()
    out_dir = RESULTS_ROOT / "baselines"
    all_units = []
    for split in splits:
        print(
            f"[baselines] {split['label']} (train_end={split['train_end']}, "
            f"horizon={split['horizon']})",
            flush=True,
        )
        train, test = load_frozen_split(split["train_end"])
        unit_df = baselines.run_baselines_origin(
            train,
            test,
            specs,
            origin=split["label"],
            stage="retrospective",
            scale_mode="minmax_individual",
            study="smart_home",
            out_dir=out_dir,
            seed=seed,
        )
        all_units.append(unit_df)
        med = unit_df.groupby("config")["rel_mae"].median()
        print(med.round(3).to_string())

    units = pd.concat(all_units, ignore_index=True)
    summary = (
        units.groupby(["origin", "config"])["rel_mae"]
        .median()
        .unstack()
        .reindex(columns=[c for c, _n, _f in specs])
    )
    summary.to_csv(out_dir / "summary_by_origin.csv")
    print(f"\n[baselines] summary by origin -> {out_dir / 'summary_by_origin.csv'}")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
