"""Fetch and freeze the smart-home data.

    python fixed_case_studies/smart_home/01_fetch_data.py

Downloads the smart-home appliance readings and the Boston temperature
context through the package loaders and writes frozen local copies with
SHA-256 sidecars. The temperature context is stored per training cutoff so
that no future information can leak into the transferred seasonality.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from fixed_case_studies import common
from fixed_case_studies.smart_home import config as cfg
from fixed_case_studies.smart_home.config import (
    PRIMARY_TRAIN_CUTOFF,
    ROLLING_ORIGINS,
)
from vangja.datasets import load_kaggle_temperature, load_smart_home_readings


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sh_df = load_smart_home_readings(column=cfg.SMART_HOME_COLUMNS, freq="D")
    sh_df.to_csv(data_dir / "smart_home_daily.csv", index=False)

    cutoffs = [PRIMARY_TRAIN_CUTOFF] + [r["train_end"] for r in ROLLING_ORIGINS]
    for cutoff in cutoffs:
        end = pd.Timestamp(cutoff) - pd.Timedelta(days=1)
        temp = load_kaggle_temperature(
            city=cfg.TEMP_CITY,
            start_date=cfg.TEMP_START,
            end_date=end,
            freq="D",
        )
        tag = cutoff.replace("-", "")
        temp.to_csv(data_dir / f"boston_temp_until_{tag}.csv", index=False)
        print(f"boston temp until {end.date()}: {len(temp)} rows")

    hashes = common.record_sidecars(data_dir)
    print(f"\nWrote {len(hashes)} files with SHA-256 sidecars.")


if __name__ == "__main__":
    main()
