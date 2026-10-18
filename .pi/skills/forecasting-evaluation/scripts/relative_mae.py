#!/usr/bin/env python3
"""Reference implementation of Relative MAE vs persistence (stdlib only).

Input CSVs are long-format (``ds,series,y`` for train/test and
``ds,series,yhat`` for predictions). For every series it computes the
persistence forecast as the last training value, then

    rel_mae = MAE(model) / MAE(persistence)

Values must already be in the scaled space used for fitting (the epsilon rule
is defined on the scaled target). Units with persistence MAE < epsilon are
excluded from the aggregate and counted; they are never silently dropped.

Prints the documented JSON shape; with ``--out`` also writes ``units.csv`` and
``aggregate.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

EPSILON = 1e-3


def sort_key(value: str):
    try:
        return (0, datetime.fromisoformat(value))
    except (ValueError, TypeError):
        return (1, str(value))


def read_rows(path: Path, needed: tuple[str, ...]) -> list[dict]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in needed if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}; "
                             f"found {reader.fieldnames}")
        return [row for row in reader]


def to_float(value: str, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: not a number: {value!r}") from exc


def relative_mae_units(train: list[dict], test: list[dict], pred: list[dict],
                       epsilon: float) -> tuple[list[dict], dict]:
    last: dict[str, float] = {}
    for row in sorted(train, key=lambda r: sort_key(r["ds"])):
        last[row["series"]] = to_float(row["y"], f"train ds={row['ds']}")
    forecasts: dict[tuple[str, str], float] = {}
    for row in pred:
        key = (row["series"], row["ds"])
        if key in forecasts:
            raise ValueError(f"duplicate prediction for series={key[0]} ds={key[1]}")
        forecasts[key] = to_float(row["yhat"], f"pred series={key[0]} ds={key[1]}")

    by_series: dict[str, list[dict]] = {}
    for row in sorted(test, key=lambda r: (r["series"], sort_key(r["ds"]))):
        by_series.setdefault(row["series"], []).append(row)

    units: list[dict] = []
    for series_name, rows in sorted(by_series.items()):
        matched = [(row, forecasts[(row["series"], row["ds"])])
                   for row in rows if (row["series"], row["ds"]) in forecasts]
        if series_name not in last or not matched:
            continue
        errors = [abs(to_float(row["y"], f"test ds={row['ds']}") - yhat)
                  for row, yhat in matched]
        persistence = last[series_name]
        pers_errors = [abs(to_float(row["y"], f"test ds={row['ds']}") - persistence)
                       for row, _ in matched]
        mae = sum(errors) / len(errors)
        pers_mae = sum(pers_errors) / len(pers_errors)
        excluded = pers_mae < epsilon or pers_mae == 0.0
        rel = None if excluded else mae / pers_mae
        units.append({
            "series": series_name,
            "n": len(matched),
            "mae": mae,
            "persistence_mae": pers_mae,
            "rel_mae": rel,
            "excluded": excluded,
        })

    scored = [u["rel_mae"] for u in units if not u["excluded"]]
    aggregate = None
    if scored:
        ordered = sorted(scored)

        def quantile(q: float) -> float:
            if len(ordered) == 1:
                return ordered[0]
            pos = q * (len(ordered) - 1)
            lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
            return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)

        aggregate = {
            "median": quantile(0.5),
            "mean": sum(ordered) / len(ordered),
            "q1": quantile(0.25),
            "q3": quantile(0.75),
            "prop_below_1": sum(1 for v in ordered if v < 1) / len(ordered),
        }
    return units, aggregate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="relative_mae.py",
        description="Relative MAE vs persistence with the epsilon exclusion rule.",
        epilog="Example: relative_mae.py --train train.csv --test test.csv "
               "--pred pred.csv --epsilon 1e-3 --out results/relmae",
    )
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--test", required=True, type=Path)
    p.add_argument("--pred", required=True, type=Path)
    p.add_argument("--epsilon", type=float, default=EPSILON,
                   help=f"scaled persistence-MAE exclusion threshold (default {EPSILON})")
    p.add_argument("--series-col", default="series")
    p.add_argument("--time-col", default="ds")
    p.add_argument("--target-col", default="y")
    p.add_argument("--pred-col", default="yhat")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    try:
        train = read_rows(args.train, (args.series_col, args.time_col, args.target_col))
        test = read_rows(args.test, (args.series_col, args.time_col, args.target_col))
        pred = read_rows(args.pred, (args.series_col, args.time_col, args.pred_col))
    except (OSError, ValueError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2

    rename = {
        args.series_col: "series", args.time_col: "ds",
        args.target_col: "y", args.pred_col: "yhat",
    }
    train = [{rename[k]: v for k, v in row.items()} for row in train]
    test = [{rename[k]: v for k, v in row.items()} for row in test]
    pred = [{rename[k]: v for k, v in row.items()} for row in pred]

    try:
        units, aggregate = relative_mae_units(train, test, pred, args.epsilon)
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2

    n_excluded = sum(1 for u in units if u["excluded"])
    report = {
        "metric": "relative_mae_vs_persistence",
        "epsilon": args.epsilon,
        "scale": "values must be on the scaled target used for fitting",
        "n_units_total": len(units),
        "n_units_scored": len(units) - n_excluded,
        "n_excluded": n_excluded,
        "aggregate": aggregate,
        "units": units,
    }

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "units.csv", "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["series", "n", "mae", "persistence_mae", "rel_mae", "excluded"])
            writer.writeheader()
            writer.writerows(units)
        (args.out / "aggregate.json").write_text(
            json.dumps({k: report[k] for k in
                        ("metric", "epsilon", "n_units_total", "n_units_scored",
                         "n_excluded", "aggregate")}, indent=2) + "\n")

    print(json.dumps(report, indent=2))
    return 0 if units else 1


if __name__ == "__main__":
    sys.exit(main())
