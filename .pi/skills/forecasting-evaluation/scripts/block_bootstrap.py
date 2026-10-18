#!/usr/bin/env python3
"""Two-way block bootstrap for paired forecast differences (stdlib only).

Resamples target series with replacement and forecast-origin *blocks* with
replacement (blocks keep overlapping horizons together), recomputes the median
paired difference each iteration, and reports percentile confidence
intervals. Mirrors ``fixed_case_studies/common.py:two_way_block_bootstrap``.

Input CSV columns (configurable): ``series``, ``origin_block``,
``diff_rel_mae``. Output JSON (also written to ``--out/bootstrap.json``)::

    {
      "n_paired_units": 366,
      "n_iter": 2000,
      "sample_median": -0.013,
      "median_diff": -0.012,
      "ci_2p5": -0.041,
      "ci_97p5": 0.017,
      "prop_diff_lt_0": 0.78,
      "negative_transfer_rate": 0.22
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


def percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        raise ValueError("empty sample")
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap(paired: list[dict], diff_col: str, series_col: str, block_col: str,
              n_iter: int, seed: int) -> dict:
    series_values = sorted({row[series_col] for row in paired})
    block_values = sorted({row[block_col] for row in paired})
    if not series_values or not block_values:
        raise ValueError("no series or origin blocks found")

    rng = random.Random(seed)
    medians: list[float] = []
    for _ in range(n_iter):
        drawn_series = {rng.choice(series_values) for _ in range(len(series_values))}
        drawn_blocks = {rng.choice(block_values) for _ in range(len(block_values))}
        diffs = sorted(
            float(row[diff_col])
            for row in paired
            if row[series_col] in drawn_series and row[block_col] in drawn_blocks
        )
        if not diffs:
            continue
        medians.append(percentile(diffs, 0.5))
    if not medians:
        raise ValueError("every bootstrap iteration was empty")
    medians.sort()
    sample_diffs = sorted(float(row[diff_col]) for row in paired)
    return {
        "n_paired_units": len(paired),
        "n_iter": n_iter,
        "n_iter_used": len(medians),
        "sample_median": percentile(sample_diffs, 0.5),
        "median_diff": percentile(medians, 0.5),
        "ci_2p5": percentile(medians, 0.025),
        "ci_97p5": percentile(medians, 0.975),
        "prop_diff_lt_0": sum(1 for m in medians if m < 0) / len(medians),
        "negative_transfer_rate": sum(1 for d in sample_diffs if d > 0) / len(sample_diffs),
        "seed": seed,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="block_bootstrap.py",
        description="Two-way block bootstrap over series and origin blocks.",
        epilog="Example: block_bootstrap.py --paired paired.csv --n-iter 2000 "
               "--seed 0 --out results/boot",
    )
    p.add_argument("--paired", required=True, type=Path)
    p.add_argument("--diff-col", default="diff_rel_mae")
    p.add_argument("--series-col", default="series")
    p.add_argument("--block-col", default="origin_block")
    p.add_argument("--n-iter", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.n_iter <= 0:
        print("--n-iter must be positive", file=sys.stderr)
        return 2
    try:
        with open(args.paired, newline="") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in (args.series_col, args.block_col, args.diff_col)
                       if c not in (reader.fieldnames or [])]
            if missing:
                print(f"{args.paired}: missing column(s) {missing}", file=sys.stderr)
                return 2
            paired = list(reader)
    except OSError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2

    try:
        result = bootstrap(paired, args.diff_col, args.series_col, args.block_col,
                           args.n_iter, args.seed)
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "bootstrap.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
