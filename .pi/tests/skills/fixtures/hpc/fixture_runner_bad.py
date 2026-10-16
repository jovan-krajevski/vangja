#!/usr/bin/env python3
"""Fixture runner that IGNORES --resume (must make checkpoint_smoke fail)."""

import argparse
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--units", default="a,b,c,d")
    parser.add_argument("--sleep", type=float, default=0.05)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    work_log = open(args.out / "work.log", "a")
    fitted = 0
    for unit in args.units.split(","):
        if args.limit and fitted >= args.limit:
            break
        time.sleep(args.sleep)
        work_log.write(f"fit {unit}\n")
        work_log.flush()
        fitted += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
