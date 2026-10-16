#!/usr/bin/env python3
"""Fixture runner implementing the Vangja checkpoint contract.

Used by .pi/tests/skills/test_checkpoint_smoke.py. It is deliberately simple:
one unit = one sleep, one ``fit <unit>`` line in work.log, one atomic update of
progress.json, and ``skip <unit>`` in skips.log for units already recorded.

  --out DIR      run directory (work.log, skips.log)
  --resume DIR   checkpoint directory holding progress.json (default: --out)
  --limit N      stop after N newly fitted units (0 = unlimited)
  --linger SEC   after reaching --limit, stay alive so a supervisor can kill
                 the process mid-run (used by checkpoint_smoke.py)
  --units        comma-separated unit names
  --sleep        seconds per fitted unit
"""

import argparse
import json
import os
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--units", default="a,b,c,d")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--linger", type=float, default=0.0)
    parser.add_argument("--sleep", type=float, default=0.15)
    args = parser.parse_args()

    ckpt = args.resume or args.out
    args.out.mkdir(parents=True, exist_ok=True)
    progress_file = ckpt / "progress.json"
    if progress_file.exists():
        completed = list(json.loads(progress_file.read_text()).get("completed", []))
    else:
        completed = []

    work_log = open(args.out / "work.log", "a")
    skip_log = open(args.out / "skips.log", "a")
    fitted = 0
    for unit in args.units.split(","):
        unit = unit.strip()
        if not unit:
            continue
        if unit in completed:
            skip_log.write(f"skip {unit}\n")
            skip_log.flush()
            continue
        if args.limit and fitted >= args.limit:
            break
        time.sleep(args.sleep)
        work_log.write(f"fit {unit}\n")
        work_log.flush()
        fitted += 1
        completed.append(unit)
        tmp = progress_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "completed": completed}))
        os.replace(tmp, progress_file)

    if args.linger > 0 and args.limit and fitted >= args.limit:
        # Stay alive so checkpoint_smoke.py can SIGTERM us mid-run: this is
        # what a 24h wall-limit kill looks like to the runner.
        time.sleep(args.linger)

    print(f"DONE completed={len(completed)} fitted={fitted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
