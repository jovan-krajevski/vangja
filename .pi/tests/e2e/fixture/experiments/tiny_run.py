#!/usr/bin/env python3
"""Deterministic mock experiment with a checkpoint/resume contract (S7 smoke test).

The S7 end-to-end sandbox run queues this command as an experiment record and then
executes it locally (no HPC host). It writes a progress marker after every
completed unit, so a kill mid-run is resumed from the checkpoint, and it emits
the two declared outputs the artifact manifest hashes.

Usage:
    python3 experiments/tiny_run.py --outputs DIR --checkpoint-dir DIR [--resume]

Exit 0 with a JSON summary on stdout. No network, no model calls, no randomness.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal

UNITS = ("fit", "predict", "score")
SEED = 20261018


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True, help="directory for the declared outputs")
    parser.add_argument("--checkpoint-dir", required=True, help="directory holding progress.json")
    parser.add_argument("--resume", action="store_true", help="continue from the progress marker")
    args = parser.parse_args()

    outputs = pathlib.Path(args.outputs)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_file = checkpoint_dir / "progress.json"

    completed: list[str] = []
    if args.resume and progress_file.exists():
        completed = list(json.loads(progress_file.read_text()).get("completed", []))

    # Fault injection for the S7 resilience matrix: SIGKILL after unit N, once.
    kill_after = os.environ.get("TINY_RUN_KILL_AFTER")
    kill_marker = os.environ.get("TINY_RUN_KILL_MARKER")
    for unit in UNITS:
        if unit in completed:
            continue
        completed.append(unit)
        progress_file.write_text(json.dumps({"version": 1, "completed": completed}) + "\n")
        if kill_after and kill_marker and len(completed) == int(kill_after) and not pathlib.Path(kill_marker).exists():
            pathlib.Path(kill_marker).write_text("killed after unit 1\n")
            os.kill(os.getpid(), signal.SIGKILL)

    numbers = {"series": "synthetic", "rel_mae": 0.125, "units": len(UNITS), "seed": SEED}
    (outputs / "numbers.json").write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    (outputs / "metrics.csv").write_text("series,rel_mae\nsynthetic,0.125\n")
    print(json.dumps({"completed": completed, "resumed": bool(args.resume), "outputs": sorted(p.name for p in outputs.iterdir())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
