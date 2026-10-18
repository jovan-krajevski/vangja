#!/usr/bin/env python3
"""Gate a Vangja run on its convergence diagnostics.

Thin strict wrapper around ``diagnose_idata.py``: exits non-zero unless the fit
passes R-hat / ESS / BFMI / divergence / tree-depth / target_accept thresholds.
``--strict`` also fails on warnings. ``--waiver-reason`` records an explicit
human waiver in the output and turns a fail into an exit 0 — use it only with a
recorded supervisor waiver.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_idata  # noqa: E402  (same-skill import)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="check_convergence.py",
        description="Fail a run that does not meet the project's convergence thresholds.",
        epilog="Example: check_convergence.py --idata trace.nc --label stocks-main",
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--idata", type=Path)
    source.add_argument("--json", type=Path)
    p.add_argument("--label", default=None)
    p.add_argument("--thresholds", type=Path, default=None)
    p.add_argument("--strict", action="store_true", help="treat warnings as failures")
    p.add_argument("--waiver-reason", default=None,
                   help="record an explicit waiver; fail becomes exit 0 (logged)")
    p.add_argument("--out", type=Path, default=None, help="write diagnostics JSON here")
    args = p.parse_args(argv)

    try:
        thresholds = diagnose_idata.load_thresholds(args.thresholds)
        if args.json:
            if not args.json.exists():
                print(f"diagnostics JSON not found: {args.json}", file=sys.stderr)
                return 2
            payload = json.loads(args.json.read_text())
        else:
            if not args.idata.exists():
                print(f"InferenceData not found: {args.idata}", file=sys.stderr)
                return 2
            payload = diagnose_idata.from_idata(args.idata)
        if args.label:
            payload["label"] = args.label
        report = diagnose_idata.evaluate(payload, thresholds)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"convergence check failed to run: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2

    if args.waiver_reason:
        report["waiver"] = {
            "reason": args.waiver_reason,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "note": "requires a supervisor waiver entry in the ledger",
        }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")

    failed = [c for c in report["checks"] if c["status"] == "fail"]
    warned = [c for c in report["checks"] if c["status"] == "warn"]
    for check in report["checks"]:
        print(f"{check['status'].upper():5} {check['check']}: {check['detail']}")

    if failed and args.waiver_reason:
        print(f"WAIVED ({len(failed)} failure(s)): {args.waiver_reason}")
        return 0
    if failed:
        print(f"CONVERGENCE FAIL: {len(failed)} check(s) failed — "
              "re-fit with a higher target_accept/reparameterisation, or record a waiver")
        return 1
    if warned and args.strict:
        print(f"CONVERGENCE FAIL (strict): {len(warned)} warning(s)")
        return 1
    print(f"CONVERGENCE PASS ({len(warned)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
