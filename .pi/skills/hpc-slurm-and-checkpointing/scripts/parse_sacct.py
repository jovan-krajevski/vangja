#!/usr/bin/env python3
"""Map ``sacct --json`` output to Vangja states and requeue actions.

State/action tables are in ``references/slurm-fields.md`` and
``research_agent/plan/06_HPC_AND_COLLECTOR.md`` §6.

Usage:
  sacct --json -j 4711 | parse_sacct.py -
  parse_sacct.py fixture.json --progress-advanced --attempt 2 --json
  parse_sacct.py fixture.json --out scratch/E-0007/status.json

Exit codes: 0 classified, 1 nothing to classify, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = "vangja-sacct-classification-1.0"

COMPLETED = {"COMPLETED"}
QUEUED = {"PENDING", "CONFIGURING", "REQUEUED", "RESIZING"}
RUNNING = {"RUNNING", "COMPLETING", "SUSPENDED", "STAGE_OUT"}
TIMED_OUT = {"TIMEOUT", "TIME_LIMIT", "DEADLINE"}
OOM = {"OUT_OF_MEMORY"}
PREEMPTED = {"NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "NODE_DOWN"}
CANCELLED = {"CANCELLED"}
FAILED = {"FAILED", "ERROR", "SPECIAL_EXIT", "REVOKED", "LAUNCH_FAILED"}

ACTION_WAIT = "wait"
ACTION_COLLECT = "collect"
ACTION_REQUEUE = "requeue"
ACTION_REQUEUE_NO_PROGRESS = "requeue-no-progress"
ACTION_OOM_RETRY = "retry-once-doubled-memory"
ACTION_FAILED_FINAL = "failed-final"
ACTION_STOP = "stop"


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def extract_state(job: dict) -> str:
    state = job.get("state")
    if isinstance(state, dict):
        current = state.get("current")
        values = _as_list(current)
        if not values:
            return "UNKNOWN"
        return str(values[0]).upper()
    if isinstance(state, list):
        return str(state[0]).upper() if state else "UNKNOWN"
    return str(state or "UNKNOWN").upper()


def extract_exit_code(job: dict) -> int | None:
    for key in ("derived_exit_code", "exit_code"):
        value = job.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            parts = value.split(":")
            try:
                return int(parts[-1])
            except ValueError:
                continue
        if isinstance(value, dict):
            for field in ("return_code", "number", "code"):
                if field in value:
                    try:
                        return int(value[field])
                    except (TypeError, ValueError):
                        return None
    return None


def extract_signal(job: dict) -> int | None:
    value = job.get("derived_exit_code") or job.get("exit_code")
    if isinstance(value, dict) and "signal" in value:
        try:
            return int(value["signal"])
        except (TypeError, ValueError):
            return None
    return None


def elapsed_seconds(job: dict) -> int | None:
    time = job.get("time")
    if not isinstance(time, dict):
        return None
    elapsed = time.get("elapsed")
    if isinstance(elapsed, int):
        return elapsed
    if isinstance(elapsed, dict) and "seconds" in elapsed:
        return int(elapsed["seconds"])
    if isinstance(elapsed, str):
        match = re.match(r"^(?:(\d+)-)?(\d+):(\d+):(\d+)$", elapsed)
        if match:
            days, hours, minutes, seconds = (int(g or 0) for g in match.groups())
            return days * 86400 + hours * 3600 + minutes * 60 + seconds
    return None


def classify(
    job: dict,
    *,
    progress_advanced: bool = False,
    attempt: int = 1,
    max_attempts: int = 6,
    memory_can_double: bool = False,
) -> dict:
    raw_state = extract_state(job)
    exit_code = extract_exit_code(job)
    signal = extract_signal(job)
    state = raw_state
    reason = ""

    if raw_state in QUEUED:
        state, action = "queued", ACTION_WAIT
    elif raw_state in RUNNING:
        state, action = "running", ACTION_WAIT
    elif raw_state in COMPLETED:
        if exit_code == 99:
            state, action = "interrupted_clean", ACTION_REQUEUE
        elif exit_code in (0, None):
            state, action = "completed", ACTION_COLLECT
        else:
            state, action = "failed", ACTION_FAILED_FINAL
    elif raw_state in TIMED_OUT:
        state = "timed_out"
        action = ACTION_REQUEUE if progress_advanced else ACTION_REQUEUE_NO_PROGRESS
    elif raw_state in OOM:
        state = "oom"
        action = ACTION_OOM_RETRY if memory_can_double else ACTION_FAILED_FINAL
        if attempt > 1 and progress_advanced:
            reason = "OOM after a previous attempt; doubling memory is allowed once"
        elif not memory_can_double:
            reason = "partition headroom unknown: --memory-can-double not given"
    elif raw_state in PREEMPTED:
        state = "preempted"
        action = ACTION_REQUEUE if progress_advanced else ACTION_REQUEUE_NO_PROGRESS
    elif raw_state in CANCELLED:
        state, action = "cancelled", ACTION_STOP
    elif raw_state in FAILED:
        if exit_code == 99:
            state, action = "interrupted_clean", ACTION_REQUEUE
        elif exit_code == 137 or signal == 9:
            state = "oom"
            action = ACTION_OOM_RETRY if memory_can_double else ACTION_FAILED_FINAL
        else:
            state, action = "failed", ACTION_FAILED_FINAL
    else:
        state, action = "unknown", ACTION_FAILED_FINAL
        reason = f"unrecognised state {raw_state!r}"

    if action in (ACTION_REQUEUE, ACTION_REQUEUE_NO_PROGRESS, ACTION_OOM_RETRY):
        if attempt >= max_attempts:
            action = ACTION_FAILED_FINAL
            reason = f"attempt {attempt} >= maxAttempts {max_attempts}"
        elif action == ACTION_REQUEUE_NO_PROGRESS and attempt > 1:
            reason = reason or (
                "no checkpoint progress; a second such attempt is failed-final"
            )

    return {
        "job_id": str(job.get("job_id", job.get("job_id_with_sched", ""))),
        "name": job.get("name", ""),
        "raw_state": raw_state,
        "state": state,
        "exit_code": exit_code,
        "signal": signal,
        "elapsed_seconds": elapsed_seconds(job),
        "node": _node_list(job),
        "action": action,
        "reason": reason,
        "progress_hint": "advanced" if progress_advanced else "unknown-or-none",
    }


def _node_list(job: dict) -> str:
    nodes = job.get("nodes")
    if isinstance(nodes, dict):
        nodes = nodes.get("list")
    if isinstance(nodes, list):
        return ",".join(str(n) for n in nodes)
    return str(nodes or "")


def load_payload(path: str) -> dict:
    text = sys.stdin.read() if path == "-" else Path(path).read_text()
    return json.loads(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify sacct --json jobs into Vangja states and actions.",
        epilog=(
            "Examples:\n"
            "  sacct --json -j 4711 | parse_sacct.py -\n"
            "  parse_sacct.py fixture.json --progress-advanced --attempt 2 --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="sacct --json file, or '-' for stdin")
    parser.add_argument("--progress-advanced", action="store_true",
                        help="the checkpoint progress file grew since the last poll")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--memory-can-double", action="store_true",
                        help="partition has headroom to double --mem after an OOM")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    args = parser.parse_args()

    payload = load_payload(args.input)
    jobs = payload.get("jobs", [])
    classified = [
        classify(
            job,
            progress_advanced=args.progress_advanced,
            attempt=args.attempt,
            max_attempts=args.max_attempts,
            memory_can_double=args.memory_can_double,
        )
        for job in jobs
    ]
    report = {"schema": SCHEMA, "n_jobs": len(classified), "jobs": classified}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for job in classified:
            print(
                f"{job['job_id']} {job['name']}: {job['state']} "
                f"(exit={job['exit_code']}, signal={job['signal']}) -> {job['action']}"
                + (f" [{job['reason']}]" if job["reason"] else "")
            )
        print(f"JOBS: {len(classified)}")
    return 0 if classified else 1


if __name__ == "__main__":
    sys.exit(main())
