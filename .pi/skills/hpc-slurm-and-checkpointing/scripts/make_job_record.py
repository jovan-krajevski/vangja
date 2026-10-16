#!/usr/bin/env python3
"""Build an HPC experiment record (plan/06 §4) with checkpoint metadata.

Refuses to produce a record that cannot be requeued: the checkpoint dir,
progress file, resume command and a smoke-test result are mandatory, and the
command must name the checkpoint flag. The idempotency key is
``sha256(branch + commit + command + seeds + data_hash)`` so resubmitting the
same work cannot create a second experiment.

Usage:
  make_job_record.py --id E-0007 --task-id T-012 --branch work/T-012 \
    --commit "$(git rev-parse HEAD)" --command "python -m runner" \
    --data-hash ABC --uv-lock-sha DEF \
    --checkpoint-dir '$SCRATCH/vangja/E-0007/checkpoints' \
    --progress-file progress.json \
    --resume-command "python -m runner --resume \\$CKPT" \
    --expected-output results/units.csv --seeds 42,20240816 \
    --smoke-ok --out scratch/E-0007

Exit codes: 0 written, 1 validation failure, 2 usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA = "vangja-experiment-record-1.0"
ID_RE = re.compile(r"^E-\d{4}$")


def idempotency_key(branch: str, commit: str, command: str, seeds, data_hash: str) -> str:
    components = [
        branch.strip(),
        commit.strip(),
        " ".join(command.split()),
        ",".join(str(s) for s in sorted(seeds, key=str)),
        data_hash.strip(),
    ]
    return hashlib.sha256("\x00".join(components).encode()).hexdigest()


def build_record(args) -> tuple[dict | None, list[str]]:
    errors: list[str] = []
    if not ID_RE.match(args.id):
        errors.append(f"--id must look like E-0007, got {args.id!r}")
    for field in ("task_id", "branch", "commit", "command", "data_hash",
                  "uv_lock_sha", "checkpoint_dir", "progress_file",
                  "resume_command"):
        if not getattr(args, field):
            errors.append(f"--{field.replace('_', '-')} is required")
    if not args.seeds:
        errors.append("--seeds is required (comma-separated ints)")
    if not args.expected_output:
        errors.append("--expected-output is required at least once")
    if not args.smoke_ok:
        errors.append("--smoke-ok is required: run scripts/checkpoint_smoke.py first")
    if args.resume_command and "--resume" not in args.resume_command:
        errors.append("--resume-command must contain '--resume'")
    if errors:
        return None, errors

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    resources = json.loads(args.resources_json) if args.resources_json else {
        "kind": "hpc",
        "partition": args.partition,
        "cpus": args.cpus,
        "memGb": args.mem_gb,
        "gpus": args.gpus,
        "wallClockHours": 24,
    }
    record = {
        "schema": SCHEMA,
        "id": args.id,
        "task_id": args.task_id,
        "pr_id": args.pr_id,
        "branch": args.branch,
        "commit_sha": args.commit,
        "data_hash": args.data_hash,
        "uv_lock_sha": args.uv_lock_sha,
        "container_sha": args.container_sha,
        "command": args.command,
        "resources": resources,
        "checkpoints": {
            "dir": args.checkpoint_dir,
            "progressFile": args.progress_file,
            "resumeCommand": args.resume_command,
            "flushOnSignal": True,
        },
        "expectedOutputs": list(args.expected_output),
        "seeds": seeds,
        "idempotencyKey": idempotency_key(
            args.branch, args.commit, args.command, seeds, args.data_hash
        ),
        "attempt": args.attempt,
        "status": "draft",
    }
    return record, []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an HPC experiment record with checkpoint metadata.",
        epilog=(
            "Example:\n"
            "  make_job_record.py --id E-0007 --task-id T-012 --branch work/T-012 \\\n"
            "    --commit $(git rev-parse HEAD) --command 'python -m runner' \\\n"
            "    --data-hash ABC --uv-lock-sha DEF \\\n"
            "    --checkpoint-dir '$SCRATCH/vangja/E-0007/checkpoints' \\\n"
            "    --progress-file progress.json \\\n"
            "    --resume-command 'python -m runner --resume $CKPT' \\\n"
            "    --expected-output results/units.csv --seeds 42 --smoke-ok\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--pr-id", default="")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--data-hash", required=True)
    parser.add_argument("--uv-lock-sha", required=True)
    parser.add_argument("--container-sha", default="")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--progress-file", default="progress.json")
    parser.add_argument("--resume-command", required=True)
    parser.add_argument("--expected-output", action="append", default=[])
    parser.add_argument("--seeds", default="")
    parser.add_argument("--smoke-ok", action="store_true")
    parser.add_argument("--resources-json", default="")
    parser.add_argument("--partition", default="")
    parser.add_argument("--cpus", type=int, default=8)
    parser.add_argument("--mem-gb", type=int, default=32)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--out", type=Path, help="directory to write record.json into")
    parser.add_argument("--json", action="store_true", help="print the record JSON")
    args = parser.parse_args()

    record, errors = build_record(args)
    if errors:
        for error in errors:
            print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    assert record is not None
    serialized = json.dumps(record, indent=2) + "\n"
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "record.json").write_text(serialized)
        print(f"wrote {args.out / 'record.json'}")
        print(f"idempotencyKey {record['idempotencyKey']}")
    if args.json or not args.out:
        sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    sys.exit(main())
