#!/usr/bin/env python3
"""Prove a runner resumes from a checkpoint (the local smoke test of plan/06 §5.4).

Procedure:

1. Start the command once with ``{limit}`` replaced by ``--limit N`` (and
   ``{resume}`` dropped), polling ``progress.json`` until N units are recorded,
   then kill it (SIGTERM, then SIGKILL after a grace period).
2. Start it again with ``{limit}`` dropped and ``{resume}`` replaced by
   ``--resume <checkpoint_dir>``.
3. Assert that every unit recorded before the kill was skipped on resume and
   was not refitted, and that the second run exited 0.
4. Write ``smoke.json`` evidence under ``--out``.

The runner contract (see SKILL.md): ``progress.json`` is
``{"version": 1, "completed": [...]}`` updated atomically after each unit;
``work.log`` gets one ``fit <unit>`` line per fitted unit; ``skips.log`` gets
``skip <unit>`` for units skipped because they were already complete.

Usage:
  checkpoint_smoke.py --out scratch/smoke-E7 --limit 1 -- \
    python fixed_case_studies/stocks/runner.py --units a,b,c {limit} {resume}

Exit codes: 0 proven, 1 not proven, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "vangja-checkpoint-smoke-1.0"


def read_progress(progress_file: Path) -> list[str]:
    if not progress_file.exists():
        return []
    try:
        payload = json.loads(progress_file.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    return list(payload.get("completed", []))


def read_log(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text().splitlines()


def build_command(command: list[str], *, limit: int | None, resume: str | None):
    out: list[str] = []
    for token in command:
        if token == "{limit}":
            if limit is not None:
                out += ["--limit", str(limit)]
        elif token == "{resume}":
            if resume is not None:
                out += ["--resume", resume]
        else:
            out.append(token)
    return out


def run_and_kill(command: list[str], progress_file: Path, limit: int,
                 timeout: float, poll: float) -> int:
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            if len(read_progress(progress_file)) >= limit:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
                return proc.returncode if proc.returncode is not None else -1
            time.sleep(poll)
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
        return proc.returncode if proc.returncode is not None else -1
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prove that a runner resumes from its checkpoint.",
        epilog=(
            "Example:\n"
            "  checkpoint_smoke.py --out scratch/smoke --limit 1 -- \\\n"
            "    python runner.py --units a,b,c,d {limit} {resume}\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", type=Path, required=True,
                        help="checkpoint/evidence directory (must be fresh)")
    parser.add_argument("--checkpoint-dir", type=Path,
                        help="checkpoint dir passed to --resume (default: --out)")
    parser.add_argument("--limit", type=int, default=1,
                        help="units to complete in the first (killed) run")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="seconds allowed for the first run before it is killed")
    parser.add_argument("--poll", type=float, default=0.02)
    parser.add_argument("--force", action="store_true",
                        help="clear progress/logs in an existing --out directory")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="runner command after '--' (use {limit} and {resume})")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a runner command after '--' is required")
    if "{resume}" not in command:
        parser.error("the command must contain the {resume} placeholder")

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    progress_file = out / "progress.json"
    work_log = out / "work.log"
    skip_log = out / "skips.log"
    checkpoint_dir = args.checkpoint_dir or out

    if read_progress(progress_file) or work_log.exists():
        if not args.force:
            print(
                f"REFUSED: {out} is not fresh (progress.json/work.log exist); "
                "use --force to clear it",
                file=sys.stderr,
            )
            return 2
        for path in (progress_file, work_log, skip_log):
            if path.exists():
                path.unlink()

    first_cmd = build_command(command, limit=args.limit, resume=None)
    first_code = run_and_kill(first_cmd, progress_file, args.limit,
                              args.timeout, args.poll)
    first_completed = read_progress(progress_file)
    work_lines_before = read_log(work_log)
    skip_lines_before = read_log(skip_log)

    resume_cmd = build_command(command, limit=None, resume=str(checkpoint_dir))
    second = subprocess.run(resume_cmd, capture_output=True, text=True,
                            timeout=args.timeout * 5)
    second_completed = read_progress(progress_file)
    work_lines_after = read_log(work_log)[len(work_lines_before):]
    skip_lines_after = read_log(skip_log)[len(skip_lines_before):]

    skipped_on_resume = {
        line.split(" ", 1)[1]
        for line in skip_lines_after
        if line.startswith("skip ") and " " in line
    }
    refit_after_resume = {
        line.split(" ", 1)[1]
        for line in work_lines_after
        if line.startswith("fit ") and " " in line
    }

    missing_skips = sorted(set(first_completed) - skipped_on_resume)
    refitted = sorted(set(first_completed) & refit_after_resume)
    advanced = set(second_completed) >= set(first_completed)
    ok = (
        bool(first_completed)
        and second.returncode == 0
        and not missing_skips
        and not refitted
        and advanced
    )
    reason = ""
    if not first_completed:
        reason = ("the first run recorded no completed unit; increase --timeout "
                  "or check the runner's progress contract")
    elif second.returncode != 0:
        reason = f"resume run exited {second.returncode}: {second.stderr[-400:]}"
    elif missing_skips:
        reason = f"recorded units were not skipped on resume: {missing_skips}"

    evidence = {
        "schema": SCHEMA,
        "ok": ok,
        "reason": reason,
        "out": str(out),
        "checkpoint_dir": str(checkpoint_dir),
        "limit": args.limit,
        "first_command": shlex.join(first_cmd),
        "resume_command": shlex.join(resume_cmd),
        "first_returncode": first_code,
        "resume_returncode": second.returncode,
        "first_completed": sorted(first_completed),
        "completed_after_resume": sorted(second_completed),
        "skipped_on_resume": sorted(skipped_on_resume),
        "missing_skips": missing_skips,
        "refitted_after_resume": refitted,
    }
    (out / "smoke.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"SMOKE {'PASS' if ok else 'FAIL'}: first={sorted(first_completed)} "
          f"skipped={sorted(skipped_on_resume)} refitted={refitted}")
    if reason:
        print(f"REASON: {reason}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
