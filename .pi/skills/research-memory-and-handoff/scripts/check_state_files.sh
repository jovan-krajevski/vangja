#!/usr/bin/env bash
# Validate the loop state files used across fresh-context sessions.
#
# Checks:
#   * .pi/state/ exists
#   * LOOP_STATE.json (if present) parses, has schema pi-loop-state/1, the full
#     key set, and a legal state name from .pi/lib/state.ts
#   * CONTEXT.md exists and is <= 200 lines (generated view, not hand-edited)
#   * every *.jsonl ledger line parses and carries schema/kind/ts/actor
#
# Usage:
#   check_state_files.sh [--root DIR] [--strict] [--help]
#
# Exit codes: 0 no failures, 1 failures, 2 usage error.
# Missing LOOP_STATE.json warns (run /loop:start to boot a run); --strict
# turns that warning into a failure.
set -uo pipefail

ROOT=""
STRICT=0

usage() {
    cat <<'EOF'
usage: check_state_files.sh [--root DIR] [--strict]

  --root DIR   repository root to validate (default: the repo containing this script)
  --strict     treat a missing LOOP_STATE.json as a failure
  --help       this message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root)
            [[ $# -ge 2 ]] || { echo "error: --root needs a value" >&2; exit 2; }
            ROOT="$2"
            shift 2
            ;;
        --strict)
            STRICT=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$ROOT" ]]; then
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
fi

exec python3 - "$ROOT" "$STRICT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
strict = sys.argv[2] == "1"
state = root / ".pi" / "state"

LOOP_STATES = [
    "UNINITIALIZED", "BOOT", "SELECT_TASK", "WORK", "WAITING_HPC", "REVIEW",
    "MERGE", "POST_MERGE", "READINESS", "PAPER", "PAPER_REVIEW", "AUDIT",
    "DECIDE", "PAUSED", "BLOCKED", "DONE",
]
REQUIRED_STATE_KEYS = {
    "schema", "run_id", "state", "paused_from", "task_id", "pr_id",
    "experiment_id", "attempt", "next_gate", "reason", "updated_at",
    "updated_by",
}
LEDGER_KEYS = {"schema", "kind", "ts", "actor"}

failures = 0
warnings = 0


def report(level: str, name: str, detail: str) -> None:
    global failures, warnings
    print(f"{level:<4}  {name}: {detail}")
    if level == "FAIL":
        failures += 1
    elif level == "WARN":
        warnings += 1


if not state.is_dir():
    report("FAIL", "state-dir", f"{state} does not exist")
else:
    report("PASS", "state-dir", str(state))

context = state / "CONTEXT.md"
if not context.is_file():
    report("WARN", "CONTEXT.md", "missing (the conductor generates it)")
else:
    lines = len(context.read_text().splitlines())
    if lines > 200:
        report("FAIL", "CONTEXT.md", f"{lines} lines (limit 200)")
    else:
        report("PASS", "CONTEXT.md", f"{lines} lines")

loop_state = state / "LOOP_STATE.json"
if not loop_state.is_file():
    report("FAIL" if strict else "WARN", "LOOP_STATE.json",
           "missing (strict mode)" if strict else "missing (created by the orchestrator)")
else:
    try:
        payload = json.loads(loop_state.read_text())
    except json.JSONDecodeError as exc:
        report("FAIL", "LOOP_STATE.json", f"invalid JSON: {exc}")
        payload = None
    if isinstance(payload, dict):
        if payload.get("schema") != "pi-loop-state/1":
            report("FAIL", "LOOP_STATE.json",
                   f"schema is {payload.get('schema')!r}, expected 'pi-loop-state/1'")
        missing = sorted(REQUIRED_STATE_KEYS - set(payload))
        if missing:
            report("FAIL", "LOOP_STATE.json", f"missing keys: {missing}")
        if payload.get("state") not in LOOP_STATES:
            report("FAIL", "LOOP_STATE.json",
                   f"unknown state {payload.get('state')!r}")
        if payload.get("state") == "PAUSED" and not payload.get("paused_from"):
            report("FAIL", "LOOP_STATE.json", "PAUSED without paused_from")
        if not missing and payload.get("state") in LOOP_STATES:
            report("PASS", "LOOP_STATE.json", f"state={payload.get('state')}")

for ledger in sorted(state.glob("*.jsonl")):
    bad = 0
    count = 0
    for number, line in enumerate(ledger.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        count += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            report("FAIL", ledger.name, f"line {number}: invalid JSON ({exc})")
            bad += 1
            continue
        if not isinstance(record, dict):
            report("FAIL", ledger.name, f"line {number}: not an object")
            bad += 1
            continue
        missing = sorted(LEDGER_KEYS - set(record))
        if missing:
            report("FAIL", ledger.name, f"line {number}: missing {missing}")
            bad += 1
    if bad == 0:
        report("PASS", ledger.name, f"{count} record(s)")

print(f"check_state_files: {failures} failure(s), {warnings} warning(s)")
sys.exit(1 if failures else 0)
PY
