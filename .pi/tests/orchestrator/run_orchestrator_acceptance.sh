#!/usr/bin/env bash
# S4 orchestrator acceptance transcript.
#
# Deterministic and offline: every cycle runs in a scratch clone under
# .pi/state/orchestrator-acceptance/ with the stub child (PI_LOOP_COMMAND), so
# this script never calls a model and never touches the real working tree.
#
# It prints:
#   1. the sandbox cycle transcript (exact commands + the merge commit message),
#   2. the pause/resume + idempotency summary (one completion per action),
#   3. the budget-cap demonstration (PAUSED + the resume command),
#   4. the .pi/loop/step.sh ↔ CLI parity check,
#   5. the real-repository invariant check (refs and update_pymc unchanged).
#
# Exit 0 only when every assertion holds.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# The node --test paths below are repo-root relative; be independent of the
# caller's cwd (`make -C .pi <target>` starts in .pi/).
cd "$ROOT"
NODE="${NODE:-node}"
CLI="$ROOT/.pi/extensions/orchestrator/cli.ts"
STUB="$ROOT/.pi/tests/fixtures/orchestrator_stub_child.mjs"
WORK="$ROOT/.pi/state/orchestrator-acceptance"
FAILURES=0

say() { printf '%s\n' "$*"; }
ok() { say "PASS  $1"; }
bad() { say "FAIL  $1"; FAILURES=$((FAILURES + 1)); }

REAL_SHA_BEFORE="$(git -C "$ROOT" rev-parse update_pymc)"
REAL_REFS_BEFORE="$(git -C "$ROOT" for-each-ref --format='%(refname) %(objectname)')"
REAL_DIRTY_BEFORE=0
# Tracked-file dirt is what the conductor refuses; untracked, gitignored archives
# (Vangja-project/) are reported but are not a dirty checkout.
[[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]] && REAL_DIRTY_BEFORE=1

make_sandbox() {
	local dir="$1"
	rm -rf "$dir"
	mkdir -p "$WORK"
	git clone --quiet --local "$ROOT" "$dir"
	git -C "$dir" checkout --quiet update_pymc
	mkdir -p "$dir/.pi/state/tasks"
	# Fast, deterministic pre-merge tests check (the scan logic is what S4 tests).
	printf '{"testsCommand":["true"]}\n' >"$dir/.pi/config/gates.json"
}

seed_task() {
	local dir="$1"
	NODE_PATH_SCRIPT="$dir/.seed_task.mjs" SANDBOX="$dir" "$NODE" --input-type=module - <<'JS'
import * as fs from "node:fs";
import { pathToFileURL } from "node:url";
import * as path from "node:path";
const root = process.env.SANDBOX;
const { appendRecord } = await import(pathToFileURL(path.join(root, ".pi/extensions/ledger/store.ts")).href);
const record = appendRecord(root, "task", {
  id: "T-001",
  title: "acceptance stub task",
  origin: "PLAN:acceptance",
  priority: "P2",
  objective: "Exercise the S4 orchestrator end to end in a scratch clone.",
  acceptance_criteria: ["STUB_TASK.md exists in the worktree"],
  deliverables: ["STUB_TASK.md"],
  compute: "none",
  status: "open",
  effort: "small",
}, { root, role: "conductor", actor: "acceptance" });
fs.writeFileSync(path.join(root, ".pi/state/tasks", `${record.id}.md`), `## ${record.id} acceptance\n`);
JS
}

step() {
	local dir="$1"; shift
	local env_mode="${1:-none}"; shift
	say "\$ PI_STUB_MODE=$env_mode node .pi/extensions/orchestrator/cli.ts step --sandbox <sandbox> --headless"
	PI_STUB_MODE="$env_mode" PI_LOOP_COMMAND="$STUB" "$NODE" "$CLI" step --sandbox "$dir" --headless --command "$STUB"
}

say "== S4 acceptance: orchestrator sandbox cycle =="
say "workdir: $WORK"
SANDBOX="$WORK/cycle"
make_sandbox "$SANDBOX"
seed_task "$SANDBOX"

say
say "-- 1. sandbox cycle transcript ------------------------------------------"
step "$SANDBOX" none
step "$SANDBOX" none
step "$SANDBOX" work
step "$SANDBOX" none
step "$SANDBOX" review
step "$SANDBOX" none
step "$SANDBOX" none            # opens the human merge gate
step "$SANDBOX" none            # waiting on the gate (no action)

GATE_ID=$(python3 - "$SANDBOX" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
records = [json.loads(l) for l in (root / ".pi/state/GATES.jsonl").read_text().splitlines() if l.strip()]
pending = [r for r in records if r["status"] == "pending"]
print(pending[-1]["id"] if pending else "")
PY
)
say "\$ node .pi/extensions/orchestrator/cli.ts gate approve $GATE_ID \"acceptance: reviewed the transcript\" --sandbox <sandbox>"
"$NODE" "$CLI" gate approve "$GATE_ID" "acceptance: reviewed the transcript" --sandbox "$SANDBOX" --headless
say "\$ PI_STUB_MODE=none node .pi/extensions/orchestrator/cli.ts step --sandbox <sandbox> --headless   # pre-merge gate + merge"
PI_STUB_MODE=none PI_LOOP_COMMAND="$STUB" "$NODE" "$CLI" step --sandbox "$SANDBOX" --headless --command "$STUB"
PI_STUB_MODE=none PI_LOOP_COMMAND="$STUB" "$NODE" "$CLI" step --sandbox "$SANDBOX" --headless --command "$STUB"
say
say "-- merge commit message --"
git -C "$SANDBOX" log --format='%h %s' -2 update_pymc
say "-- tag --"
git -C "$SANDBOX" tag --list 'run/*'

if git -C "$SANDBOX" log -1 --format=%s update_pymc | grep -qE '^merge PR-001 \(review RV-001\)'; then
	ok "merge commit message contains the review id"
else
	bad "merge commit message must contain the review id"
fi
if [[ "$(git -C "$SANDBOX" tag --list 'run/*/merge/1' | wc -l | tr -d ' ')" == "1" ]]; then
	ok "exactly one run/<run_id>/merge/1 tag"
else
	bad "expected exactly one run/*/merge/1 tag"
fi
if [[ "$(git -C "$SANDBOX" rev-list --count "$REAL_SHA_BEFORE..update_pymc")" == "2" ]]; then
	ok "the sandbox update_pymc advanced by exactly the task commit + the merge commit"
else
	bad "the sandbox update_pymc did not advance by exactly two commits"
fi
GATE_RUNS=$(ls "$SANDBOX/.pi/state/gate-runs" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$GATE_RUNS" == "1" ]]; then
	ok "exactly one pre-merge gate run recorded"
else
	bad "expected exactly one pre-merge gate run, found $GATE_RUNS"
fi
if python3 - "$SANDBOX" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
events = [json.loads(l) for l in (root / ".pi/state/EVENTS.jsonl").read_text().splitlines() if l.strip()]
merges = [e for e in events if e.get("action_key") == "merge:PR-001:RV-001" and e.get("action_phase") == "complete"]
assert len(merges) == 1, f"{len(merges)} merge completions"
assert merges[0].get("gate_ok") is True, "gate_ok must be true"
assert merges[0].get("merge_sha"), "merge_sha missing"
print("gate_ok + merge_sha recorded in exactly one completion event")
PY
then ok "the merge event records the gate result and the merge sha"; else bad "merge event evidence"; fi

say
say "-- 2. pause/resume + idempotency ---------------------------------------"
"$NODE" --test .pi/tests/orchestrator/test_pause_resume.test.ts >"$WORK/pause-resume.tests.txt" 2>&1
if [[ $? -eq 0 ]]; then
	ok "replay/kill/pause-resume tests green ($(grep -c '^✔' "$WORK/pause-resume.tests.txt") cases)"
else
	bad "pause/resume tests"
	tail -20 "$WORK/pause-resume.tests.txt"
fi
say "-- 3. budget caps --------------------------------------------------------"
"$NODE" --test .pi/tests/orchestrator/test_budget.test.ts >"$WORK/budget.tests.txt" 2>&1
if [[ $? -eq 0 ]]; then
	ok "six budget caps pause with the reason + resume command"
	grep -E "cap reached" "$WORK/budget.tests.txt" | head -1 || true
else
	bad "budget tests"
	tail -20 "$WORK/budget.tests.txt"
fi

say
say "-- 4. .pi/loop/step.sh ↔ CLI parity ------------------------------------"
"$NODE" --test .pi/tests/orchestrator/test_cli_status.test.ts >"$WORK/cli.tests.txt" 2>&1
if [[ $? -eq 0 ]]; then
	ok "step.sh and the CLI append identical events for the same starting state"
else
	bad "step.sh/CLI parity"
	tail -20 "$WORK/cli.tests.txt"
fi

say
say "-- 5. real repository invariant ----------------------------------------"
REAL_SHA_AFTER="$(git -C "$ROOT" rev-parse update_pymc)"
REAL_REFS_AFTER="$(git -C "$ROOT" for-each-ref --format='%(refname) %(objectname)')"
if [[ "$REAL_SHA_AFTER" == "$REAL_SHA_BEFORE" ]]; then ok "real update_pymc SHA unchanged ($REAL_SHA_AFTER)"; else bad "real update_pymc moved"; fi
if [[ "$REAL_REFS_AFTER" == "$REAL_REFS_BEFORE" ]]; then ok "no branch or tag created in the real repository"; else bad "real repository refs changed"; fi
if [[ "$REAL_DIRTY_BEFORE" == "1" ]]; then
	say "NOTE  the real checkout had tracked-file changes before this run (build in progress);"
	say "      this script asserts only that it changed nothing (see the two checks above)."
else
	if [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]]; then ok "real working tree clean (no tracked-file changes)"; else bad "real working tree is dirty"; fi
fi
say "git status --porcelain (informational; Vangja-project/ is the pre-existing untracked archive):"
git -C "$ROOT" status --porcelain | sed 's/^/      /'

say
if [[ "$FAILURES" -eq 0 ]]; then
	python3 -c "import shutil, sys; shutil.rmtree(sys.argv[1], ignore_errors=True)" "$WORK"
	say "acceptance: all S4 orchestrator checks passed"
	exit 0
fi
say "acceptance: $FAILURES check(s) failed"
exit 1
