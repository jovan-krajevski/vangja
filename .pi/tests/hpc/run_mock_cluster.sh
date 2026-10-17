#!/usr/bin/env bash
# S5 mock-cluster acceptance.
#
# Deterministic and offline: every scenario runs against the mock cluster in
# `.pi/tests/hpc/fixtures/mockbin/` (ssh/scp/rsync/sbatch/squeue/sacct/scancel
# shims backed by a scenario file), and the literal `/loop:step` check runs
# inside a working-tree copy under $TMPDIR with the child stub — no model call,
# no network, no real HPC host.
#
# Usage: bash .pi/tests/hpc/run_mock_cluster.sh
#
# Prints:
#   1. the per-test matrix from `node --test --test-reporter=tap`,
#   2. the required-scenario matrix (the S5 prompt's scenarios 1–9),
#   3. the raw single-slot refusal and hpc-disabled evidence,
#   4. the literal `/loop:step` transcript for a queued experiment with no host.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NODE="${NODE:-node}"
MOCKBIN="$ROOT/.pi/tests/hpc/fixtures/mockbin"
STUB="$ROOT/.pi/tests/fixtures/orchestrator_stub_child.mjs"
WORK="${TMPDIR:-/tmp}/vangja-hpc.$$"
FAILURES=0

say() { printf '%s\n' "$*"; }
ok() { say "PASS  $1"; }
bad() { say "FAIL  $1"; FAILURES=$((FAILURES + 1)); }

mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

# ---------------------------------------------------------------------------
# 1. node --test matrix
# ---------------------------------------------------------------------------
say "== S5 mock-cluster: unit + scenario matrix =="
TAP="$WORK/hpc.tap"
(
	cd "$ROOT" && PATH="$MOCKBIN:$PATH" "$NODE" --test --test-reporter=tap \
		.pi/tests/hpc/test_hpc_disabled.test.ts \
		.pi/tests/hpc/test_hpc_config.test.ts \
		.pi/tests/hpc/test_hpc_submit_gate.test.ts \
		.pi/tests/hpc/test_hpc_lifecycle.test.ts \
		.pi/tests/hpc/test_hpc_retry_rules.test.ts
) >"$TAP" 2>&1
TESTS_STATUS=$?
say ""
while IFS= read -r line; do
	say "$line"
done < <(grep -E '^(ok|not ok) ' "$TAP" || true)
say ""
if [[ "$TESTS_STATUS" -eq 0 ]]; then
	ok "node --test (all offline HPC tests + the recorded-fixture parity check)"
else
	bad "node --test (exit $TESTS_STATUS; see $TAP)"
fi

# ---------------------------------------------------------------------------
# 2. required-scenario matrix
# ---------------------------------------------------------------------------
say ""
say "== required scenarios (research_agent/prompts/05_hpc_and_collector.md) =="
require_test() {
	local label="$1" needle="$2"
	local line escaped
	# Escape ERE metacharacters: names contain literal '(' / ')' / '.'.
	escaped="$(printf '%s' "$needle" | sed -e 's/[][\.^$*+?(){}|\\]/\\&/g')"
	line="$(grep -E "^ok .* - .*${escaped}" "$TAP" | head -1 || true)"
	if [[ -n "$line" ]]; then
		ok "$label"
	else
		bad "$label (no passing test matching '${needle}')"
	fi
}
require_test "1  disabled: every tool {enabled:false}, no ssh" "enabled:false"
require_test "1b disabled: collector one hpc-disabled event" "hpc-disabled event"
require_test "1c disabled: loop skips WAITING_HPC (unit)" "skips WAITING_HPC"
require_test "2  happy path: submit → collect + hashed manifest" "happy path"
require_test "3  timeout exit 99: checkpoints first, requeue attempt 2" "timeout (exit 99)"
require_test "4  OOM: one doubled-memory requeue, then failed-final" "OOM requeues once"
require_test "5  no progress across two attempts" "no progress across two attempts"
require_test "6  missing declared output: collection-failed + task" "missing declared output"
require_test "7  missing checkpoint metadata named field" "queue-time refusals"
require_test "8  missing smoke-test evidence refused" "submit gate refuses missing smoke-test"
require_test "9  no key material in outputs/ledgers/evidence" "fake key material"
require_test "S  single-slot: gate checks + second submit refused" "submit gate checks"
require_test "R  retry table: collector/stuck/code-bug/collection-failed" "retry table: collector statuses"
require_test "P  sacct classifier parity with parse_sacct.py" "classifySacctJob matches parse_sacct.py"
require_test "C  config schema: host:\"\" disabled, amendment for >1 slot" "hpc.example.json parses"

# ---------------------------------------------------------------------------
# 3. acceptance snippets (disabled 8/8; second-submit refusal names the active id)
# ---------------------------------------------------------------------------
say ""
say "== acceptance snippets =="
SNIPPET_OUT="$WORK/snippets.txt"
if (cd "$ROOT" && PATH="$MOCKBIN:$PATH" "$NODE" .pi/tests/hpc/acceptance_snippets.mjs) >"$SNIPPET_OUT" 2>&1; then
	while IFS= read -r line; do say "      $line"; done <"$SNIPPET_OUT"
	ok "disabled 8/8 + second-submit refusal"
else
	say "$(cat "$SNIPPET_OUT")"
	bad "acceptance snippets"
fi

# ---------------------------------------------------------------------------
# 4. literal `pi --print "/loop:step"` with a queued experiment and no HPC host
# ---------------------------------------------------------------------------
say ""
say "== literal /loop:step with a queued experiment and no HPC host =="
SANDBOX="$WORK/loop-step"
mkdir -p "$SANDBOX"
# Copy the working tree (extension code included) without git/state: the point
# is the code under test, not the committed snapshot.
# This setup copy is the harness's own; MOCKBIN can already be on PATH when a
# caller exported it, in which case the mock rsync would shadow the real one
# and refuse the --exclude flags. Bypass the mock explicitly.
# A failed setup copy must be fatal: without the extension code the two
# `/loop:step` calls fall through to the model and can "pass" by accident.
if ! command -p rsync -a --exclude '.git' --exclude '.venv' --exclude 'Vangja-project' \
	--exclude '.pi/state' --exclude '__pycache__' "$ROOT/" "$SANDBOX/"; then
	bad "sandbox copy for the literal /loop:step check failed; section 4 cannot run"
	exit 1
fi
mkdir -p "$SANDBOX/.pi/state"
if ! SANDBOX="$SANDBOX" "$NODE" "$ROOT/.pi/tests/hpc/seed_disabled_loop.mjs"; then
	bad "seed_disabled_loop.mjs failed; section 4 cannot run"
	exit 1
fi
rm -f "$SANDBOX/.pi/config/hpc.json"   # host: "" — the HPC path is disabled

# `-a` is required: the sandbox is a fresh path with no saved trust decision,
# and project trust gates `.pi/extensions`. Without it the loop extension never
# loads, `/loop:step` is sent to the model as an ordinary prompt, and the
# command never runs (the events stay empty). One step must never call a model.
run_loop_step() {
	(cd "$SANDBOX" && PATH="$MOCKBIN:$PATH" \
		PI_LOOP_ROOT="$SANDBOX" PI_LOOP_REAL_ROOT="$ROOT" PI_LOOP_COMMAND="$STUB" \
		pi --print -a "/loop:step")
}

say "\$ cd <sandbox> && pi --print -a \"/loop:step\"   # 1st step: bookkeep"
run_loop_step >"$WORK/loop-step-1.txt" 2>&1
STEP1=$?
say "\$ cd <sandbox> && pi --print -a \"/loop:step\"   # 2nd step: local fallback"
run_loop_step >"$WORK/loop-step-2.txt" 2>&1
STEP2=$?
if [[ "$STEP1" -ne 0 || "$STEP2" -ne 0 ]]; then
	bad "pi /loop:step exited $STEP1/$STEP2"
	say "$(cat "$WORK/loop-step-1.txt" "$WORK/loop-step-2.txt")"
else
	ok "pi /loop:step ran twice offline"
fi
EVENTS="$SANDBOX/.pi/state/EVENTS.jsonl"
if grep -q '"hpc_disabled":true' "$EVENTS"; then
	ok "one hpc-disabled event in EVENTS.jsonl"
else
	bad "no hpc-disabled event"
fi
if grep -q '"to":"WAITING_HPC"' "$EVENTS"; then
	bad "the loop entered WAITING_HPC while HPC was disabled"
else
	ok "the loop never entered WAITING_HPC"
fi
if grep -q 'work-local:' "$EVENTS"; then
	ok "the loop proceeded to a local phd_candidate run"
else
	bad "the loop did not proceed to the local fallback"
fi
if grep -q '"state":"WAITING_HPC"' "$SANDBOX/.pi/state/LOOP_STATE.json"; then
	bad "LOOP_STATE.json is WAITING_HPC"
else
	ok "final loop state is not WAITING_HPC ($(python3 -c "import json;print(json.load(open('$SANDBOX/.pi/state/LOOP_STATE.json'))['state'])"))"
fi
say "-- raw hpc-disabled event --"
grep -m1 'hpc_disabled' "$EVENTS" | python3 -c 'import json,sys; r=json.loads(sys.stdin.readline()); print(json.dumps({k:r[k] for k in ("summary","experiment_id","hpc_disabled")}, indent=0))'
say "-- raw second-step event --"
grep -m1 'work-local:' "$EVENTS" | python3 -c 'import json,sys; r=json.loads(sys.stdin.readline()); print(json.dumps({k:r.get(k) for k in ("action_key","summary","agent","exit_state")}, indent=0))'

# ---------------------------------------------------------------------------
say ""
if [[ "$FAILURES" -eq 0 ]]; then
	say "all S5 mock-cluster checks passed"
	exit 0
fi
say "$FAILURES S5 mock-cluster check(s) FAILED"
exit 1
