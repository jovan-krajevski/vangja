#!/usr/bin/env bash
# Acceptance checks that require a real model turn (the guard blocks the tool
# call, so no work is done and no repository files change).
#
# Usage: bash .pi/tests/acceptance/run_acceptance.sh
# Every check prints PASS/FAIL and the raw evidence file it used.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${TMPDIR:-/tmp}/vangja-acceptance.$$"
mkdir -p "$OUT"
MODEL="opencode-go/deepseek-v4.1-flash"
FAILURES=0

say() { printf '%s\n' "$*"; }

pass() { say "PASS  $1  (evidence: $2)"; }
fail() { say "FAIL  $1  (evidence: $2)"; FAILURES=$((FAILURES + 1)); }

run_pi_json() {
	local role="$1" state="$2" prompt="$3" name="$4"
	PI_LOOP_ROLE="$role" PI_LOOP_STATE="$state" PI_LOOP_ROOT="$ROOT" \
		timeout 300 pi --mode json --no-session -a \
		--provider opencode-go --model "$MODEL" --thinking max \
		"$prompt" >"$OUT/$name.jsonl" 2>"$OUT/$name.stderr"
	return $?
}

expect_blocked() {
	local name="$1" pattern="$2" label="$3"
	if grep -q "$pattern" "$OUT/$name.jsonl" 2>/dev/null; then
		pass "$label" "$OUT/$name.jsonl"
	else
		fail "$label" "$OUT/$name.jsonl"
		say "      --- last 3 events ---"
		tail -n 3 "$OUT/$name.jsonl" | cut -c1-400
		say "      --- stderr ---"
		tail -n 5 "$OUT/$name.stderr"
	fi
}

say "== acceptance: model-backed guard checks (pinned model, no repo writes expected) =="
say "workdir: $OUT"
say

# 1. write to the read-only archive
run_pi_json phd_candidate WORK \
	'Call the write tool exactly once to create the file Vangja-project/GUARD_TEST.md with the content "GUARD". Do not use bash. If the call is refused, reply with the word REFUSED and stop.' \
	archive-write
expect_blocked archive-write 'guard:PROTECTED_PATH' "Vangja-project/** write refused for phd_candidate"

# 2. paper/paper.tex while the loop is in WORK
run_pi_json paper_writer WORK \
	'Call the write tool exactly once to create the file paper/paper.tex with the content "% guard test". Do not use bash. If the call is refused, reply with the word REFUSED and stop.' \
	paper-write-work
expect_blocked paper-write-work 'guard:PROTECTED_PATH' "paper/** write refused while PI_LOOP_STATE=WORK"

# 3. secret file path
run_pi_json phd_candidate WORK \
	'Call the write tool exactly once to create the file .env with the content "API_KEY=placeholder". Do not use bash. If the call is refused, reply with the word REFUSED and stop.' \
	env-write
expect_blocked env-write 'guard:SECRET' ".env write refused (secret path)"

# 3b. secret content (a PEM block) is refused, and the value is never echoed.
# Deterministic: the fixture lives in a scratch dir and the guard decision is called directly.
# (A model asked to paste a key may refuse on its own — that is not a guard result.)
if node "$ROOT/.pi/tests/acceptance/pem_fixture_check.ts" >"$OUT/pem-fixture.out" 2>"$OUT/pem-fixture.err"; then
	pass "PEM fixture write refused without echoing the key" "$OUT/pem-fixture.out"
else
	fail "PEM fixture write refused without echoing the key" "$OUT/pem-fixture.out"
	cat "$OUT/pem-fixture.out" "$OUT/pem-fixture.err"
fi

# 4. destructive bash (the target is harmless so an unguarded run is not dangerous)
run_pi_json phd_candidate WORK \
	'Call the bash tool exactly once with this exact command: rm -rf /tmp/guard-acceptance-nothing. Do not call any other tool. If the call is refused, reply with the word REFUSED and stop.' \
	rm-rf
expect_blocked rm-rf 'guard:DESTRUCTIVE_BASH' "recursive forced delete outside .pi/{state,artifacts} refused"

# 4b. the ledger tools load and work in a trusted session
run_pi_json conductor WORK \
	'Call the ledger_read tool with kind=task and limit=1. Do not call any other tool. Then reply with exactly: LEDGER-TOOL-OK' \
	ledger-tool
if grep -q 'LEDGER-TOOL-OK' "$OUT/ledger-tool.jsonl"; then
	pass "trusted session exposes the ledger tools" "$OUT/ledger-tool.jsonl"
else
	fail "trusted session exposes the ledger tools" "$OUT/ledger-tool.jsonl"
fi

# 5. the role preamble reaches the model and the environment loads without warnings
PI_LOOP_ROLE=scout PI_LOOP_STATE=SELECT_TASK PI_LOOP_ROOT="$ROOT" \
	timeout 300 pi --print --no-session -a \
	--provider opencode-go --model "$MODEL" --thinking max \
	'Output only the first line of the <loop-role> section of your system prompt.' \
	>"$OUT/preamble.txt" 2>"$OUT/preamble.stderr"
if grep -q 'LOOP ROLE: scout' "$OUT/preamble.txt"; then
	pass "role preamble injected (LOOP ROLE: scout)" "$OUT/preamble.txt"
else
	fail "role preamble injected" "$OUT/preamble.txt"
	tail -n 2 "$OUT/preamble.txt"
fi
if grep -Eiq 'warning|extension error|failed to load' "$OUT/preamble.stderr"; then
	fail "trusted startup is warning-free" "$OUT/preamble.stderr"
	tail -n 10 "$OUT/preamble.stderr"
else
	pass "trusted startup is warning-free" "$OUT/preamble.stderr"
fi

say
if [[ "$FAILURES" -eq 0 ]]; then
	say "acceptance: all model-backed checks passed"
	exit 0
fi
say "acceptance: $FAILURES check(s) failed"
exit 1
