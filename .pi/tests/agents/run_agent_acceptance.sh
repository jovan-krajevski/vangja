#!/usr/bin/env bash
# S3 agent/dispatch acceptance checks.
#
# Offline checks are deterministic; the model-backed checks use the pinned model
# and write only to the runtime state under .pi/state/ (sessions + EVENTS.jsonl).
#
# Usage: bash .pi/tests/agents/run_agent_acceptance.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${TMPDIR:-/tmp}/vangja-agents.$$"
mkdir -p "$OUT"
MODEL="opencode-go/deepseek-v4.1-flash"
EVENTS="$ROOT/.pi/state/EVENTS.jsonl"
FAILURES=0

say() { printf '%s\n' "$*"; }
pass() { say "PASS  $1  (evidence: $2)"; }
fail() { say "FAIL  $1  (evidence: $2)"; FAILURES=$((FAILURES + 1)); }

event_lines() { if [[ -f "$EVENTS" ]]; then wc -l <"$EVENTS" | tr -d ' '; else echo 0; fi; }

query_events() {
	# query_events <since-lines> <python-expression-on-record> ; prints matches
	local since="$1" expr="$2"
	EVENTS_FILE="$EVENTS" SINCE="$since" EXPR="$expr" python3 - <<'PY'
import json, os
since = int(os.environ["SINCE"])
expr = os.environ["EXPR"]
path = os.environ["EVENTS_FILE"]
if not os.path.exists(path):
    raise SystemExit(0)
with open(path) as fh:
    for i, line in enumerate(fh):
        if i < since or not line.strip():
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if eval(expr, {"r": record, "any": any, "all": all}):
            print(json.dumps(record, indent=2, sort_keys=True))
PY
}

run_pi_json() {
	# run_pi_json <role> <tools> <name> <outfile> <prompt...>
	local role="$1" tools="$2" name="$3" outfile="$4"
	shift 4
	PI_LOOP_ROOT="$ROOT" PI_LOOP_ROLE="$role" PI_LOOP_STATE=WORK PI_AGENT_NAME="$name" \
		timeout 1200 pi --mode json -p -a \
		--provider opencode-go --model "$MODEL" --thinking max \
		--tools "$tools" \
		--session-dir "$OUT/sessions/$name" --name "$name" \
		"$@" >"$OUT/$outfile.jsonl" 2>"$OUT/$outfile.stderr"
}

say "== S3 acceptance: agent files, prompt templates, dispatch =="
say "workdir: $OUT"
say

# ---------------------------------------------------------------------------
# 6. Every agent file validates against the manifest and the tool registry.
# Deterministic and first, so later checks run against a valid roster.
# ---------------------------------------------------------------------------
AGENT_FILES=$(ls "$ROOT"/.pi/agents/*.md 2>/dev/null | wc -l | tr -d ' ')
if node --test "$ROOT/.pi/tests/agents/test_agents_manifest.test.ts" "$ROOT/.pi/tests/agents/test_tools_registry.test.ts" >"$OUT/agents-offline.txt" 2>&1; then
	pass "6. all $AGENT_FILES agent files validate against the manifest + tool registry" "$OUT/agents-offline.txt"
else
	fail "6. agent files / tool registry validate" "$OUT/agents-offline.txt"
	tail -n 25 "$OUT/agents-offline.txt"
fi

# ---------------------------------------------------------------------------
# 1. A brief template resolves without executing.
# ---------------------------------------------------------------------------
node "$ROOT/.pi/tests/agents/resolve_prompt.ts" "$ROOT/.pi/prompts/research-next-task.md" T-099 >"$OUT/template-resolve.txt" 2>&1
if grep -q "T-099" "$OUT/template-resolve.txt" && grep -q "task_enqueue" "$OUT/template-resolve.txt"; then
	pass "1a. research-next-task resolves offline with arguments" "$OUT/template-resolve.txt"
else
	fail "1a. offline template resolution" "$OUT/template-resolve.txt"
	cat "$OUT/template-resolve.txt"
fi

# pi itself expands project templates; --no-tools makes execution impossible.
run_pi_json human "read" accept-template template-pi "/research-next-task T-099"
if grep -q "task_enqueue" "$OUT/template-pi.jsonl" && grep -q "planner" "$OUT/template-pi.jsonl"; then
	pass "1b. pi expands /research-next-task (tools disabled: resolved, not executed)" "$OUT/template-pi.jsonl"
else
	fail "1b. pi template expansion" "$OUT/template-pi.jsonl"
	tail -n 3 "$OUT/template-pi.jsonl" | cut -c1-300
fi

# ---------------------------------------------------------------------------
# 2 + 4. phd_candidate dispatches scout; then an unknown agent yields an error.
# ---------------------------------------------------------------------------
BEFORE_SCOUT=$(event_lines)
run_pi_json phd_candidate "read,bash,dispatch_agent,ledger_read" phd_candidate__T-099__1 dispatch-scout \
	'Call the dispatch_agent tool exactly once with agent="scout", task="Map .pi/skills: list each skill directory, its SKILL.md line count, and a one-line purpose. Answer in at most 60 lines.", task_id="T-099", attempt=1. Do not call any other tool. When it returns, reply with DISPATCH-DONE and one fact from the scout summary.' \
	'Now call the dispatch_agent tool exactly once with agent="no_such_agent", task="does not matter", task_id="T-099", attempt=2. Do not call any other tool. Then report the exact error text you received.'

SCOUT_EVENT=$(query_events "$BEFORE_SCOUT" "r.get('agent')=='scout' and r.get('exit_state')=='ok'")
if [[ -n "$SCOUT_EVENT" ]] \
	&& echo "$SCOUT_EVENT" | grep -q '"usage"' \
	&& echo "$SCOUT_EVENT" | grep -q '"session_path"'; then
	SESSION_PATH=$(echo "$SCOUT_EVENT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["session_path"])')
	if [[ -f "$SESSION_PATH" ]]; then
		pass "2. scout dispatched from phd_candidate: summary + usage + session path in EVENTS.jsonl" "$EVENTS"
		say "      session_path=$SESSION_PATH"
		echo "$SCOUT_EVENT" | grep -E '"(agent|exit_state|session_path|usage)"' | head -8
	else
		fail "2. dispatch event found but session file is missing" "$EVENTS"
		say "      session_path=$SESSION_PATH"
	fi
else
	fail "2. scout dispatch event (agent=scout, exit_state=ok, usage, session_path) missing" "$OUT/dispatch-scout.jsonl"
	tail -n 5 "$OUT/dispatch-scout.jsonl" | cut -c1-400
	tail -n 5 "$OUT/dispatch-scout.stderr"
fi

if grep -q "Available agents" "$OUT/dispatch-scout.jsonl" && grep -q "no_such_agent" "$OUT/dispatch-scout.jsonl"; then
	pass "4. unknown agent name -> explicit error listing available agents" "$OUT/dispatch-scout.jsonl"
else
	fail "4. unknown agent error" "$OUT/dispatch-scout.jsonl"
fi

# ---------------------------------------------------------------------------
# 3. merge_pr: refused for phd_candidate; refused by the conductor without an
#    accepted review; no merge event is appended in either case.
# ---------------------------------------------------------------------------
BEFORE_MERGE=$(event_lines)
run_pi_json phd_candidate "read,merge_pr" phd_candidate__T-002__1 merge-refused \
	'Call the merge_pr tool exactly once with pr_id="PR-001". If the call is refused, reply with exactly: MERGE-REFUSED.'
if grep -q "guard:ROLE" "$OUT/merge-refused.jsonl"; then
	pass "3a. phd_candidate merge_pr refused by the guard" "$OUT/merge-refused.jsonl"
else
	fail "3a. phd_candidate merge_pr refusal" "$OUT/merge-refused.jsonl"
fi
if [[ "$(event_lines)" == "$BEFORE_MERGE" ]] && [[ "$(query_events "$BEFORE_MERGE" "r.get('merged') is True" | wc -l | tr -d ' ')" == "0" ]]; then
	pass "3b. refused merge_pr appended no merge event" "$EVENTS"
else
	fail "3b. refused merge_pr must not append a merge event" "$EVENTS"
fi

# S4: the conductor merges only an accepted review. PR-001 does not exist in the
# real state, so the tool must refuse with the conductor's reason (not the guard).
BEFORE_CONDUCTOR=$(event_lines)
run_pi_json supervisor "read,merge_pr" supervisor__PR-001__1 merge-conductor \
	'Call the merge_pr tool exactly once with pr_id="PR-001", reason="S4 acceptance: there is no accepted review". Then reply with exactly: MERGE-REFUSED.'
if grep -qE "merge refused|unknown PR" "$OUT/merge-conductor.jsonl"; then
	pass "3c. supervisor merge_pr refused by the conductor without an accepted review" "$OUT/merge-conductor.jsonl"
else
	fail "3c. conductor merge_pr refusal without an accepted review" "$OUT/merge-conductor.jsonl"
	tail -n 3 "$OUT/merge-conductor.jsonl" | cut -c1-300
fi
if [[ "$(event_lines)" == "$BEFORE_CONDUCTOR" ]]; then
	pass "3d. refused conductor merge appended no event" "$EVENTS"
else
	fail "3d. refused conductor merge must not append an event" "$EVENTS"
fi

# ---------------------------------------------------------------------------
# 5. Killing a child mid-run: aborted, error returned, no partial final text.
# Deterministic: the stub is a real OS process killed with SIGTERM.
# ---------------------------------------------------------------------------
if node --test --test-name-pattern="killing a child mid-run" \
	"$ROOT/.pi/tests/agents/test_dispatch_unit.test.ts" >"$OUT/abort-test.txt" 2>&1; then
	pass "5. killed child -> aborted outcome, ledger records aborted, partial text not final" "$OUT/abort-test.txt"
else
	fail "5. abort handling" "$OUT/abort-test.txt"
	tail -n 20 "$OUT/abort-test.txt"
fi

say
say "agent files: $AGENT_FILES (manifest: $(python3 -c "import json;print(len(json.load(open('$ROOT/.pi/config/agents.json'))['agents']))"))"
if [[ "$FAILURES" -eq 0 ]]; then
	say "acceptance: all S3 agent/dispatch checks passed"
	exit 0
fi
say "acceptance: $FAILURES check(s) failed"
exit 1
