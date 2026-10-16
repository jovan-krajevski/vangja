#!/usr/bin/env bash
# Unified offline test runner for the .pi environment.
#
# Usage: bash .pi/tests/run.sh [all|node|skills|orchestrator]
#   node          the TypeScript/Node suite (settings, guards, ledger, status, child pi)
#   skills        validate_skills.sh + the Python skill-script tests
#   orchestrator  the S4 orchestrator suite (state machine, sandbox cycle, budget,
#                 pause/resume, gates) plus the acceptance transcript script
#   all           node + skills (default)
#
# No network, no model calls. Exit non-zero if any suite fails.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NODE="${NODE:-node}"
SUITE="${1:-all}"

TEST_FILES=(
	.pi/tests/test_settings.test.ts
	.pi/tests/test_schemas.test.ts
	.pi/tests/test_guards_policy.test.ts
	.pi/tests/test_guards_wiring.test.ts
	.pi/tests/test_ledger.test.ts
	.pi/tests/test_ledger_concurrency.test.ts
	.pi/tests/test_status.test.ts
	.pi/tests/test_child_pi.test.ts
	.pi/tests/agents/test_agents_manifest.test.ts
	.pi/tests/agents/test_tools_registry.test.ts
	.pi/tests/agents/test_dispatch_unit.test.ts
	.pi/tests/agents/test_prompt_templates.test.ts
)

ORCHESTRATOR_TESTS=(
	.pi/tests/orchestrator/test_state_machine.test.ts
	.pi/tests/orchestrator/test_budget.test.ts
	.pi/tests/orchestrator/test_gates.test.ts
	.pi/tests/orchestrator/test_cli_status.test.ts
	.pi/tests/orchestrator/test_pause_resume.test.ts
	.pi/tests/orchestrator/test_sandbox_cycle.test.ts
)

run_node() {
	echo "== node --test (offline) =="
	(cd "$ROOT" && "$NODE" --test "${TEST_FILES[@]}")
}

run_orchestrator() {
	echo "== node --test orchestrator (offline, sandbox only) =="
	(cd "$ROOT" && "$NODE" --test "${ORCHESTRATOR_TESTS[@]}")
	local status=$?
	echo
	echo "== orchestrator acceptance transcript =="
	bash "$ROOT/.pi/tests/orchestrator/run_orchestrator_acceptance.sh"
	local acceptance=$?
	if [[ "$status" -ne 0 || "$acceptance" -ne 0 ]]; then
		return 1
	fi
	return 0
}

run_skills() {
	echo "== validate skills =="
	bash "$ROOT/.pi/skills/validate_skills.sh"
	local validate_status=$?
	echo
	echo "== python -- skill script tests =="
	(cd "$ROOT" && PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
		-s .pi/tests/skills -t .pi/tests/skills -v)
	local unit_status=$?
	if [[ "$validate_status" -ne 0 || "$unit_status" -ne 0 ]]; then
		return 1
	fi
	return 0
}

status=0
case "$SUITE" in
	all)
		run_node || status=1
		echo
		run_skills || status=1
		;;
	node) run_node || status=1 ;;
	skills) run_skills || status=1 ;;
	orchestrator) run_orchestrator || status=1 ;;
	*)
		echo "usage: bash .pi/tests/run.sh [all|node|skills|orchestrator]" >&2
		exit 2
		;;
esac
exit "$status"
