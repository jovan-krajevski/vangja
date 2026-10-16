#!/usr/bin/env bash
# Unified offline test runner for the .pi environment.
#
# Usage: bash .pi/tests/run.sh [all|node|skills]
#   node    the TypeScript/Node suite (settings, guards, ledger, status)
#   skills  validate_skills.sh + the Python skill-script tests
#   all     both (default)
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
)

run_node() {
	echo "== node --test (offline) =="
	(cd "$ROOT" && "$NODE" --test "${TEST_FILES[@]}")
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
	*)
		echo "usage: bash .pi/tests/run.sh [all|node|skills]" >&2
		exit 2
		;;
esac
exit "$status"
