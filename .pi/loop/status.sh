#!/usr/bin/env bash
# status.sh — thin shim around the orchestrator CLI's `status` command.
#
# Prints the same summary as the TUI widget: state, current task/PR/experiment,
# budget, open gates, recent events, and what the next step would do.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
CLI="${ROOT}/.pi/extensions/orchestrator/cli.ts"

if [[ ! -f "${CLI}" ]]; then
	cat >&2 <<EOF
status.sh: not implemented until S4.

  expected : ${CLI}
  will run : node ${CLI} status --headless \$@
  prompt   : research_agent/prompts/04_orchestrator_and_state.md

Fallback (S0, read-only, no dependencies):
  node ${ROOT}/.pi/tests/check_env.ts --status
EOF
	exit 2
fi

exec node "${CLI}" status --headless "$@"
