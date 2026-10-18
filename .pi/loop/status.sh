#!/usr/bin/env bash
# status.sh — thin shim around the orchestrator CLI's `status` command.
#
# Pure pass-through (no logic lives here). Prints the same summary as the TUI
# widget: state, current task/PR/experiment + attempt, budget vs caps, open
# human gates, the last 5 events, a stale child lock if any, and the nextAction.
#
# Usage: bash .pi/loop/status.sh [--json] [--sandbox <path>]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
CLI="${ROOT}/.pi/extensions/orchestrator/cli.ts"

if [[ ! -f "${CLI}" ]]; then
	cat >&2 <<EOF
status.sh: the orchestrator CLI is missing.

  expected : ${CLI}
  fallback : node ${ROOT}/.pi/tests/check_env.ts --status
EOF
	exit 2
fi

exec node "${CLI}" status --headless "$@"
