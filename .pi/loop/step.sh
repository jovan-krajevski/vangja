#!/usr/bin/env bash
# step.sh — thin shim around the orchestrator CLI (headless).
#
# Until S4 lands, this exits with a clear "not implemented" failure instead of
# silently doing nothing. Once `.pi/extensions/orchestrator/cli.ts` exists this
# shim is a pure pass-through; no logic lives here.
#
# Usage:
#   bash .pi/loop/step.sh            # one transition
#   bash .pi/loop/step.sh --forever  # keep stepping until PAUSED/BLOCKED/DONE
#   bash .pi/loop/step.sh --dry-run  # log the intended action only
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
CLI="${ROOT}/.pi/extensions/orchestrator/cli.ts"

if [[ ! -f "${CLI}" ]]; then
	cat >&2 <<EOF
step.sh: not implemented until S4.

  expected : ${CLI}
  will run : node ${CLI} step --headless \$@
  prompt   : research_agent/prompts/04_orchestrator_and_state.md

Until then the loop cannot run; the environment tests are available with:
  make -C .pi verify-env
EOF
	exit 2
fi

exec node "${CLI}" step --headless "$@"
