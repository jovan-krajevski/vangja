#!/usr/bin/env bash
# step.sh — thin shim around the orchestrator CLI (headless).
#
# Pure pass-through: no loop logic lives here (plan/05_EXTENSIONS_AND_TOOLS.md §2).
#
# Usage:
#   bash .pi/loop/step.sh                    # one action
#   bash .pi/loop/step.sh --forever          # keep stepping until PAUSED/BLOCKED/DONE
#   bash .pi/loop/step.sh --dry-run          # log the intended action only
#   bash .pi/loop/step.sh --sandbox <path>   # run against a scratch clone
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
CLI="${ROOT}/.pi/extensions/orchestrator/cli.ts"

if [[ ! -f "${CLI}" ]]; then
	cat >&2 <<EOF
step.sh: the orchestrator CLI is missing.

  expected : ${CLI}

The loop cannot run without it. Restore the file (see .pi/loop/README.md) or
run 'make -C .pi verify-env' to see what is broken.
EOF
	exit 2
fi

exec node "${CLI}" step --headless "$@"
