#!/usr/bin/env bash
# collect.sh — thin shim around the orchestrator CLI's collector command (S5).
#
# The collector shepherds HPC jobs: it polls the active experiment, records
# evidence, fetches checkpoints/outputs, and never submits a second job.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
CLI="${ROOT}/.pi/extensions/orchestrator/cli.ts"

if [[ ! -f "${CLI}" ]]; then
	cat >&2 <<EOF
collect.sh: not implemented until S5.

  expected : ${CLI}
  will run : node ${CLI} collect --headless \$@
  prompts  : research_agent/prompts/04_orchestrator_and_state.md (orchestrator)
             research_agent/prompts/05_hpc_and_collector.md (collector)
EOF
	exit 2
fi

exec node "${CLI}" collect --headless "$@"
