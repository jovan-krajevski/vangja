#!/usr/bin/env bash
# collect.sh — thin shim around the orchestrator CLI's collector command (S5).
#
# The collector shepherds HPC jobs: it polls the active experiment, records
# evidence, fetches checkpoints/outputs, and never submits a second job. Until
# S5 the CLI reports the HPC gate state and exits 2 (never a silent success).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"
CLI="${ROOT}/.pi/extensions/orchestrator/cli.ts"

if [[ ! -f "${CLI}" ]]; then
	cat >&2 <<EOF
collect.sh: the orchestrator CLI is missing.

  expected : ${CLI}
  note     : the collector itself arrives in S5
             (research_agent/prompts/05_hpc_and_collector.md)
EOF
	exit 2
fi

exec node "${CLI}" collect --headless "$@"
