#!/bin/bash
#SBATCH --job-name=vangja-ablation
#SBATCH --time=24:00:00
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# Adjust --partition to match your HPC cluster's queue names:
# #SBATCH --partition=compute

# =========================================================================
# Vangja Ablation Study — SLURM Submission Script
#
# Usage:
#   sbatch case_studies/submit_ablation.slurm <case_study> [mode]
#
# Arguments:
#   <case_study>  Directory name under case_studies/ (e.g. "smart_home")
#   [mode]        One of: fast, full, classical (default: full)
#
# Examples:
#   sbatch case_studies/submit_ablation.slurm smart_home fast
#   sbatch case_studies/submit_ablation.slurm smart_home full
#   sbatch case_studies/submit_ablation.slurm smart_home classical
# =========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CASE_STUDY="${1:?Usage: sbatch submit_ablation.slurm <case_study> [fast|full|classical]}"
MODE="${2:-full}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE_DIR="${REPO_ROOT}/case_studies/${CASE_STUDY}"
SIF="${REPO_ROOT}/case_studies/vangja.sif"

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
if [[ ! -d "${CASE_DIR}" ]]; then
    echo "ERROR: Case study directory not found: ${CASE_DIR}" >&2
    exit 1
fi

if [[ ! -f "${SIF}" ]]; then
    echo "ERROR: Singularity image not found: ${SIF}" >&2
    echo "Build it first: singularity build case_studies/vangja.sif case_studies/singularity.def" >&2
    exit 1
fi

case "${MODE}" in
    fast)      SCRIPT="ablation_fast.py" ;;
    full)      SCRIPT="ablation_full.py" ;;
    classical) SCRIPT="classical_baselines.py" ;;
    *)
        echo "ERROR: Unknown mode '${MODE}'. Use: fast, full, or classical." >&2
        exit 1
        ;;
esac

SCRIPT_PATH="${CASE_DIR}/${SCRIPT}"
if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "ERROR: Script not found: ${SCRIPT_PATH}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Job information
# ---------------------------------------------------------------------------
echo "========================================"
echo "Vangja Ablation Study"
echo "========================================"
echo "Job ID:        ${SLURM_JOB_ID}"
echo "Node:          $(hostname)"
echo "CPUs:          ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Case study:    ${CASE_STUDY}"
echo "Mode:          ${MODE}"
echo "Script:        ${SCRIPT}"
echo "Started at:    $(date --iso-8601=seconds)"
echo "========================================"

# Create logs directory
mkdir -p "${REPO_ROOT}/logs"

# ---------------------------------------------------------------------------
# Run the ablation inside the Singularity container
# ---------------------------------------------------------------------------
# Bind the entire repository so the script can read source code, data,
# and write results back to the host filesystem.
singularity exec \
    --bind "${REPO_ROOT}:${REPO_ROOT}" \
    --pwd "${REPO_ROOT}" \
    "${SIF}" \
    python "${SCRIPT_PATH}"

EXIT_CODE=$?

echo "========================================"
echo "Finished at:   $(date --iso-8601=seconds)"
echo "Exit code:     ${EXIT_CODE}"
echo "========================================"

exit ${EXIT_CODE}
