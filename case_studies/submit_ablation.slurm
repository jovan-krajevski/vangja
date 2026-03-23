#!/bin/bash
#SBATCH --job-name=vangja-ablation
#SBATCH --time=24:00:00
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err
#SBATCH --nodes=1
#SBATCH --nodelist=gpgpu01
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=openlab-queue

# =========================================================================
# Vangja Ablation Study — SLURM Submission Script
#
# Usage:
#   sbatch case_studies/submit_ablation.slurm <case_study> <pipeline>
#
# Arguments:
#   <case_study>  Directory name under case_studies/ (e.g. "smart_home")
#   <pipeline>    One of: vangja, baselines (default: vangja)
#
# Examples:
#   sbatch case_studies/submit_ablation.slurm smart_home vangja
#   sbatch case_studies/submit_ablation.slurm smart_home baselines
# =========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CASE_STUDY="${1:?Usage: sbatch submit_ablation.slurm <case_study> <pipeline>}"
PIPELINE="${2:-vangja}"

REPO_ROOT="$(pwd)"
echo "${REPO_ROOT}"
CASE_DIR="${REPO_ROOT}/case_studies/${CASE_STUDY}"
SIF="${REPO_ROOT}/case_studies/vangja.sif"

ls "${REPO_ROOT}"

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

case "${PIPELINE}" in
    vangja)    SCRIPT="run_vangja.py" ;;
    baselines) SCRIPT="run_baselines.py" ;;
    *)
        echo "ERROR: Unknown pipeline '${PIPELINE}'. Use: vangja or baselines." >&2
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
echo "Pipeline:      ${PIPELINE}"
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
