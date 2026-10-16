#!/bin/bash
# Job template for one Vangja experiment. Copy to scratch, fill every
# {{PLACEHOLDER}}, and submit with `sbatch --parsable`. Do not invent a new
# script: the USR1/exit-99 contract below is what makes requeue safe.
#
# Contract: exit 0 = complete, exit 99 = cleanly interrupted (checkpoint
# flushed, safe to requeue), anything else = failure. See the
# hpc-slurm-and-checkpointing skill and plan/06_HPC_AND_COLLECTOR.md.
{{SBATCH_DIRECTIVES}}
#SBATCH --job-name={{EXPERIMENT_ID}}
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output={{LOG_DIR}}/%j.out
#SBATCH --error={{LOG_DIR}}/%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_DIR="{{REMOTE_REPO_DIR}}"
CKPT="{{CHECKPOINT_DIR}}"
mkdir -p "$CKPT" "{{LOG_DIR}}"

# --- checkpoint flush on the wall-limit warning --------------------------
# The runner updates progress.json after every completed unit; on USR1 we
# only need to leave the process cleanly with code 99.
trap 'echo "USR1 received: flushing checkpoint"; python3 -c "import sys; sys.exit(99)"' USR1

echo "========================================"
echo "experiment:  {{EXPERIMENT_ID}}"
echo "commit:      {{COMMIT_SHA}}"
echo "started:     $(date --iso-8601=seconds)"
echo "checkpoints: ${CKPT}"
echo "========================================"

cd "$REPO_DIR"
git checkout --detach "{{COMMIT_SHA}}" >/dev/null
test -f uv.lock

# Frozen environment. No network beyond this sync.
uv sync --frozen --no-dev

# --- run -----------------------------------------------------------------
# `--resume "$CKPT"` makes the second and later attempts skip completed
# units. THE COMMAND MUST ACCEPT --resume; verify with the checkpoint smoke
# test before submitting.
set +e
{{CONTAINER_PREFIX}} uv run python "{{ENTRYPOINT}}" \
    {{ENTRYPOINT_ARGS}} \
    --resume "$CKPT"
EXIT_CODE=$?
set -e

echo "========================================"
echo "finished:  $(date --iso-8601=seconds)"
echo "exit code: ${EXIT_CODE}"
echo "========================================"

# 0 = complete; 99 = interrupted-clean; anything else = failure.
exit "${EXIT_CODE}"
