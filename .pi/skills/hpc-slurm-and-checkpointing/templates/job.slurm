#!/bin/bash
# Vangja SLURM job template. `hpc_submit` renders every placeholder token and
# uploads the result; do not hand-edit a rendered copy in place.
#
# Exit-code contract: 0 = complete, 99 = cleanly interrupted (checkpoints
# flushed, safe to requeue), anything else = failure. The USR1 trap below is
# what makes the requeue path safe at the 24-hour wall limit.
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --cpus-per-task={{CPUS}}
#SBATCH --mem={{MEM_GB}}G
#SBATCH --partition={{PARTITION}}
#SBATCH --gres={{GRES}}
#SBATCH --output={{LOG_DIR}}/slurm-%j.out
#SBATCH --error={{LOG_DIR}}/slurm-%j.err
#SBATCH --comment={{KIND}}/{{EXPERIMENT_ID}}/attempt-{{ATTEMPT}}

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

RUN_DIR={{RUN_DIR}}
CKPT={{CHECKPOINT_DIR}}
mkdir -p "$CKPT" "{{LOG_DIR}}"

# `--signal=B:USR1@300` gives us a 5-minute warning before the 24h wall limit.
# The runner writes progress.json after every completed unit; here we only
# need to leave a marker and exit 99 so the collector knows the state is
# consistent and the job may be requeued.
on_usr1() {
    echo "USR1 received: wall limit in 5 minutes; flushing checkpoints and exiting 99 (safe to requeue)"
    touch "$CKPT/INTERRUPTED"
    exit 99
}
trap on_usr1 USR1

echo "=== {{EXPERIMENT_ID}} attempt {{ATTEMPT}} ==="
echo "run dir:     $RUN_DIR"
echo "checkpoints: $CKPT"
echo "started:     $(date --iso-8601=seconds 2>/dev/null || date)"
echo "========================================"

cd "$RUN_DIR"

# Frozen environment. No network beyond this sync; never resolve dependencies
# at run time (the lock hash is checked by the submission gate).
uv sync --frozen --no-dev

# The resume command is the experiment record's `checkpoints.resumeCommand`.
# `--resume "$CKPT"` makes this and every later attempt skip completed units.
set +e
{{CONTAINER_PREFIX}}{{RESUME_COMMAND}}
EXIT_CODE=$?
set -e

echo "========================================"
echo "finished:  $(date --iso-8601=seconds 2>/dev/null || date)"
echo "exit code: ${EXIT_CODE}"
echo "========================================"

# 0 = complete · 99 = interrupted-clean · anything else = failure.
exit "${EXIT_CODE}"
