---
name: hpc-slurm-and-checkpointing
description: Submit, monitor, requeue, or collect Vangja work on SLURM, or prove a runner resumes from a checkpoint. Use when writing a job script or experiment record, interpreting squeue/sacct output, handling TIMEOUT/OOM/NODE_FAIL, forwarding a git agent to a cluster, or when a job may be killed at the 24-hour wall limit. Encodes the single-active-job rule and the checkpoint-resume smoke test.
---

# HPC, SLURM, and checkpointing

The authority is `research_agent/plan/06_HPC_AND_COLLECTOR.md`; this skill is
the operational distillation. Read `references/slurm-fields.md` before
parsing status, and copy `templates/job.slurm` rather than writing a new one.

## Hard rules

1. **One job at a time.** While any experiment is `queued`, `running`,
   `collecting` or `requeue_requested`, no new submission is allowed. The
   queue gate refuses a second one; the submission tool does too.
2. **24-hour wall limit.** Every job is submitted with `--time=24:00:00` and
   must behave as if it will be killed at exactly 24h. `#SBATCH --signal=B:USR1@300`
   gives a 5-minute warning.
3. **Checkpoint or it does not run.** A submission is refused without a
   checkpoint dir, a progress marker, a `--resume` command, and a passing
   local smoke test (`scripts/checkpoint_smoke.py`).
4. **Exit code 99 = cleanly interrupted.** The USR1 trap (`--signal=B:USR1@300`)
   flushes a marker and exits 99. 0 = complete, 99 = requeue, anything else =
   failure. `hpc_submit` refuses a record whose `checkpoints.flushOnSignal` is
   not `true`.
5. **HPC disabled when `host` is empty** (`.pi/config/hpc.example.json`). Then
   `hpc_*` tools return `{enabled:false}`, the collector writes one
   `hpc-disabled` event, the conductor skips `WAITING_HPC`, and the work runs
   locally with the same checkpoint contract; never attempt SSH.
6. **Git agent forwarding only.** `ssh -A -o ForwardAgent=yes` with a private
   per-run agent that holds only the git key. If the key cannot be added,
   remote git is skipped (`git-forwarding-off`) and the submission is refused
   rather than guessed.

## Checkpoint contract (enforced)

The runner must:

1. Write `progress.json` after **each completed unit**, atomically:
   `{"version": 1, "completed": ["2023-01-01:main", ...]}`.
2. Accept `--resume <checkpoint_dir>` and skip units already completed —
   verify it, do not assume (`fixed_case_studies/stocks/runner.py:307` is the
   checkpoint hit; `runner.py:442-453` caches NUTS source fits as pkl).
3. Write declared `expectedOutputs` incrementally, so a 24h kill leaves
   usable partial results.
4. Write nothing outside `remoteScratchDir/<experiment_id>/`.

## Commands

The HPC tools in `.pi/extensions/hpc/` implement this contract; the shell
equivalents below are for inspection and for building a record by hand.

```bash
# 0. Tool path (preferred): experiment_queue → hpc_submit (gate-checked) →
#    hpc_status → hpc_fetch. hpc_submit renders the job script itself.

# 1. Render a job script by hand (placeholders come from the record:
#    {{JOB_NAME}} {{EXPERIMENT_ID}} {{ATTEMPT}} {{RUN_DIR}} {{SCRATCH}} {{CHECKPOINT_DIR}}
#    {{LOG_DIR}} {{CPUS}} {{MEM_GB}} {{PARTITION}} {{GRES}} {{CONTAINER_PREFIX}} {{RESUME_COMMAND}})
cp .pi/skills/hpc-slurm-and-checkpointing/templates/job.slurm scratch/E-0007.slurm

# 2. Prove resume works locally BEFORE submitting
python .pi/skills/hpc-slurm-and-checkpointing/scripts/checkpoint_smoke.py \
  --out scratch/smoke-E-0007 --limit 1 -- \
  python fixed_case_studies/stocks/runner.py --units a,b,c {limit} {resume}

# 3. Build the experiment record (refuses without checkpoint metadata)
python .pi/skills/hpc-slurm-and-checkpointing/scripts/make_job_record.py \
  --id E-0007 --task-id T-012 --branch work/T-012-seeds \
  --commit "$(git rev-parse HEAD)" --command "python -m ..." \
  --data-hash "$(sha256sum data/... | cut -d' ' -f1)" \
  --checkpoint-dir '$SCRATCH/vangja/E-0007/checkpoints' \
  --progress-file progress.json \
  --resume-command "python ... --resume \$CKPT" \
  --expected-output results/units.csv --seeds 42,20240816 \
  --out scratch/E-0007

# 4. Submit (collect the job id from --parsable) and record it
sbatch --parsable --job-name=E-0007 --time=24:00:00 ...

# 5. Poll / collect
squeue -u "$USER" -j <jobid> -o '%T %M %R'
sacct --json -j <jobid> | python .pi/skills/hpc-slurm-and-checkpointing/scripts/parse_sacct.py -
```

`checkpoint_smoke.py` understands two placeholders in the command:
`{limit}` (replaced by `--limit N` on the first run, dropped on resume) and
`{resume}` (dropped on the first run, replaced by `--resume <dir>` on
resume). It kills the first run once `--limit` units are recorded, resumes,
and asserts that every recorded unit was skipped and none was refitted.

## Requeue rules (§6 of `research_agent/plan/06_HPC_AND_COLLECTOR.md`)

| Situation | Action |
|---|---|
| exit 0, outputs exist | `collect` |
| exit 99 / TIMEOUT / DEADLINE, progress advanced | `requeue`, attempt+1 |
| PREEMPTED / NODE_FAIL, progress advanced | `requeue`, attempt+1 |
| OOM | retry **once** with doubled memory if the partition allows, else failed-final |
| no progress between two consecutive attempts | `failed-final` + `BLOCKED` |
| attempt > 6 (`maxAttemptsPerExperiment`) | `failed-final` + `BLOCKED` |
| non-zero exit from a code bug | `failed-final`; hand the stderr tail to a task |

Never requeue silently: each transition is appended with the raw
`squeue`/`sacct` evidence. Never "repair" a missing output, never interpret a
metric, never edit code from the collector role.

## Failure modes

- `squeue` empty but `sacct` shows COMPLETED → collect from `sacct`, not
  from a memory of the job.
- Job "vanishes": accounting lag can delay `sacct`; poll, do not requeue on
  the first empty answer.
- Node failure with checkpoints: requeue is safe *only* after fetching
  checkpoints and recording the progress delta.
- Container path changed → `container_sha` mismatch → refuse, do not run with
  a different image than the record names.
- Job exceeds 24h by design (a monolithic NUTS fit): split into stages that
  checkpoint; "resubmit from scratch" is not a retry policy.
