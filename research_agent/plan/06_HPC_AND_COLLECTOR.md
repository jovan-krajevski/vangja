# 06 — HPC, SLURM, and the collector protocol

> Config and state live at the repository root under `.pi/`: `config/hpc.json`,
> `state/EXPERIMENTS.jsonl`, `artifacts/<experiment_id>/`.

Hard rules (from the project owner):

1. **One job at a time.** While an experiment is `queued`, `running`,
   `requeueing`, or `collecting`, no further submission is allowed. The loop waits
   for it to finish (and be collected) before submitting another.
2. **Wall limit is 24 hours.** Every job is submitted with `--time=24:00:00` and
   must assume it can be killed at 24h.
3. **Checkpoint or it does not run.** A job without a resume command, a checkpoint
   directory, and a progress marker is refused at the queue gate.

If `.pi/config/hpc.json` has an empty `host`, the entire HPC path is disabled:
`hpc_*` tools return `{enabled:false}`, the collector exits immediately with an
`hpc-disabled` event, and `phd_candidate` must run locally or mark the task
`BLOCKED` (it must never attempt SSH).

## 1. Configuration

`.pi/config/hpc.json` (gitignored; `hpc.example.json` committed):

```json
{
  "host": "", "user": "", "port": 22,
  "remoteRepoDir": "", "remoteScratchDir": "",
  "hpcKey": "~/.ssh/id_ed25519",
  "gitKey": "~/.ssh/vangja_github",
  "forwardGitAgent": true,
  "container": "",
  "partition": "", "cpus": 8, "memGb": 32, "gpus": 0,
  "wallClockHours": 24,
  "pollSeconds": 60,
  "maxPolls": 40,
  "maxAttemptsPerExperiment": 6,
  "maxParallelJobs": 1
}
```

Validation on load: required fields present when enabled; keys exist and are
readable; `maxParallelJobs > 1` is rejected unless an amendment file says
otherwise; the config is never printed with key material.

## 2. SSH and agent forwarding

- The `hpc` extension starts a **private `ssh-agent` per loop run**
  (`ssh-agent -s`), adds **only** `gitKey`, and passes `SSH_AUTH_SOCK` to its
  children. The user's own agent is never touched.
- Connect with `ssh -A -o IdentitiesOnly=yes -o ForwardAgent=yes
  -o StrictHostKeyChecking=accept-new -p <port> -i <hpcKey> user@host`.
- If `gitKey` is missing or cannot be added → `forwardGitAgent` is treated as
  unavailable, remote `git` is skipped, and the tool reports `git-forwarding-off`.
  If the remote checkout then cannot be created and rsync is also unavailable →
  the submission is refused (never guess).
- All remote commands are executed with an explicit timeout and are logged
  verbatim (minus credentials) to the experiment record.

## 3. Submission protocol

`hpc_submit(experiment_id)` refuses unless all hold:

1. HPC enabled; no experiment active (single-slot rule).
2. The record validates against the schema in §4.
3. The record carries **checkpoint metadata** and a **checkpoint smoke test**
   entry proving resume works locally (`--limit 1` run, or an equivalent dry run).
4. The remote work tree is a clean detached checkout of the record's `commit_sha`
   (`git fetch origin && git checkout --detach <sha> && git status --porcelain`
   empty).
5. `uv.lock` SHA matches the record; container (if any) exists and its SHA-256
   matches the record.
6. Estimated wall clock < 24h; otherwise the record must be explicitly staged
   (`stages: N`) and the command must accept a stage index.

Submission uses `sbatch --parsable --job-name=<exp_id> --time=24:00:00
--cpus-per-task=… --mem=…G [--partition=…] [--gres=gpu:…]` with the rendered
template (`.pi/skills/hpc-slurm-and-checkpointing/templates/job.slurm`). The job
script:

```bash
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300        # 5 min warning before the wall limit
set -euo pipefail
export PYTHONUNBUFFERED=1
CKPT="${SCRATCH_DIR}/checkpoints"
mkdir -p "$CKPT"
trap 'echo "USR1 received"; python -c "import sys;sys.exit(99)"' USR1   # flush + exit 99
# runner writes checkpoints itself after every unit of work
uv sync --frozen --no-dev
uv run python "$ENTRYPOINT" --resume "$CKPT" … || exit $?
```

Exit-code convention: `0` = complete; `99` = cleanly interrupted (checkpoint
flushed, safe to requeue); anything else = failure (collector records
`DerivedExitCode`, stderr tail, and whether checkpoints advanced).

## 4. Experiment record (one JSONL line, append-only)

```json
{
  "id": "E-0007", "task_id": "T-012", "pr_id": "PR-003",
  "branch": "work/T-012-seeds", "commit_sha": "…",
  "data_hash": "…", "uv_lock_sha": "…", "container_sha": "…",
  "command": "uv run python fixed_case_studies/stocks/04_run_confirmation.py …",
  "resources": {"partition": "", "cpus": 8, "memGb": 32, "gpus": 0, "wallClockHours": 24},
  "checkpoints": {
    "dir": "$SCRATCH/vangja/E-0007/checkpoints",
    "progressFile": "progress.json",
    "resumeCommand": "<same command plus --resume $CKPT>",
    "flushOnSignal": true
  },
  "expectedOutputs": ["results/…/units_*.csv", "results/…/manifest_*.json"],
  "seeds": [42, 20240816],
  "idempotencyKey": "sha256(branch+commit+command+seeds+data_hash)",
  "attempt": 1, "status": "draft"
}
```

Lifecycle:

```
draft → queued → running → collecting → collected
                    │            └→ collection-failed
                    ├→ timed_out → requeue_requested → queued(attempt+1)
                    ├→ interrupted_clean (exit 99) → requeue_requested
                    ├→ failed (OOM/NODE_FAIL/other) → retry once if checkpoints advanced, else failed-final
                    └→ cancelled (only on explicit instruction)
```

`queued|running|collecting|requeue_requested` are the "active" set that blocks new
submissions. `collected`, `failed-final`, `cancelled` are terminal.

## 5. Checkpoint contract (enforced)

Every HPC task must:

1. Write a machine-readable progress marker after each completed unit of work
   (e.g. `{"completed": ["2023-01-01:main"], "version": 1}`), and update it
   atomically.
2. Accept `--resume <checkpoint_dir>` and skip completed units (this is already
   how `fixed_case_studies` runners behave — verify it, do not assume).
3. Declare `expectedOutputs` that are written incrementally, so a 24h kill still
   leaves usable partial results.
4. Pass a **local smoke test** proving resume works: run 1–2 units, kill the
   process, re-run with `--resume`, and show that completed units are skipped and
   the output set is consistent. This evidence is recorded in the job record.
5. Never write results outside `remoteScratchDir/<experiment_id>/`.

If a runner cannot satisfy this (e.g. a monolithic NUTS fit), it must be split
into stages that can, or the task escalates to the human. "Resubmit from scratch"
is not an acceptable retry policy.

## 6. Collector protocol

Invoked when the active set is non-empty. Each poll:

1. `hpc_status` → `squeue` for the job, `sacct --json` for finished ones. Raw
   output (trimmed) is appended as evidence.
2. State transitions, each with evidence:
   - `queued → running`: `squeue` shows `R`, or `sacct` `Start` set.
   - `running → collecting`: `sacct` State complete and outputs exist, or exit 99
     with checkpoints present.
   - `running → timed_out`: State ∈ {TIMEOUT, TIME_LIMIT, DEADLINE}; or exit 99;
     or `PREEMPTED`/`NODE_FAIL` after progress.
   - `running → failed`: OOM, non-zero exit with no checkpoint progress.
3. On timeout/interruption/failure-with-progress: fetch **checkpoints first**,
   record the progress delta, then write `requeue_requested` with the evidence and
   the suggested `attempt+1`. Never requeue silently: the event is explicit, and
   the conductor performs the resubmission.
4. On completion: `hpc_fetch` declared outputs + checkpoints; verify every file
   against `expectedOutputs`; SHA-256 each; write
   `artifacts/<experiment_id>/manifest.json` (including `sacct` accounting:
   Elapsed, MaxRSS, State, ExitCode, NodeList, AllocCPUS, Start/End) and mark
   `collected`. If any output is missing → `collection-failed` with the missing
   list; the conductor opens a task for `phd_candidate` and does not guess.
5. Heartbeat: if nothing changed, append a `heartbeat` event with the poll count.
   After `maxPolls` with no transition, escalate (`BLOCKED`: "job E-#### stuck
   in <state>; squeue/sacct evidence attached").

Collector prohibitions: no code edits, no config edits, no re-running science, no
"repairing" a missing output, no cancelling without an instruction record, no
metric/result interpretation. Its entire job is state, evidence, and files.

Retry rules for the conductor (not the collector):

| Situation | Action |
|---|---|
| exit 99 / TIMEOUT, progress advanced | requeue, attempt+1 |
| preempted/node failure, progress advanced | requeue, attempt+1 |
| OOM | requeue once with doubled memory if within partition limits, else failed-final |
| no progress between two consecutive attempts | `failed-final` + `BLOCKED` |
| `attempt > maxAttemptsPerExperiment` (6) | `failed-final` + `BLOCKED` |
| non-zero exit caused by a code bug | `failed-final`; open a `phd_candidate` task with the stderr tail |

## 7. Local fallback

When HPC is disabled or a task is small, `phd_candidate` runs the same command
locally with the same checkpoint contract and the same manifest; the experiment
record uses `resources.kind = "local"`. The rest of the pipeline is identical, so
the audit cannot tell "HPC" from "local" except through the record — which is the
point.

## 8. What the collector hands back

`collected` records make three things available to `phd_candidate` without any
hidden state: the manifest path (with SHA-256 and accounting), the raw outputs,
and the event history. The next `phd_candidate` session is pointed at exactly
those paths by the conductor — no conversation history is reused.
