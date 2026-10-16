# SLURM fields, states, and the requeue classifier

`scripts/parse_sacct.py` implements exactly the mapping below; this file is
the human-readable spec. Field names follow `sacct --json` (Slurm 23.x/24.x);
the parser also accepts the flat string forms so older clusters work.

## 1. Jobs array

```json
{"jobs": [{"job_id": 4711, "name": "E-0007",
           "state": {"current": ["TIMEOUT"], "reason": "TimeLimit"},
           "exit_code": {"return_code": 0, "signal": 0},
           "derived_exit_code": {"return_code": 99, "signal": 0},
           "time": {"elapsed": 86400, "start": "2026-01-01T09:00:00",
                    "end": "2026-01-02T09:00:00"},
           "nodes": "gpgpu02", "alloc_cpus": 8, "max_rss": "21.4G",
           "job_id_with_sched": "4711", "account": "openlab"}]}
```

Fields used by the record written at collection time (plan/06 §6.4):
`Elapsed`, `MaxRSS`, `State`, `ExitCode`, `NodeList`, `AllocCPUS`,
`Start`/`End`. Store them verbatim; the audit re-reads them.

Accepted shape variations the parser tolerates:

| Field | Accepted |
|---|---|
| `state` | `"TIMEOUT"`, `{"current": "TIMEOUT"}`, `{"current": ["TIMEOUT"]}` |
| `exit_code` / `derived_exit_code` | `"0:99"`, `{"return_code": 0, "signal": 0}`, `{"number": 99, "set": true}` |
| `time.elapsed` | integer seconds or `"1-00:00:00"` |
| `nodes` | `"n01"` or `{"list": ["n01"]}` |

## 2. State mapping

| sacct state(s) | Our state | Meaning |
|---|---|---|
| `PENDING`, `CONFIGURING`, `REQUEUED` (waiting) | `queued` | not started |
| `RUNNING`, `COMPLETING`, `SUSPENDED` | `running` | active |
| `COMPLETED` with exit 0 | `completed` | outputs should exist |
| `COMPLETED` with exit 99 | `interrupted_clean` | checkpoint flushed |
| `TIMEOUT`, `TIME_LIMIT`, `DEADLINE` | `timed_out` | wall limit hit |
| `OUT_OF_MEMORY` | `oom` | memory limit hit |
| `NODE_FAIL`, `PREEMPTED`, `BOOT_FAIL` | `preempted` | infrastructure |
| `FAILED`, `CANCELLED` (non-zero, not 99) | `failed` | inspect stderr |
| `CANCELLED` explicitly | `cancelled` | only on instruction |

Exit-code convention: **0 = complete, 99 = cleanly interrupted** (the job
flushed its checkpoint and is safe to requeue), anything else = failure.
Signal 9/137 usually means the OOM killer or a SIGKILL.

## 3. Classifier output

`parse_sacct.py` maps each job to `{state, exit_code, action, reason,
progress_hint}`. `action` is one of:

| action | When | What the conductor does |
|---|---|---|
| `wait` | queued/running | poll again; append a heartbeat |
| `collect` | completed | fetch declared outputs + checkpoints, hash, write the artifact manifest |
| `requeue` | interrupted_clean, timed_out, preempted **and** progress advanced | record the progress delta, append `requeue_requested`, resubmit at attempt+1 |
| `requeue-no-progress` | as above but progress did not advance | counts toward `failed-final` if it repeats |
| `retry-once-doubled-memory` | oom, first occurrence, partition allows | resubmit once with doubled `--mem`; otherwise `failed-final` |
| `failed-final` | failed, or attempts exhausted | open a task with the stderr tail, mark `BLOCKED` |
| `stop` | cancelled | terminal; only after an explicit instruction record |

Flags that change the classification:

- `--progress-advanced` — the checkpoint progress file grew since the last
  poll. Without it, timeout/preempt maps to `requeue-no-progress` (triaged as
  a probable bug).
- `--attempt N` (default 1) and `--max-attempts M` (default 6): at
  `N >= M`, everything that would requeue becomes `failed-final`.
- `--memory-can-double` — the partition headroom check. Off → OOM is
  `failed-final` immediately.

## 4. Polling discipline

1. `squeue -u "$USER" -j <jobid> -o '%T %M %R'` first; use `sacct --json`
   for finished jobs.
2. Accounting lag: after a job leaves the queue, `sacct` may be empty for a
   few seconds. Poll up to `maxPolls` (default 40, 60s apart) before
   escalating; never requeue on a single empty answer.
3. `maxParallelJobs > 1` is rejected unless an amendment file says otherwise:
   the single-active-job rule is a scientific control, not a convenience.
4. Every transition stores the raw command output (trimmed) as evidence —
   the ledger must be checkable without trusting prose.

## 5. Agent forwarding and the remote checkout

The remote work tree is a **clean detached checkout of the record's
`commit_sha`**:

```bash
ssh -A -o IdentitiesOnly=yes -o ForwardAgent=yes -p "$PORT" -i "$HPC_KEY" \
    "$USER@$HOST" \
    "git fetch origin && git checkout --detach $SHA && git status --porcelain"
```

- `git status --porcelain` must be empty; a dirty tree is a refusal.
- Treat missing git forwarding as an unavailable capability; if the checkout
  cannot be created and rsync is unavailable, refuse the submission.
- `uv.lock` SHA and `container_sha` must match the record, or the run would
  not be reproducible by construction.

## 6. Container notes (from `case_studies/singularity.def`)

- Images are built locally (`sudo singularity build case_studies/vangja.sif case_studies/singularity.def`)
  and transferred; the base is pinned (`continuumio/miniconda3:v25.11.1`).
- Record `sha256sum case_studies/vangja.sif` in the experiment record.
- `%test` runs `import vangja; import pymc; import arviz`; a failing `%test`
  image must not be submitted.
- In-job use: `singularity exec --bind "$REPO_ROOT:$REPO_ROOT" --pwd "$REPO_ROOT" "$SIF" python ...`.
  GPU jobs add `--nv` and a `--gres=gpu:...` request; never rely on GPU
  availability for correctness.
