# Prompt 05 — Build stage S5: HPC tools, experiment ledger, collector

Prerequisite: S0–S4 green. Read `research_agent/plan/06_HPC_AND_COLLECTOR.md` in
full, `plan/05_EXTENSIONS_AND_TOOLS.md` §5 (tools + gates) and §7 (role matrix),
and `plan/03_AGENTS.md` §1.2 (collector contract).

The hard rules: **one active job at a time**, **24h wall limit**, **checkpoint or
it does not run**, and `host: ""` means the whole path is disabled.

## Deliverables

1. `.pi/extensions/hpc/index.ts` — tools with the exact behaviour of
   `plan/05_EXTENSIONS_AND_TOOLS.md` §5: `hpc_config`, `hpc_status`,
   `hpc_submit`, `hpc_checkpoints`, `hpc_fetch`, `hpc_tail`, `hpc_cancel`,
   `hpc_requeue`. Requirements:
   - Config load + validation against `.pi/config/hpc.example.json` schema;
     `host: ""` → every tool returns `{enabled:false}` and performs no I/O;
     `maxParallelJobs != 1` rejected unless `.pi/state/amendments/` contains a
     matching amendment.
   - Private `ssh-agent` per run: start `ssh-agent -s`, `ssh-add` **only**
     `gitKey`, pass `SSH_AUTH_SOCK` to children, `ssh -A -o IdentitiesOnly=yes
     -i <hpcKey>`. If `gitKey` missing → `forwardGitAgent` unavailable, report
     `git-forwarding-off`, never fall back to the user's agent.
   - `hpc_submit` gate (all must pass, each failure names the missing item):
     no active job; record validates; checkpoint block present (dir,
     `progressFile`, `resumeCommand`, `flushOnSignal`); **local resume smoke-test
     evidence present**; remote clean detached checkout of `commit_sha`;
     `uv.lock` hash match; container hash match; wall clock ≤ `wallClockHours`.
   - Submit with `sbatch --parsable` from the template; record job id + raw
     output; never print key material.
   - `hpc_fetch` copies declared outputs **and checkpoints**, SHA-256s every file,
     writes `.pi/artifacts/<id>/manifest.json` including `sacct` accounting, and
     fails loudly on missing declared outputs.
   - `hpc_requeue` only after the previous attempt is confirmed ended; creates
     `attempt+1`, same `idempotencyKey`.
2. `.pi/skills/hpc-slurm-and-checkpointing/templates/job.slurm` — final template
   with `--time=24:00:00`, `--signal=B:USR1@300`, trap → flush + exit 99,
   `uv sync --frozen`, optional container exec, unbuffered logs.
3. `.pi/agents/collector.md` — final version encoding: poll → status transitions
   with raw evidence → on timeout/preempt/exit-99 fetch checkpoints first and
   write `requeue_requested` → on success fetch + verify + mark `collected` →
   heartbeat; prohibitions (no code, no resubmit, no interpretation, no repair).
4. Experiment ledger integration: `experiment_queue` enforces the single-slot
   rule and the checkpoint schema; `experiment_update` restricted to collector +
   conductor; `.pi/state/EXPERIMENTS.md` view shows id/attempt/status/progress.
5. Conductor wiring: `WAITING_HPC` state, requeue decisions per
   `plan/06_HPC_AND_COLLECTOR.md` §6 (retry table), `BLOCKED` on no-progress,
   max attempts, or stuck polls; `hpc-disabled` skips `WAITING_HPC` entirely.
6. Mock-cluster test harness `.pi/tests/hpc/fixtures/mockbin/` with shims for
   `ssh`, `scp`, `rsync`, `sbatch`, `squeue`, `sacct`, driven by recorded
   transcripts; a `run_mock_cluster.sh` that puts the shims first on `PATH`.

## Mock scenarios (each a test)

1. Disabled: empty host → submit refused with `hpc: disabled`, collector exits
   with one `hpc-disabled` event, no ssh attempted (assert by wrapping `ssh` in a
   shim that fails the test if called).
2. Happy path: queued → running → completed → outputs + checkpoints fetched →
   `collected`; manifest contains `sacct` fields and SHA-256 for every file.
3. Timeout at 24h (exit 99): checkpoints fetched, progress recorded,
   `requeue_requested`; a **second submit before the first ended is refused**;
   after the requeue the same test asserts `attempt: 2` with the same
   idempotency key.
4. OOM: one requeue with doubled memory; second OOM → `failed-final` + `BLOCKED`.
5. No progress across two attempts → `failed-final` + `BLOCKED`.
6. Missing declared output after completion → `collection-failed`, task opened,
   no fabricated result.
7. Missing checkpoint metadata at queue time → refused with the exact missing
   field.
8. Checkpoint smoke-test missing → refused.
9. Secrets: config with a fake key → the key never appears in any tool output,
   ledger record, or session file (grep assertion).

## Acceptance tests

- `bash .pi/tests/hpc/run_mock_cluster.sh` → all scenarios green, printed matrix.
- `hpc_submit` twice in one test → the second refusal message names the active
  experiment id.
- `PATH=... pi --print "/loop:step"` with a queued experiment and no HPC host →
  the loop skips `WAITING_HPC` and proceeds; evidence in the events.

## Do not

- Do not contact any real host, and do not write real credentials anywhere.
- Do not weaken the single-slot invariant for convenience.

## Final report

Append `## S5 — HPC tools, experiment ledger, collector` to
`research_agent/REPORT_BUILD.md` with the scenario matrix and raw output.
