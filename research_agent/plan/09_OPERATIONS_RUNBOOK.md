# 09 — Operations runbook

Everything below happens at the repository root; runtime state is in `.pi/state/`.

> Cross-checked against the implementation in S7 (decision D-003): command names,
> gate storage (`GATES.jsonl`, not `HUMAN_GATES.json`), the `hpc gate` wording,
> and `collect.sh` (no experiment id) were corrected to match the code.

## 1. First run

```bash
cd /path/to/vangja
uv sync --extra test --extra datasets --extra reproducibility
pi                                # trust the project when asked
/loop:verify-env                  # must be green (model pin, resources, schemas)
/loop:status                      # state is UNINITIALIZED before the first step
```

Optional HPC: `cp .pi/config/hpc.example.json .pi/config/hpc.json`, fill
`host/user/remoteRepoDir/remoteScratchDir/hpcKey/gitKey`; leave `host` empty to
disable the HPC path completely. Check with `/loop:verify-env`: its `hpc gate`
line reads `not-configured` (no host: experiments run locally), `ok` (host
configured and validated), or `blocked` (config present but invalid — the loop
will not guess).

Optional PRs on GitHub: `gh auth status` must succeed; otherwise PRs are recorded
locally and nothing is pushed.

## 2. Start, pause, resume

| Action | Command | Effect |
|---|---|---|
| start | `/loop:start` | conductor begins stepping from the recorded state |
| one step | `/loop:step` | exactly one transition (useful for watching) |
| pause | `/loop:pause [reason]` | finishes the running child, writes `PAUSED`, stops scheduling; HPC jobs keep running |
| resume | `/loop:resume` | re-enters the paused state; completed steps are never repeated |
| stop | `touch .pi/state/STOP` | hard stop checked before every step (survives restarts) |
| status | `/loop:status --json` | machine-readable snapshot for scripts |

Headless (tmux/cron): `bash .pi/loop/step.sh --forever` behaves like
`/loop:start`; `bash .pi/loop/status.sh` prints the summary. Both read the same
`.pi/config/agents.json`, so the TUI and headless modes cannot diverge.

**Pausing with a job in flight is safe.** State is written after every transition;
the collector re-reads `squeue`/`sacct` on resume. Nothing needs to be re-run.

## 3. Human gates

Gates live in the `.pi/state/GATES.jsonl` ledger and are shown by `/loop:status`
and `/loop:gate list`.

| Gate | When | Default |
|---|---|---|
| `merge` | every accepted PR before it lands on `update_pymc` | on (`requireHumanMerge`) |
| `submission` | when the decision rule says DONE — the sign-off to finish the run | on |
| `confirmation` | defined for the first confirmation-stage run; opened by whoever reaches it, not by the conductor | manual |
| `audit` | defined for starting an audit cycle; the conductor reaches AUDIT through the paper merge without this gate | manual |
| `enqueue` | opened by `amendment_record` when a protected protocol path must change | on for amendments |

`/loop:gate list`, `/loop:gate approve <id> <note>` (the note is the audit trail),
`/loop:gate reject <id> <why>`. Rejecting a `merge` gate sends the PR back to
WORK with the note; rejecting the `submission` gate returns the loop to
SELECT_TASK. It never stops the run.

## 4. Reading state

| File | Meaning |
|---|---|
| `.pi/state/LOOP_STATE.json` | current state, current task/PR/experiment, attempt counters, budget |
| `.pi/state/CONTEXT.md` | generated ≤200-line summary — read this first after any break |
| `.pi/state/EVENTS.jsonl` | every transition, action, child invocation, usage record |
| `.pi/state/TASKS.md` / `PRS.md` / `REVIEWS.md` / `EXPERIMENTS.md` | human-readable views |
| `.pi/state/BUDGET.json` | sessions, wall clock, cost consumed vs caps |
| `.pi/state/GATES.jsonl` | every human gate (pending/approved/rejected) |
| `.pi/state/READINESS.json` | the supervisor's readiness verdict; consumed once and archived to `READINESS.verdict.json` |
| `.pi/state/sessions/<run_id>/…` | full pi session transcripts — replay any agent step |
| `.pi/artifacts/<experiment_id>/manifest.json` | what a run produced, with SHA-256 and SLURM accounting |
| `.pi/audit/cycle-<n>/` | the audit: scope, gates, findings, verification, report, grade |

## 5. Intervening without breaking the loop

- **Redirect a task:** edit `.pi/state/tasks/T-###.md` before its `WORK` step, or
  reject the PR with instructions (the review note becomes the next brief).
- **Add a task:** write a brief from the template and call the `task_enqueue`
  tool (the ledger records it), or drop a note in `GATES.jsonl` with
  `action: "enqueue"`.
- **Kill a bad experiment:** record a decision naming the experiment and the
  cancellation (`decision_record`), then `hpc_cancel` — a cancellation is refused
  without an instruction record. The conductor owns requeue (`hpc_requeue`); never
  edit `EXPERIMENTS.jsonl` by hand.
- **Change the model/budget/venue:** edit `.pi/config/*.json`, then
  `/loop:verify-env` (the guard will refuse child processes that violate the pin).
- **Emergency stop:** `touch .pi/state/STOP`; kill the pi process; remote SLURM
  jobs continue (they are checkpointed). Remove `STOP` and `/loop:resume` later.

## 6. HPC day-to-day

- Only one job at a time, by design. `/loop:status` shows the active job, its
  attempt number, and the last `sacct` state.
- A 24h job that is killed at the wall limit produces `timed_out` →
  `requeue_requested`; the collector fetches checkpoints first, then the conductor
  submits attempt+1. You will see this as two experiment events with the same
  `id` and `attempt: 1, 2`.
- Watch progress: `.pi/state/EXPERIMENTS.md` lists the checkpoint progress marker
  per attempt. If the marker does not move between attempts, the loop blocks for
  you instead of burning the queue.
- Collect manually if needed: `bash .pi/loop/collect.sh` runs one collector pass
  for the active experiment (the shim takes no experiment id; the collector is
  bound to the one unfinished record).

## 7. Budgets and cost

`.pi/config/budgets.json` caps sessions/day, parallel workers (default 4), HPC
jobs (hard 1), task wall clock, consecutive failures, no-progress cycles, audit
cycles, and cost. On a cap the loop writes `PAUSED` with the reason and the exact
resume command. Raise the cap deliberately and record why in
`.pi/state/DECISIONS.md`.

## 8. Recovery recipes

| Symptom | Do |
|---|---|
| loop paused on `maxConsecutiveFailures` | read the last 3 events + session transcripts, fix the cause or enqueue a task, `/loop:resume` |
| PR failed the pre-merge gate | read `.pi/state/reviews/RV-###.md`; either waive (supervisor, with reason) or send back for changes |
| protected-path write refused | if the change is legitimate, record an amendment, then retry; otherwise stop and inspect the agent's reasoning in its session |
| audit `uncalibrated` | narrow the workstream briefs (the calibration report names which findings were missed) and re-run those workstreams; the `UNCALIBRATED` marker under `.pi/audit/calibration/latest/` is authoritative over any stale `recall.json` |
| artifact hash mismatch | treat as an S0 finding; re-collect the job; never use the artifact until reconciled |
| HPC credentials missing | loop writes `BLOCKED` with `hpc-disabled`/`key-missing`; provide the key path in config and resume |
| two jobs submitted | impossible by design; if seen, stop the loop immediately and file a bug against the conductor (single-slot invariant) |

## 9. What a "good day" looks like

1. `/loop:status`: one merge accepted, one task in `WORK`, one experiment
   collected, no open S0/S1.
2. `.pi/state/REVIEWS.md`: reviews show commands actually run by the supervisor,
   not just prose.
3. `.pi/state/EXPERIMENTS.md`: progress markers advancing; no attempt without
   progress.
4. `.pi/state/BUDGET.json`: under caps.
5. Session transcripts: every child invocation used the pinned model and thinking
   level (the guard enforces it, but check after config edits).
