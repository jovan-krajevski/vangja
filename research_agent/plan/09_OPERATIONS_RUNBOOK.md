# 09 — Operations runbook

Everything below happens at the repository root; runtime state is in `.pi/state/`.

## 1. First run

```bash
cd /path/to/vangja
uv sync --extra test --extra datasets --extra reproducibility
pi                                # trust the project when asked
/loop:verify-env                  # must be green (model pin, resources, schemas)
/loop:status                      # state should be IDLE/BOOT
```

Optional HPC: `cp .pi/config/hpc.example.json .pi/config/hpc.json`, fill
`host/user/remoteRepoDir/remoteScratchDir/hpcKey/gitKey`; leave `host` empty to
disable the HPC path completely. Check with `/loop:verify-env` (it reports
`hpc: disabled` or `hpc: ok`).

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

Gates are listed in `.pi/state/HUMAN_GATES.json` and shown by `/loop:status`.

| Gate | When | Default |
|---|---|---|
| `merge` | every accepted PR before it lands on `update_pymc` | on |
| `confirmation` | before the first confirmation-stage run | on |
| `audit` | before starting an audit cycle | on |
| `submission` | before anything labelled ready-to-submit | on |

`/loop:gate list`, `/loop:gate approve <id> [note]`, `/loop:gate reject <id> <why>`.
Rejecting a gate sends the loop back with your note attached to the task; it does
not stop the run.

## 4. Reading state

| File | Meaning |
|---|---|
| `.pi/state/LOOP_STATE.json` | current state, current task/PR/experiment, attempt counters, budget |
| `.pi/state/CONTEXT.md` | generated ≤200-line summary — read this first after any break |
| `.pi/state/EVENTS.jsonl` | every transition, action, child invocation, usage record |
| `.pi/state/TASKS.md` / `PRS.md` / `REVIEWS.md` / `EXPERIMENTS.md` | human-readable views |
| `.pi/state/BUDGET.json` | sessions, wall clock, cost consumed vs caps |
| `.pi/state/sessions/<run_id>/…` | full pi session transcripts — replay any agent step |
| `.pi/artifacts/<experiment_id>/manifest.json` | what a run produced, with SHA-256 and SLURM accounting |
| `.pi/audit/cycle-<n>/` | the audit: scope, gates, findings, verification, report, grade |

## 5. Intervening without breaking the loop

- **Redirect a task:** edit `.pi/state/tasks/T-###.md` before its `WORK` step, or
  reject the PR with instructions (the review note becomes the next brief).
- **Add a task:** write a brief from the template and run `ledger task_enqueue`,
  or drop a note in `state/HUMAN_GATES.json` with `action: "enqueue"`.
- **Kill a bad experiment:** `/loop:gate reject <job-id> "why"` then instruct
  `hpc_cancel` (the ledger records both). Never edit `EXPERIMENTS.jsonl` by hand.
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
- Collect manually if needed: `bash .pi/loop/collect.sh E-0007` runs one collector
  pass for one experiment.

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
| audit `uncalibrated` | narrow the workstream briefs (the calibration report names which findings were missed) and re-run those workstreams |
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
