---
description: Implement one task brief and open its PR
argument-hint: "<T-###>"
---
Brief: `${1:-<T-###>}`. Read `.pi/state/CONTEXT.md` and the task brief (in the
ledger or `.pi/state/tasks/`). If the task is missing, list open tasks and stop.

Run this as a `phd_candidate` work unit: one fresh process, the pinned model,
`PI_LOOP_BRIEF` set to the brief. The agent must read the files it changes, plan
with `planner` for non-trivial work, get a `methodologist` verdict for
statistical changes, run the narrowest test first and the full suite at the end,
and dispatch `critic` on the diff before opening the PR.

Acceptance evidence: `pr_open` with the machine-readable footer (`task_id`,
`head_sha`, `base_sha`, `branch`, `artifacts`, `tests`, `claims`, `hpc_jobs`),
the exact commands + observed output in the record, and a clean worktree at
`head_sha`. If compute is needed, the record must carry the experiment record
with checkpoints, `--resume`, and a passing local resume smoke test.

Stop on `BLOCKED` with the smallest decision the human must make. Never merge.
