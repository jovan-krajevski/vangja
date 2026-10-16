---
description: Prepare an experiment record, prove resume, and queue it
argument-hint: "<T-###>"
---
Task: `${1:-<T-###>}`. Run this as a `phd_candidate` work unit with
`experimenter` guidance in scope.

Assemble the experiment record exactly per `plan/06_HPC_AND_COLLECTOR.md` §4:
task_id, branch, commit_sha, data_hash, exact command, resources, seeds,
expectedOutputs, checkpoint directory + progress file, `--resume` command,
idempotency key, and the frozen config reference. Then prove resume locally:
kill the cell mid-run, restart with `--resume`, and record the units skipped vs
re-fit (no re-fitting of completed units).

Acceptance evidence: the record plus a resume-smoke log with the two commands and
their observed completed/skipped/refit sets, followed by `experiment_queue`.
A queue call without checkpoint metadata or without the smoke proof must be
refused. One job at a time: if any experiment is active, report the wait and stop.
Never choose a configuration on the test horizon and never touch `src/` configs.
