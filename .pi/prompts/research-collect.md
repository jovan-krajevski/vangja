---
description: Poll and collect an open experiment
argument-hint: "[E-###]"
---
Collect `${1:-all open experiments}`. Run this as a `collector` work unit —
never as `phd_candidate`.

Read `EXPERIMENTS.jsonl` and `.pi/config/hpc.json`. If `hpc.json` has no host,
append one `hpc-disabled` event and stop cleanly. Otherwise poll `squeue`/`sacct`
and record the raw trimmed output; on normal completion fetch and SHA-256 every
declared output before marking `collected`. On TIMEOUT/TIME_LIMIT/PREEMPTED/exit
99, fetch the checkpoints **first**, record the progress delta, then append
`requeue_requested`.

Acceptance evidence: one `experiment_update` per transition with raw cluster
evidence, fetched artifacts with hashes, and a failure record for anything that
did not collect. Never edit code/config, never resubmit silently, never run two
jobs at once, never fabricate a missing number.
