---
description: Show the loop state, open tasks/jobs/gates, budget, and recent events
argument-hint: "[--json]"
---
Read-only status. Do not change any state and do not dispatch a model.

Assemble from `.pi/state/CONTEXT.md`, `state_get`, the `TASKS`/`PRS`/`REVIEWS`/
`EXPERIMENTS` ledgers, `GATES`, `BUDGET.json`, and the last events
(`ledger_read kind=event limit=10 tail=true`). Render, in order: loop state +
run id; current task/PR/experiment (with attempt); open tasks with status;
active HPC jobs; pending human gates (`GATES` pending + `HUMAN_GATES.json`); the
last review decision; and the next required action.

Acceptance evidence: the rendered status text (or `--json` object) matches
`LOOP_STATE.json` and the ledger tails; no file in `.pi/state/` is modified
(check with `git status` / mtimes if asked).
