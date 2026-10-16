---
description: Pick the next task from the backlog and enqueue a checkable brief
argument-hint: "[origin-hint]"
---
Read `.pi/state/CONTEXT.md`, `.pi/state/TASKS.md`, and open findings
(`ledger_read kind=finding`). Priority order: open S0/S1 findings, then
REVIEW/PLAN P0/P1 items, then protocol gaps, then paper-evidence gaps.

Assemble the brief with the `research_agent/plan/03_AGENTS.md` §3 template:
origin, priority, effort class, objective, acceptance criteria (each a command or
a rubric line), deliverables (exact paths), evidence required, compute, out of
scope, protected paths, budget, dependencies. The origin hint is `${@:-none}`.

Dispatch the `planner` worker with the assembled objective; enqueue the returned
body with `task_enqueue`. Reject any brief without acceptance criteria, with an
unknown origin, or with a test-horizon choice in it.

Acceptance evidence: a new `T-###` open in `.pi/state/TASKS.jsonl` plus its line
in `TASKS.md`; every acceptance criterion must be checkable by a command.

MODE: if the arguments contain `--dry-run`, print the assembled brief and stop —
no tool calls, no model dispatch.
