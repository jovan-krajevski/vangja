---
name: research-memory-and-handoff
description: Start, end, pause, or hand off a Vangja loop session. Use when opening a fresh-context session, writing a task brief or ADR, leaving work resumable mid-way, summarising a run for the next agent, or checking that the state files (CONTEXT, ledgers, LOOP_STATE) are valid. Files are the memory; the conversation is not.
---

# Research memory and handoff

Fresh sessions start with zero context; only files survive. Read this before
writing a handoff, a brief, or a decision record.

## What to read first in a fresh session (in this order)

1. `.pi/state/CONTEXT.md` — the derived state view (state, task, PR,
   experiment, budget, recent events, recent reviews). It is generated from
   the ledgers; do not edit it.
2. The task brief (the state-directory task file the conductor names) —
   objective, acceptance criteria, deliverables, out-of-scope.
3. `AGENTS.md` — non-negotiables and gotchas.
4. The skills the brief names.
5. Prior reviews of the same PR (`REVIEWS.jsonl`/`REVIEWS.md`) and any open
   findings touching the same code.
6. `research_agent/plan/09_OPERATIONS_RUNBOOK.md` if anything about the loop
   itself is unclear.

If `CONTEXT.md` names no task, the session is idle: do not invent work;
record an event or stop.

## Writing rules (what belongs where)

| Artifact | Where | Contains |
|---|---|---|
| Event | ledger (`ledger_append`) | what happened, in one sentence, with evidence |
| Task brief | the task file in the state directory (via `task_enqueue`) | the template in `templates/brief.md` |
| Handoff | `templates/handoff.md` as the final comment of the PR or the task file | state, evidence, next action, blockers |
| ADR | ledger `decision_record` | context, decision, alternatives, consequences (`templates/adr.md`) |
| Review | ledger `review_write` | decision, rubric scores, evidence, required changes |
| Finding | audit register | nine-field schema (`scientific-audit-and-adversarial-review`) |
| Durable code fact | one line in `AGENTS.md` | only if it will change future work |

**Numbers never live in prose alone.** A handoff or event that quotes a number
must name the artifact path + manifest SHA (and config/seed). If the artifact
does not exist yet, the number does not exist yet.

## Leaving work resumable mid-way

1. Commit the work in progress on the task branch (WIP commits are fine).
2. Write a handoff with: `head_sha`, what is done, what is *proven* (command +
   output), what is not, the next concrete action, and any blocked decision.
3. Record an event; if blocked on a human, set the state via the conductor
   with `BLOCKED` and the smallest decision needed.
4. Do not delete the worktree, checkpoints, or scratch outputs the next
   session will need; name their paths explicitly.
5. If a fit/experiment is mid-flight, leave the checkpoint dir and its
   progress file; never "clean up" a partially completed run.

## Session hygiene

- One task per session where possible; when the task changes, write the
  handoff first.
- Re-read the brief after any pause; do not trust the conversation's memory of
  the acceptance criteria.
- Record failed attempts as events with the error — a later session must not
  repeat a dead end.
- Keep `.pi/state/CONTEXT.md` under 200 lines (it is generated; if it grows,
  the ledgers need projection, not hand-editing).
- Never edit `.pi/state/**` with `write`/`edit` (guard-protected); use the
  ledger tools so changes are locked and append-only.

## Checking the state files

```bash
bash .pi/skills/research-memory-and-handoff/scripts/check_state_files.sh
bash .pi/skills/research-memory-and-handoff/scripts/check_state_files.sh --strict
```

It validates `LOOP_STATE.json` (schema, key set, legal state name),
`CONTEXT.md` (≤200 lines), and every ledger JSONL (each line an object with
`schema`/`kind`/`ts`/`actor`). Missing `LOOP_STATE.json` is a warning until
the orchestrator (S4) exists; `--strict` makes it an error.

## Failure modes

- A handoff that says "continue as discussed" → useless; name the file, the
  command, the expected output.
- A decision made but not recorded → the next session re-litigates it; record
  an ADR with the alternatives you rejected.
- Numbers in a handoff without artifact paths → treated as unverified by the
  next session (and by the audit); re-derive or drop them.
- `CONTEXT.md` hand-edited → the next generation overwrites it and the lie is
  gone; use the ledgers.
