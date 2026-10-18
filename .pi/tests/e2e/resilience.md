# Resilience matrix — kill, restart, exactly-once

`node .pi/tests/e2e/resilience.ts` (also `make -C .pi resilience`). It builds one
scratch clone per kill point with the synthetic task + fixture from
`smoke.ts`, drives the real conductor to the boundary, kills it, restarts with
the operator commands (`pause` → `resume`), continues, and asserts exactly-once
semantics for branches, PRs, merges, experiments and artifacts.

Every scenario prints `KILLPOINT <name>: PASS|FAIL — detail`; the script exits
non-zero if any fails. Nothing touches the real repository (the clones live under
`.pi/state/e2e-tests/resilience-*`, which is gitignored).

| # | Kill point | Mechanism | Exactly-once assertion |
|---|---|---|---|
| 1 | before child spawn | a stale lock (dead pid) with the next action key is planted; the conductor must report it, clear it, and spawn once | one `stale` event; one `work:T-001:1` start event; one experiment record |
| 2 | during child, before any effect | the stub `SIGKILL`s itself immediately (`PI_E2E_MODE=work-kill-early`) | one failed completion; the retry uses a new key (`work:T-001:2`); one experiment record; no branch/PR duplicate |
| 3 | after child (completion recorded) | the loop is restarted after the child finished and before the next step | one `work:T-001:3` start event; one PR; the restart advances WORK → REVIEW without a second spawn |
| 4 | before ledger append (effect exists, no record) | the stub creates the branch + commit, then `SIGKILL`s before `appendRecord` (`PI_E2E_KILL=before-record`) | one branch; the resumed attempt adopts the commit; exactly one PR; one start event per attempt key |
| 5 | after ledger append (record exists, no completion) | the stub writes the PR record, then `SIGKILL`s (`PI_E2E_MODE=work-then-kill`) | one PR; one start event for the killed attempt; the restart goes straight to REVIEW (no duplicate spawn) |
| 6 | during merge | the merge lands by hand with the conductor's exact message, then the loop restarts (the crash window between `git merge --no-ff` and the completion event) | one merge commit; one `merge` completion event with `already_merged: true`; no second commit |
| 7 | during collection | the experiment `SIGKILL`s itself after the first checkpoint unit (`TINY_RUN_KILL_AFTER=1`) | one experiment record; the resumed run completes 3/3 checkpointed units; one artifact manifest; the progress marker advances to 3 |

## What is a real kill and what is simulated

- Scenarios 2, 4, 5 and 7 use a real `SIGKILL` of the child process at the
  documented boundary (the stub kills itself exactly there, so the timing is
  deterministic rather than race-dependent).
- Scenario 1 reproduces the documented crash window *after the lock, before the
  child process exists*; the pid is a dead one, which is what the conductor sees
  after a real kill.
- Scenario 6 reproduces the crash window *after the effect, before the
  completion event*: the merge commit exists with the conductor's message but no
  event records it. This is the same shape the S4 verification exercised for the
  effect→event window.
- Scenario 3 is a pure replay: the completion event exists, so the action is a
  no-op and the loop advances.

## Restart semantics

Every scenario restarts with `pause` + `resume` (the operator commands from
`plan/09_OPERATIONS_RUNBOOK.md`), which re-enters the state recorded in
`LOOP_STATE.json`; completed action keys are never repeated because the
completion event is the idempotency key. Scenarios where the loop is not PAUSED
still run `resume`, which reports `loop is not PAUSED … nothing to resume` — the
restart path is identical.

## Evidence

```
$ node .pi/tests/e2e/resilience.ts
KILLPOINT before-child-spawn: PASS — stale=1 starts(work:T-001:1)=1 experiments=1
KILLPOINT during-child-before-effect: PASS — failed=1 starts(work:1)=1 starts(work:2)=1 experiments=1
KILLPOINT after-child: PASS — starts(work:T-001:3)=1 PRs=1
KILLPOINT before-ledger-append: PASS — branches=1 PRs=1 starts(work:3)=1 starts(work:4)=1
KILLPOINT after-ledger-append: PASS — PRs=1 starts(work:3)=1 state=REVIEW
KILLPOINT during-merge: PASS — merge commits=1 completions=1 already_merged=true
KILLPOINT during-collection: PASS — experiments=1 status=collected progress=[{"attempt":1,"completed":3}] checkpoint=["fit","predict","score"]

RESILIENCE PASS: 7/7 kill point(s) recovered with exactly-once semantics
```

The happy-path end-to-end run (including the tampered artifact, the seeded audit
and the decision rule) is `smoke.ts` / `make -C .pi smoke`; see
`research_agent/REPORT_BUILD.md` § S7 for its transcript.
