# Prompt 04 — Build stage S4: orchestrator, state machine, git/PR pipeline

Prerequisite: S0–S3 green. Read `research_agent/plan/02_ARCHITECTURE.md` in full
(states, gates, budgets, model pin, memory protocol) and `plan/03_AGENTS.md` §3–4.

## Deliverables

1. `.pi/extensions/orchestrator/` with these modules:
   - `state-machine.ts` — `nextAction(state, ledgers): Action` as a **pure
     function** covering every transition in `plan/02_ARCHITECTURE.md` §2,
     including `PAUSED`, `BLOCKED`, `HUMAN_GATE`, `WAITING_HPC`, budget stops, and
     the `DONE` conditions. Table-driven; no I/O in this module.
   - `execute.ts` — performs exactly one action: build the child `pi` argv from
     `.pi/config/agents.json`, spawn it, parse the JSON stream (final message,
     tool calls, usage, errors), append the completion event with the exact argv
     and prompt hashes, and update `LOOP_STATE.json` atomically.
   - `gates.ts` — pre-merge gate runner (tests, protected paths, test diff,
     integrity scan, provenance, hygiene, claims), HPC gate hook (delegates to S5),
     human gates (`merge`, `confirmation`, `audit`, `submission`).
   - `git.ts` — worktrees under `.pi/state/worktrees/<task_id>/`, branch
     `work/T-###-slug` from `update_pymc`, commit helpers, PR record creation,
     `merge --no-ff` with the review ID in the message, tags
     `run/<run_id>/merge/<n>`, `audit-cycle/<n>`, `submission/<n>`; never
     force-push, never rebase `update_pymc`, refuse to operate on a dirty main
     checkout.
   - `budget.ts` — accounting from child usage; caps; `PAUSED` on exceed.
   - `context.ts` — deterministic `state/CONTEXT.md` (≤200 lines) and the derived
     `TASKS.md`/`PRS.md`/`REVIEWS.md`/`EXPERIMENTS.md` views.
   - `commands.ts` — `/loop:start`, `/loop:step`, `/loop:pause`, `/loop:resume`,
     `/loop:status [--json]`, `/loop:verify-env`, `/loop:gate`.
   - `sandbox.ts` — `--sandbox <path>` redirects the repo root for a scratch clone
     (used by tests and dry runs); `--dry-run` logs actions without side effects.
2. Orchestrator CLI: `node .pi/extensions/orchestrator/cli.ts <command>` (or the
   extension's own entry) with `start|step|pause|resume|status|verify-env|gate
   list|gate approve|gate reject`. `.pi/loop/step.sh`, `status.sh`, and
   `collect.sh` become thin wrappers around it (no duplicated logic).
3. `.pi/loop/README.md` — the human entry points and what each does.
4. Tests `.pi/tests/orchestrator/`:
   - state-machine table tests (every state × condition → expected action),
   - sandbox end-to-end: a trivial synthetic task in a scratch repo goes
     `SELECT_TASK → WORK → PR → REVIEW(accept) → MERGE → POST_MERGE`; assert the
     merge only happens after the gate run, and that the sandbox `update_pymc`
     advanced while the real one did not,
   - pause/resume: kill between each pair of transitions, resume, assert no
     duplicated side effects (one merge, one job, one PR),
   - idempotency: re-run `execute` for a completed action → no-op,
   - budget: exceed each cap → `PAUSED` with the reason,
   - human gate: `requireHumanMerge: true` → loop pauses before merge; `gate
     approve` → resume completes the merge,
   - protected-path and dirty-tree refusals,
   - `--dry-run` writes nothing except its own log.

## State/child invocation contract

- Child argv is fully determined by `.pi/config/agents.json` + the action; it is
  recorded in the event before spawning.
- Env for children: `PI_LOOP_ROLE`, `PI_AGENT_NAME`, `PI_LOOP_STATE`, `PI_LOOP_RUN`,
  `PI_LOOP_SANDBOX` (when set), plus the private `SSH_AUTH_SOCK` when HPC runs.
- `status` must print: state, current task/PR/experiment + attempt, budget, open
  human gates, last 5 events, and `nextAction` (what a `step` would do).
- `PAUSED` is resumable from the recorded state; `step` never starts a second
  child while one is alive; stale PID/lock files are detected and reported.

## Acceptance tests (paste output into `REPORT_BUILD.md`)

1. `bash .pi/tests/run.sh orchestrator` → all green.
2. Sandbox cycle transcript showing the exact commands and the merge commit
   message containing the review ID.
3. Pause/resume matrix: for each transition boundary, kill → resume → the ledger
   shows exactly one completion event per action.
4. Budget cap test: sessions, wall clock, cost, consecutive failures,
   no-progress, audit cycles.
5. `./.pi/loop/step.sh` and `/loop:step` produce identical events for the same
   starting state (parity check).
6. `git status` in the real repo after the whole test run: clean, no branches or
   tags created, `update_pymc` SHA unchanged.

## Do not

- Do not implement HPC (S5) beyond a `hpc_gate()` interface stub that returns
  `not-configured`.
- Do not run the loop against the real repo in tests; always use the sandbox.
- Do not `git push` anywhere; `gh` may be used only when explicitly enabled by
  config `useGitHub: true` (default false), and never in tests.

## Final report

Append `## S4 — orchestrator, state machine, git/PR pipeline` to
`research_agent/REPORT_BUILD.md`.
