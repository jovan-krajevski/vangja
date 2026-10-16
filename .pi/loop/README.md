# `.pi/loop` — headless entry points

The loop runs in two places: inside pi (`/loop:*` commands) and headless from a
terminal, cron, or tmux. These scripts are the headless entry points and contain
**no logic**: each one execs the orchestrator CLI the TUI commands also call, so
the two cannot diverge.

```
bash .pi/loop/step.sh [--forever] [--max-steps N] [--dry-run] [--sandbox P]
  →  node .pi/extensions/orchestrator/cli.ts step --headless …

bash .pi/loop/status.sh [--json] [--sandbox P]
  →  node .pi/extensions/orchestrator/cli.ts status --headless …

bash .pi/loop/collect.sh [--sandbox P]      # stage S5: exits 2 until the collector exists
  →  node .pi/extensions/orchestrator/cli.ts collect --headless …
```

Every command takes its configuration from `.pi/config/*.json`; none of them
reads a private config. `--sandbox <path>` redirects the whole loop to a scratch
clone and is refused if the path is the real repository.

## What each command does

| Command | Behaviour |
|---|---|
| `step` | exactly one action (`execute.ts`): spawn a primary agent, bookkeep a transition, open a human gate, run the pre-merge gate and merge, or pause/block |
| `start` | `step` in a loop until PAUSED/BLOCKED/DONE, no progress, or `--max-steps` |
| `pause [reason]` | writes PAUSED, records the reason; a running child is left alone, HPC jobs keep running |
| `resume` | returns to `paused_from` and reports the next action; completed actions are never repeated |
| `status` | state, current task/PR/experiment + attempt, budget vs caps, open human gates, last 5 events, stale child lock, and the `nextAction` |
| `verify-env` | `.pi/tests/check_env.ts` plus the orchestrator's own checks (HPC gate state, human-gate config) |
| `gate list\|approve\|reject` | human gates: `merge`, `confirmation`, `audit`, `submission`, `enqueue` |
| `collect` | S5 collector entry point (exits 2 until it exists — a stub that succeeds would make a dead loop look alive) |
| `peek` | read-only: prints the action a `step` would perform |

## Always available

```bash
make -C .pi verify-env          # environment + orchestrator tests (offline)
bash .pi/tests/run.sh orchestrator   # the S4 acceptance tests + transcript script
```

## Invariants enforced here

- **One child at a time**: a pid lock under `.pi/state/locks/child.json`; a live
  child blocks a second spawn, a dead pid is reported as a stale lock and cleared.
- **Completion events are the idempotency key**: the effect is applied before the
  event, so a replayed action is a no-op and a killed action resumes.
- **The base branch moves only by `git merge --no-ff`** with the review id in the
  message, from a clean checkout, with a tag `run/<run_id>/merge/<n>`.
- **Human gates never merge**: with `requireHumanMerge`, the loop opens the gate
  and returns; approving lets the next step merge.
- **`--dry-run` writes exactly one file**, `.pi/state/DRYRUN.jsonl`.
- **Never push, never rebase, never rewrite history**: the conductor has no such
  code path (asserted by `.pi/tests/orchestrator/test_gates.test.ts`).
