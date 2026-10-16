# `.pi/loop` — headless entry points

The loop runs in two places: inside pi (`/loop:*` commands, implemented in S4)
and headless from a terminal or tmux. These three scripts are the headless
entry points; they contain **no logic**. Each one is a thin shim that execs the
same orchestrator CLI the TUI commands call, so the two cannot diverge:

```
bash .pi/loop/step.sh [--forever] [--dry-run]   →  node .pi/extensions/orchestrator/cli.ts step --headless …
bash .pi/loop/status.sh [--json]                →  node .pi/extensions/orchestrator/cli.ts status --headless …
bash .pi/loop/collect.sh                        →  node .pi/extensions/orchestrator/cli.ts collect --headless …
```

## Status: stubs (S0)

`.pi/extensions/orchestrator/cli.ts` does not exist yet — it is built in stage
S4 (prompt `research_agent/prompts/04_orchestrator_and_state.md`); `collect` also
needs S5. Until then each script prints exactly what it *would* run and exits
with code 2. This is deliberate: a stub that silently succeeds would make a dead
loop look alive.

Until S4, use:

```bash
make -C .pi verify-env          # environment tests (this is what S0 provides)
node .pi/tests/check_env.ts --status   # read-only summary of state/config
```

## Contract for S4

- `step.sh` → `step --headless` performs exactly one transition, respects
  `.pi/state/STOP`, and never starts a second child while one is alive.
- `status.sh` → `status --headless` prints state, current task/PR/experiment +
  attempt, budget, open human gates, the last 5 events, and `nextAction`.
- `collect.sh` → `collect --headless` is the S5 collector entry point.
- Every command takes its configuration from `.pi/config/*.json`; none of them
  reads a private config.
