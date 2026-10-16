# `.pi/extensions` — the loop's executable behavior

Each directory is one pi extension (`index.ts` entry point, loaded with `-e` by
the conductor and by every dispatched child). All logic that can be tested
without pi lives in sibling modules (no runtime imports of pi, typebox, or
pi-tui), so `node --test` exercises it directly.

| Extension | Registers | Stage | Notes |
|---|---|---|---|
| `guards/` | no tools; blocks model drift, protected paths, secrets, destructive bash, and forbidden role/tool combinations; injects the `<loop-role>` preamble | S0 | decisions in `policy.ts`/`bash.ts`/`protected.ts` |
| `ledger/` | `ledger_read`, `ledger_append`, `task_enqueue`, `task_close`, `pr_open`, `review_write`, `experiment_queue`, `experiment_update`, `decision_record`, `waiver_record`, `amendment_record`, `state_get`, `state_set` | S0/S3 | storage + validation in `store.ts`/`handlers.ts` |
| `status/` | TUI status line/widget and ledger renderers | S0 | read-only |
| `dispatch/` | `dispatch_agent`, `dispatch_parallel` (≤4), `dispatch_chain` | S3 | spawn engine in `lib.ts`; role allowlist in `.pi/lib/agents.ts` |
| `orchestrator/` | `merge_pr` + the `/loop:*` commands | S4 | `state-machine.ts` (pure `nextAction`), `execute.ts` (one action, pid lock, JSON-stream parse), `gates.ts` (pre-merge + human gates + HPC hook), `git.ts` (worktrees/merge/tags), `budget.ts`, `context.ts`, `commands.ts`, `sandbox.ts`, `cli.ts` |

`hpc/`, `verify/`, `rubric/`, `auditkit/`, and `paper/` arrive in S5/S6; the
tool names are declared in `.pi/config/tools.json` with `kind: "planned"` so
agent manifests can reference them today. The orchestrator's HPC gate is a stub
(`gates.ts` → `hpcGate`) that reports `not-configured`; S5 replaces it.

### The orchestrator's entry points

- TUI/pi: `/loop:start|step|pause|resume|status|verify-env|gate` (`index.ts`);
- headless: `.pi/loop/*.sh` → `node .pi/extensions/orchestrator/cli.ts <command>`;
- both call the same `commands.ts`, so they cannot diverge (parity is tested).

## `dispatch/` provenance and licence

`dispatch/lib.ts` and `dispatch/index.ts` are adapted from pi's bundled
**subagent** example
(`examples/extensions/subagent/`, shipped with
`@earendil-works/pi-coding-agent`, version recorded by `pi --version`).
That package is licensed **MIT** (`package.json` → `"license": "MIT"`; the
bundled `LICENSE` in the package root). The adaptation keeps the JSON-mode
spawn/parse pattern but adds: the pinned provider/model/thinking argv, the
roster-driven `--tools`/`--skill` allowlists, per-child session directories,
the per-role worker allowlist, one validated `dispatch` event per child, the
50 KB model-visible cap, and `structuredContent`. The example itself is not
vendored here; only the derived implementation is.

## Contract notes

- Scheme validation happens twice: the guard blocks the `tool_call`, and the
  handler re-checks the role capability before spawning (`plan/05 §7`).
- A child never inherits another child's session: every dispatch gets its own
  `--session-dir`, no `--continue`/`--resume`/`--session` is ever passed, and
  the session path is recorded in `EVENTS.jsonl`.
- Failures are never swallowed: stderr tail + `stopReason` are returned, and a
  failed/aborted child's partial text is explicitly labelled as not a result.
