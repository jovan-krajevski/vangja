# Prompt 00 — Build stage S0: scaffold, config, guards, ledger, status

You are building the local pi environment for the Vangja research loop. Read
`research_agent/README.md`, `research_agent/plan/{00_GOAL_AND_CONSTRAINTS,02_ARCHITECTURE,03_AGENTS,05_EXTENSIONS_AND_TOOLS}.md`
and the repository `AGENTS.md` first. Then locate the pi installation
(`readlink -f "$(command -v pi)"` → its package root) and read its
`docs/{configuration,settings,extensions,tui,cli,json}.md` plus
`examples/extensions/{hello,permission-gate,protected-paths,todo,subagent/index,subagent/agents}.ts`.
Everything you build lives **at the repo root under `.pi/`**; the plan and
prompts stay in `research_agent/`.

## Deliverables

1. `.pi/settings.json`
   - `defaultProvider: "opencode-go"`, `defaultModel: "deepseek-v4.1-flash"`,
     `defaultThinkingLevel: "max"`,
     `enabledModels: ["opencode-go/deepseek-v4.1-flash"]`,
     `modelThinkingLevels: {"opencode-go/deepseek-v4.1-flash": "max"}`.
   - `defaultTools: ["+codemode"]`, `disableModelInvocation` left alone.
   - Compaction: `compaction.reserveTokens: 32768`,
     `compaction.keepRecentTokens: 120000`.
   - Do **not** disable builtins or add packages; resource discovery is
     conventional (`.pi/{skills,prompts,extensions}`).
   - `$schema: "https://pi.dev/schemas/settings.schema.json"`.
2. `.pi/config/`
   - `model.json` — the canonical pin (provider, model, thinking, contextWindow
     1000000, `enforce: true`).
   - `agents.json` — manifest: for each agent a record
     `{name, file, role, tools[], skills[], effort, maxParallel, model}`; start
     with the full roster from `plan/03_AGENTS.md` (§1 primary, §2 workers); files
     themselves arrive in stage S3.
   - `budgets.json` — exactly the defaults in `plan/02_ARCHITECTURE.md` §6.
   - `project.json` — repo root, base branch `update_pymc`, venue default
     `Journal of Forecasting`, `requireHumanMerge: true`, `dryRun: false`,
     sandbox path, session dir, artifact dir.
   - `hpc.example.json` — the schema in `plan/06_HPC_AND_COLLECTOR.md` §1, with
     `host: ""`, `maxParallelJobs: 1`, `wallClockHours: 24`.
   - `README.md` documenting every field and which files are versioned vs
     runtime-only.
3. `.pi/extensions/guards/index.ts`
   - Model/thinking pin on `session_start` and `before_agent_start`: block the run
     with an explicit error naming `.pi/config/model.json` if either drifts.
   - Inject a 5-line role preamble (role from `PI_LOOP_ROLE`, current state, task,
     "read your brief first", "no claim without an artifact ID").
   - Protected-path guard on `write`/`edit` tool calls:
     `Vangja-project/**`, `research_agent/plan/**`, `fixed_case_studies/protocol/**`,
     `paper/**` unless `PI_LOOP_STATE=PAPER`, `AGENTS.md` unless
     `PI_LOOP_ROLE=supervisor`, direct `.git/**` writes, `state/**` (ledger tools
     only). Denials must state the allowed alternative.
   - Destructive-bash guard: `git push --force`, `git filter-branch`, rebase on
     shared branches, `reset --hard` with uncommitted changes, `rm -rf` outside
     `.pi/state` and `.pi/artifacts`, writes to `~/.ssh`, `curl | sh`.
   - Secret scan on write/edit and on `bash` output: block/redact PEM keys,
     `.env`, `auth.json`, `hpc.json`, bearer tokens; never echo key material.
   - Role gates: `merge_pr` requires `supervisor`; `hpc_submit`/`hpc_cancel`/
     `hpc_requeue` require the matching role; `state_set` requires `conductor`.
4. `.pi/extensions/ledger/index.ts` — the tools and kinds exactly as in
   `plan/05_EXTENSIONS_AND_TOOLS.md` §3 (`ledger_read`, `ledger_append`,
   `task_enqueue`, `task_close`, `pr_open`, `review_write`, `experiment_queue`,
   `experiment_update`, `decision_record`, `waiver_record`, `amendment_record`,
   `state_get`, `state_set`). JSONL files under `.pi/state/`; every append under a
   lockfile; write-temp+fsync+rename for `LOOP_STATE.json`; schema validation;
   derived `*.md` views via a deterministic serializer; bounded reads.
5. `.pi/extensions/status/index.ts` — status-line widget: state, task, active
   experiment + attempt, budget, next gate. Also a `renderResult` for ledger
   tools so writes are visible.
6. `.pi/tests/` — a runner (`pnpm`/`bun`/`node` + `tsx`, or plain `pi --mode json`
   child scripts — pick the smallest thing that works with what is installed) and
   tests: settings load, schemas validate, ledger round-trip + concurrency,
   guard blocks (wrong model, protected path, secret, force-push), role matrix.
   Add `make -C .pi verify-env` (a small Makefile) that runs them plus
   `pi --print --model opencode-go/deepseek-v4.1-flash --thinking max
   "/loop:verify-env"` once the command exists in S4 — stub with a clear
   "not implemented until S4" failure until then.
7. `.pi/loop/` — `step.sh`, `status.sh`, `collect.sh` as thin shims that exec the
   orchestrator CLI with `--headless` (stub messages until S4/S5), and a
   `.pi/loop/README.md`.
8. `.pi/README.md` — what the environment is, how to verify it, what is state.
9. `.gitignore` additions: `.pi/state/`, `.pi/artifacts/*/`,
   `.pi/config/hpc.json`, `.pi/**/node_modules/`, `.pi/loop/worktrees/`,
   `.pi/sessions/`, `*.sif`, `*.pdf` outside `paper/`.

## Requirements

- TypeScript extensions use `jiti`, so no build step; import types from
  `@earendil-works/pi-coding-agent` (host-provided — `peerDependencies` only).
- Every tool result has model-facing `content` and a `details` object; truncate
  large outputs and say where the full file is.
- All errors are explicit and actionable, and name the config file or path
  involved. No silent fallbacks.
- Keep each file under ~400 lines; split modules as needed.
- No network calls in this stage.

## Acceptance tests (run, then paste raw output into `research_agent/REPORT_BUILD.md`)

1. `make -C .pi verify-env` → all green.
2. Wrong model: launch a child `pi` with `--model opencode-go/gpt-5.6-luna` and a
   trivial prompt; the guard blocks and the message names `model.json`.
3. Protected path: an agent-role child tries to `write` `Vangja-project/x.md`;
   refused. Then `paper/paper.tex` with `PI_LOOP_STATE=WORK`; refused.
4. Ledger: 200 concurrent appends from 4 processes; line count = 200, JSON valid,
   no torn lines; `LOOP_STATE.json` survives 100 concurrent read/writes.
5. Secret: a fixture file with a PEM block cannot be written/committed.
6. Role gates: `merge_pr` as `phd_candidate` and `hpc_submit` as `collector` are
   refused with the matrix reason.
7. `pi --print` in the repo loads the environment with zero warnings
   (paste the startup diagnostics).

## Do not

- Do not implement skills, agents, the orchestrator state machine, or HPC yet
  (S1–S6) — only the interfaces and stubs those stages need.
- Do not modify anything outside `.pi/**`, `research_agent/**`, `.gitignore`.
- Do not install global npm packages, run `pi install`, or push to a remote.

## Final report

Append to `research_agent/REPORT_BUILD.md`:

```
## S0 — scaffold and guards
Created: <paths>
Acceptance: <test> → <observed result>
Deviations from plan/08 §S0: <none | list with reason>
Open items for later stages: <list>
```
