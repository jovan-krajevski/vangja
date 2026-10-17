# .pi — the Vangja research-agent environment

Everything the loop runs lives here and is versioned, except runtime state under
`.pi/state/` and collected artifacts under `.pi/artifacts/` (both gitignored).
`research_agent/` holds the plan and the build prompts; this directory is the
implementation.

## Layout

| Path | What it is | Versioned |
|---|---|---|
| `settings.json` | pi project settings: model pin, thinking level, `enabledModels`, `+codemode`, compaction | yes |
| `config/model.json` | **canonical model pin** (provider, model, thinking, 1M window, `enforce`) | yes |
| `config/agents.json` | agent manifest: role, file, tools, skills, effort, maxParallel, model | yes |
| `config/tools.json` | tool registry: concrete tools, families, stage; agent tools are validated against it | yes |
| `config/budgets.json` | budget caps (exact `plan/02_ARCHITECTURE.md` §6 defaults) | yes |
| `config/project.json` | repo root, base branch, venue, human-merge gate, sandbox/session/artifact dirs | yes |
| `config/hpc.example.json` | HPC schema with placeholders (copy to `hpc.json` to enable) | yes |
| `config/hpc.json` | your real HPC host/keys | **no** (gitignored) |
| `lib/` | shared, dependency-free TypeScript: config loading, locked JSONL IO, record schemas, roles, secrets, state | yes |
| `extensions/guards/` | model pin, protected paths, destructive bash, secrets, role gates, role preamble | yes |
| `extensions/ledger/` | append-only ledger tools + derived `*.md` views | yes |
| `extensions/dispatch/` | isolated worker sessions (`dispatch_agent`, `dispatch_parallel` ≤4, `dispatch_chain`); provenance in `extensions/README.md` | yes |
| `extensions/orchestrator/` | conductor: state machine, CLI, gates, git/PR pipeline, budget, sandbox, `merge_pr`, AUDIT/DECIDE wiring | yes |
| `extensions/verify/` | S6 deterministic verification: tests, artifact hashes, reproduction with numeric diff, traceability, guards, `run_gates` | yes |
| `extensions/rubric/` | S6 rubric trees: load/validate/autofill/score/render | yes |
| `extensions/auditkit/` | S6 finding schema, dedupe register, report assembly, grade/decision, seeded canary, calibration scorer | yes |
| `extensions/status/` | TUI status line/widget + ledger `renderResult` | yes |
| `loop/` | headless shims for the orchestrator (`step.sh`, `status.sh`, `collect.sh`) | yes |
| `tests/` | offline `node --test` suite, fixtures, acceptance scripts | yes |
| `state/` | `LOOP_STATE.json`, `*.jsonl` ledgers, derived `*.md` views, sessions, worktrees | **no** (runtime) |
| `artifacts/` | collected experiment outputs + manifests | **no** (runtime; manifests are the committed part once S5 lands — see open items) |
| `agents/`, `prompts/` | 16 agent definitions and 11 argument-driven slash-command templates (S3) | yes |
| `tests/agents/` | agent/roster/tool-registry tests plus the offline dispatch engine tests and the S3 acceptance script | yes |
| `rubrics/` | five rubric trees + `SCHEMA.md` (`pr_review`, `readiness`, `paper`, `audit`, `decision`) | yes |
| `audit/` | S6 audit templates + the frozen calibration expectation set and the real historical run | yes |
| `calibrate.sh` | calibration runner: fixture mode and the read-only historical-snapshot finder run | yes |
| `skills/` | six research & methodology skills (S1): manifests, protocol, Bayesian workflow, evaluation, anti-gaming, provenance | yes |

## Verify

```bash
make -C .pi test           # unit + integration suite (offline, ~10s)
make -C .pi skills-test    # validate .pi/skills + python skill-script tests
make -C .pi env-check      # config/settings/roster/secrets checks with PASS lines
make -C .pi verify-env     # node + python skill suite + env checks + orchestrator checks
make -C .pi agents-accept  # S3 model-backed acceptance (dispatch a scout, role refusals)
make -C .pi hpc-accept     # S5 mock-cluster acceptance transcript
make -C .pi audit-accept   # S6 audit acceptance transcript (offline fixtures)
make -C .pi audit-calibrate # score the committed strong calibration fixture
```

`verify-env` is green: 277 node tests, 156 python tests, 15/15 environment
checks, 4/4 orchestrator checks. `verify-env-strict` is an alias.

## Audit harness (S6)

```bash
# deterministic gates for cycle N (writes .pi/audit/cycle-N/gates.json)
node .pi/extensions/orchestrator/cli.ts audit-gates N

# five rubric trees validate with weights summing to 1
node --test .pi/tests/audit/test_rubrics.test.ts

# seeded-defect canary: 8/8 S0/S1, no negative-control false positive
node --experimental-strip-types .pi/tests/audit/acceptance_probe.ts grade <fixture-copy>

# calibration: fixture modes are offline; --run-finders uses the pinned model
.pi/calibrate.sh --fixture .pi/tests/audit/fixtures/calibration/strong.json --out /tmp/cal
.pi/calibrate.sh --weakened --out /tmp/cal-weak        # exit 10 + UNCALIBRATED
.pi/calibrate.sh --run-finders --narrowed --out /tmp/cal-real

# apply rubrics/decision.json to a completed cycle (writes .pi/state/DECISION.json)
node .pi/extensions/orchestrator/cli.ts audit-decide N
```

The real calibration against `main@8028c51` is archived under
`.pi/audit/calibration/historical-8028c51/`; it is **UNCALIBRATED** (recall 6/7
but one confirmed rejected lead, F-015), so the conductor refuses to grade until
the M brief is narrowed and S re-derives `WS-S-003`.

The model pin is checked from both sides:

```bash
pi --print --no-session -na -e .pi/extensions/guards/index.ts \
   --provider opencode-go --model opencode-go/gpt-5.6-luna --thinking max "hi"
# exit 3, no model output, stderr names .pi/config/model.json
```

## What is state

Runtime state is **only** in `.pi/state/`:

- `LOOP_STATE.json` — current state (validated against `plan/02_ARCHITECTURE.md` §2),
  current task/PR/experiment, attempt, next gate; written atomically.
- `EVENTS.jsonl`, `TASKS.jsonl`, `PRS.jsonl`, `REVIEWS.jsonl`,
  `EXPERIMENTS.jsonl`, `FINDINGS.jsonl`, `DECISIONS.jsonl`, `WAIVERS.jsonl`,
  `AMENDMENTS.jsonl`, `BUDGET.jsonl`, `GATES.jsonl` — append-only ledgers,
  written under a lock directory and fsynced.
- `CONTEXT.md`, `TASKS.md`, `PRS.md`, `REVIEWS.md`, `EXPERIMENTS.md`,
  `DECISIONS.md` — generated views; never edit them.
- `sessions/`, `worktrees/`, `STOP` — sessions for replay, git worktrees, and
  the hard stop flag.

Updates repeat the record `id`; the current value of an entity is the record
with the highest `seq` for that id. Nothing rewrites history.

## Security notes

- A human session without `PI_LOOP_ROLE` is treated as role `human`: everything
  is readable, but `AGENTS.md` and the role-gated tools stay out of reach. To
  edit `AGENTS.md` from a human session, start it with
  `PI_LOOP_ROLE=supervisor pi`.
- Every session is pinned to `opencode-go/deepseek-v4.1-flash`, thinking `max`.
  The guard drops input, blocks tools, and exits 3 on drift; the message names
  `.pi/config/model.json`.
- Protected paths: `Vangja-project/**`, `research_agent/plan/**`,
  `fixed_case_studies/protocol/**` (amendment only), `paper/**` (PAPER state
  only), `AGENTS.md` (supervisor/conductor), `.git/**`, `.pi/state/**`.
- Secrets: writes matching `.env`, `auth.json`, `hpc.json`, `*.pem`, `id_*` are
  refused, and the same rule applies to shell write targets and to
  `git add`/`git commit` (a staged secret file is refused; the guard reads the
  index, read-only); PEM blocks, bearer tokens, AWS/GitHub/OpenAI keys in write
  content or bash command text are refused; bash output is redacted before it
  reaches the model. Key material is never echoed back.
- Bash: force pushes, history rewrites, shared-branch rebases, `reset --hard`
  with uncommitted work, `rm -rf` outside `.pi/state`/`.pi/artifacts`,
  `~/.ssh` writes, `curl | sh`, and environment exfiltration are refused.
- Role gates: `merge_pr` needs supervisor/conductor, `hpc_submit`
  phd_candidate/conductor, `hpc_requeue` collector, `state_set` conductor,
  `review_write`/`waiver_record`/`amendment_record` supervisor, and so on —
  the matrix is `plan/05_EXTENSIONS_AND_TOOLS.md` §7 and is enforced twice (guard
  on tool call, handler on execution).

## Open items for later stages

- `make verify-env` prints `STUB (S4)` in place of the `/loop:verify-env` child
  check (see above).
- `.pi/artifacts/*/` is gitignored wholesale per the S0 spec; the plan wants
  manifests committed. S5 must either narrow the ignore rule or write manifests
  to a versioned path (`.pi/artifacts/manifests/`).
- `paper_*` tools are declared `planned` in `.pi/config/tools.json`; S7
  registers them. `verify`, `rubric`, and `auditkit` tools are implemented.
- `tests/**` protection (strengthen-only) is implemented in the pre-merge scan
  and the audit `test-diff` gate, not in the write guard.
- The audit's real calibration run is UNCALIBRATED by design: see
  `.pi/audit/calibration/historical-8028c51/REPORT.md` for the two narrowings
  required before a graded cycle.
