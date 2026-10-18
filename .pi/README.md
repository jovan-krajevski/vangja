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
| `extensions/understand/` | `/understand_loop` guided tour (12 chapters, one at a time), progress in `.pi/state/UNDERSTAND_LOOP.json`, TUI progress widget | yes |
| `loop/` | headless shims for the orchestrator (`step.sh`, `status.sh`, `collect.sh`) | yes |
| `tests/` | offline `node --test` suite, fixtures, acceptance scripts | yes |
| `tests/e2e/` | S7 end-to-end sandbox smoke test, resilience matrix, model-pin audit, calibration wrapper | yes |
| `state/` | `LOOP_STATE.json`, `*.jsonl` ledgers, derived `*.md` views, sessions, worktrees | **no** (runtime) |
| `artifacts/` | collected experiment outputs + manifests | **no** (runtime; manifests are the committed part once S5 lands — see open items) |
| `agents/`, `prompts/` | 16 agent definitions and 11 argument-driven slash-command templates (S3) | yes |
| `tests/agents/` | agent/roster/tool-registry tests plus the offline dispatch engine tests and the S3 acceptance script | yes |
| `rubrics/` | five rubric trees + `SCHEMA.md` (`pr_review`, `readiness`, `paper`, `audit`, `decision`) | yes |
| `audit/` | S6 audit templates + the frozen calibration expectation set and the real historical run | yes |
| `calibrate.sh` | calibration runner: fixture mode and the read-only historical-snapshot finder run | yes |
| `skills/` | 15 skills: the six S1 research & methodology skills (manifests, protocol, Bayesian workflow, evaluation, anti-gaming, provenance) plus nine S2 engineering/writing/review/process skills | yes |

## Verify

```bash
make -C .pi test           # unit + integration suite (offline, ~80s)
make -C .pi skills-test    # validate .pi/skills + python skill-script tests
make -C .pi env-check      # config/settings/roster/secrets checks with PASS lines
make -C .pi model-pin      # every recorded child argv used the pinned model + thinking
make -C .pi verify-env     # node + python skill suite + env checks + orchestrator checks + model pin
make -C .pi smoke          # end-to-end sandbox loop (BOOT → audit → decision; ~20s)
make -C .pi resilience     # kill/restart matrix, exactly-once assertions (~25s)
make -C .pi calibration    # offline audit-calibration modes + refusal paths
make -C .pi all            # verify-env + test + smoke
make -C .pi agents-accept  # S3 model-backed acceptance (dispatch a scout, role refusals)
make -C .pi hpc-accept     # S5 mock-cluster acceptance transcript
make -C .pi audit-accept   # S6 audit acceptance transcript (offline fixtures)
make -C .pi audit-calibrate # score the committed strong calibration fixture
```

`verify-env` is green: 304 node tests, 163 python tests, 15/15 environment
checks, 4/4 orchestrator checks, and 0 model-pin violations. `verify-env-strict`
is an alias. `smoke` and `resilience` are deterministic (a stub child replaces the
model) and never touch the real repository — they clone it under `.pi/state/`,
run with `--sandbox`, and assert the real `update_pymc` SHA and refs are
unchanged.

## End-to-end smoke test (S7)

`make -C .pi smoke` builds a scratch clone with a synthetic task ("add a
docstring and a passing test to a fixture module"), a mock experiment with a
checkpoint/resume contract, and the S6 seeded-defect audit tree, then drives the
real conductor through BOOT → WORK → REVIEW → MERGE → READINESS → PAPER →
PAPER_REVIEW → MERGE → AUDIT → DECIDE. It prints one `CHECK <name>: PASS|FAIL`
line per assertion:

- the transition sequence matches `plan/02_ARCHITECTURE.md` §2;
- exactly one work branch / PR / review / merge commit (plus exactly one paper PR
  and merge, the route into AUDIT);
- exactly one experiment record and one collected artifact with verified hashes;
- a tampered copy of the artifact is caught by the readiness check *before* PAPER
  and the loop returns to SELECT_TASK;
- the deterministic audit gates run before any finder, the seeded canary detects
  8/8 defects with no false positives, and DECIDE writes a grade with a blocking
  fix list;
- the sandbox `update_pymc` moved and the real one did not;
- every child invocation is recorded with argv, prompt hash, model, and usage.

The last run is archived (checks + transcript) under
`.pi/state/e2e-tests/smoke-<stamp>.json`; the resilience matrix and its kill
points are documented in `.pi/tests/e2e/resilience.md`.

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
- Protected paths: `Vangja-project/**`, `research_agent/plan/**` (read-only except
  a human S7 session editing a path named by a recorded S7 decision — the
  `planEditCovers` exception), `fixed_case_studies/protocol/**` (amendment only),
  `paper/**` (PAPER state only), `AGENTS.md` (supervisor/conductor), `.git/**`,
  `.pi/state/**`.
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

- `.pi/artifacts/*/` is gitignored wholesale per the S0 spec, so artifact
  *manifests* are not committed either. The S5 fetch writes
  `.pi/artifacts/<experiment_id>/manifest.json` (SHA-256 per file + sacct
  accounting); if the plan wants manifests versioned, narrow the ignore rule.
- `paper_build` and `paper_diff_summary` are still declared `planned` in
  `.pi/config/tools.json`; the PAPER phase in the S7 smoke run exercises the
  state machine and PR route but does not rebuild the manuscript.
- `tests/**` protection (strengthen-only) is implemented in the pre-merge scan
  and the audit `test-diff` gate, not in the write guard.
- The audit's real calibration run is UNCALIBRATED by design: see
  `.pi/audit/calibration/historical-8028c51/REPORT.md` for the two narrowings
  required before a graded cycle. The smoke test's sandbox supplies a synthetic
  calibrated result so the decision path can be exercised; the S6 harness owns
  the refusal-path tests.
- The S7 smoke run seeds the audit tree *after* the paper merge (the defects are
  the audit's target, not a PR's diff), and its paper PR touches `docs/` rather
  than `paper/**` so the uncited-number defect survives to the audit. A real
  paper revision runs the full claims pipeline instead.
- Runtime state under `.pi/state/**` (test sandboxes, worktrees, calibration
  snapshots) is excluded from the static integrity/protocol scanners as of S7;
  before that a kept sandbox could make the real-repo scan dirty.
