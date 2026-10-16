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
| `config/budgets.json` | budget caps (exact `plan/02_ARCHITECTURE.md` §6 defaults) | yes |
| `config/project.json` | repo root, base branch, venue, human-merge gate, sandbox/session/artifact dirs | yes |
| `config/hpc.example.json` | HPC schema with placeholders (copy to `hpc.json` to enable) | yes |
| `config/hpc.json` | your real HPC host/keys | **no** (gitignored) |
| `lib/` | shared, dependency-free TypeScript: config loading, locked JSONL IO, record schemas, roles, secrets, state | yes |
| `extensions/guards/` | model pin, protected paths, destructive bash, secrets, role gates, role preamble | yes |
| `extensions/ledger/` | append-only ledger tools + derived `*.md` views | yes |
| `extensions/status/` | TUI status line/widget + ledger `renderResult` | yes |
| `loop/` | headless shims for the orchestrator (`step.sh`, `status.sh`, `collect.sh`) | yes |
| `tests/` | offline `node --test` suite, fixtures, acceptance scripts | yes |
| `state/` | `LOOP_STATE.json`, `*.jsonl` ledgers, derived `*.md` views, sessions, worktrees | **no** (runtime) |
| `artifacts/` | collected experiment outputs + manifests | **no** (runtime; manifests are the committed part once S5 lands — see open items) |
| `agents/`, `prompts/`, `rubrics/`, `audit/` | arriving in S2–S6 | yes |
| `skills/` | six research & methodology skills (S1): manifests, protocol, Bayesian workflow, evaluation, anti-gaming, provenance | yes |

## Verify

```bash
make -C .pi test           # unit + integration suite (offline, ~10s)
make -C .pi skills-test    # validate .pi/skills + python skill-script tests
make -C .pi env-check      # config/settings/roster/secrets checks with PASS lines
make -C .pi verify-env     # all three, plus an explicit STUB banner for the S4 command
make -C .pi verify-env-strict  # same, but fails until /loop:verify-env exists (S4)
```

`verify-env` is green in S0. It prints `STUB (S4)` for the
`pi --print "/loop:verify-env"` check, because the orchestrator does not exist
yet; `verify-env-strict` turns that stub into the failure CI should see once the
orchestrator lands.

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
- `pr_open` validates the record shape only; the clean-worktree-at-head-SHA check
  belongs to the S4 orchestrator gate.
- `tests/**` protection (strengthen-only) is implemented in the S6 pre-merge
  scan, not in the write guard.
