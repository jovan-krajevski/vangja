# `.pi/config` — every field, and what is versioned

All loaders live in `.pi/lib/config.ts`. Every loader throws `CONFIG_INVALID`
naming the file when a field is missing or malformed; nothing is defaulted
silently. `model.json`, `budgets.json`, `project.json`, `agents.json`, and
`hpc.example.json` are **versioned**. `hpc.json` is **runtime-only**
(gitignored) because it names hosts and key paths.

---

## `model.json` — the canonical pin (hard requirement)

| Field | Type | Meaning |
|---|---|---|
| `provider` | string | Provider id passed to `--provider`. |
| `model` | string | Model id passed with `--model` as `provider/model`. |
| `thinking` | `off\|minimal\|low\|medium\|high\|xhigh\|max` | Thinking level passed to `--thinking`. |
| `contextWindow` | number | Declared window; must be `1000000` while `enforce` is true. |
| `enforce` | boolean | When true the guard blocks any session that drifts. |

The guard extension compares the *active* session against this file on
`session_start`, `model_select`, `thinking_level_select`, and
`before_agent_start`, and also drops input and blocks tool calls. A missing or
malformed file fails closed with a message naming `model.json`.

## `budgets.json` — caps (exact `plan/02_ARCHITECTURE.md` §6)

| Field | Default | Meaning |
|---|---|---|
| `maxSessionsPerDay` | 120 | Child `pi` sessions started per day. |
| `maxParallelWorkers` | 4 | Dispatched workers in flight. |
| `maxParallelHpcJobs` | 1 | **Hard single-slot rule**; raising it needs an amendment. |
| `maxTaskWallClockMinutes` | 360 | Per-task wall clock. |
| `maxHpcWallClockHours` | 48 | Per-experiment HPC wall clock (the job itself is 24h). |
| `maxConsecutiveFailures` | 3 | Circuit breaker. |
| `maxNoProgressCycles` | 2 | Circuit breaker. |
| `maxAuditCycles` | 5 | Audit loop cap. |
| `maxCostUsdPerRun` | 250 | Cost cap; exceeding it pauses the loop. |

Exceeding a cap must set `PAUSED` with a human-readable reason (S4).

## `project.json` — project policy

| Field | Meaning |
|---|---|
| `schema` | `pi-project/1`. |
| `repoRoot` | `"."` = the directory containing `.pi/`; relative paths resolve from it. |
| `baseBranch` | The only branch the supervisor/conductor merges into (`update_pymc`). |
| `venue` | Default venue for the paper rubric (no deadline by default). |
| `requireHumanMerge` | `true` pauses before every merge into the base branch. |
| `dryRun` | `true` logs intended actions without side effects (sandbox runs). |
| `sandboxPath` | Scratch clone used by `--sandbox` tests/dry runs. |
| `sessionDir` | Where child sessions are persisted for replay. |
| `artifactDir` | Collected experiment outputs + manifests. |
| `stateDir` | Ledgers, `LOOP_STATE.json`, derived views. |
| `worktreeDir` | Per-task git worktrees. |

## `agents.json` — the roster

| Field | Meaning |
|---|---|
| `schema` | `pi-agents/1`. |
| `pinnedModel` | Must equal `provider/model` from `model.json`. |
| `pinnedThinking` | Must equal `thinking` from `model.json`. |
| `agents[]` | One entry per agent (see below). |

Each agent entry:

| Field | Meaning |
|---|---|
| `name` | Agent name; unique; matches the frontmatter `name` in `file`. |
| `file` | Markdown definition (`.pi/agents/<name>.md`), created in S3. |
| `role` | One of `LOOP_ROLES` in `.pi/lib/roles.ts`; drives the guard matrix. |
| `tools` | Tool names/groups the agent may use (groups `dispatch`, `ledger`, `hpc`, `verify`, `task`, `paper`, `rubric`, `auditkit`, `merge` are resolved according to `plan/05_EXTENSIONS_AND_TOOLS.md`). |
| `skills` | Skill directory names under `.pi/skills/` (S1–S2). |
| `effort` | `small \| medium \| large`; the effort class from `plan/03_AGENTS.md` §4. |
| `maxParallel` | Maximum concurrent instances of this agent. |
| `model` | Always the pinned model; frontmatter may not override it. |

The roster is `plan/03_AGENTS.md`: three primary agents (`phd_candidate`,
`collector`, `supervisor`), the four audit phases (`audit_finder`,
`audit_verifier`, `audit_synthesizer`, `audit_adjudicator`), and nine workers
(`scout`, `planner`, `methodologist`, `experimenter`, `repro_engineer`,
`literature_scout`, `paper_writer`, `critic`, `meta_review`). The `audit` primary
in the plan is realised as the four phase agents; there is no separate `audit`
definition.

## `hpc.example.json` — HPC schema (versioned; placeholders only)

Copy to `hpc.json` (gitignored) to enable the HPC path. If `host` is empty the
whole HPC path is disabled: `hpc_*` tools return `enabled:false`, the collector
exits with an `hpc-disabled` event, and agents never attempt SSH.

| Field | Meaning |
|---|---|
| `host`, `user`, `port` | SSH target. Empty `host` = disabled. |
| `remoteRepoDir`, `remoteScratchDir` | Remote checkout and scratch roots. |
| `hpcKey` | Private key for the cluster (`~/.ssh/...`). |
| `gitKey` | Key added to a private `ssh-agent` for `-A` forwarding. |
| `forwardGitAgent` | Whether to attempt agent forwarding (`git fetch` on the cluster). |
| `container` | Optional `.sif` path; its SHA-256 is recorded in job records. |
| `partition`, `cpus`, `memGb`, `gpus` | SLURM resources. |
| `wallClockHours` | Must stay ≤ 24 for the job wall limit. |
| `pollSeconds`, `maxPolls` | Collector poll cadence and escalation threshold. |
| `maxAttemptsPerExperiment` | Requeue cap (default 6). |
| `maxParallelJobs` | **Must be 1**; `>1` is rejected unless a recorded amendment exists. |

## Runtime-only files (never committed)

| Path | Why |
|---|---|
| `.pi/config/hpc.json` | Hosts and key paths. |
| `.pi/state/**` | Ledgers, state, sessions, worktrees. |
| `.pi/artifacts/*/**` | Collected outputs (S5 decides how manifests are versioned). |
