# 02 — Architecture: the loop, the gates, the invariants

> Convention: every path written `state/…`, `artifacts/…`, `audit/…`,
> `config/…`, or `agents/…` in this plan is relative to the repository-root
> `.pi/` directory (e.g. `state/LOOP_STATE.json` = `.pi/state/LOOP_STATE.json`).

## 1. Layers

```
            human (reviewer of this plan; gatekeeper at 4 points)
              │  /loop:start | /loop:pause | /loop:resume | /loop:status
              ▼
  conductor  (deterministic TypeScript, no LLM): state machine, git, budgets,
             gates, integrity checks, child-process invocation, ledgers
              │  spawns fresh `pi` processes (one per work unit)
              ▼
  primary agents:  phd_candidate → collector → supervisor → audit
  worker agents:   scout, planner, methodologist, experimenter, repro_engineer,
                   literature_scout, paper_writer, critic, audit_finder,
                   audit_verifier, audit_synthesizer, audit_adjudicator
              │
              ▼
  durable state:   state/*.jsonl, state/tasks|prs|reviews/, sessions/, artifacts/
```

The conductor is not an agent. It never interprets science, never writes prose,
and never decides "is this good enough" beyond running the rubrics. Every such
judgement is an agents' job with a rubric and a written output.

## 2. State machine

States and transitions. `[G]` = human gate possible, `[H]` = HPC involved.

```
BOOT ──▶ SELECT_TASK ──▶ WORK[H] ──▶ REVIEW ──accept──▶ MERGE[G] ──▶ POST_MERGE
           ▲                 ▲            │ changes_requested ──▶ WORK (same task)
           │                 │            │ rejected ──▶ SELECT_TASK (task closed)
           │                 └────────────┘
           │
           ├── (all readiness criteria green) ──▶ READINESS ──fail──▶ SELECT_TASK
           │                                          │ pass
           │                                          ▼
           │                                        PAPER ──▶ PAPER_REVIEW ──accept──▶ MERGE[G]
           │                                                                              │
           │                                                    AUDIT ◀──────────────────┘
           │                                                      │
           └──── more work needed ◀── DECIDE ◀───────────────────┘
                                      │ finish
                                      ▼
                                    DONE (human sign-off)
   any state ──▶ PAUSED (manual, budget, or gate) ──▶ resume to same state
   any state ──▶ BLOCKED (missing credential/data/decision) ──▶ human resolves
```

- `BOOT`: verify environment (guard active, model pinned, resources load, repo
  clean, expected base commit), create `run_id`, snapshot `update_pymc` SHA.
- `SELECT_TASK`: `phd_candidate` runs its `planner` worker over the backlog —
  open audit S0/S1 findings first, then `Vangja-project/REVIEW/PLAN.md` P0/P1
  items, then protocol gaps, then paper-evidence gaps. Emits `state/tasks/T-###.md`.
- `WORK`: implement in a worktree, run the narrowest tests, commit atomically,
  write the PR record. If compute is needed: write experiment records and submit
  (→ `WAITING_HPC`, a sub-state of `WORK`).
- `REVIEW`: supervisor runs the PR rubric + verification commands, writes
  `state/reviews/RV-###.md`, decides accept / changes / reject.
- `MERGE`: conductor performs `git merge --no-ff` into `update_pymc`, tags
  `run/<run_id>/merge/<n>`; human gate by default.
- `POST_MERGE`: status bookkeeping — close the finding/plan item, update
  `state/CONTEXT.md`, `CHANGELOG.md`, and the task index.
- `READINESS`: supervisor evaluates the readiness rubric (§5.3). Pass → `PAPER`.
- `PAPER`: `phd_candidate` + `paper_writer` revise `paper/`, regenerate all
  tables/figures/numbers through `paper_build`, update `paper/CLAIMS.json`.
- `PAPER_REVIEW`: supervisor uses the paper rubric (claims backed by artifacts,
  scope honesty, limitations, AI-use disclosure, venue fit).
- `AUDIT`: the full audit pipeline (`plan/07_AUDIT_HARNESS.md`), output grade +
  findings + fixes.
- `DECIDE`: supervisor applies the decision rule (§5.4).
- `PAUSED` / `BLOCKED` / `DONE`: terminal or interrupt states.

Every transition appends one line to `state/EVENTS.jsonl`
(`ts, run_id, from, to, actor, task_id, pr_id, summary, session_path, usage`)
and rewrites `state/LOOP_STATE.json` atomically (temp file + `rename`). Steps are
idempotent by completion event: the conductor never repeats a completed step.

## 3. Work units: task briefs, PRs, reviews

**Task brief** (`state/tasks/T-###.md`, template in `.pi/prompts/` and enforced by
a schema check):

```
id, title, origin (finding ID | PLAN item | gap), priority (P0..P3),
objective (one paragraph), acceptance criteria (checkable list),
deliverables (files/artifacts), out of scope, evidence required,
compute (none | local | HPC with size), budget (wall clock, sessions),
dependencies, protected paths touched.
```

Briefs are the subagent contract from the research notes: objective, output
format, tools, boundaries. A brief without acceptance criteria is rejected by the
conductor before any work starts.

**PR record** (`state/prs/PR-###.md` + real GitHub PR when `gh` is available):
summary, task ID, head SHA, base SHA, evidence (commands + observed output),
artifacts produced (paths + hashes), tests added, risks/uncertainty, claims
changed, checklist (protocol/guards), and a machine-readable footer:

```json
{"task_id":"T-012","head_sha":"…","base_sha":"…","artifacts":["artifacts/…/manifest.json"],
 "tests":["tests/…"],"claims":[],"hpc_jobs":["E-004"]}
```

**Review** (`state/reviews/RV-###.md`): decision, rubric scores, findings with
file:line evidence, required changes, verification commands the supervisor
actually ran, and — for accept — a merge recommendation with residual risk. The
supervisor must run at least the acceptance-criteria commands itself; reading the
PR body is not verification.

## 4. Effort scaling (from Anthropic's practice)

| Work unit | Default effort | Parallel workers |
|---|---|---|
| Documentation/typo/small fix | 1 session, ≤15 tool calls | — |
| Code fix with tests | 1 session + 1 critic, ≤60 calls | 2 |
| Experiment design + run + analysis | planner + experimenter + methodologist + critic | 3 |
| Paper section | writer + literature_scout + critic | 3 |
| Audit workstream | 1 finder session, ≤120 calls | up to 8 in parallel |
| Verification pass | 1 session per affected finding batch | up to 4 |
| Synthesis/adjudication | 1 session each | 1 |

The conductor caps tool calls indirectly via wall-clock and session budgets; the
briefs state the effort class so agents self-limit.

## 5. Gates

### 5.1 Pre-merge (conductor, deterministic — all must pass)

1. `uv run pytest` green, or failures explicitly waived by the supervisor with a
   recorded reason (`state/waivers/W-###.md`) that names the failing test and why
   it does not invalidate the merge.
2. Protected-path diff scan clean: `Vangja-project/**`,
   `fixed_case_studies/protocol/**`, `research_agent/plan/**`, `AGENTS.md`
   (supervisor-only), `paper/**` (PAPER state only), `.pi/**`
   (config changes need an explicit PR + review).
3. Test-diff scan: tests added or strengthened only. Deletions/weakenings
   (removed assertions, relaxed tolerances, skipped tests, changed seeds in
   frozen tests) are flagged and require a waiver.
4. Integrity scan: no selection on test metrics; no changes to splits, metric
   definitions, exclusion rules, or frozen configs without a protocol amendment
   (`state/amendments/A-###.md`, supervisor-accepted, human-visible).
5. Provenance: every new artifact has a manifest with commit SHA, seeds, versions,
   data hashes; no result files outside `artifacts/` or `results/`.
6. Hygiene: no secrets, no data dumps, `.gitignore` respected, no
   `git push --force`, no history rewrite.
7. Claims: any number added to prose/paper has a `CLAIMS.json` entry pointing at
   an artifact ID.

### 5.2 HPC gate (single-slot, checkpoint-mandatory)

Before submitting: the experiment record names the SHA, the exact command, the
seeds, expected outputs, the 24h wall limit, a data hash, a checkpoint directory,
a `--resume` command, and a *local smoke test proving resume works*. **At most
one job may be active at a time**; a submission is refused while any other job is
`queued|running|collecting|requeue_requested`. A timed-out or preempted job is
requeued only after its checkpoints are fetched and its progress is recorded (see
`plan/06_HPC_AND_COLLECTOR.md`). Local/HPC runs must be reproducible from the
ledger alone.

### 5.3 Readiness rubric (`rubrics/readiness.json`) — before paper writing

- All S0/S1 findings closed or explicitly waived with evidence.
- All P0/P1 `REVIEW/PLAN.md` items closed.
- Confirmation runs complete with all `PROTOCOL.md` §11 artifacts; hash of frozen
  origins verified.
- Baselines run on identical data; context excluded from target aggregates.
- Uncertainty: calibration analysis done; MAP-heuristic disclosure ready.
- Reproducibility: `make verify` (paper build) passes on the artifacts present.
- Environment pinned (`uv.lock` + `pyproject.toml`), CI-equivalent test run green.
- Paper-relevant negative/limitation findings documented.

### 5.4 Audit decision rule (`rubrics/decision.json`) — after audit

| Grade | Action |
|---|---|
| strong reject, reject | reopen P0 tasks from the findings; new audit cycle after fixes |
| weak reject, borderline | fix all S0/S1 and the S2s named as blocking; re-audit |
| weak accept | fix the "feasible fixes" list (all S0/S1 + blocking S2); re-audit unless the fixes are purely editorial |
| accept, strong accept | if the report lists no blocking fixes → `DONE` (human sign-off); otherwise fix and re-audit |
| anything else | `BLOCKED` (audit failed to produce a usable grade — human decides) |

Max 5 audit cycles, then `BLOCKED` for human direction regardless of grade.

## 6. Budgets (`.pi/config/budgets.json`, defaults)

```json
{
  "maxSessionsPerDay": 120,
  "maxParallelWorkers": 4,
  "maxParallelHpcJobs": 1,
  "maxTaskWallClockMinutes": 360,
  "maxHpcWallClockHours": 48,
  "maxConsecutiveFailures": 3,
  "maxNoProgressCycles": 2,
  "maxAuditCycles": 5,
  "maxCostUsdPerRun": 250
}
```

Accounting from child-process usage events → `state/BUDGET.json`. Exceeding a cap
sets `PAUSED` with a human-readable reason and the exact resume command.

## 7. Model and context pinning (hard requirement)

- Canonical config `.pi/config/model.json`:
  `{"provider":"opencode-go","model":"deepseek-v4.1-flash","thinking":"max","contextWindow":1000000}`.
- The conductor always invokes children with
  `--provider opencode-go --model opencode-go/deepseek-v4.1-flash --thinking max`
  (the `--provider` flag also blocks accidental fuzzy model matches).
- `.pi/settings.json` pins `defaultProvider`, `defaultModel`,
  `defaultThinkingLevel: "max"`, `enabledModels: ["opencode-go/deepseek-v4.1-flash"]`,
  `modelThinkingLevels: {"opencode-go/deepseek-v4.1-flash": "max"}`.
- Guard extension on `session_start`/`before_agent_start`: if the active model id
  or thinking level differs, it blocks the session with an explicit error naming
  the config file. No agent frontmatter may name a different model.
- Context: 1M is the model's window — no overrides. Compaction stays enabled with
  a large kept history (`compaction.reserveTokens` 32768,
  `compaction.keepRecentTokens` 120000) so long audits do not lose their evidence
  chain; all long-lived state still lives in files.
- The environment check (`loop/status.sh --verify-env`) fails loudly if any of
  these drift.

## 8. Fresh context + memory protocol

- One `pi` process per work unit, `--session-dir state/sessions/<run_id>/<agent>/`,
  `--name <agent>__<task_id>__<attempt>`; sessions are persisted for replay but
  never continued as context.
- The only inputs an agent gets: its agent definition, the brief, the specific
  files the brief names, and the repo (read tools).
- The conductor writes `state/CONTEXT.md` (≤200 lines, deterministic, generated
  from ledgers: current state, open tasks, open findings, recent merges, budget).
  Agents read it first; humans read it instead of grep-ing jsonl.
- `state/DECISIONS.md` holds ADR-style decisions (dated, with alternatives and
  consequences). `state/HUMAN_GATES.json` holds pending human decisions.
