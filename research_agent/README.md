# research_agent — a senior research group for the Vangja paper

**This folder is a plan, not an implementation.** Nothing here executes yet. It
specifies (a) the multi-agent research environment we want pi to build, and (b)
the prompts that make pi build it. Review the plan, run the prompts, then run the
loop.

## The idea

Turn the Vangja project into a self-driving research group that works the way a
good empirical-science lab does:

- **`phd_candidate`** does the work — codes, experiments, analysis, PRs, and (at
  the end) the paper. Always on a fresh branch → PR into `update_pymc`. Always in
  a **brand-new context**; its memory is the repository, the task brief, and the
  ledgers on disk. Can drive an HPC over SSH/SLURM when credentials are provided.
- **`collector`** shepherds HPC jobs: reads the experiment ledger, watches
  `squeue`/`sacct`, fetches finished results, writes status back so the
  `phd_candidate` can continue. Never runs science of its own, never invents a
  result.
- **`supervisor`** is the gate: reviews every PR (accept / request changes /
  reject), merges accepted work into `update_pymc`, decides when the project is
  audit-ready, triggers paper writing, then triggers the audit, then decides
  continue vs finish from the audit grade.
- **`audit`** is the adversarial reviewer, modelled on the historical
  `Vangja-project/audit` process but stronger: independent workstream sessions
  that re-derive everything from source, a separate verification session that
  tries to refute each finding, a synthesis session, a deterministic
  reproduction gate, and a final conference-style grade with a fix list.

Everything is resumable: state lives in append-only ledgers plus one small state
file, so `pause` now / `resume` tomorrow works — including while SLURM jobs run.

**HPC rules are strict:** exactly **one job at a time** (no second submission
while one is queued, running, or being collected), a **24-hour wall limit**, and
**checkpoint-or-it-does-not-run** — every job must carry a checkpoint directory, a
`--resume` command, and a local smoke test proving resume works, and a timed-out
job is requeued only after its checkpoints are collected. If no host is
configured, agents never attempt SSH.

## Design principles (why it looks like this)

Distilled from how frontier-lab agent systems actually work (citations in
`plan/01_RESEARCH_NOTES.md`):

1. **Separate generation from verification.** The entity that produces a result
   never grades it (FunSearch/AlphaEvolve evaluators, OpenAI's prover/verifier
   protocol, PaperBench rubrics, our own historical WS-V pass).
2. **Orchestrator + fresh-context workers.** One lead agent decomposes and
   delegates; subagents get narrow briefs (objective, output schema, tools,
   boundaries) and their own context window (Anthropic's multi-agent research
   system).
3. **Verify mechanically wherever possible.** Here the "Lean checker" is:
   `pytest` + deterministic recomputation of every reported number from frozen
   artifacts + SHA-256 provenance + the frozen confirmation-origin protocol. The
   audit's strongest tool is *run the thing again*.
4. **Rubric decomposition + calibration of the judge.** PaperBench-style
   hierarchical rubrics; our historical 89 audit findings and 4 rejected leads
   become a golden calibration set for the audit agent.
5. **Population + selection, not one-shot answers.** Tasks produce candidate
   fixes/configs; a verifier keeps the winners (AlphaEvolve / AI-Scientist-v2
   tree search, applied to engineering choices rather than algorithms).
6. **Durable, artifact-based memory.** No hidden conversational state: task
   briefs, PR bodies, review files, experiment ledgers, audit findings, and
   decision records are the interface. Sessions are persisted so a human can
   replay any agent run.
7. **Human gates where irreversibility or public claims begin** (merge into
   `update_pymc`, confirmation runs, audit start, submission) — because the
   Navier-Stokes episode shows what happens when announcement outruns
   independent verification.
8. **Anti-gaming is a first-class threat model.** DeepMind found ~14% of AI
   agents "cheating" on scored tasks; our guardrails detect test weakening,
   selection on test data, seed shopping, failure trimming, and frozen-file edits.

## What the build creates

Everything pi-related is installed **at the repository root** (project `.pi/`),
not inside `research_agent/`. `research_agent/` holds only the plan and the build prompts. Project
instructions for any pi session live in the root `AGENTS.md` (the old
`.github/copilot-instructions.md` has been retired).

```
research_agent/
  README.md
  plan/              # this plan (read-only during the build)
  prompts/           # the build prompts, run in order

<vangja repo root>/
  AGENTS.md          # terse project instructions (already in place)
  .pi/               # LOCAL pi resources — auto-discovered after project trust
    settings.json    # model pin, resource paths, tool defaults
    config/          # agents.json manifest, model.json, budgets.json,
                     # project.json, hpc.example.json (hpc.json is gitignored)
    agents/          # phd_candidate.md, collector.md, supervisor.md, audit*.md, workers/*.md
    skills/          # one dir per skill, each with SKILL.md (+ scripts/templates)
    prompts/         # slash-command templates (/research:*, /audit:*, /loop:*)
    extensions/      # orchestrator, guards, ledger, dispatch, hpc, verify,
                     # rubric, auditkit, paper, status
    rubrics/         # readiness, PR-review, audit, and paper rubrics (JSON + MD)
    tests/           # validation suite for the pi environment
    loop/            # headless step runner + status (reads .pi/config/agents.json)
    state/           # LOOP_STATE.json, *jsonl ledgers, sessions/, worktrees/ (gitignored)
    artifacts/       # collected runs: bodies gitignored, manifests + hashes committed
    audit/           # audit cycles (committed)
```

## How to build it

**Status: S7 build complete on 2026-10-18 — GO** (self-verified in this
session; independent verification pending, as for S0–S6). The environment is
fully committed and self-contained: a fresh clone of the S7 commit runs
`make -C .pi verify-env` green with no local state. Evidence per stage is in
`REPORT_BUILD.md`; the final GO/NO-GO block is at the end of that file.

The build is deliberately staged so each step is reviewable and testable:

1. **Read the plan.** Start with `plan/00_GOAL_AND_CONSTRAINTS.md`, then
   `plan/02_ARCHITECTURE.md`, `plan/03_AGENTS.md`, `plan/07_AUDIT_HARNESS.md`.
   Everything else is supporting detail. Open questions for you are at the end of
   `plan/00_GOAL_AND_CONSTRAINTS.md`.
2. **Run the prompts in order**, each in a fresh pi session at the repo root:
   ```
   pi @research_agent/prompts/00_scaffold_and_guards.md
   pi @research_agent/prompts/01_skills_research.md
   pi @research_agent/prompts/02_skills_ops.md
   pi @research_agent/prompts/03_agents_and_prompts.md
   pi @research_agent/prompts/04_orchestrator_and_state.md
   pi @research_agent/prompts/05_hpc_and_collector.md
   pi @research_agent/prompts/06_audit_harness.md
   pi @research_agent/prompts/07_verify_smoke_and_docs.md
   ```
   (Or open pi and paste `/prompts/…` contents.) Each prompt ends with an
   acceptance checklist; do not proceed until it passes. Prompt 07 runs the
   end-to-end smoke test and prints the go/no-go summary.

   Verified commands after S7 (use these; the runbook was corrected to match):
   ```bash
   make -C .pi all           # verify-env + test + smoke (offline, ~5 min)
   make -C .pi verify-env    # every stage's suite, skills, schemas, model pin
   make -C .pi smoke         # end-to-end sandbox loop, 20/20 checks (~20s)
   make -C .pi resilience    # kill/restart matrix, 7/7 (~25s)
   make -C .pi calibration   # offline calibration modes + refusal paths
   pi --print "/loop:verify-env"   # 4/4 orchestrator checks (print mode)
   ```
   `collect.sh` takes no experiment id (one pass for the active experiment), and
   human gates live in `.pi/state/GATES.jsonl` — both were wrong in plan/09
   before the S7 cross-check (decision D-003).
3. **Configure, then dry-run.** Copy `.pi/config/hpc.example.json` →
   `.pi/config/hpc.json` only if you have an HPC (leave `host` empty and the whole
   HPC path is disabled). Set budgets in `.pi/config/budgets.json`. Run the loop
   in `--dry-run --step` mode a few times and inspect `.pi/state/`.
4. **Start.** Inside pi: `/loop:start` (or `bash .pi/loop/step.sh --forever` from
   tmux). Watch with `/loop:status`. Pause with `/loop:pause`; resume with
   `/loop:resume`.

## What only you can provide

| Thing | Where | If absent |
|---|---|---|
| HPC host/user/port/remote dir | `.pi/config/hpc.json` | HPC path disabled; agents run locally and say so |
| HPC private key (path) | `~/.ssh/…` + `hpc.json` | no SSH attempted |
| Git key for agent forwarding | `~/.ssh/…` + `hpc.json` | no `-A`, no remote git; rsync-only if you supply a fallback |
| `gh` auth (or a token) | environment | PRs are recorded locally; no GitHub API calls |
| Token/cost budget, wall-clock limits | `.pi/config/budgets.json` | conservative defaults apply |
| Target venue + submission deadline | `.pi/config/project.json` | defaults: *Journal of Forecasting*, no deadline |
| Confirmation-data decision | `.pi/state/HUMAN_GATES.json` | loop pauses at the gate and asks |

## Safety model

- `update_pymc` is written **only** by the supervisor/conductor, only after an
  accepted review with green verification; agents work in git worktrees on
  `work/T-###` branches.
- Protected paths (guard extension): `paper/paper.tex` (only in paper phase),
  `fixed_case_studies/protocol/*` (frozen after hash commit), `tests/**`
  (strengthen-only; any removal is flagged), `research_agent/plan/**`,
  `Vangja-project/**` (read-only), `AGENTS.md` (supervisor-approved edits only).
- No `git push --force`, no history rewrite, no deleting branches without a
  recorded review, no secrets or data commits, no network calls in experiment
  code.
- Every agent invocation is logged with exact argv, prompt hashes, repo SHA,
  model id, usage, and produced files; child sessions are persisted under
  `.pi/state/sessions/` for replay.
- Circuit breakers: max consecutive failures, no-progress detection, budget cap,
  and a hard stop flag (`.pi/state/STOP`).

## Model policy (hard requirement)

Every agent in every session uses `opencode-go/deepseek-v4.1-flash`, thinking
level `max`, full 1M-token context. This is enforced three ways: the manifest
passes `--model`/`--thinking` explicitly; a guard extension aborts a session on
model/thinking drift; settings pin defaults and `enabledModels`. Details in
`plan/02_ARCHITECTURE.md` §7.

## Documents

| File | Contents |
|---|---|
| `plan/00_GOAL_AND_CONSTRAINTS.md` | Objective, definition of "best paper", constraints, open decisions |
| `plan/01_RESEARCH_NOTES.md` | Web research: OpenAI swarm/Navier–Stokes, Anthropic, DeepMind, Sakana, PaperBench, AI co-scientist (with links) |
| `plan/02_ARCHITECTURE.md` | Loop state machine, git/PR/worktree policy, budgets, model pinning |
| `plan/03_AGENTS.md` | Every agent: role, inputs, outputs, tools, stop conditions, briefs |
| `plan/04_SKILLS.md` | Skill inventory with what/why/how-built for each |
| `plan/05_EXTENSIONS_AND_TOOLS.md` | Extensions, custom tools, schemas, guardrails |
| `plan/06_HPC_AND_COLLECTOR.md` | SSH/agent-forwarding, SLURM, experiment ledger, collector protocol |
| `plan/07_AUDIT_HARNESS.md` | The audit: workstreams, rubrics, verification, synthesis, grades, calibration |
| `plan/08_BUILD_PLAN.md` | Staged build with acceptance tests and file manifests |
| `plan/09_OPERATIONS_RUNBOOK.md` | Day-to-day: start/pause/resume, intervene, read state, recover, cost |
| `prompts/` | The eight build prompts |
