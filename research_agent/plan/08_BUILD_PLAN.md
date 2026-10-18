# 08 — Build plan: what the prompts create, and how we know it works

Eight stages, each implemented by one prompt file in `research_agent/prompts/`
run in order in a fresh pi session at the repo root. A stage is not done until
its acceptance tests pass and `research_agent/REPORT_BUILD.md` has an entry with
the commands and observed output.

## Ground rules for the builder sessions

- Work only inside `.pi/**` and `research_agent/**` (plus `.gitignore`,
  `AGENTS.md` only if the prompt says so). **Never** modify `src/`, `tests/`,
  `fixed_case_studies/`, `paper/`, `case_studies/`, `docs/`, or
  `Vangja-project/**` during the build.
- Do not install anything globally; do not use `pi install`; every resource is a
  file in this repo. Extension dependencies, if any, go in
  `.pi/extensions/package.json` (with `peerDependencies` for pi-provided packages).
- Do not push to any remote, do not create GitHub PRs, do not touch the HPC.
- Run everything with the pinned model: `opencode-go/deepseek-v4.1-flash`,
  thinking `max`.
- If a prompt's prerequisite is missing, stop and report — do not improvise a
  different architecture.

## Stage map

| Stage / prompt | Creates | Acceptance test |
|---|---|---|
| **S0** `00_scaffold_and_guards.md` | `.pi/settings.json`, `.pi/config/{agents,model,budgets,project,hpc.example}.json`, `.pi/extensions/{guards,ledger,status}/`, `.pi/tests/`, `.pi/loop/` skeleton, `.gitignore` entries | `/loop:verify-env` green (model pin, resources load, tool names unique, schemas valid); guard test: a child `pi` started with a wrong model/thinking is blocked; ledger append/read round-trip with a lock; secret scan blocks a fixture secret |
| **S1** `01_skills_research.md` | skills 1–6 of `plan/04_SKILLS.md` with references/scripts/tests | each skill validates (`validate_skills.sh`), each bundled script has a passing test, `scan_gaming.py` detects the seeded fixture set |
| **S2** `02_skills_ops.md` | skills 7–15 | same validation; `lint_pymc_names.py` catches a bad fixture name; `parse_sacct.py` handles recorded fixtures; `bib_check.py` flags an unverifiable citation fixture |
| **S3** `03_agents_and_prompts.md` | `.pi/agents/*.md` (4 primary + workers), `.pi/prompts/*.md`, `.pi/extensions/dispatch/` | every agent file parses; dispatch a `scout` worker in a scratch dir and get output + usage logged; the role matrix blocks `phd_candidate` from `merge_pr` and `collector` from `hpc_submit` |
| **S4** `04_orchestrator_and_state.md` | `.pi/extensions/orchestrator/`, `.pi/loop/{step.sh,status.sh}`, worktree + PR + merge tools, budget accounting, gates, dry-run mode | in a **sandbox clone** of the repo (scratch origin, not the real one): a full simulated cycle (select→work→PR→review→merge) runs; kill mid-step and resume; replay causes no duplicate merges/jobs; budget cap pauses; protected-path write is refused; `update_pymc` untouched without an accepted review |
| **S5** `05_hpc_and_collector.md` | `.pi/extensions/hpc/`, `.pi/agents/collector.md` (final), slurm template, experiment ledger tools, mock-cluster tests | with `host` empty: all `hpc_*` tools return `enabled:false`, collector exits cleanly; with a mock `ssh`/`sbatch`/`squeue`/`sacct` on `PATH`: submit → running → timeout(exit 99) → checkpoints fetched → requeue → collected; a second submit while one is active is **refused**; a record without checkpoint metadata is refused; no key material in any output |
| **S6** `06_audit_harness.md` | `.pi/rubrics/*`, `.pi/extensions/{verify,rubric,auditkit}/`, audit agent defs, audit cycle templates + `_PROTOCOL.md`, calibration mode | seeded-defect sandbox copy: audit finds ≥80% of seeded S0/S1 defects, awards ≤ `weak reject`, and the grade trace names the gates; calibration run on the historical snapshot subset produces recall/false-positive numbers and refuses to grade if recall < 0.6 |
| **S7** `07_verify_smoke_and_docs.md` | `.pi/tests/` final suite, `.pi/loop/README.md`, `research_agent/REPORT_BUILD.md`, updated `research_agent/README.md` (status), `research_agent/plan/09_OPERATIONS_RUNBOOK.md` cross-check | `make -C .pi verify-env` green; end-to-end dry-run loop in a sandbox reaches `READINESS` with a trivial synthetic task; every stage's acceptance evidence is in `REPORT_BUILD.md`; a human can follow the README without asking questions |

## Stage detail

### S0 — scaffold and guards
Files: `.pi/settings.json` (model pin, resource paths, `defaultTools` including
`+codemode`, compaction settings), four config files with schemas,
`.pi/config/README.md` documenting each field, `guards`, `ledger`, `status`
extensions, `.pi/tests/{test_settings,test_ledger,test_guards,test_schemas}.ts`
(or `.py` driving `pi --mode json`), `validate_environment` command.
Also: `.gitignore` entries for `.pi/state/`, `.pi/artifacts/*/`,
`.pi/config/hpc.json`, `.pi/**/node_modules/`, worktrees, session dumps; and
`.pi/README.md` explaining what is versioned vs runtime.

### S1/S2 — skills
One PR-sized unit per skill (skill dir + tests + one line in
`.pi/skills/README.md`). Skills must be written from the project's actual
material (`AGENTS.md`, `fixed_case_studies/`, `Vangja-project/REVIEW/PROTOCOL.md`,
the historical audit's `_PROTOCOL.md`) — not generic advice. Where a skill
bundles a script, the script is the source of truth and has tests.

### S3 — agents and prompts
Each agent file is ≤120 lines: role, inputs, procedure, output contract,
prohibitions, stop conditions. Prompt templates are thin wrappers that assemble a
brief and dispatch. The `dispatch` extension is adapted from pi's bundled
subagent example (which is Apache/MIT-licensed; verify and record the licence) and
extended with role checks, per-role worker allowlists, persistent sessions, and
usage accounting.

### S4 — orchestrator and state
The state machine of `plan/02_ARCHITECTURE.md` §2 implemented once as a pure
`nextAction(state, ledgers)` function with table-driven tests, plus `execute()`
side effects. Worktrees under `state/worktrees/`; sandbox mode
(`--sandbox <path>`) redirects the repo root so the whole loop can be exercised
without touching the real `update_pymc`. Dry-run mode logs intended actions
without side effects.

### S5 — HPC and collector
`hpc.ts` with the exact tools and gates of `plan/05_EXTENSIONS_AND_TOOLS.md` §5;
mock-cluster tests use shims (a `tests/fixtures/mockbin/` directory on `PATH`)
that replay `sbatch`/`squeue`/`sacct`/`rsync`/`ssh` transcripts, including a
timeout-then-requeue scenario. The single-slot invariant gets a dedicated test
that tries to submit twice and asserts the refusal.

### S6 — audit harness
Rubric JSON with schema + `rubric_autofill` for gate-backed leaves;
`auditkit` with schema validation and dedupe; the ten workstream briefs as
templates; the four-phase agent defs; sandbox "seeded defect" fixtures (a copy of
a small synthetic project with known bugs: wrong sign, selection on test, weakened
test, missing artifact, unlabelled retrospective data) with an expected-findings
list; a calibration script that runs the calibration subset against the frozen
historical snapshot.

### S7 — verify, smoke, docs
Full environment verification (`make -C .pi verify-env`), the sandbox end-to-end
loop, a resilience matrix (kill/restart at each stage), and the documentation
pass. Output: `research_agent/REPORT_BUILD.md` — what was built, evidence per
acceptance test, known limitations, and the exact commands to start the loop.

## Definition of done for the build

- All eight acceptance rows green with evidence.
- `.pi` environment loads with **zero** warnings in pi.
- The sandbox loop reaches a mock audit and stops with a recorded grade.
- No file outside `.pi/**`, `research_agent/**`, and `.gitignore` was modified
  (`git status` evidence recorded).
- `REPORT_BUILD.md` lists every deviation from this plan with its reason.

## Suggested effort

S0 ~1 session · S1 ~2 · S2 ~2 · S3 ~1–2 · S4 ~2–3 · S5 ~2 · S6 ~2–3 · S7 ~1.
Long stages may be split into smaller PRs, but the acceptance tests stay whole.
