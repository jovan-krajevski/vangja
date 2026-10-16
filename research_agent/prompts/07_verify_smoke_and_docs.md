# Prompt 07 — Build stage S7: full verification, smoke test, documentation

Prerequisite: S0–S6 green. This stage adds no new architecture; it proves the
environment works as a whole and leaves it documented.

## Deliverables

1. `.pi/Makefile` with targets:
   - `verify-env` — every stage's tests + `validate_skills.sh` + schema checks +
     model-pin audit of `.pi/state/EVENTS.jsonl` (every recorded child argv used
     `opencode-go/deepseek-v4.1-flash` and `--thinking max`),
   - `test` — unit/integration suite only,
   - `smoke` — the end-to-end sandbox run below,
   - `calibration` — the audit calibration run,
   - `all` — `verify-env test smoke`.
2. **End-to-end sandbox smoke test** (`.pi/tests/e2e/`): create a scratch clone of
   the repo with a synthetic tiny task in the backlog
   ("add a docstring and a passing test to a fixture module" plus a mock
   experiment that "runs" locally with a checkpoint). Drive the loop with
   `--sandbox` until it reaches the audit; assert:
   - the transition sequence matches `plan/02_ARCHITECTURE.md` §2,
   - exactly one branch, one PR record, one review, one merge commit, one
     experiment record, one collected artifact,
   - a tampered copy of the artifact is caught before the paper step,
   - the seeded-defect audit produces a grade and a fix list,
   - the sandbox `update_pymc` moved; the **real** repo `update_pymc` SHA did not,
   - every child process is recorded with its argv, prompt hashes, model, usage.
3. **Resilience matrix** (`.pi/tests/e2e/resilience.md` + script): kill the loop at
   each of {before child spawn, during child, after child, before ledger append,
   after ledger append, during merge, during collection}, restart with `resume`,
   and assert exactly-once semantics for merges, jobs, and PRs.
4. Documentation:
   - `.pi/README.md` final: what is here, what is versioned vs runtime, how to
     verify, how to start/pause/resume, security notes.
   - `research_agent/REPORT_BUILD.md` final: one `GO/NO-GO` summary, the
     acceptance evidence for all eight stages, every deviation from the plan with
     its reason, known limitations, and the exact commands to start the loop.
   - Update the "How to build it" section of `research_agent/README.md` with the
     verified status line (`Status: S7 verified on <date> — GO`) and any command
     corrections discovered while running the tests.
   - Cross-check `research_agent/plan/09_OPERATIONS_RUNBOOK.md` against the real
     command names; fix the plan if a command differs (this is the one build stage
     allowed to edit `research_agent/plan/`).
5. A final `git status` and `git diff --stat` report proving nothing outside
   `.pi/**`, `research_agent/**`, `.gitignore` was modified.

## Acceptance tests

1. `make -C .pi all` → green, with raw output captured.
2. The e2e assertions listed above, each with its check printed.
3. Resilience matrix: every kill point resumes with exactly-once effects.
4. Model-pin audit: 100% of recorded child invocations use the pinned model and
   `max` thinking; zero violations.
5. `pi --print "/loop:verify-env"` inside the repo → green, zero warnings.
6. A fresh clone of the repo (or a `git worktree` at the same commit) loads the
   environment and runs `make -C .pi verify-env` green — proving the environment
   is fully committed and self-contained.
7. `grep -r` for key material across `.pi/` → nothing but the example placeholders.

## Final report format

`research_agent/REPORT_BUILD.md` ends with:

```
## GO / NO-GO
Status: GO | NO-GO
Evidence: <one line per stage with the command and result>
Environment: pi <version>, model opencode-go/deepseek-v4.1-flash, thinking max
Start the loop: <exact commands>
Known limitations: <list>
Deviations from plan: <list with reasons>
```

## Do not

- Do not start the real loop, submit to the HPC, or push anywhere.
- Do not "fix" failing acceptance tests by weakening them; fix the environment or
  report a NO-GO with the cause.

## Stop condition

If any acceptance test cannot pass, stop and produce a `NO-GO` report with the
smallest reproducible failure and the proposed fix. Do not proceed to running the
research loop.
