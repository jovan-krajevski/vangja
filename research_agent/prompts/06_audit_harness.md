# Prompt 06 — Build stage S6: audit harness, rubrics, gates, calibration

Prerequisite: S0–S5 green. Read `research_agent/plan/07_AUDIT_HARNESS.md` in full
and the historical material it builds on:
`Vangja-project/audit/{README.md,DEFINITIONS.md,stage-2/_PROTOCOL.md,stage-2/README.md}`,
`Vangja-project/audit/stage-2/ws-v-verification/REPORT.md`,
`Vangja-project/audit/FINDINGS.md`, `Vangja-project/audit/FINAL_REPORT.md`.

## Deliverables

1. `.pi/rubrics/` — JSON rubric trees + `SCHEMA.md`:
   - `pr_review.json` (code quality, tests, evidence, protocol compliance,
     provenance, claims, scope discipline, risk),
   - `readiness.json` (every bullet in `plan/02_ARCHITECTURE.md` §5.3),
   - `paper.json` (structure, claims↔artifacts, baselines honesty, limitations,
     calibration disclosure, reproducibility statement, AI-use disclosure, venue fit),
   - `audit.json` (one branch per workstream + cross-cutting: evidence quality,
     coverage, refutation effort, reproducibility, integrity),
   - `decision.json` (grade → action mapping of §5.4).
   Leaves: `type: binary|scale|auto`; `auto` names a gate result; weights sum to 1
   per branch; every leaf has a one-line grading criterion.
2. `.pi/extensions/verify/index.ts` — `verify_tests`, `verify_artifact`,
   `verify_reproduce`, `verify_traceability`, `verify_guards`, `run_gates`
   emitting `.pi/audit/cycle-<n>/gates.json` with the gates of
   `plan/07_AUDIT_HARNESS.md` §5. `verify_reproduce` executes the artifact's
   recorded `repro_command` in a temp dir, compares numbers within a stated
   tolerance, and reports per-number diffs.
3. `.pi/extensions/rubric/index.ts` — `rubric_load`, `rubric_autofill`,
   `rubric_score` (LLM leaves require an evidence pointer per score),
   `rubric_validate`, `rubric_render`.
4. `.pi/extensions/auditkit/index.ts` — `finding_validate` (nine fields +
   severity/confidence/status/reproducible; reject a finding with no command),
   `finding_register` (dedupe by normalized location+title hash; cross-refs),
   `findings_render` (MD + JSON), `audit_report_assemble`, `audit_grade`
   (blocking rules of §7 encoded, with a trace explaining the cap).
5. `.pi/audit/_templates/` — `SCOPE.md`, `_PROTOCOL.md`, per-workstream
   `BRIEF.md` for all ten workstreams (M/S/P/D/I/R/H/E/C/L with hard boundaries,
   coverage checklists, ≤25-finding cap, and the "no command → not a finding"
   rule), `VERIFY.md`, `FINAL_REPORT.md`, `ADJUDICATED_REPORT.md`,
   `ROOT_ISSUE_REGISTER.csv`, `QUESTIONS_FOR_AUTHORS.md`, `GRADE.md`.
6. Finalise the audit agent definitions (from S3) so they encode `plan/07` §8:
   finder sessions are independent and may not read other workstreams' findings;
   the verifier reopens cited lines and re-runs commands; the synthesizer may
   only consume verified IDs; the adjudicator resolves disputes and maps every
   root issue to an owner (`code|paper|protocol|docs|env`) and a fix.
7. Orchestrator wiring: `AUDIT` state runs `run_gates` → finders (parallel ≤8) →
   verifier → synthesizer + `meta_review` → adjudicator; writes
   `.pi/audit/cycle-<n>/`; then `DECIDE` reads `GRADE.md` + `findings.json` and
   applies `rubrics/decision.json`.
8. `calibrate.sh` — runs the calibration subset (workstreams S, M, R) in
   `calibration` mode against a **read-only worktree of the historical snapshot**
   (`main@8028c51`, extracted under `.pi/state/calibration/`), with the expected
   findings list from `plan/07` §7 and the four rejected Phase-A leads as negative
   controls. Produces `.pi/audit/calibration/<ts>/{REPORT.md,recall.json}` with
   per-item hit/miss and false positives; if recall < 0.6 or any rejected lead is
   confirmed, it writes `UNCALIBRATED` and the orchestrator refuses to grade
   (it may still run finders with narrowed briefs).
9. Seeded-defect fixtures `.pi/tests/audit/fixtures/` — a small synthetic repo
   containing at least: wrong potential sign; a runner that selects configs on
   test MAPE; a weakened test (assertion removed); a missing artifact for a
   reported number; a retrospective dataset labelled "confirmation"; a baseline
   mislabelled "Prophet"; a hard-coded favourable seed; a hard-coded number in
   the paper with no claims entry. Plus `expected.json` listing the defect →
   severity → workstream that should find it.

## Acceptance tests (paste into `REPORT_BUILD.md`)

1. `rubric_validate` on all five rubrics → valid; weights sum correctly.
2. `run_gates` on the seeded fixture repo → all gates report; a tampered artifact
   fails `hash_ok` and `verify_reproduce` prints the numeric drift.
3. Audit on the seeded fixture repo → ≥80% of seeded S0/S1 detected, grade ≤
   `weak reject`, and `GRADE.md` cites the failing gates that capped it.
4. `validate_findings.py` rejects: missing command, unknown severity, duplicate
   finding (deduped with a cross-ref), status `verified` with confidence `low`.
5. `audit_grade` unit tests: each blocking rule, each grade boundary, and the
   "accept requires all gates green + no S0/S1" rule.
6. Calibration against the historical snapshot: report recall over the expected
   list and false positives on the rejected leads; assert the `UNCALIBRATED`
   refusal path with a deliberately weakened brief (fixture mode).
7. `make -C .pi verify-env` green with the audit tests included.

## Do not

- Do not modify `Vangja-project/**`; the calibration worktree is read-only
  (`git worktree add --detach`, never commit there).
- Do not let the synthesizer add primary findings; assert this in a test
  (a synthesizer-created finding without a verified ID must be rejected).
- Do not download anything.

## Final report

Append `## S6 — audit harness` to `research_agent/REPORT_BUILD.md` with the
seeded-defect detection matrix and the calibration recall/false-positive numbers.
