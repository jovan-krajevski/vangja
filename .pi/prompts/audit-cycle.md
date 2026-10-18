---
description: Run one full audit cycle over a frozen snapshot (find→verify→synthesise→adjudicate)
argument-hint: "[cycle-number]"
---
Cycle: `${1:-<n>}`. Phase 1: freeze the snapshot tag/SHA and write
`audit/cycle-<n>/SCOPE.md` (paper SHA/PDF hash, artifact index, protocol hashes,
ledger snapshot), then run the deterministic gates. Dispatch one `audit_finder`
per workstream (M/S/P/D/I/R/H/E/C/L per `plan/07_AUDIT_HARNESS.md` §3) in
parallel, each with its own brief, boundary, checklist, and cap.

Then, as separate fresh work units and never reusing a finder's context:
`audit_verifier` over every S0/S1/S2 finding (try to refute),
`audit_synthesizer` on verified findings only (report + rubric + grade; it
dispatches `meta_review` for coverage gaps), and `audit_adjudicator` (disputes,
root-issue register, grade-trace).

Acceptance evidence: `SCOPE.md`, `gates.json`, `<ws>/FINDINGS.md` +
`findings.json` per workstream, `VERIFY.md`, `FINAL_REPORT.md` + `rubric.json` +
`GRADE.md`, `ADJUDICATED_REPORT.md` + `ROOT_ISSUE_REGISTER.csv`, and calibration
recall ≥ 0.6 before the grade counts. Finders must not read
`Vangja-project/audit/**`; the adjudicator must not copy it either.
