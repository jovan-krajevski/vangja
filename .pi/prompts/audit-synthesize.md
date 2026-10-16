---
description: Synthesise the verified audit record into a report and grade
argument-hint: "[cycle]"
---
Synthesise cycle `${1:-<n>}`. Run this as an `audit_synthesizer` work unit using
`VERIFY.md`/verdicts and `gates.json` only — no new primary inspection.

Score every rubric leaf with an evidence pointer (`rubric_score`), apply the
blocking rule mechanically (confirmed S0, or confirmed S1 in M/S/P/C, or a failed
gate caps the grade at `weak reject`; `accept` requires no confirmed S0/S1, all
gates green, traceability complete, limitations present), and record the rule
trace. Write the author-facing report plus a fix list with owner/effort. Dispatch
`meta_review` for the coverage/"not checked" section before finishing.

Acceptance evidence: `audit/cycle-<n>/FINAL_REPORT.md`, `rubric.json`, `GRADE.md`
with verified finding ids only, plus the meta-review coverage table. Never state
a weakened/refuted finding as fact and never soften the blocking rule.
