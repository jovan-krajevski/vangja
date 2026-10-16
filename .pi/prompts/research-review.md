---
description: Review one PR against its acceptance criteria
argument-hint: "<PR-###>"
---
Review target: `${1:-<PR-###>}`. Run this as a `supervisor` work unit.

Read the PR record and the diff at its recorded `head_sha`; then run every
acceptance criterion yourself and paste the raw observed output. Run the
deterministic pre-merge gates (tests, protected-path scan, test-diff scan,
integrity scan, artifact provenance, claims traceability, frozen-protocol rule).
Re-derive at least one headline number from its artifact or dispatch `critic` to
try to falsify it. For statistical changes, require the `methodologist` verdict.

Acceptance evidence: `review_write` (`RV-###`) with decision, rubric scores, the
commands actually run, findings with `file:line`, required changes, and residual
risk. On accept, record the merge recommendation and call `merge_pr` (S3: logged
by the stub; S4: conductor executes). Never review work you authored.
