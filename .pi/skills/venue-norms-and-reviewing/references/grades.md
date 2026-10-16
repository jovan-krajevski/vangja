# Grade scale and blocking rules

Source of truth: `research_agent/plan/07_AUDIT_HARNESS.md` §7. This file is a
distillation; if they disagree, the plan wins and this file is fixed.

## Ladder (highest to lowest)

| Rank | Grade | Journal mapping | When it applies |
|---|---|---|---|
| 1 | `strong accept` | accept | no confirmed S0/S1, gates green, traceability complete, limitations present, significant advance |
| 2 | `accept` | accept | same, without the "significant advance" bar |
| 3 | `weak accept` | minor revision | correct and reproducible; only non-blocking findings |
| 4 | `borderline` | major revision | an unfixed confirmed S1 outside methodology/statistics/UQ/claims |
| 5 | `weak reject` | reject | confirmed S0, or a confirmed S1 in methodology/statistics/UQ/claims, or a failed gate without an inapplicability reason |
| 6 | `reject` | reject | major defects not fixable in one revision |
| 7 | `strong reject` | reject | results unsupported or unreproducible at the root |

## Blocking rules (apply before the ladder)

1. Any **confirmed S0** → cap `weak reject`.
2. Any **confirmed S1 in M/S/P/C** (methodology, statistics, probabilistic UQ,
   claims↔artifact traceability) → cap `weak reject`.
3. Any **unfixed confirmed S1** elsewhere → cap `borderline`.
4. A **failed deterministic gate** → cap `weak reject`, unless the adjudicator
   records why the gate is inapplicable (a waiver is not an inapplicability
   reason).
5. `accept`/`strong accept` additionally require: all gates green,
   traceability complete, limitation disclosures present.
6. An **uncalibrated** audit run (historical recall < 0.6, or a confirmed
   previously-rejected lead) may not grade the paper; the conductor narrows
   the briefs and re-runs the affected workstreams.

Severity definitions (historical audit `DEFINITIONS.md`): S0 correctness-defining ·
S1 claim-affecting · S2 documentation/usability/reproducibility degradation ·
S3 cosmetic. Confidence: `high` reproduced directly · `medium` inferred from
code · `low` hypothesis needing data. A finding without an exact command (or
`file:line` plus an observed output) is not a finding.

## Fix list (required with every grade)

Each item: `{finding_id, owner (code|paper|protocol|docs|env), fix, effort
(small|medium|large), feasibility (feasible|needs-human)}`. A grade whose fix
list is empty and non-`accept` is rejected by the conductor.

## Calibration

Before grading, the audit must have run the calibration subset against the
historical snapshot (`main@8028c51`) and reported recall over the expected
historical S1 findings and the count of confirmed rejected leads. Record both
numbers in `GRADE.md`; recall < 0.6 ⇒ uncalibrated.
