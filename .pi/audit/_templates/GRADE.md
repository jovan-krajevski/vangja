# Grade — cycle {{cycle}}

**Grade:** {{grade}} · **Base:** {{base_grade}} · **Journal:** {{journal}}
**Capped:** {{capped}} · **Reason:** {{reason}}

## Rule trace

| # | Rule | Effect | Result | Evidence |
|---:|---|---|---|---|
| 1 | rubric-base | base | | |
| 2 | blocking-s0 | cap | | |
| 3 | blocking-core-s1 | cap | | |
| 4 | blocking-other-s1 | cap | | |
| 5 | failed-gate | cap | | |
| 6 | accept-preconditions | require | | |
| 7 | calibration | refuse | | |

A failed deterministic gate caps at `weak reject` unless the adjudicator records
why it is inapplicable. `accept`/`strong accept` require no confirmed S0/S1, all
gates green, complete traceability, and the limitation disclosures present.

## Failing gates

- `<gate-id>` — <detail>

## Blocking fixes

| Finding | Severity | Owner | Effort | Feasible |
|---|---|---|---|---|
| | | | | |

## Fix list

| Finding | Severity | Owner | Effort | Feasible | Blocking | Fix |
|---|---|---|---|---|---|---|
| | | | | | | |

## Decision

Action per `rubrics/decision.json`: `<action>` (re-audit: yes/no, done: yes/no).
