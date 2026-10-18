# Audit protocol — cycle {{cycle}}

Applies to every phase. Read `SCOPE.md` first.

## 1. Phases (fresh contexts, files as the only channel)

| Phase | Agent | May read | May write | Must not |
|---|---|---|---|---|
| 1 find | `audit_finder` (one per WS) | snapshot, own brief, gates.json, skills | `<ws>/FINDINGS.md`, `<ws>/findings.json` | read another WS's findings; verify; grade |
| 2 verify | `audit_verifier` | every `<ws>/findings.json`, snapshot | `VERIFY.md`, `verify/VERIFY_<ws>.md` | discover new primary findings; grade |
| 3 synthesise | `audit_synthesizer` + `meta_review` | verified ids, gates, rubric | `FINAL_REPORT.md`, `rubric.json`, `GRADE.md` | add a primary finding; score without evidence |
| 4 adjudicate | `audit_adjudicator` | all of the above + `QUESTIONS_FOR_AUTHORS.md` | `ADJUDICATED_REPORT.md`, `ROOT_ISSUE_REGISTER.csv` | invent a resolution; change a rule |

## 2. Finding schema (all nine fields, always)

```
id, title, location, snapshot, command, expected, observed, consequence,
severity (S0|S1|S2|S3), confidence (high|medium|low), status
(verified|to-verify|refuted|unresolved), reproducible ("yes" | "no: why"),
evidence[], cross_refs[], owner (code|paper|protocol|docs|env), found_in_cycle
```

- **No command → not a finding.** "Static reading" is not a command. If it cannot
  be reproduced, record `to-verify` with the failed command or drop it.
- `verified` requires `reproducible: yes` and confidence high/medium.
- `unresolved` needs a row in `QUESTIONS_FOR_AUTHORS.md` tagged
  `[blocking]` or `[non-blocking]`.
- IDs are `<WS>-NNN` and unique across the cycle; the register dedupes by
  normalized location+title and records the duplicate as a cross-ref.
- A finding outside your workstream is a `cross-ref: <WS>-NNN` lead, not a body.

## 3. Independence

- A finder may not read another workstream's findings, the historical audit, or
  the calibration expectations. Rediscover or do not report.
- The verifier's default hypothesis is that the finding is wrong. It reopens the
  cited lines, re-runs the command, and records the raw output.
- The synthesizer may only cite verified ids. A synthesizer-authored record
  without a verified id is rejected by `audit_report_assemble`.
- The adjudicator resolves disputes with evidence; an unresolved dispute is
  escalated, never smoothed over.

## 4. Anti-flooding

Cap each workstream at 25 findings ordered by severity. Everything else is an
appendix candidate list; candidates are not findings and do not enter the grade.
A workstream that reports outside its boundary is penalised in the audit rubric.

## 5. Severity

- **S0** correctness-defining: a reported result is wrong, or the method cannot
  support any claim.
- **S1** claim-affecting: a published or plausible claim is unsupported or
  overstated.
- **S2** documentation/usability/reproducibility degradation.
- **S3** cosmetic.

## 6. Append-only within a cycle

Corrections are new files (`VERIFY_CORRECTIONS.md`) or new history entries on the
same id; no finding body is silently edited. `findings.json` is the machine record;
`FINDINGS.md` is rendered from it.

## 7. Snapshot discipline

Everything is read at `{{sha}}`. A finding whose location does not exist at that
SHA is refuted or re-located, not asserted. Never edit the snapshot.
