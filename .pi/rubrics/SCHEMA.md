# Rubric schema (`vangja-rubric/1`)

Rubrics make the supervisor's decision a **rule application**, not a judgement
call (plan/07_AUDIT_HARNESS.md §7). Five trees live here:

| File | Kind | Used by | Branches |
|---|---|---|---|
| `pr_review.json` | `rubric` | supervisor in `REVIEW` | code quality, tests, evidence, protocol compliance, provenance, claims, scope discipline, risk |
| `readiness.json` | `rubric` | supervisor in `READINESS` | every bullet of plan/02 §5.3 |
| `paper.json` | `rubric` | supervisor in `PAPER_REVIEW` | structure, claims↔artifacts, baselines honesty, limitations, calibration disclosure, reproducibility statement, AI-use disclosure, venue fit |
| `audit.json` | `rubric` | audit synthesizer | one branch per workstream M/S/P/D/I/R/H/E/C/L + evidence quality, coverage, refutation effort, reproducibility, integrity |
| `decision.json` | `decision` | audit adjudicator / conductor | grade → action mapping of plan/02 §5.4 |

## Tree shape

```json
{
  "schema": "vangja-rubric/1",
  "id": "pr_review",
  "title": "PR review rubric",
  "kind": "rubric",
  "scale": ["0", "0.5", "1"],
  "branches": [
    {
      "id": "tests",
      "title": "Tests",
      "weight": 0.15,
      "children": [
        {
          "id": "tests_added",
          "text": "Every code change adds or updates a test",
          "criterion": "1 only when a failing-before/passing-after test covers the change.",
          "type": "binary",
          "weight": 0.5
        },
        {
          "id": "tests_pass",
          "text": "The suite passes at the head SHA",
          "criterion": "Autofilled from the pre-merge tests gate.",
          "type": "auto",
          "auto": "tests_pass",
          "weight": 0.5
        }
      ]
    }
  ]
}
```

## Invariants (`rubric_validate`)

1. `schema` is `vangja-rubric/1`; `id` and `title` are non-empty.
2. `kind` is `rubric` or `decision`.
3. Branch ids and leaf ids are unique across the tree; every branch has children.
4. Every leaf has `text`, a one-line `criterion` (≤240 chars), a positive
   `weight`, and `type` ∈ `binary | scale | auto`.
5. **Weights sum to 1** inside every branch and across the top-level branches
   (tolerance 1e-9). They are relative importance, not a count.
6. An `auto` leaf names exactly one deterministic source from the table below.
7. A non-auto leaf is scored by an LLM and **must** carry an evidence pointer
   (finding id, gate id, or artifact path). `rubric_score` rejects a score
   without one.
8. A leaf that cannot be resolved is reported as `unresolved`; the total is
   `null` until every leaf has evidence. It is never silently scored 0.

## Leaf types

- `binary` — 0 or 1, with a written criterion that says when 1 holds.
- `scale` — a number in [0, 1]; optional `levels` restricts the allowed values.
- `auto` — computed from `gates.json` (or the calibration result); never scored
  by an LLM. A skipped gate is `unresolved`, because a skip proves nothing.

## Auto sources

| `auto` | Source | 1 means |
|---|---|---|
| `tests_pass` | gate `tests` | the suite is green |
| `hash_ok` | gate `artifacts-complete#hash_ok` | every declared artifact hash matches |
| `protocol_hashes_ok` | gate `protocol-hashes` | frozen files match their sidecars |
| `confirmation_frozen` | gate `confirmation-frozen` | origins predate results |
| `no_test_selection` | gate `no-test-selection` | no selection on the test horizon |
| `test_diff_clean` | gate `test-diff` | tests only strengthened |
| `no_failures_hidden` | gate `no-failures-hidden` | failures/availability reconcile with N |
| `context_excluded` | gate `context-excluded` | context series absent from target aggregates |
| `artifacts_complete` | gate `artifacts-complete` | required artifacts exist and validate |
| `reproduce_headline` | gate `reproduce-headline` | top-N numbers recompute within tolerance |
| `traceability_ok` | gate `traceability` | CLAIMS.json covers the paper and SHAs are ancestors |
| `env_pinned` | gate `env-pinned` | lockfile + pins consistent |
| `gaming_scan_clean` | gate `gaming-scan` | the anti-gaming scan is clean or explained |
| `canary_detected` | gate `canary` | seeded defects are still detected |
| `calibration_ok` | calibration recall | recall ≥ 0.6 and no rejected lead confirmed |
| `guards_clean` | `verify_guards` | protected-path/test-diff scans clean |
| `artifact_hashes_ok` | `verify_artifact` | a specific manifest verifies |

## Grade derivation

`auditkit.audit_grade(rubric, findings, gates, calibration)`:

1. Base grade from the rubric total (boundaries 0.90 / 0.80 / 0.68 / 0.55 /
   0.40 / 0.25 → strong accept … strong reject).
2. Cap at `weak reject` for any confirmed S0, any confirmed S1 in M/S/P/C, or
   any failed gate not waived/inapplicable.
3. Cap at `borderline` for an unfixed confirmed S1 outside M/S/P/C.
4. `accept`/`strong accept` additionally require no confirmed S0/S1, all gates
   green, complete traceability, and limitation disclosures.
5. Recall < 0.6 or a confirmed rejected Phase-A lead → `uncalibrated`; the
   orchestrator refuses to grade.
6. More than 5 audit cycles → `BLOCKED` for human direction.

Every cap appends a trace line naming the evidence; `GRADE.md` must cite the
failing gate(s) that capped the grade.

## `decision.json`

```json
{
  "schema": "vangja-rubric/1",
  "kind": "decision",
  "scale": ["strong reject", "…", "strong accept"],
  "max_cycles": 5,
  "mapping": {
    "weak reject": {"action": "fix-s0-s1", "reaudit": true, "done": false}
  },
  "blocking_rules": {"confirmed_s0": "any confirmed S0 caps at weak reject"}
}
```

The seven grade keys are mandatory and their actions are fixed by plan/02 §5.4
(`reopen-p0`, `fix-s0-s1`, `fix-feasible`, `fix-blocking-or-done`). A missing or
renamed grade is a validation error.
