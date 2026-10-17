# Gaming catalogue — patterns seen in this project

Sources: `research_agent/plan/07_AUDIT_HARNESS.md` §6, `Vangja-project/REVIEW/FINDINGS.md`
(F-1, F-3, F-9, F-11, F-13, F-14, F-16), and the historical audit findings cited
inline. Each entry: pattern → detection → required countermeasure.

## 1. Configuration selection on the test horizon

- **Seen:** F-1 — a large grid selected per target on the retrospective test
  horizon.
- **Detect:** `idxmin`/`idxmax`/`argmin`/`argmax`/`nsmallest`/`nlargest`,
  builtin `min`/`max`/`sorted` over test/metric tokens, or
  `sort_values(...).head(1)`
  over a frame with `test`/confirmation metric columns; per-target "best"
  config maps.
- **Countermeasure:** one frozen global config; selection only on development
  training windows (`HYPERPARAMETER_SELECTION.md`). If a result was selected on
  test data, it is not a result.

## 2. Seed shopping / unrecorded seeds

- **Seen:** F-11 — no seeds, no `target_accept`, unpinned PyMC.
- **Detect:** seeds absent from artifacts; seed constants changed in frozen
  configs; seeds chosen after results (diff adds/removes a `FINALIST_SEEDS`
  entry).
- **Countermeasure:** `BASE_SEED=42`, `seed_for`, `source_seed_for`; record
  seeds in `fit_info` and every manifest.

## 3. Test weakening

- **Seen:** generic anti-pattern; guarded by `tests/**` strengthen-only policy.
- **Detect:** deleted `assert` lines; `skip`/`xfail` added to tests; tolerance
  relaxed (`pytest.approx(abs=...)`); a test deleted instead of fixed.
- **Countermeasure:** never weaken or delete an existing test; if the test is
  wrong, state it explicitly and fix it deliberately with evidence.

## 4. Failure trimming

- **Seen:** historical runs where failed units simply did not appear.
- **Detect:** `except: pass` around recording; `dropna()` before `to_csv`/save;
  a results table whose unit count changes without a reported failure.
- **Countermeasure:** `record_failure` → `failures.csv`; failure rate next to
  every accuracy table; `availability` records missing series.

## 5. Post-hoc exclusions

- **Detect:** `exclude`/`excluded` logic changed after results; exclusion sets
  differing between arms; units dropped because they hurt one arm.
- **Countermeasure:** exclusion rule fixed before results (ε on scaled target);
  per-unit exclusion applied identically to all arms; report counts.

## 6. Metric redefinition / denominator clipping

- **Seen:** F-13 — MAPE degenerate near zero used for selection.
- **Detect:** changed `REL_MAE_EPSILON` (must be `1e-3`); `clip`/`max` on the
  persistence denominator; swapping the primary aggregate away from median;
  reintroducing MAPE as primary.
- **Countermeasure:** relative MAE vs persistence is primary; ε excludes and
  counts; MAPE is a safeguarded continuity metric only.

## 7. Split / data-handling changes

- **Seen:** F-9 — interpolation before split, calendar-day invention.
- **Detect:** `HORIZON_DAYS`/window edits; `interpolate=True`; scaling fitted
  across the split; confirmation origins edited after hashing.
- **Countermeasure:** trading-days-only data; split-first, within-split
  interpolation only; `CONFIRMATION_ORIGINS.csv` hash-verified before scoring.

## 8. Context leakage into targets

- **Seen:** F-3 (context scored as a target), F-16 (context used both for
  transfer and as a group without accounting).
- **Detect:** `^GSPC`/`source`/`Boston` rows inside `units_*.csv` or target
  aggregates; `CONTEXT_SERIES` not applied in a scoring function.
- **Countermeasure:** `CONTEXT_SERIES = {"source", "^GSPC"}` excluded
  everywhere (`fixed_case_studies/common.py:51`); tests verify the exclusion.

## 9. Ill-named baselines

- **Seen:** F-14 — "Prophet"/"TimeSeers" arms were Vangja configurations.
- **Detect:** labels in result files that do not match the implementation.
- **Countermeasure:** run the real package or relabel honestly; the no-transfer
  arm is `no_transfer`.

## 10. Horizon cropping

- **Detect:** different horizons or evaluation windows between arms; a
  shortened horizon in a diff.
- **Countermeasure:** frozen `HORIZON_DAYS`; any `horizon_days` override must
  be `__h`-tagged and excluded from reported aggregates.

## 11. Artifact-less numbers

- **Detect:** a number in prose/table with no artifact path + manifest SHA; a
  manifest whose commit is not an ancestor of the audited tag; results produced
  on a branch other than `update_pymc`.
- **Countermeasure:** claim discipline (`scientific-writing-latex`); manifests
  validate (`check_manifest.py`).

## 12. Covariance-preservation over-claim

- **Seen:** F-8 / WS-S-008 — mean-field ADVI source presented as joint transfer.
- **Detect:** covariance claims without a NUTS source fit and saved covariance
  matrices.
- **Countermeasure:** `PROTOCOL.md` §8; use the honest four-level framing
  (main contribution / secondary / package capability / no claim).

## Reporting a finding

```json
{
  "id": "AF-012",
  "location": "fixed_case_studies/stocks/runner.py:301",
  "command": "python .pi/skills/.../scan_gaming.py --root .",
  "observed": "units_<tag>.csv written after .dropna() without a failure row",
  "consequence": "failed stocks silently absent from the median; result biased",
  "severity": "S1",
  "confidence": "high",
  "status": "verified",
  "refutation": "if failures.csv contains the corresponding rows, close"
}
```

Severity is assigned by consequence, not by intent. A finding is closed only by
evidence (a re-run, a diff, a counterexample) or by a recorded waiver.
