---
name: scientific-integrity-and-anti-gaming
description: Check a change or a claim for research-gaming patterns before it is merged or reported. Use before claiming a result, when reviewing a diff that touches tests, metrics, splits, exclusions or seeds, and when investigating suspicious numbers. Contains the project's gaming catalogue, severity rules, and a scanner with a seeded fixture matrix.
---

# Scientific integrity and anti-gaming

House rules: `AGENTS.md` § Non-negotiables (tests). The loop's failure mode is
not a wrong number; it is a number that looks better
because a rule was bent. This skill lists the bends this project has actually
seen, the severity scale, and the countermeasure for each. The catalogue is
`references/gaming-catalogue.md`; the scanner is `scripts/scan_gaming.py`.

## When to use

- Before claiming or reporting any result.
- Reviewing any diff touching `tests/`, metrics, splits, seeds, exclusions, or
  aggregation.
- Any result that is unexpectedly good, or any table whose units changed
  between revisions.

## Severity scale (plan `07_AUDIT_HARNESS.md` §7)

| Severity | Meaning | Examples |
|---|---|---|
| S0 | correctness-defining; invalidates the result | test-set selection; metric reassignment; split edit |
| S1 | claim-affecting | seed shopping; failure trimming; denominator clipping; frozen-config edit without amendment |
| S2 | documentation / usability | missing failure counts; unlabelled heuristic intervals |
| S3 | cosmetic | naming, formatting |

A finding needs: `location` (file:line), the command/scan output, the
consequence, and a severity. Evidence first; no finding without a reproduction.

## Scan

```bash
python .pi/skills/scientific-integrity-and-anti-gaming/scripts/scan_gaming.py \
  --root .                       # static scan
python .pi/skills/scientific-integrity-and-anti-gaming/scripts/scan_gaming.py \
  --root . --patch change.patch   # diff scan
python .pi/skills/scientific-integrity-and-anti-gaming/scripts/scan_gaming.py \
  --root . --fail-on S2 --out results/scan
```

Exit code is 1 when any finding at or above `--fail-on` (default `S1`) exists.
The fixture matrix under
`fixtures/seeded/` has one seeded instance per detected category; `fixtures/clean/`
must produce zero findings (enforced by
`.pi/tests/skills/test_scan_gaming.py`).

## Categories the scanner detects

| Category | What it looks like |
|---|---|
| `test-metric-selection` | `idxmin`/`idxmax`/`argmin`/`argmax` over a frame carrying `test`/confirmation metrics |
| `test-weakening-skip` | `@pytest.mark.skip`, `@unittest.skip`, added `pytest.skip`/`xfail` in a diff |
| `test-weakening-assertion-removed` | deleted `assert` lines in a patch |
| `test-weakening-tolerance` | `pytest.approx(..., abs=...)` with a loose tolerance |
| `seed-change-frozen-config` | seed constants changed in a frozen `config.py` / diff |
| `failure-trimming` | `except ...: pass` around result recording; `dropna().to_csv(...)` before reporting |
| `exclusion-change` | changed exclusion logic in a patch |
| `metric-reassignment` | changed `REL_MAE_EPSILON` / primary-metric definition |
| `denominator-clipping` | `np.clip`/`max` applied to the persistence denominator |
| `hashed-file-mismatch` | a `*.sha256` sidecar that no longer matches |
| `horizon-cropping` | shortened `HORIZON_DAYS`/window in a patch |

## Countermeasures (required)

1. **Freeze before looking.** Confirmation origins are hashed before results;
   selection uses development/training data only.
2. **Report exclusions and failures.** `failures.csv` and `availability` are
   part of the artifact; a table without them is incomplete.
3. **Named baselines.** A Vangja arm is never relabelled "Prophet"/"TimeSeers"
   (FINDINGS F-14).
4. **Seeds in artifacts.** Seeds chosen after seeing results are seed shopping;
   finalist seeds are `(42, 20240816, 7)` (`fixed_case_studies/common.py:49`).
5. **Same units across arms.** Compare on the intersection; never re-exclude
   after seeing which arm it helps.
6. **No denominator tricks.** The ε rule excludes and counts; it never clips.
7. **Test changes are deliberate.** Never weaken or delete an existing test to
   make a change pass; if a test is wrong, say so and fix it explicitly with
   evidence.
8. **Amendment, not edit.** Frozen files change only through a recorded
   amendment (`experiment-protocol-design`).
9. **Artifact-backed claims.** Every number cites an artifact path + manifest
   SHA; a number without one is a finding.

## False-positive discipline

The scanner flags *patterns*, and a pattern can be legitimate. Triage every hit:

- A skip with a runtime guard (`if baselines.ARIMA is None: pytest.skip(...)`)
  is not test weakening; record why in the finding and close it.
- Provenance code may swallow its own errors so a fit cannot break; but code
  that records *results* must not.
- `idxmax` over `elpd` on training data is the approved selection procedure.
- When unsure, read the surrounding function before assigning severity.

Report triaged false positives to the skill owner so the scanner can be
narrowed — silently ignoring a hit is itself an anti-pattern.
