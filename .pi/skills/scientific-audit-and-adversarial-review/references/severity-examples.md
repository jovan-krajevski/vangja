# Severity and confidence — calibrated examples

Real findings from the historical audit (`Vangja-project/audit/stage-2/**`,
89 findings, 85 confirmed / 0 rejected by WS-V). Use these to keep today's
scale the same; do not copy the findings themselves.

## S0 — correctness-defining (a result is wrong, or no claim is supportable)

No historical finding was graded S0: the 89-finding register is 27 S1 · 42 S2 ·
9 S3 (6 ungraded "—"), and `Severity: S0` never appears
(`grep -rn 'Severity:\*\* S0' Vangja-project/audit/stage-2/*/FINDINGS.md` → no
match). S0 is a live category for today's audit, not an inherited one: raise it
only when the result is wrong or no claim can be supported — and say which
claim dies with it.

Signals to listen for: sign errors, estimator computing a different quantity
than claimed, an invariance that does not hold, a metric that cannot order the
arms correctly.

## S1 — claim-affecting (a claim is unsupported or overstated)

- **WS-S-001**: `parametric` transfer centres the prior on the posterior
  **mean**, while the paper argues for and prints the **MAP/mode**
  (`linear_trend.py:204`, `fourier_seasonality.py:196`, `flat_trend.py:141`).
  The code is internally consistent; the paper's claim is false → S1, high.
- **WS-S-002** (`linear_trend.py`, complete and individual pooling): the
  trend-transfer potential had the wrong sign/form (complete `+|·|`,
  individual `+(·)²`), so regularisation pulled the target slope *away* from
  the context slope; the partial branch matched the paper. Historically
  graded **S1**, not S0, because the headline case studies did not exercise
  the buggy branches — only the stocks `individual`-pool ablation rows did
  (`FINDINGS.md:203`; same defect, lower consequence).
- **WS-S-003**: the Fourier seasonal regularisation scale `lam` differed
  across pooling modes (a ≈22× ratio for the smart-home yearly term) and was
  undocumented vs. the paper, with the partial-pooling form active for a
  headline result → S1 (the inconsistency is certain; the intended fix
  direction is by analogy with the complete branch).
- **WS-H-001**: the smart-home "best" and "reference" tables report different
  Vangja configurations (per-series test-best vs one global config) and the
  text does not distinguish them → S1 for the claim (selection-bias severity
  owned by WS-M/WS-D), high.
- **WS-R-005**: `main`'s `submit_ablation.slurm` maps every mode to scripts
  that do not exist on `main`, so every documented reproduction command
  aborts at the path check → S1 (reproducibility claim), high. (The current
  `update_pymc` branch maps to `run_vangja.py` etc., which do exist — check
  the branch you are auditing.)
- **F-14 (WS-M)**: "Prophet"/"TimeSeers" baselines were Vangja
  configurations → S1, high.
- **WS-R-014**: "all experiments can be reproduced" while the branch, commit,
  PyMC version, seeds, and data snapshot are all absent → S1.

## S2 — documentation / usability / reproducibility degradation

- **WS-H-008**: the `load_nyc_temperature` docstring says "daily maximum
  temperature in Fahrenheit"; the body returns a daily **mean** in the
  source's units with no conversion → S2 (behaviour claim owned by WS-D).
- **WS-H-004**: paper says stocks yearly `series_order=5`; both runner
  scripts use 6 → S2 spec↔code mismatch.
- **WS-R-007/008**: reproducibility checklist contradicts the artefacts
  (base image tag wrong, deps unpinned, seeds absent, no image hash;
  checkpointing features documented that no runner implements) → S2.
- **WS-H-005**: the README documents a file layout that exists on neither
  branch → S2.

## S3 — cosmetic

- **WS-H-016**: "absolute reduction of 78.2%" is a *relative* reduction → S3.
- **WS-H-017**: "9% to 77%" where 77.95% rounds to 78% → S3.
- Typo in a filename that has no functional consequence → S3 (but WS-H-007
  upgraded it to S2 because it silently dropped a page from the built docs —
  severity is about consequence, not size).

## Confidence

- **high** — you ran the command / opened the exact lines and reproduced it.
- **medium** — inferred from code/docs without executing; say what would
  confirm it.
- **low** — hypothesis; needs data or a run. `low` + `verified` is a
  contradiction; use `to-verify`.

## Refuted leads are results too

- **F-012** ("README uses `map`, `__init__` uses `mapx`") was **refuted** on
  the audited branch (WS-H-017/WS-I-001): the README consistently uses
  `mapx`, a real dispatched method. Record refutations with the same rigour —
  the calibration score penalises confirming a known-rejected lead.
- **WS-V** weakened four findings by re-counting their numbers (e.g.
  WS-S-009's "6 parameters with ESS < 200" was actually 0; the qualitative
  point survived). "Weakened" means: the mechanism is real, the quoted
  magnitude was wrong — fix the magnitude, keep the finding.

## What is NOT a finding

- "This could be a problem" with no location, command, or observed output.
- Style preferences with no documented norm.
- A duplicate of another workstream's finding (use `cross-ref`).
- Anything whose only evidence is a model-generated summary.
