# Limitations / disclosures checklist

Fill every item; delete nothing. Each row must be true of the submitted
artifacts, not of an earlier draft. Cross-references are to
`fixed_case_studies/README.md` §6–§7 and the historical findings unless stated.

## Required statements

- [ ] **Horizon framing.** The 4× horizon is the stocks study
      (91-day train / 365-day horizon); smart home is ≈2.9×. Never generalise
      "4× horizons" to both.
- [ ] **Retrospective vs confirmatory.** Stocks confirmation origins are
      hash-committed (`CONFIRMATION_ORIGINS.csv` + `.sha256`); smart-home
      numbers are retrospective and illustrative only.
- [ ] **Frozen configuration.** The study measures a defensible method, not
      an optimised configuration; the frozen values were chosen a priori
      (development evidence disclosed). No best-of-grid number as headline.
- [ ] **Metric.** Relative MAE vs persistence, per target×origin unit,
      aggregate = median; units with persistence MAE `< 1e-3` on the scaled
      target are excluded and the excluded count is reported.
- [ ] **Baselines.** Prophet/TimeSeers are relabelled or removed unless the
      official packages were actually run (F-14).
- [ ] **Transfer machinery.** `parametric` = posterior mean/sd matching, not
      MAP; `prior_from_idata` covariance evidence requires MCMC/VI draws, and
      mean-field ADVI is not such evidence.
- [ ] **MAP intervals.** The MAP `predict_uncertainty` path is a
      residual-calibrated heuristic (`sqrt(1 + h/n)` widening), not a
      posterior predictive; no frequentist coverage claim.
- [ ] **Universe provenance.** S&P membership reconstructed from Wikipedia's
      changes table (accurate ~1997+); the index mechanically overlaps the
      targets; high-weight sensitivity uses a frozen mega-cap approximation,
      not historical weights. Survivorship of the *target* universe is
      handled by fixing membership at the origin.
- [ ] **Data revision.** Provider data may change; SHA-256 sidecars freeze
      the *used* data. No "bit-for-bit" reproduction of a live download.
- [ ] **Failures/exclusions.** Failed fits, unavailable series, and excluded
      units appear in the failure/availability files and reconcile with N.
- [ ] **Dependence.** Monthly origins have overlapping horizons; the
      two-way block bootstrap models this but there are few independent
      origin blocks — paired results are descriptive-with-uncertainty.
- [ ] **Deep learning.** No head-to-head transfer-learning DNN baseline was
      run; the scope statement says so.

## Mandatory submission statements

- [ ] **AI use.** Which model/agent produced or checked what, and that every
      number was recomputed from committed artifacts by a deterministic
      script.
- [ ] **Reproducibility.** Commands, environment fingerprint
      (`freeze_report.py`), data hashes, artifact locations (paths + SHAs).
- [ ] **Code/data availability.** Repository, licence, and how to obtain the
      data with the frozen revisions.
