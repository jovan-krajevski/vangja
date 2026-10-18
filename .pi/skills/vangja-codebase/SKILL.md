---
name: vangja-codebase
description: Change or review the Vangja package internals in src/, tests/, or fixed_case_studies/. Use when adding or fixing a component, touching pooling or transfer regularisation, changing fit()/predict()/fit_info, wiring a dataset loader, or when a Vangja model raises a PyMC name, shape, or sampling error. Expands AGENTS.md with exact file:line contracts and a symptom→cause→fix catalogue.
---

# Vangja codebase

Read `AGENTS.md` first; it is terse by design and this skill expands it. The
three references here are the contract:

- `references/architecture.md` — component contract, operator semantics,
  pooling, transfer potentials, `fit()`/`fit_info`, naming rules.
- `references/datasets.md` — every loader, and the scaling rules that decide
  whether a metric is comparable.
- `references/failure-modes.md` — symptom → cause → fix, each with a real
  case and the test that guards it.

## When to use

- Editing `src/vangja/**` or `tests/**`.
- A fit fails with a shape error, a name error, an ADVI start error, or a
  NUTS dispatch surprise.
- Adding a component, a pooling mode, a transfer method, a loader, or a metric.
- Reviewing a diff that touches `fixed_case_studies/**` runners or configs.

## Non-negotiables (from AGENTS.md, expanded)

- **Tests move with code.** Every change adds/updates a test. Never weaken or
  delete an existing test to make a change pass. `tests/test_regularization.py`
  is the contract for transfer potentials and dispatch; `tests/test_time_series.py`
  for the base class; do not edit either to fit new behaviour without saying
  explicitly what the old expectation got wrong.
- **Name PyMC variables `{type}_{i} - {param}`.** No commas, brackets, or
  dots — never encode priors/config in names. Run
  `scripts/lint_pymc_names.py` on the diff.
- **Non-complete pooling always has a per-group dim**, even `n_groups == 1`:
  initvals are arrays; `_predict_columns` values are 1-D `(horizon,)`;
  predictions index `[group_code]`. Run `scripts/check_pooling_shapes.py`.
- **Transfer is not `tune()`.** Build a new model and
  `fit(..., idata=base.trace)`. `parametric` = posterior **mean/sd** matching
  (not the MAP mode). `prior_from_idata` needs MCMC/VI source draws;
  mean-field ADVI gives a diagonal prior and is **not** evidence of
  covariance preservation.
- **Reproducibility:** `random_seed` always, `target_accept` for NUTS; never
  pass `step` together with `nuts_sampler`. `fit_info` is recorded, not
  decorative.

## Working rules

1. Find the narrowest test first: `uv run pytest tests/test_regularization.py -k trend -q`,
   then `uv run pytest tests/test_time_series.py -q`, then the full suite.
2. `uv run black . && uv run ruff check .` before handing the diff on.
3. Prefer a numeric probe (compile the pytensor expression and evaluate it)
   over prose when a formula is in question — see `tests/test_regularization.py`
   `_compile_potential`.
4. When you learn something durable, add one line to `AGENTS.md` §
   Vangja/PyMC gotchas and a row to `references/failure-modes.md`.
5. Never inline data loading: `src/vangja/datasets/**` is the only data access
   layer, and the fixed case studies call it.

## Bundled scripts

```bash
# Flag PyMC variable names that violate the naming rule (diff mode or repo mode)
python .pi/skills/vangja-codebase/scripts/lint_pymc_names.py --diff -   # stdin
python .pi/skills/vangja-codebase/scripts/lint_pymc_names.py --root src/vangja

# Flag per-group shape contract violations, from a static scan or a JSON report
python .pi/skills/vangja-codebase/scripts/check_pooling_shapes.py --root src/vangja
python .pi/skills/vangja-codebase/scripts/check_pooling_shapes.py --report shape_report.json
```

Both scripts exit 0 when clean, 1 when they have findings, and print
`<SEV> <rule> <file>:<line> <message>`. They are heuristics: triage every hit
against the references before acting. They are stdlib-only and offline.

## Review checklist (code diff)

- [ ] Names follow `{type}_{i} - {param}`; no config/priors in names.
- [ ] Non-complete pooling: array initvals, 1-D prediction columns, `[group_code]` indexing.
- [ ] Potentials: trend is `-loss·Σ(slope−slope_transferred)²` in **all** pooling modes; seasonality is the one-sided per-grid-point cap, gated on `period > 2·n_obs`.
- [ ] `delta_tune_method` (never `tune_method`) dispatches the changepoint transfer.
- [ ] `fit()` gains `random_seed`/`target_accept` support, `fit_info` records them.
- [ ] New loader goes through `src/vangja/datasets` and documents scaling/units.
- [ ] Tests added; `black`/`ruff` clean; no test weakened.
