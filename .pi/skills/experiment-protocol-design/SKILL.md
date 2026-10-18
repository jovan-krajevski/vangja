---
name: experiment-protocol-design
description: Design, freeze, or audit the rules of a Vangja study before results are produced. Use when choosing configurations, defining splits and origins, adding or changing metrics, exclusions, baselines or analyses, when a frozen file might change, or when deciding whether something is a protocol amendment. Contains the project's frozen protocol distilled with file and line references.
---

# Experiment protocol design

House rules: `AGENTS.md` § Non-negotiables. The study's defence is that no
decision was made after seeing the test horizon.
This skill states the frozen rules, the files that encode them, and the
amendment path. Scan for violations with
`scripts/check_protocol_compliance.py`; the full distilled protocol is in
`references/vangja-protocol.md`.

## When to use

- Designing a new origin set, split, metric, exclusion, or baseline.
- Asked to "try a few settings" or "pick the best" — read §"Selection" first.
- Reviewing a diff that touches `fixed_case_studies/stocks/config.py`,
  `fixed_case_studies/smart_home/config.py`, `fixed_case_studies/common.py`, or
  `fixed_case_studies/protocol/CONFIRMATION_ORIGINS.csv`.

## Core rules

1. **One frozen global configuration per study.** Exactly one config plus a
   small, pre-registered ablation list; no grid search
   (`fixed_case_studies/stocks/config.py:5`). No per-target best config.
2. **Dev and confirmation are separate.** The 24 retrospective origins
   (`DEV_ORIGINS`, 2013-01..2014-12, `config.py:25`) are development only. New
   confirmation origins live in
   `fixed_case_studies/protocol/CONFIRMATION_ORIGINS.csv`; its SHA-256 is
   committed in `CONFIRMATION_ORIGINS.sha256` and verified by
   `fixed_case_studies/common.py:load_confirmation_origins` (`common.py:596`),
   which **refuses to score** on mismatch.
3. **No configuration selection on the test horizon.** Selection evidence is
   training-window only: prior-predictive coverage, PSIS-LOO `elpd_loo` on the
   training window, ±50 % robustness sweeps, LFO-CV + CRPS, stacking weights
   (`fixed_case_studies/HYPERPARAMETER_SELECTION.md`).
4. **Primary metric = median Relative MAE vs persistence**, per
   target × origin unit; `REL_MAE_EPSILON = 1e-3` on the scaled target
   (`fixed_case_studies/common.py:44`, `unit_metrics` at `common.py:185`).
   Excluded units are counted, never deleted.
5. **Context series are never targets.** `CONTEXT_SERIES = {"source", "^GSPC"}`
   (`common.py:51`); `GC=F` is the negative-control context. They may not enter
   any target average, median, ranking or table.
6. **Failed fits are reported.** `record_failure` appends to `failures.csv`
   (`common.py:558`); failure rate accompanies any accuracy table.
7. **Baselines are named honestly.** A Vangja configuration is never called
   "Prophet" or "TimeSeers" (`Vangja-project/REVIEW/FINDINGS.md` F-14); the
   no-transfer arm is labelled `no_transfer`.
8. **Dependence-aware comparison only.** Two-way block bootstrap over series
   and origin blocks (`common.py:two_way_block_bootstrap`, `common.py:400`);
   no pooled Diebold–Mariano, no stock-only bootstrap.
9. **Covariance claims need a covariance-capable source.** Mean-field ADVI
   gives a diagonal transfer prior; do not claim covariance preservation from
   it (`fixed_case_studies/stocks/runner.py:180-182`; FINDINGS F-8). Use NUTS source
   posteriors for joint-transfer claims.
10. **The paper may only cite artifacts produced on `update_pymc`** and each
    number cites an artifact path plus manifest SHA.

## Amend, don't edit

Changing a frozen value (config, split, metric, exclusion, epsilon,
confirmation origins) is an **amendment**, not an edit:

1. Update the justification in `fixed_case_studies/README.md` and/or
   `fixed_case_studies/HYPERPARAMETER_SELECTION.md`.
2. Re-hash `CONFIRMATION_ORIGINS.csv` if it changed and commit both.
3. Record it with `amendment_record` (opens a human gate) before any new
   results are produced; the JSONL ledgers are in `.pi/state/`.
4. Re-run affected development stages; confirmation runs once on the frozen
   configuration.

## Compliance scan

```bash
python .pi/skills/experiment-protocol-design/scripts/check_protocol_compliance.py --repo .
python .pi/skills/experiment-protocol-design/scripts/check_protocol_compliance.py --repo . --diff HEAD~1
python .pi/skills/experiment-protocol-design/scripts/check_protocol_compliance.py --repo . --json --out results/protocol_report.json
```

Detects (file:line in the report): configuration selection over test metrics
(`idxmin`/`idxmax`/`argmin`/`argmax` on a frame carrying test metrics), edits to
hashed files (sidecar mismatch), confirmation results present without a
hash-valid origin file, context series inside target aggregates, and — in
`--diff` mode — changes to frozen split/metric/exclusion constants. Exit code 1
on any S1 finding. `Vangja-project/` is excluded by default because it is the
read-only archive; pass `--include-readonly` to scan it.

## Review checklist

- [ ] Does the diff change a frozen constant? If yes, is there an amendment?
- [ ] Is any selection criterion computed on `test`/confirmation rows?
- [ ] Are excluded units and failed fits counted and reported?
- [ ] Are context series filtered from every aggregate?
- [ ] Does every reported number point at an artifact + manifest SHA?
- [ ] Are baselines named as what they actually are?
