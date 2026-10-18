# WS-H — Documentation & hygiene — cycle {{cycle}}

> Finder brief. Read `SCOPE.md` and `_PROTOCOL.md` first. You are an independent
> reviewer: do not read another workstream's findings, the historical audit, or
> the calibration expectations. Rediscover or do not report.

## Objective

Find, with a reproducing command, every documentation & hygiene defect in the frozen
snapshot at `{{sha}}` that affects a claim, a result, or the integrity of the
artifact set. Write only to `.pi/audit/cycle-{{cycle}}/h/`.

## Boundary (hard)

Documentation & hygiene. Anything outside this boundary is a `cross-ref: <WS>-NNN` lead in your
appendix, not a finding body. Do not verify or grade your own findings.

## Coverage checklist

- [ ] README/docs claims match behaviour (features that exist, commands that work).
- [ ] Docstrings and units: daily max vs mean, Kelvin vs Fahrenheit, per-group vs total.
- [ ] Naming honesty: a baseline/estimator is called what it is.
- [ ] Notebooks and docs build: Sphinx toctree, stale cells, code that no longer runs.
- [ ] Licence, packaging metadata, repository layout, .gitignore, no secrets or data dumps.
- [ ] Every example in the docs is executed or explicitly marked as illustrative.

## Evidence per finding (all nine fields)

`location` (file:line or artifact id) · `snapshot` (cycle@SHA) · `command` (exact,
re-runnable) · `expected` (per paper/spec/sound practice) · `observed` (numbers) ·
`consequence` · `severity` S0–S3 · `confidence` high/medium/low · `status`
verified/to-verify/refuted/unresolved · `reproducible` yes | no: why · `evidence[]`.

**No command → not a finding.** A static-reading-only claim is `to-verify` at best.

## What to inspect

- [ ] README, docs/, notebooks
- [ ] docstrings and units
- [ ] licence and packaging metadata
- [ ] naming of baselines and estimators

## Cap and ordering

At most **25 findings**, ordered S0 → S3. Everything else goes to
`APPENDIX_CANDIDATES.md` (one line each) and does not enter the register.

## Output

- `.pi/audit/cycle-{{cycle}}/h/FINDINGS.md`
- `.pi/audit/cycle-{{cycle}}/h/findings.json` (via `finding_register`)
- A coverage statement: what you inspected, what you did not, and why.

## Prohibitions

- Never copy the historical audit; it is the calibration scorer's reference.
- Never use the test horizon to pick a configuration, metric, or exclusion.
- Never edit the snapshot, another workstream's files, or the paper.
- Never report a count without the command that produces it.

## Stop conditions

- Checklist covered and both files validate: stop.
- Cap or boundary reached: record the lead, stop.
- A candidate cannot be reproduced: downgrade with the failed command, or drop it.
