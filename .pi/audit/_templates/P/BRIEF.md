# WS-P — Probabilistic UQ & calibration — cycle {{cycle}}

> Finder brief. Read `SCOPE.md` and `_PROTOCOL.md` first. You are an independent
> reviewer: do not read another workstream's findings, the historical audit, or
> the calibration expectations. Rediscover or do not report.

## Objective

Find, with a reproducing command, every probabilistic uq & calibration defect in the frozen
snapshot at `{{sha}}` that affects a claim, a result, or the integrity of the
artifact set. Write only to `.pi/audit/cycle-{{cycle}}/p/`.

## Boundary (hard)

Probabilistic UQ & calibration. Anything outside this boundary is a `cross-ref: <WS>-NNN` lead in your
appendix, not a finding body. Do not verify or grade your own findings.

## Coverage checklist

- [ ] Posterior predictive validity: does the predictive distribution cover held-out targets?
- [ ] Coverage / sharpness / CRPS on a held-out horizon, with the number of units and the interval levels stated.
- [ ] MAP-heuristic uncertainty is disclosed as such; no interval is called a posterior unless it is one.
- [ ] Sampler diagnostics: R-hat, ESS bulk/tail, divergences, tree depth, BFMI for every reported fit.
- [ ] ADVI mean-field: a diagonal approximation cannot support a covariance-preservation claim.
- [ ] Calibration of the transfer prior: does the transferred prior match the target posterior it claims to approximate?

## Evidence per finding (all nine fields)

`location` (file:line or artifact id) · `snapshot` (cycle@SHA) · `command` (exact,
re-runnable) · `expected` (per paper/spec/sound practice) · `observed` (numbers) ·
`consequence` · `severity` S0–S3 · `confidence` high/medium/low · `status`
verified/to-verify/refuted/unresolved · `reproducible` yes | no: why · `evidence[]`.

**No command → not a finding.** A static-reading-only claim is `to-verify` at best.

## What to inspect

- [ ] posterior predictive checks
- [ ] calibration/coverage scripts and outputs
- [ ] sampler diagnostics CSVs
- [ ] MAP/ADVI disclosure in paper and docs

## Cap and ordering

At most **25 findings**, ordered S0 → S3. Everything else goes to
`APPENDIX_CANDIDATES.md` (one line each) and does not enter the register.

## Output

- `.pi/audit/cycle-{{cycle}}/p/FINDINGS.md`
- `.pi/audit/cycle-{{cycle}}/p/findings.json` (via `finding_register`)
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
