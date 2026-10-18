# WS-S — Statistics & math — cycle {{cycle}}

> Finder brief. Read `SCOPE.md` and `_PROTOCOL.md` first. You are an independent
> reviewer: do not read another workstream's findings, the historical audit, or
> the calibration expectations. Rediscover or do not report.

## Objective

Find, with a reproducing command, every statistics & math defect in the frozen
snapshot at `{{sha}}` that affects a claim, a result, or the integrity of the
artifact set. Write only to `.pi/audit/cycle-{{cycle}}/s/`.

## Boundary (hard)

Statistics & math. Anything outside this boundary is a `cross-ref: <WS>-NNN` lead in your
appendix, not a finding body. Do not verify or grade your own findings.

## Coverage checklist

- [ ] Formulas vs code: trend, changepoints, Fourier terms, operators, scaling.
- [ ] Transfer moments: `parametric` = posterior-moment matching (mean/sd), not the MAP mode; `prior_from_idata` preserves covariance.
- [ ] Pooling parameterisation: complete/partial/individual, non-centered parameterisation, shrinkage hyperpriors, dims.
- [ ] Regularisation potentials: sign, one-sidedness, `loss_factor` scaling, per-pooling-mode consistency.
- [ ] Estimator correctness: MAP vs NUTS vs ADVI, `target_accept`, dispatch, non-differentiable kinks.
- [ ] Numeric probes with fixtures; check tests actually assert the mathematics.

## Evidence per finding (all nine fields)

`location` (file:line or artifact id) · `snapshot` (cycle@SHA) · `command` (exact,
re-runnable) · `expected` (per paper/spec/sound practice) · `observed` (numbers) ·
`consequence` · `severity` S0–S3 · `confidence` high/medium/low · `status`
verified/to-verify/refuted/unresolved · `reproducible` yes | no: why · `evidence[]`.

**No command → not a finding.** A static-reading-only claim is `to-verify` at best.

## What to inspect

- [ ] `src/vangja/**` component and time-series code
- [ ] `tests/test_*.py` numerical assertions
- [ ] transfer and pooling definitions
- [ ] numeric probes for every formula claim

## Cap and ordering

At most **25 findings**, ordered S0 → S3. Everything else goes to
`APPENDIX_CANDIDATES.md` (one line each) and does not enter the register.

## Output

- `.pi/audit/cycle-{{cycle}}/s/FINDINGS.md`
- `.pi/audit/cycle-{{cycle}}/s/findings.json` (via `finding_register`)
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
