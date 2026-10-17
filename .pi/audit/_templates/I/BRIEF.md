# WS-I — Implementation & software quality — cycle {{cycle}}

> Finder brief. Read `SCOPE.md` and `_PROTOCOL.md` first. You are an independent
> reviewer: do not read another workstream's findings, the historical audit, or
> the calibration expectations. Rediscover or do not report.

## Objective

Find, with a reproducing command, every implementation & software quality defect in the frozen
snapshot at `{{sha}}` that affects a claim, a result, or the integrity of the
artifact set. Write only to `.pi/audit/cycle-{{cycle}}/i/`.

## Boundary (hard)

Implementation & software quality. Anything outside this boundary is a `cross-ref: <WS>-NNN` lead in your
appendix, not a finding body. Do not verify or grade your own findings.

## Coverage checklist

- [ ] API correctness: public signatures, types, defaults, dispatch tables, error handling.
- [ ] Tests: numerical-correctness tests, not only smoke tests; no weakened or contradictory tests.
- [ ] Dead paths and unused parameters; documented knobs that do nothing.
- [ ] Dependency behaviour: imports that the declared dependency set cannot satisfy.
- [ ] Runtime probes with the pinned environment; reproduce every crash before reporting it.
- [ ] Packaging: sdist contents, version/tag consistency, Python-version support.

## Evidence per finding (all nine fields)

`location` (file:line or artifact id) · `snapshot` (cycle@SHA) · `command` (exact,
re-runnable) · `expected` (per paper/spec/sound practice) · `observed` (numbers) ·
`consequence` · `severity` S0–S3 · `confidence` high/medium/low · `status`
verified/to-verify/refuted/unresolved · `reproducible` yes | no: why · `evidence[]`.

**No command → not a finding.** A static-reading-only claim is `to-verify` at best.

## What to inspect

- [ ] `src/**`, `tests/**`, packaging
- [ ] dispatch tables and sampling backends
- [ ] runtime probes on the pinned stack
- [ ] sdist/tag/CI configuration

## Cap and ordering

At most **25 findings**, ordered S0 → S3. Everything else goes to
`APPENDIX_CANDIDATES.md` (one line each) and does not enter the register.

## Output

- `.pi/audit/cycle-{{cycle}}/i/FINDINGS.md`
- `.pi/audit/cycle-{{cycle}}/i/findings.json` (via `finding_register`)
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
