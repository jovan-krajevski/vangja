# WS-C — Claims ↔ artifacts traceability — cycle {{cycle}}

> Finder brief. Read `SCOPE.md` and `_PROTOCOL.md` first. You are an independent
> reviewer: do not read another workstream's findings, the historical audit, or
> the calibration expectations. Rediscover or do not report.

## Objective

Find, with a reproducing command, every claims ↔ artifacts traceability defect in the frozen
snapshot at `{{sha}}` that affects a claim, a result, or the integrity of the
artifact set. Write only to `.pi/audit/cycle-{{cycle}}/c/`.

## Boundary (hard)

Claims ↔ artifacts traceability. Anything outside this boundary is a `cross-ref: <WS>-NNN` lead in your
appendix, not a finding body. Do not verify or grade your own findings.

## Coverage checklist

- [ ] Every number/claim in `paper/**` maps to `paper/CLAIMS.json` and an artifact with a manifest SHA.
- [ ] Every artifact SHA is an ancestor of the audited tag.
- [ ] Statement conformance: the code computes what the paper says it computes (estimator, prior, pooling, horizon).
- [ ] No hand-typed table/figure numbers; generated includes are the source.
- [ ] Claims that changed since the previous cycle are listed with the reason.
- [ ] A number with no artifact is a finding; a claim with no number is checked for overstatement.

## Evidence per finding (all nine fields)

`location` (file:line or artifact id) · `snapshot` (cycle@SHA) · `command` (exact,
re-runnable) · `expected` (per paper/spec/sound practice) · `observed` (numbers) ·
`consequence` · `severity` S0–S3 · `confidence` high/medium/low · `status`
verified/to-verify/refuted/unresolved · `reproducible` yes | no: why · `evidence[]`.

**No command → not a finding.** A static-reading-only claim is `to-verify` at best.

## What to inspect

- [ ] `paper/**` numbers and tables
- [ ] `paper/CLAIMS.json`
- [ ] artifact manifests and ancestry
- [ ] statement-conformance probes

## Cap and ordering

At most **25 findings**, ordered S0 → S3. Everything else goes to
`APPENDIX_CANDIDATES.md` (one line each) and does not enter the register.

## Output

- `.pi/audit/cycle-{{cycle}}/c/FINDINGS.md`
- `.pi/audit/cycle-{{cycle}}/c/findings.json` (via `finding_register`)
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
