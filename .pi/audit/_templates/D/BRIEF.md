# WS-D — Data handling & leakage — cycle {{cycle}}

> Finder brief. Read `SCOPE.md` and `_PROTOCOL.md` first. You are an independent
> reviewer: do not read another workstream's findings, the historical audit, or
> the calibration expectations. Rediscover or do not report.

## Objective

Find, with a reproducing command, every data handling & leakage defect in the frozen
snapshot at `{{sha}}` that affects a claim, a result, or the integrity of the
artifact set. Write only to `.pi/audit/cycle-{{cycle}}/d/`.

## Boundary (hard)

Data handling & leakage. Anything outside this boundary is a `cross-ref: <WS>-NNN` lead in your
appendix, not a finding body. Do not verify or grade your own findings.

## Coverage checklist

- [ ] Splits and origins: development vs confirmation separation, frozen origins, no test rows in training windows.
- [ ] Scaling: fit on train only, per series, no leakage across the split.
- [ ] Interpolation and joins: split-first, within-split only; no future prices in the training window.
- [ ] Context cutoffs: `max(T_X) = max(T_C)`, context re-fit per window, context series excluded from target aggregates.
- [ ] Availability and exclusions: delisted tickers, missing days, failed units — all reported, never dropped silently.
- [ ] Data hashes: every input identifiable by a recorded hash; unavailable data is reported as unavailable.

## Evidence per finding (all nine fields)

`location` (file:line or artifact id) · `snapshot` (cycle@SHA) · `command` (exact,
re-runnable) · `expected` (per paper/spec/sound practice) · `observed` (numbers) ·
`consequence` · `severity` S0–S3 · `confidence` high/medium/low · `status`
verified/to-verify/refuted/unresolved · `reproducible` yes | no: why · `evidence[]`.

**No command → not a finding.** A static-reading-only claim is `to-verify` at best.

## What to inspect

- [ ] `src/vangja/datasets/**`
- [ ] split/scale/resample code
- [ ] context-series handling
- [ ] availability/failure accounting

## Cap and ordering

At most **25 findings**, ordered S0 → S3. Everything else goes to
`APPENDIX_CANDIDATES.md` (one line each) and does not enter the register.

## Output

- `.pi/audit/cycle-{{cycle}}/d/FINDINGS.md`
- `.pi/audit/cycle-{{cycle}}/d/findings.json` (via `finding_register`)
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
