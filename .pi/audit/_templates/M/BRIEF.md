# WS-M — Methodology & experimental design — cycle {{cycle}}

> Finder brief. Read `SCOPE.md` and `_PROTOCOL.md` first. You are an independent
> reviewer: do not read another workstream's findings, the historical audit, or
> the calibration expectations. Rediscover or do not report.

## Objective

Find, with a reproducing command, every methodology & experimental design defect in the frozen
snapshot at `{{sha}}` that affects a claim, a result, or the integrity of the
artifact set. Write only to `.pi/audit/cycle-{{cycle}}/m/`.

## Boundary (hard)

Methodology & experimental design. Anything outside this boundary is a `cross-ref: <WS>-NNN` lead in your
appendix, not a finding body. Do not verify or grade your own findings.

## Coverage checklist

- [ ] Design validity: does the horizon framing, target/context split, and attribution match the protocol?
- [ ] Selection: no configuration, threshold, or exclusion chosen on the test horizon; validation split exists and is recorded.
- [ ] Baselines: every baseline is the package it claims to be, on identical data/splits, with comparable tuning effort.
- [ ] Uncertainty in comparisons: paired tests / CIs / multiplicity correction for a search grid.
- [ ] Negative transfer and generalisation limits are measured, not just acknowledged.
- [ ] Paper↔code correspondence for every experimental claim (this side of the claim is owned here; C owns traceability).

## Evidence per finding (all nine fields)

`location` (file:line or artifact id) · `snapshot` (cycle@SHA) · `command` (exact,
re-runnable) · `expected` (per paper/spec/sound practice) · `observed` (numbers) ·
`consequence` · `severity` S0–S3 · `confidence` high/medium/low · `status`
verified/to-verify/refuted/unresolved · `reproducible` yes | no: why · `evidence[]`.

**No command → not a finding.** A static-reading-only claim is `to-verify` at best.

## What to inspect

- [ ] `paper/CLAIMS.json` claim statements (methodology side)
- [ ] `fixed_case_studies/protocol/**`
- [ ] all runner selection code
- [ ] every result table's selection provenance

## Cap and ordering

At most **25 findings**, ordered S0 → S3. Everything else goes to
`APPENDIX_CANDIDATES.md` (one line each) and does not enter the register.

## Output

- `.pi/audit/cycle-{{cycle}}/m/FINDINGS.md`
- `.pi/audit/cycle-{{cycle}}/m/findings.json` (via `finding_register`)
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
