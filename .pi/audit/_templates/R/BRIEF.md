# WS-R — Reproducibility & provenance — cycle {{cycle}}

> Finder brief. Read `SCOPE.md` and `_PROTOCOL.md` first. You are an independent
> reviewer: do not read another workstream's findings, the historical audit, or
> the calibration expectations. Rediscover or do not report.

## Objective

Find, with a reproducing command, every reproducibility & provenance defect in the frozen
snapshot at `{{sha}}` that affects a claim, a result, or the integrity of the
artifact set. Write only to `.pi/audit/cycle-{{cycle}}/r/`.

## Boundary (hard)

Reproducibility & provenance. Anything outside this boundary is a `cross-ref: <WS>-NNN` lead in your
appendix, not a finding body. Do not verify or grade your own findings.

## Coverage checklist

- [ ] Environment: lockfile, pinned PyMC/NumPy/Numba, container hashes, CI-equivalent run.
- [ ] Seeds: recorded and fixed before the run for every sampler and runner.
- [ ] SHAs: every artifact's manifest commit is an ancestor of the audited tag.
- [ ] Manifests: schema valid, files hashed, missing outputs explicit.
- [ ] Ledger integrity: events append-only, completion events idempotent, no gaps.
- [ ] One-command reproduction: the documented command actually rebuilds every headline number.

## Evidence per finding (all nine fields)

`location` (file:line or artifact id) · `snapshot` (cycle@SHA) · `command` (exact,
re-runnable) · `expected` (per paper/spec/sound practice) · `observed` (numbers) ·
`consequence` · `severity` S0–S3 · `confidence` high/medium/low · `status`
verified/to-verify/refuted/unresolved · `reproducible` yes | no: why · `evidence[]`.

**No command → not a finding.** A static-reading-only claim is `to-verify` at best.

## What to inspect

- [ ] `pyproject.toml`, `uv.lock`, container definitions
- [ ] seeds in `fit()` and every runner
- [ ] artifact manifests and commit SHAs
- [ ] loop ledgers and gate runs

## Cap and ordering

At most **25 findings**, ordered S0 → S3. Everything else goes to
`APPENDIX_CANDIDATES.md` (one line each) and does not enter the register.

## Output

- `.pi/audit/cycle-{{cycle}}/r/FINDINGS.md`
- `.pi/audit/cycle-{{cycle}}/r/findings.json` (via `finding_register`)
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
