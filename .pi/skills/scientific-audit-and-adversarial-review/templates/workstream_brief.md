# Workstream brief — {{WS}} · {{NAME}}

- Snapshot: `audit-cycle/<n>` tag @ {{SHA}}; read-only. Do not edit the code.
- Owner: {{OWNER}} (cross-boundary leads → `cross-ref`, never a finding body)
- Find cap: 25 findings, ordered by severity; extras go to an appendix list.
- Effort cap: {{EFFORT}} (stop and write the coverage gap if exceeded).

## Scope (hard boundaries)

{{SCOPE}}

## Out of scope

{{OUT_OF_SCOPE}}

## Coverage checklist (every box must be answered or explicitly skipped)

{{CHECKLIST}}

## Evidence required for every finding

- exact `file:line` (or artifact path + config/seed), and the snapshot SHA;
- the exact command run (or "static reading of lines X–Y");
- observed output (trimmed) or numbers;
- severity (S0–S3) + confidence (high/medium/low) + status
  (verified/to-verify/refuted/unresolved);
- for `unresolved`: a question in `QUESTIONS_FOR_AUTHORS.md` tagged
  `[blocking]`/`[non-blocking]`.

## Output contract

- `{{WS}}/FINDINGS.md` — human-readable, schema-ordered (S0 first).
- `{{WS}}/findings.json` — machine-readable list, valid under
  `validate_findings.py --workstream {{WS}}`.
- A closing "What I did not check" section.

## Prohibitions

- No code edits, no re-running science to produce new results, no copying
  another workstream's finding, no severity inflation, no finding without a
  command, no reliance on prior sessions' conclusions.
