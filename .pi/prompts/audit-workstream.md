---
description: Run one audit finder workstream
argument-hint: "<M|S|P|D|I|R|H|E|C|L> [cycle]"
---
Workstream: `${1:-<ws>}`, cycle `${2:-<n>}`. Run this as one `audit_finder` work
unit on the frozen snapshot; read `SCOPE.md` and write `BRIEF.md` from the §3
scope table before probing.

Probe with the narrowest falsifying command per hypothesis; fill the nine-field
schema (`location`, `snapshot`, `command`, `expected`, `observed`, `consequence`,
`severity`, `confidence`, `status`); no command → not a finding. Cap at 25
findings ordered by severity; cross-boundary leads become `cross-ref: <WS>-NNN`.
Record unresolved questions with `[blocking]`/`[non-blocking]`.

Acceptance evidence: `audit/cycle-<n>/<ws>/FINDINGS.md` and `findings.json`,
schema-valid via `auditkit`, plus the coverage statement (what was inspected,
what was not). Never read `Vangja-project/audit/**` and never copy a historical
finding; never verify or synthesise your own work.
