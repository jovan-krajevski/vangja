---
description: Refute or confirm the audit findings
argument-hint: "[cycle]"
---
Verify cycle `${1:-<n>}`. Run this as an `audit_verifier` work unit over
`audit/cycle-<n>/*/findings.json`, starting with S0/S1/S2.

For each finding: reopen the cited `file:line`/artifact at the snapshot SHA,
re-run the cited command, and try to refute it by changing one assumption at a
time. Record `confirmed` / `weakened` / `refuted` / `unresolved` with the exact
command and raw output, and correct the nine fields when your evidence differs.
Unrunnable findings are `unresolved`, never confirmed from prose. Use
`Vangja-project/audit/**` only to notice gaps, never to confirm.

Acceptance evidence: `audit/cycle-<n>/VERIFY.md` with per-finding verdicts +
commands + observed output, a severity × verdict count table, and the unresolved
list. Add no new primary claims, no grade, no fix list — synthesis is phase 3.
