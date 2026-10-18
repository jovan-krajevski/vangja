# Historical calibration — `main@8028c51`

This is the **real** calibration run of the audit harness against the frozen
historical snapshot, not a fixture. It was produced by
`.pi/calibrate.sh --run-finders --narrowed` (pinned model, 900 s per-finder wall
clock) with the read-only worktree `.pi/state/calibration/snapshot`.

- Workstreams: S, M, R (the calibration subset).
- Expected findings: `../expected.json` (WS-S-001/002/003, WS-M-001/002,
  WS-R-001/004) and four rejected Phase-A leads (F-012, F-014, F-015, F-016).
- Result: **UNCALIBRATED** — recall 6/7 = 0.857 (threshold 0.6), but the M
  finder confirmed **F-015** (the superseded "Prophet under-tuned" fairness
  framing) in `M-010` alongside the stronger `M-002` identity finding.
- Consequence: the orchestrator refuses to grade (`UNCALIBRATED`); the narrower
  M brief must report the identity finding only, and S must re-derive
  `WS-S-003` (the `lam` inconsistency), before a graded cycle.
- The fixture-mode runs in `.pi/tests/audit/fixtures/calibration/` separately
  prove the calibrated path (strong fixture, recall 1.0) and the weakened-brief
  refusal path (recall 0.29 → exit 10).

`REPORT.md` and `recall.json` were written by
`auditkit/calibrate_cli.ts`; the three `findings/` registers are the finder
output the scorer consumed.

## Archived probes

The finders wrote their numeric probes under the runtime cycle directory
`.pi/audit/cycle-900/<ws>/probes/` (gitignored). They are archived verbatim under
`probes/<ws>/` so a reader can inspect what produced the observations. The
`command` fields in `findings/*/findings.json` name the runtime path; replace
`.pi/audit/cycle-900/<ws>/probes` with
`.pi/audit/calibration/historical-8028c51/probes/<ws>` to re-run them.
