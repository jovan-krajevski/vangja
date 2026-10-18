# Calibration brief — workstreams S, M, R

> Used by `.pi/calibrate.sh --run-finders`. The finders do not see the expected
> list; the scorer holds it (`.pi/audit/calibration/expected.json`).

- Snapshot: read-only worktree at `.pi/state/calibration/snapshot`, `main@8028c51`.
- Workstreams: S, M, R only (the cheap calibration subset).
- Output: `finding_register(cycle=900, workstream=<WS>)` in the collector root.
- Rules: the normal WS brief applies; nine fields; no command → not a finding;
  cap 25; do not read `.pi/audit/calibration/**` or `Vangja-project/audit/**`.
- The scorer computes recall over the expected historical S1 list and flags any
  confirmed rejected Phase-A lead. Recall < 0.6 or a confirmed rejected lead
  writes `UNCALIBRATED`, and the orchestrator refuses to grade.
