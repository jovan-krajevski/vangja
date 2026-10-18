# Calibration — UNCALIBRATED

- Mode: finder
- Snapshot: 8028c5108d926c58ec4e9b29f72842f2b9d8d095 (read-only worktree: yes)
- Findings source: .pi/audit/calibration/historical-8028c51/findings
- Expected source: .pi/audit/calibration/expected.json
- Generated: 2026-10-17T14:56:16.782Z
- Note: real calibration run: workstreams S/M/R against main@8028c51; narrowed brief after the first miss; 900s per-finder wall clock. UNCALIBRATED because the M finder confirmed the superseded F-015 fairness framing alongside the stronger WS-M-002 identity finding.

**Recall:** 6/7 = 0.857 (threshold 0.6)
**Rejected Phase-A leads confirmed:** 1
**Verdict:** the orchestrator refuses to grade; narrow the briefs and re-run S/M/R

## Per-workstream recall

| Workstream | Expected | Hit | Recall |
|---|---:|---:|---:|
| S | 3 | 2 | 0.67 |
| M | 2 | 2 | 1.00 |
| R | 2 | 2 | 1.00 |

## Expected findings

| Expected | Workstream | Result | Finding |
|---|---|---|---|
| WS-S-001 | S | HIT | S-006 |
| WS-S-002 | S | HIT | S-001 |
| WS-S-003 | S | MISS (FourierSeasonality regularization scale lam is inconsistent across pooling modes) | — |
| WS-M-001 | M | HIT | M-001 |
| WS-M-002 | M | HIT | M-002 |
| WS-R-001 | R | HIT | R-001 |
| WS-R-004 | R | HIT | R-002 |

## Rejected Phase-A leads (negative controls)

| Lead | Result | Finding |
|---|---|---|
| F-012 | not confirmed (correct) | — |
| F-014 | not confirmed (correct) | — |
| F-015 | CONFIRMED (false positive): M-010 | case_studies/smart_home/run.py:12-19: The Vangja runner searches 2592 configurations (2 x 2 x 2 x 6 x 3 x 3 x 2 x 3), all selected on the test horizon. The paper describes the primary Prophet baseline as 'A default configuration Prophet m |
| F-016 | not confirmed (correct) | — |

## Why this run may not grade

- 1 rejected Phase-A lead(s) were confirmed: F-015.

Narrowed briefs:
- WS-S: re-derive WS-S-003 with the command named in the historical register.
- WS-M: re-derive F-015 (superseded framing; report the stronger identity finding, not this one) with the command named in the historical register.
- Every finding must be re-derived from the snapshot; the historical text is not a source.
