# Seeded audit fixture

Synthetic, deterministic fixture for the S6 canary gate and the audit acceptance
tests. It contains eight deliberate defects (D1–D8) and one negative control
(N1); `expected.json` says which probe must find which defect. No real data,
no model fits, and no network access.

Run the probes with:

```bash
node .pi/tests/audit/run_canary.ts
```

The canary gate (`run_gates`, id `canary`) runs the same probes on every audit
cycle to prove the harness is still sensitive before it is trusted to grade.
