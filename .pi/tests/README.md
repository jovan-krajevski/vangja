# `.pi/tests` — the environment test suite

Runs with Node's built-in test runner and TypeScript type stripping. No build
step, no dependencies, no network.

```bash
make -C .pi test          # 30 tests: settings, schemas, guards, ledger, status, child pi
make -C .pi env-check     # config/roster/secrets checks with PASS lines
make -C .pi verify-env    # both
make -C .pi accept        # model-backed acceptance checks (~4 short model turns)
```

| File | What it proves |
|---|---|
| `test_settings.test.ts` | `.pi/settings.json` pins model/thinking/enabledModels/compaction, uses conventional discovery, and every key exists in pi's settings schema. |
| `test_schemas.test.ts` | Every `.pi/config/*.json` validates; budgets equal `plan/02` §6; the agent roster is complete and pinned; HPC stays single-slot. |
| `test_guards_policy.test.ts` | Model-pin drift, protected paths (+ amendments, PAPER, supervisor), secret paths and content, destructive bash, shell write targets, the full role matrix, the 5-line preamble. |
| `test_guards_wiring.test.ts` | The real guard extension registered against a fake `ExtensionAPI`: drift blocks input/tools and exits 3, paths/secrets/bash/roles are refused with the alternative named, bash output is redacted, the preamble section is injected. |
| `test_ledger.test.ts` | Append/read round-trip, append-only projections, brief and experiment gates, lifecycle transitions, role denials, state machine, deterministic views, bounded reads. |
| `test_ledger_concurrency.test.ts` | 200 appends from 4 processes → 200 untorn unique-seq lines; `LOOP_STATE.json` survives 100 concurrent read/writes. |
| `test_status.test.ts` | TUI/headless status lines for uninitialized, working, and paused states. |
| `test_child_pi.test.ts` | Real `pi` children with a drifted model/thinking are blocked before any model turn (offline; exit 3, no stdout, stderr names `model.json`). |
| `check_env.ts` | Environment summary with PASS/FAIL lines; `--status` prints the headless status fallback. |
| `fixtures/` | Worker scripts used by the concurrency test. |
| `acceptance/run_acceptance.sh` | Model-backed checks: protected/secret/destructive tool calls refused in a real session, role preamble visible to the model, trusted startup warning-free. |

Scratch roots are created under the system temp directory; the suite never writes
inside the repository.
