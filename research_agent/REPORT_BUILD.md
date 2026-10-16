# Build report — Vangja research-agent environment

Append-only. One section per build stage (S0–S7), written by the prompts in
`research_agent/prompts/`. Each entry: what was created, the acceptance command
and its observed result, deviations from `research_agent/plan/08_BUILD_PLAN.md`,
and open items.

The final entry is the `GO / NO-GO` summary required by prompt 07.

---

## S0 — scaffold and guards

Built in one session with the pinned model (`opencode-go/deepseek-v4.1-flash`,
thinking `max`), on branch `update_pymc`, working tree clean except for `.pi/**`,
`research_agent/**`, and `.gitignore`.

### Created

Configuration (versioned):

- `.pi/settings.json` — model pin, `enabledModels`, `defaultThinkingLevel: max`,
  `defaultTools: ["+codemode"]`, `compaction.reserveTokens: 32768`,
  `compaction.keepRecentTokens: 120000`, `$schema`.
- `.pi/config/model.json` — canonical pin: `opencode-go/deepseek-v4.1-flash`,
  `max`, `contextWindow: 1000000`, `enforce: true`.
- `.pi/config/budgets.json` — the nine caps of `plan/02_ARCHITECTURE.md` §6 verbatim.
- `.pi/config/project.json` — repo root `.`, base branch `update_pymc`, venue
  *Journal of Forecasting*, `requireHumanMerge: true`, `dryRun: false`, sandbox /
  session / artifact / state / worktree dirs.
- `.pi/config/agents.json` — the full roster: 3 primary + 4 audit phases + 9 workers
  (16 agents), each `{name, file, role, tools[], skills[], effort, maxParallel, model}`;
  every model is the pin; agent `.md` files arrive in S3.
- `.pi/config/hpc.example.json` — `host: ""`, `wallClockHours: 24`, `maxParallelJobs: 1`.
- `.pi/config/README.md` — every field documented, versioned vs runtime-only.

Shared library (dependency-free, `node`-importable):

- `.pi/lib/{errors,paths,io,config,roles,secrets,records,views,state}.ts` — typed
  errors that name the file; repo-root/state-path resolution (`PI_LOOP_ROOT`
  override); lock-directory + fsync + temp/rename IO; config loaders/validators;
  the role×capability matrix; secret patterns/redaction; append-only record
  schemas, ids, projections and experiment lifecycle; deterministic Markdown
  views; the validated state machine.

Extensions (TypeScript, jiti, no build step):

- `.pi/extensions/guards/{index,policy,protected,decision,bash}.ts` — model/thinking
  pin on `session_start`/`model_select`/`thinking_level_select`/`before_agent_start`
  (input dropped, tools blocked, process exit 3), 5-line `<loop-role>` preamble,
  protected paths, destructive-bash policy, secret path/content scan, bash-output
  redaction, role gates.
- `.pi/extensions/ledger/{index,handlers,store}.ts` — `ledger_read`,
  `ledger_append`, `task_enqueue`, `task_close`, `pr_open`, `review_write`,
  `experiment_queue`, `experiment_update`, `decision_record`, `waiver_record`,
  `amendment_record`, `state_get`, `state_set`; JSONL under `.pi/state/`, every
  append locked, validated, fsynced; `LOOP_STATE.json` written temp+fsync+rename;
  derived `CONTEXT.md`/`TASKS.md`/`PRS.md`/`REVIEWS.md`/`EXPERIMENTS.md`/`DECISIONS.md`;
  bounded reads that name the full ledger.
- `.pi/extensions/status/{index,status}.ts` — TUI status line + widget (state, task,
  experiment+attempt, budget, gates, STOP) and a `renderResult` for ledger tools.

Tests and tooling:

- `.pi/tests/` — 64 tests across `test_settings`, `test_schemas`,
  `test_guards_policy`, `test_guards_wiring`, `test_ledger`,
  `test_ledger_concurrency`, `test_status`, `test_child_pi`, plus fixtures and
  `check_env.ts`; `acceptance/run_acceptance.sh` + `acceptance/pem_fixture_check.ts`
  for the model-backed checks; `README.md`.
- `.pi/Makefile` — `test`, `env-check`, `verify-env`, `verify-env-strict`, `accept`, `all`.
- `.pi/loop/{step,status,collect}.sh` + `README.md` — thin shims that exec the S4/S5
  orchestrator CLI and fail loudly with "not implemented until S4/S5" until it exists.
- `.pi/README.md` — what is here, what is versioned, how to verify, security notes,
  open items.
- `.gitignore` — `.pi/state/`, `.pi/artifacts/*/`, `.pi/config/hpc.json`,
  `.pi/**/node_modules/`, `.pi/loop/worktrees/`, `.pi/sessions/`, `*.sif`, `*.pdf`.

### Acceptance

**1. `make -C .pi verify-env` → all green (exit 0; S4 child-command check is an explicit stub).**

```
$ make -C .pi verify-env
make: Entering directory '/home/jovan/repos/vangja/.pi'
== node --test (offline) ==
cd /home/jovan/repos/vangja && node --test .pi/tests/test_settings.test.ts .pi/tests/test_schemas.test.ts .pi/tests/test_guards_policy.test.ts .pi/tests/test_guards_wiring.test.ts .pi/tests/test_ledger.test.ts .pi/tests/test_ledger_concurrency.test.ts .pi/tests/test_status.test.ts .pi/tests/test_child_pi.test.ts
✔ a child pi with the wrong model is blocked and names model.json (639.877802ms)
✔ a child pi with the wrong thinking level is blocked (395.750085ms)
✔ model pin accepts the pinned model and rejects drift (2.308861ms)
✔ protected paths are refused with an allowed alternative (1.028789ms)
✔ protected-path exceptions: amendment, PAPER state, supervisor AGENTS.md (0.383793ms)
✔ normal working paths stay writable (0.270811ms)
✔ secret-bearing paths are refused (1.576531ms)
✔ secret content is detected and redacted without echoing the value (2.227638ms)
✔ destructive bash: force push, history rewrite, rebase, reset (3.004912ms)
✔ destructive bash: rm -rf, ~/.ssh writes, curl|sh, env exfiltration, chmod 777 (2.076994ms)
✔ shell writes into protected paths are refused (3.215019ms)
✔ role matrix matches plan/05 §7 for every capability (0.937607ms)
✔ role preamble is exactly five lines and names the role, state, and task (0.459266ms)
✔ guards never resolve relative paths against the wrong root (0.282633ms)
[guard:MODEL_PIN] model/thinking drift: model is "gpt-5.6-luna" but the pin requires "deepseek-v4.1-flash". The pin is enforced for every session (plan/02_ARCHITECTURE.md §7). Fix .pi/config/model.json, or relaunch with "--provider opencode-go --model opencode-go/deepseek-v4.1-flash --thinking max".
[guard:MODEL_PIN] model/thinking drift: thinking level is "low" but the pin requires "max". The pin is enforced for every session (plan/02_ARCHITECTURE.md §7). Fix .pi/config/model.json, or relaunch with "--provider opencode-go --model opencode-go/deepseek-v4.1-flash --thinking max".
[guard:MODEL_PIN] cannot load the model pin: required config file is missing: /tmp/vangja-guard-qI4hqH/.pi/config/model.json. Create it from the schema in .pi/config/README.md.. Every session must run with the pin recorded in .pi/config/model.json (plan/02_ARCHITECTURE.md §7).
✔ a drifting model is blocked before any model turn (11.566628ms)
✔ a drifting thinking level is blocked and the message names model.json (1.94759ms)
✔ a missing or malformed model pin fails closed (1.679946ms)
✔ the pinned model passes and normal tool calls are allowed (16.754113ms)
✔ protected paths are refused on write and edit with the alternative named (2.27116ms)
✔ AGENTS.md is supervisor-only (3.174783ms)
✔ paper/** opens only in the PAPER state (2.214864ms)
✔ secret paths and secret content are refused without echoing the value (2.102502ms)
✔ destructive bash commands are refused (4.285967ms)
✔ role gates refuse merge_pr for phd_candidate and hpc_submit for collector (2.378062ms)
✔ the role preamble is injected as a five-line <loop-role> section (1.277738ms)
✔ PI_LOOP_STATE is authoritative for the preamble of a child process (0.943708ms)
✔ bash output is redacted, never echoed (0.983423ms)
✔ round-trip: append then read, with assigned id and seq (9.548785ms)
✔ append-only updates repeat the id and projectLatest keeps the newest (3.711885ms)
✔ task_enqueue rejects briefs without acceptance criteria or a known origin (3.130129ms)
✔ schema validation rejects a malformed record and names the field (1.224126ms)
✔ pr_open requires the machine-readable footer fields (1.906102ms)
✔ review_write is supervisor/conductor only and validates the decision (1.421289ms)
✔ experiment_queue enforces the single-slot rule and checkpoint metadata (2.768517ms)
✔ experiment_update only allows legal lifecycle transitions (4.870619ms)
✔ decisions, waivers, and amendments record ids and open a human gate (2.990375ms)
✔ state_get/state_set validate the state machine and write atomically (2.285407ms)
✔ state_set is conductor-only (0.442675ms)
✔ derived views are written deterministically (1.827805ms)
✔ ledger_read is bounded and reports where the full ledger lives (5.475118ms)
✔ unknown ledger kinds are refused with the valid list (0.375598ms)
✔ 200 concurrent appends from 4 processes yield 200 valid lines (591.598488ms)
✔ LOOP_STATE.json survives 100 concurrent reads and writes (375.568863ms)
✔ model.json loads and matches the pin (2.496485ms)
✔ model.json rejects a smaller context window while enforce=true (1.177679ms)
✔ model.json rejects a missing field naming the file (0.688868ms)
✔ budgets.json equals the documented defaults exactly (2.020087ms)
✔ budgets.json rejects an incomplete file (0.487559ms)
✔ project.json carries the base branch, venue, and gates (0.528185ms)
✔ agents.json has the full roster from plan/03_AGENTS.md (1.365994ms)
✔ hpc.example.json has the disabled-by-default schema (0.810748ms)
✔ loadHpcConfig reports disabled when hpc.json is absent (2.054372ms)
✔ loadHpcConfig rejects maxParallelJobs > 1 without a recorded amendment (2.968464ms)
✔ no committed config file contains a real credential (0.879277ms)
✔ settings.json parses and pins the model (2.115596ms)
✔ settings.json enables codemode on top of the default tools (0.47242ms)
✔ compaction keeps a large recent history and reserves the response budget (0.32901ms)
✔ settings.json uses conventional discovery (no explicit resource lists) (0.373224ms)
✔ every top-level settings key is known to pi's settings schema (172.16987ms)
✔ the model pin file agrees with settings.json and the agents manifest (0.319612ms)
✔ the repository root used by the tests is the real repo (0.155242ms)
✔ an uninitialized loop reports UNINITIALIZED without inventing state (3.308996ms)
✔ a working loop summarises state, task, experiment, budget, and gates (12.930999ms)
✔ a paused loop reports where it will resume from (3.776046ms)
ℹ tests 64
ℹ suites 0
ℹ pass 64
ℹ fail 0
ℹ cancelled 0
ℹ skipped 0
ℹ todo 0
ℹ duration_ms 1195.322868
== environment checks ==
cd /home/jovan/repos/vangja && node .pi/tests/check_env.ts
PASS  pi binary: pi 1.3.0 on PATH
PASS  settings pin: model pin, enabledModels, +codemode, and compaction values match plan/02 §7
PASS  model.json: opencode-go/deepseek-v4.1-flash thinking=max window=1000000 enforce=true
PASS  budgets.json: all 9 caps equal plan/02_ARCHITECTURE.md §6
PASS  project.json: Journal of Forecasting; base=update_pymc; requireHumanMerge=true; dryRun=false
PASS  agents.json: 16 agents, all on opencode-go/deepseek-v4.1-flash thinking=max; 0/16 definition files present (the rest arrive in S3)
PASS  hpc.example.json: host="" wall=24h singleSlot=1
PASS  hpc.json (runtime): disabled: no /home/jovan/repos/vangja/.pi/config/hpc.json; the HPC path is disabled
PASS  loop shims: step.sh, status.sh, collect.sh present and executable (S4/S5 stubs)
PASS  gitignore coverage: 6 runtime paths ignored
PASS  no key material under .pi: no PEM blocks or provider tokens in versioned .pi files
PASS  state root: no LOOP_STATE.json yet — the orchestrator (S4) creates it on BOOT

12/12 checks passed

STUB (S4): 'pi --print "/loop:verify-env"' is not available; the orchestrator
           (research_agent/prompts/04_orchestrator_and_state.md) has not been built.
           Use 'make -C .pi verify-env-strict' in CI to fail on this stub.

verify-env: environment tests green.
verify-env: NOTE the S4 stub above — the orchestrator does not exist yet.
make: Leaving directory '/home/jovan/repos/vangja/.pi'
```

**2. Wrong model (`opencode-go/gpt-5.6-luna`) → blocked, message names `model.json`, no model turn.**

```
$ PI_LOOP_ROOT=$PWD pi --print --no-session -na -e .pi/extensions/guards/index.ts \
    --provider opencode-go --model opencode-go/gpt-5.6-luna --thinking max \
    "Reply with exactly: SHOULD-NOT-RUN"
exit=3
stdout=[]
stderr:
[guard:MODEL_PIN] model/thinking drift: model is "gpt-5.6-luna" but the pin requires "deepseek-v4.1-flash". The pin is enforced for every session (plan/02_ARCHITECTURE.md §7). Fix .pi/config/model.json, or relaunch with "--provider opencode-go --model opencode-go/deepseek-v4.1-flash --thinking max".
```

The same command with `--model opencode-go/deepseek-v4.1-flash --thinking low` exits 3
with `thinking level is "low" but the pin requires "max"` (see `test_child_pi.test.ts`).

**3. Protected paths refused in real sessions (agent role, `--mode json`, tool result unescaped).**

```
$ bash .pi/tests/acceptance/run_acceptance.sh
== acceptance: model-backed guard checks (pinned model, no repo writes expected) ==
workdir: /tmp/vangja-acceptance.99563

PASS  Vangja-project/** write refused for phd_candidate  (evidence: /tmp/vangja-acceptance.99563/archive-write.jsonl)
PASS  paper/** write refused while PI_LOOP_STATE=WORK  (evidence: /tmp/vangja-acceptance.99563/paper-write-work.jsonl)
PASS  .env write refused (secret path)  (evidence: /tmp/vangja-acceptance.99563/env-write.jsonl)
PASS  PEM fixture write refused without echoing the key  (evidence: /tmp/vangja-acceptance.99563/pem-fixture.out)
PASS  recursive forced delete outside .pi/{state,artifacts} refused  (evidence: /tmp/vangja-acceptance.99563/rm-rf.jsonl)
PASS  trusted session exposes the ledger tools  (evidence: /tmp/vangja-acceptance.99563/ledger-tool.jsonl)
PASS  role preamble injected (LOOP ROLE: scout)  (evidence: /tmp/vangja-acceptance.99563/preamble.txt)
PASS  trusted startup is warning-free  (evidence: /tmp/vangja-acceptance.99563/preamble.stderr)

acceptance: all model-backed checks passed
```

Raw tool results from that evidence directory (`Vangja-project/**` with
`PI_LOOP_ROLE=phd_candidate`; `paper/paper.tex` with `PI_LOOP_STATE=WORK`):

```
[guard:PROTECTED_PATH] "Vangja-project/GUARD_TEST.md" is inside Vangja-project/, the read-only historical audit archive Allowed alternative: read Vangja-project/** as evidence, and write your own analysis under .pi/artifacts/ or a path named by your task brief
[guard:PROTECTED_PATH] "paper/paper.tex" may only be written in the PAPER state (current state: WORK) Allowed alternative: wait for the conductor to enter PAPER (PI_LOOP_STATE=PAPER) before editing paper/**; until then put results in artifacts/ or a task branch file named by your brief
```

Neither file existed afterwards (`ls Vangja-project/GUARD_TEST.md .env` → no such
file; `paper/paper.tex` unchanged).

**4. Ledger concurrency: 200 appends / 4 processes; `LOOP_STATE.json` 100 concurrent read/writes.**

```
$ node --test .pi/tests/test_ledger_concurrency.test.ts
✔ 200 concurrent appends from 4 processes yield 200 valid lines (471.754758ms)
✔ LOOP_STATE.json survives 100 concurrent reads and writes (349.77874ms)
ℹ tests 2   ℹ pass 2   ℹ fail 0

$ scratch=/tmp/vangja-concurrency-I6qoNc
$ wc -l < $scratch/.pi/state/EVENTS.jsonl
200
$ jq -c . $scratch/.pi/state/EVENTS.jsonl | wc -l
200
$ jq -r .seq $scratch/.pi/state/EVENTS.jsonl | sort -n | uniq | wc -l
200
$ ls -d $scratch/.pi/state/*.lock 2>/dev/null | wc -l
0
$ jq -r .actor $scratch/.pi/state/EVENTS.jsonl | sort -u | tr '\n' ' '
worker-w0 worker-w1 worker-w2 worker-w3
```

The concurrency test asserts every line parses, `schema=pi-ledger/1`, seqs are
unique and contiguous 1..200, all four workers appear, no lock directory remains,
and the derived view was written. The state test asserts the final file parses as
`pi-loop-state/1`, is `BOOT`, names one of the writers, and leaves no lock.

**5. Secret fixture (PEM block) cannot be written, and the value is never echoed.**

```
$ node .pi/tests/acceptance/pem_fixture_check.ts
fixture written: /tmp/vangja-pem-fixture-g8lDFf/fixture-pem.txt (149 bytes)
guard denial: [guard:SECRET] content for "src/vangja/leaked-key.ts" contains secret-like material (pem-private-key at offset 0 (114 chars)); the values are not echoed
PASS: PEM content refused and the key material was not echoed
```

Path-based secret files are refused too:

```
[guard:SECRET] ".env" matches a secret-bearing path pattern (/(^|\/)\.env(\.[^/]*)?$/i). Credential files stay outside the repository (e.g. ~/.ssh, or an OS keyring); the committed configuration is .pi/config/hpc.example.json with placeholders. Allowed alternative: keep credentials outside the repository; commit only examples with placeholders
```

`check_env.ts` additionally greps every versioned `.pi/**` file (excluding the
scanner's own pattern definitions and runtime state) and reports
`PASS no key material under .pi`.

**6. Role gates: `merge_pr` as `phd_candidate` and `hpc_submit` as `collector` refused with the matrix reason.**

```
$ node --test --test-name-pattern="role" .pi/tests/test_guards_policy.test.ts .pi/tests/test_guards_wiring.test.ts
✔ role matrix matches plan/05 §7 for every capability (1.267859ms)
✔ role preamble is exactly five lines and names the role, state, and task (0.243139ms)
✔ role gates refuse merge_pr for phd_candidate and hpc_submit for collector (9.918898ms)
✔ the role preamble is injected as a five-line <loop-role> section (0.799947ms)
ℹ tests 4   ℹ pass 4   ℹ fail 0
```

The refusal text is asserted to match `[guard:ROLE]` and
`allowed roles: supervisor, conductor` (merge_pr) / `allowed roles: phd_candidate,
conductor` (hpc_submit); `state_set` is refused for any non-conductor role. The
policy test additionally checks the whole matrix table (30 capability×role cases
plus every matrix entry).

**7. `pi --print` in the repository loads the environment with zero warnings.**

```
$ PI_LOOP_ROLE=scout PI_LOOP_STATE=SELECT_TASK PI_LOOP_ROOT=$PWD pi --print -a --no-session \
    --provider opencode-go --model opencode-go/deepseek-v4.1-flash --thinking max \
    "Output only the first line of the <loop-role> section of your system prompt."
exit=0
stdout: LOOP ROLE: scout | STATE: SELECT_TASK | TASK: none | EXPERIMENT: none | NEXT GATE: none
stderr bytes: 0
```

Startup diagnostics: `pi` 1.3.0; the trusted run loaded `.pi/settings.json` (model
pin, `+codemode`, compaction), the three extensions, and the role preamble (proven
by the model echoing it); stderr was empty — zero warnings, zero extension errors.
A second trusted session called `ledger_read` successfully
(`tool_execution_end toolName=ledger_read isError=false`), proving the ledger tools
load and execute through jiti under the pinned model.

### Deviations from plan/08 §S0

1. **`verify-env` exits 0 while the `/loop:verify-env` child check is stubbed.**
   The prompt requires both "`make -C .pi verify-env` → all green" (acceptance 1)
   and a "clear 'not implemented until S4' failure" for the orchestrator command.
   Those cannot both hold for one exit code, so `verify-env` runs every S0 test,
   prints an explicit `STUB (S4)` banner, and exits 0; `make -C .pi
   verify-env-strict` is the CI variant that fails (exit 1 from the recipe; make
   reports 2) until S4 lands. Acceptance 1
   is therefore green **with the stub visibly reported**, not silently skipped.
2. **`pr_open` validates the record shape only.** The plan's `pr_open` requires a
   clean worktree at the head SHA, which needs the S4 orchestrator's git layer and
   cannot run while this build session itself has a dirty tree. The git check is
   an S4 gate; the ledger records what it is told and the orchestrator must
   enforce it (noted in the handler and in `.pi/README.md` open items).
3. **`hpc.json` single-slot validation reads `AMENDMENTS.jsonl`** rather than an
   `amendments/` directory (`maxParallelJobs > 1` is rejected unless a recorded
   amendment covering `maxParallelJobs` exists). The plan does not specify the
   lookup mechanism; this keeps the ledger as the single source of truth.
4. **`.pi/lib/` added as a shared module directory** (not named in plan/08, which
   lists only `extensions/`). It exists so the policies and schemas are
   pi-free and unit-testable with `node`; `.pi/lib` is outside `.pi/extensions`
   so pi does not try to load it as an extension.
5. **`tests/**` strengthen-only protection is not in the write guard** (plan/02
   §5.1 puts the test-diff scan in the S6 pre-merge gate). Flagged as an open item.
6. **`.pi/artifacts/*/` is ignored wholesale** per the prompt's `.gitignore` list,
   which conflicts with plan/02 §5.1 "manifests + hashes committed". S5 must
   narrow the rule or write manifests to a versioned path. Flagged as an open item.

### Open items for later stages

- S1/S2: skills do not exist yet, so `check_env` reports `0/16` agent definition
  files present (expected until S3).
- S3: `.pi/agents/*.md`, `.pi/prompts/*.md`; the agents manifest already records the
  full roster, but the tool-group names (`dispatch`, `ledger`, `hpc`, `verify`,
  `task`, `paper`, `rubric`, `auditkit`, `merge`) must be resolved to concrete tool
  allowlists when child argv is built.
- S4: orchestrator CLI (`step|status|pause|resume|verify-env|gate`) so the three
  `loop/*.sh` shims stop being stubs; `verify-env-strict` becomes the CI target.
  The conductor must set `PI_LOOP_STATE` for every child (the guard's preamble and
  `paper/**` rule prefer it over `LOOP_STATE.json`).
- S5: artifact manifest versioning (see deviation 6) and the `.pi/config/hpc.json`
  enable path.
- S6: `tests/**` strengthen-only scan, plus the anti-gaming scan hooked into the
  guard's pre-merge gate.
- S7: `REPORT_BUILD.md` go/no-go, docs pass, fresh-clone verification.
- Minor: `plan/09_OPERATIONS_RUNBOOK.md` says "`.pi/state/HUMAN_GATES.json`"; S0
  writes pending gates to `GATES.jsonl` + the derived `CONTEXT.md` and leaves
  `HUMAN_GATES.json` to S4. Align the runbook in S7.
