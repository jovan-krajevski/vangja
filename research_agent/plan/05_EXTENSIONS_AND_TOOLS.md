# 05 — Extensions and custom tools

> All of this is installed at the **repository root** under `.pi/`; paths written
> `config/…`, `agents/…`, `skills/…`, `state/…` are relative to it.

All extensions live in `.pi/extensions/<name>/index.ts` (TypeScript,
loaded with `-e` and/or listed in `.pi/settings.json`). The conductor sets
`PI_LOOP_ROLE` and `PI_AGENT_NAME` for every child process; tools enforce the
permission matrix in §7 from those values. Nothing here talks to the network
except `hpc_*` (SSH) and `dispatch` (model API via `pi`).

## 1. Layout

```
.pi/extensions/
  orchestrator/   # conductor: /loop:* commands, state machine, child processes
  guards/         # model pin, protected paths, destructive-op and integrity guards
  ledger/         # append-only state tools (tasks, PRs, reviews, experiments, events)
  dispatch/       # subagent dispatch (isolated contexts), adapted from pi's example
  hpc/            # SSH + SLURM tools with single-slot + checkpoint enforcement
  verify/         # tests, artifact hashes, deterministic reproduction, traceability
  rubric/         # rubric trees: load / autofill deterministic leaves / record scores
  auditkit/       # finding schema, registry/dedupe, report assembly, grade computation
  paper/          # paper build, claims registry, table/figure regeneration
  status/         # TUI status line + human-gate notifications
```

## 2. `orchestrator` — the conductor

Commands (also usable from the RPC/headless driver):

| Command | Effect |
|---|---|
| `/loop:start` | load state, verify env, begin stepping; respects `state/STOP` and gates |
| `/loop:step` | run exactly one transition (the unit the headless runner uses) |
| `/loop:pause [reason]` | finish the current child process, write `PAUSED`, stop scheduling |
| `/loop:resume` | re-enter the state recorded before pause; never repeats completed steps |
| `/loop:status [--json]` | render `LOOP_STATE.json`, open tasks/jobs/gates, budget, last events |
| `/loop:verify-env` | model pin, resource load, schemas, protected paths, tool collisions, drift vs manifest |
| `/loop:gate list|approve|reject` | human gates (`merge`, `confirmation`, `audit`, `submission`) |

Implementation contract: `nextAction(state, ledgers) → Action` is a pure function;
`execute(action)` performs exactly one side effect and appends its completion
event. Actions are keyed by `(state, task_id, attempt)`; a completion event makes
the action a no-op on replay. All state writes are `write tmp → fsync → rename`
under a `proper-lockfile`-style lock. The same pure function is implemented in
`loop/step.sh` (jq) so the loop runs headless without the TUI; the TUI command is
a thin wrapper so behaviour cannot diverge.

Child invocation (always):

```
pi --mode json -p --no-… (persistent session via --session-dir)
   --provider opencode-go --model opencode-go/deepseek-v4.1-flash --thinking max
   --skill <skills named by the brief> -e <extensions> --prompt-template <templates>
   --append-system-prompt <agents/<agent>.md body> --tools <allowlist>
   "TASK: ..." 
```

Sessions are persisted under `state/sessions/<run_id>/<agent>/<task>/<attempt>/`;
the JSON stream is parsed for final message, tool calls, usage, and errors; all are
appended to `EVENTS.jsonl` together with the exact argv and prompt hashes.

Scheduler rules: one `WORK` per task; git operations serialized; **at most one
HPC experiment may be `queued|running|requeueing` at any time** (§5); audit
finders parallel (≤8); never start a new step while a previous child is alive.

## 3. `ledger` — state tools

| Tool | Input | Notes |
|---|---|---|
| `ledger_read(kind, filter?)` | kind ∈ event/task/pr/review/experiment/finding/decision/waiver/amendment/budget/gate | bounded; returns parsed records |
| `ledger_append(kind, record)` | JSON object | append-only, locked, fsynced; `actor` from env; schema validated |
| `task_enqueue(brief)` | brief per `plan/03_AGENTS.md` §3 | rejects briefs without acceptance criteria or with unknown origin IDs |
| `task_close(task_id, status, evidence)` | | conductor/supervisor |
| `pr_open(footer)` | PR record + machine-readable footer | requires a clean worktree at the head SHA |
| `review_write(record)` | decision ∈ accept/changes/reject + rubric scores + evidence | supervisor only |
| `experiment_queue(record)` | §5 schema | refuses if any job is active; refuses without checkpoint metadata |
| `experiment_update(id, status, evidence)` | collector (and conductor for requeue) | appends, never overwrites |
| `decision_record(adr)` | ADR body | `state/DECISIONS.md` |
| `waiver_record(waiver)` | names the failed gate + reason | requires supervisor role |
| `amendment_record(amendment)` | protocol change + justification | requires supervisor role; triggers a `HUMAN_GATES` entry |

Kinds are separate JSONL files under `state/`; each append also updates the
derived, human-readable `state/*.md` views (`CONTEXT.md`, `TASKS.md`,
`EXPERIMENTS.md`, `REVIEWS.md`) through a deterministic serializer.

## 4. `guards` — invariants

- **Model pin** (`session_start`, `before_agent_start`): if model ≠
  `opencode-go/deepseek-v4.1-flash` or thinking ≠ `max`, block with a message
  naming `.pi/config/model.json`. Also injects a 5-line role preamble (role, current
  state, task, "read brief first", "no claims without artifacts").
- **Protected paths** (`tool_call` on `write`/`edit`): `Vangja-project/**`,
  `research_agent/plan/**`, `fixed_case_studies/protocol/**` (unless a recorded
  amendment), `paper/**` (unless state = PAPER), `AGENTS.md` (unless supervisor),
  `.git/**` direct edits, `state/**` (ledger tools only).
- **Destructive bash**: blocks `git push --force`, `git filter-branch`, `git
  rebase` on shared branches, `rm -rf` outside allowed dirs, `git reset --hard`
  with uncommitted work, `chmod 777`, writing to `~/.ssh`, `curl|sh`, and any
  command that exfiltrates env vars. Confirmations via `ctx.ui.confirm` when the
  project is trusted.
- **Secret scan**: blocks commits/paths matching `.env`, `*.pem`, private keys,
  `auth.json`, `hpc.json`, tokens; redacts detected credentials from tool results.
- **Integrity scan** (pre-merge, called by the conductor): runs
  `.pi/skills/scientific-integrity-and-anti-gaming/scripts/scan_gaming.py` plus the
  test-diff scan; a hit blocks the merge and opens a finding for the supervisor.
- **HPC gate**: `hpc_submit` is refused when any experiment is active, when the
  record lacks checkpoint/resume metadata, or when the remote SHA ≠ the record
  SHA.
- **Merge gate**: `merge_pr` requires supervisor role, a green pre-merge gate run
  (or a recorded waiver), and the human gate if enabled.

## 5. `hpc` — SSH/SLURM tools (single-slot, checkpoint-first)

Config `.pi/config/hpc.json` (gitignored; example committed):

```json
{
  "host": "", "user": "", "port": 22,
  "remoteRepoDir": "", "remoteScratchDir": "",
  "hpcKey": "~/.ssh/id_ed25519",
  "gitKey": "~/.ssh/vangja_github",
  "forwardGitAgent": true,
  "partition": "", "cpus": 8, "memGb": 32,
  "wallClockHours": 24,
  "container": "",              // path to .sif, optional
  "pollSeconds": 60,
  "maxAttemptsPerExperiment": 6,
  "maxParallelJobs": 1          // HARD: do not raise without a recorded amendment
}
```

| Tool | Behaviour |
|---|---|
| `hpc_config` | returns `{enabled, hostRedacted, …}`; **if `host` is empty, all other tools return `{enabled:false}` and do nothing** |
| `hpc_status(jobs?)` | runs `squeue -u $USER --format=…` and `sacct -j … --json`; stores raw output as evidence |
| `hpc_submit(experiment_id)` | gate checks (§4) → tries a private agent (`ssh-agent`, add only `gitKey`) → `ssh -A -o IdentitiesOnly=yes -i hpcKey` → remote `git fetch origin && git checkout --detach <sha>` into `remoteRepoDir/runs/<experiment_id>` → `sbatch --parsable` → records job id, queue, estimated start, exit-state of the submit |
| `hpc_checkpoints(experiment_id)` | lists checkpoint files with sizes/mtimes/hashes (works while the job runs) |
| `hpc_fetch(experiment_id, paths?)` | rsync/scp of declared outputs **and** checkpoints, then per-file SHA-256; verifies the collected set against the record's `expectedOutputs` |
| `hpc_tail(experiment_id, stream, lines)` | tail of slurm out/err |
| `hpc_cancel(experiment_id, reason)` | destructive; only when a record instructs it |
| `hpc_requeue(experiment_id)` | allowed only after the previous attempt is confirmed ended; creates attempt+1 with the same idempotency key |

Forwarding detail: one dedicated `ssh-agent` per loop run, `gitKey` added to it,
`SSH_AUTH_SOCK` set for the ssh children, `-A` enables remote git; if `gitKey`
is missing or not forwarded, remote git is skipped and the loop reports it
(rsync-only mode if `remoteRepoDir` is reachable).

## 6. `verify`, `rubric`, `auditkit`, `paper`, `status`

**verify**

| Tool | Behaviour |
|---|---|
| `verify_tests(paths?)` | `uv run pytest -q <paths>`; returns pass/fail counts, first failures, duration; records the command |
| `verify_artifact(path)` | manifest exists, schema valid, declared files exist, SHA-256 matches; returns missing/mismatched list |
| `verify_reproduce(artifact_id)` | runs the artifact's `repro_command` in a fresh temp dir against the frozen inputs; numeric compare within tolerance; returns diff summary |
| `verify_traceability()` | checks `paper/CLAIMS.json`: every claim has ≥1 artifact whose manifest SHA is an ancestor of `update_pymc`; every number found in `paper/**` LaTeX has an entry; returns dangling/uncited lists |
| `verify_guards()` | runs the protected-path, test-diff, and gaming scans on a given diff range |

**rubric** — JSON trees (`rubrics/*.json`) with nodes `{id, text, type:
"binary"|"scale"|"auto", auto: "tests_pass"|"hash_ok"|"…", weight}`. `rubric_load`,
`rubric_autofill` (deterministic leaves from verify results), `rubric_score`
(LLM leaves require an evidence pointer per score), `rubric_validate`. Grades are
derived by `auditkit` from rubric scores + blocking findings.

**auditkit** — `finding_validate` (nine-field schema; requires location,
command, expected, observed, consequence, severity, confidence, status),
`finding_register` (dedupe by normalized location+claim hash; cross-refs, never
duplicates), `findings_render` (MD + JSON), `audit_report_assemble`,
`audit_grade`.

**paper** — `paper_build` (regenerates every table/figure/number into
`paper/generated/**` from artifacts, then builds the PDF), `claims_add`,
`claims_check`, `paper_diff_summary` (which claims changed and why).

**status** — TUI: state, current task, active HPC job, budget, next human gate;
on `BLOCKED` or a gate, a notification plus a line in `state/HUMAN_GATES.json`.

## 7. Permission matrix (enforced via `PI_LOOP_ROLE`)

| Capability | phd_candidate | collector | supervisor | audit_* | conductor |
|---|---|---|---|---|---|
| read/edit/write/bash | ✓ | bash (inspection) | bash (tests/git) | read/bash (inspection) | ✓ |
| dispatch workers | ✓ | — | ✓ (critic) | ✓ (audit workers) | — |
| ledger_read / append | ✓ / ✓ | ✓ / ✓ (experiments) | ✓ / ✓ | ✓ / ✓ (findings) | ✓ / ✓ |
| task_enqueue | ✓ | — | ✓ | — | ✓ |
| pr_open | ✓ | — | — | — | ✓ |
| review_write | — | — | ✓ | — | ✓ |
| merge_pr | — | — | ✓ (executes via conductor) | — | ✓ |
| experiment_queue | ✓ | — | — | — | ✓ |
| experiment_update | — | ✓ | — | — | ✓ |
| hpc_submit | ✓ (gate-checked) | — | — | — | ✓ |
| hpc_status/fetch/tail | ✓ | ✓ | ✓ | ✓ (read) | ✓ |
| hpc_cancel/requeue | — | requeue only | ✓ | — | ✓ |
| verify_* | ✓ | hash only | ✓ | ✓ | ✓ |
| rubric_score | — | — | ✓ | ✓ | ✓ |
| auditkit_write | — | — | ✓ | ✓ | ✓ |
| state_set | — | — | — | — | ✓ |
