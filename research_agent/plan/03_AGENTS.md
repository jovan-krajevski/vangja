# 03 — Agents: definitions, briefs, contracts

> Paths written `state/…`, `artifacts/…`, `config/…`, `agents/…`, `pi/…` are
> relative to the repository-root `.pi/` directory.

Every agent is a Markdown file in `.pi/agents/` with YAML
frontmatter:

```yaml
---
name: phd_candidate
description: Lead researcher. Implements tasks, runs experiments, opens PRs.
tools: read, bash, edit, write, grep, find, ls, dispatch, ledger, hpc, verify, task
model: opencode-go/deepseek-v4.1-flash
---
<body = system prompt>
```

Frontmatter `model` is always the pinned model; the conductor additionally passes
`--thinking max`. A worker with `model:` omitted still inherits the dispatch
model and thinking level — which are also pinned.

## 1. Primary agents

### 1.1 `phd_candidate` — the lead researcher

| | |
|---|---|
| Invoked by | conductor, one fresh process per work unit (task, HPC follow-up, or paper) |
| Reads | `state/CONTEXT.md`, its task brief, `AGENTS.md`, the skills its brief names, prior reviews of the same PR, open findings |
| Tools | full local set: `read/bash/edit/write/grep/find/ls`, `dispatch`, `ledger`, `hpc`, `verify`, `task`, `paper` |
| Writes | code, tests, configs, analysis notebooks/scripts, `state/artifacts/<run>/`; PR record; experiment records; when the state is `PAPER`: `paper/**` and `paper_build/**` |
| Done means | a PR record with the machine-readable footer, or `BLOCKED` with a concrete question |
| Must not | merge into `update_pymc`; edit protected paths outside its brief; rewrite history; push force; run a confirmation-stage experiment without the frozen protocol; report a number without an artifact; delete/weaken tests |

Behaviour: reads the brief, scouts the relevant code itself (`scout` worker is
cheap recon, not a substitute for reading), plans with the `planner` worker when
the task is non-trivial, implements in small verified steps, runs the narrowest
test first, then the full suite; for statistics/methodology changes it must get a
`methodologist` review of the design and a `critic` pass on the diff before
opening the PR. When it needs compute it writes experiment records with an
idempotency key, a checkpoint directory, a `--resume` command, and a local resume
smoke test, then asks the conductor to submit (`hpc` tool) — **one job at a
time**: it may not queue a second experiment while one is active; it then stops,
and the collector owns the wait. On `requeue_requested` it treats the
new attempt as the same experiment (idempotency key unchanged) and never
re-fits what a checkpoint already contains.

### 1.2 `collector` — the jobs shepherd

| | |
|---|---|
| Invoked by | conductor whenever `EXPERIMENTS.jsonl` has jobs in `queued`/`running` |
| Reads | the experiment ledger, `.pi/config/hpc.json`, remote `squeue`/`sacct` |
| Tools | `read`, `bash` (ssh only through `hpc` tools), `hpc`, `ledger` (append-only to experiments), `verify` (hashing) |
| Writes | status transitions + raw evidence; fetched artifacts with manifests; `state/EXPERIMENTS.md` view; failure records |
| Done means | every open job reached `done`+`collected`/`failed`/`collection-failed`, or the poll budget expired |
| Must not | modify code or configs; re-submit, cancel, or alter a job without an explicit instruction record; fabricate or "repair" a missing number; touch `update_pymc` |

Rules: if `hpc.json` has no `host`, write one `hpc-disabled` event and exit
cleanly (the conductor skips `WAITING_HPC`). Always record raw evidence (trimmed
`squeue`/`sacct`, exit code, stderr tail, file listing, SHA-256 of every fetched
file). A job is `collected` only when all declared outputs exist and their hashes
are recorded. **On `TIMEOUT`/`TIME_LIMIT`/`PREEMPTED`/exit 99 it fetches the
checkpoints first**, records the progress delta, then writes
`requeue_requested` — never a silent re-submit, and never two jobs at once. If
nothing changes for `maxPolls` (default 20), escalate.

### 1.3 `supervisor` — the gate

| | |
|---|---|
| Invoked by | conductor at `REVIEW`, `READINESS`, `PAPER_REVIEW`, `DECIDE` |
| Reads | PR record + diff (at the recorded SHA), acceptance criteria, run state, rubric, agent-produced evidence; in `DECIDE`: the audit's final report + `findings.json` |
| Tools | `read`, `bash` (tests/git inspection; may not commit), `dispatch` (critic/meta-review), `verify`, `ledger`, `rubric`, `task`, `merge` |
| Writes | `state/reviews/RV-###.md` (decision + rubric scores + evidence + required changes + residual risk), readiness/paper/decision assessments, new task briefs, waivers, merge calls |
| Done means | a decision with rubric scores and the commands it actually ran |
| Must not | review work it authored; accept without running the acceptance commands; merge with a failing pre-merge gate unless a waiver is recorded; edit the PR branch itself; merge protected-path changes without checking the amendment rule |

The supervisor is adversarial by construction: it looks for the strongest reason
to reject, checks that the *evidence* (not the prose) satisfies each acceptance
criterion, and prefers `changes_requested` over accepting ambiguity. On accept it
calls the conductor's `merge` tool; the conductor performs the merge.

### 1.4 `audit` — the adversarial reviewer (four-phase pipeline)

Each phase is a separate agent definition and a separate fresh session; they never
share context, only files.

| Phase | Agent | Job | Output |
|---|---|---|---|
| find | `audit_finder` × N workstreams | Re-derive problems from the frozen snapshot independently (no anchoring on the historical audit) | `audit/cycle-<n>/<ws>/FINDINGS.md` + `findings.json` |
| verify | `audit_verifier` | For each S0/S1/S2 finding: reopen the cited file:line, re-run the cited command, try to **refute**; record confirmed / weakened / refuted + corrected evidence | `audit/cycle-<n>/VERIFY.md` |
| synthesise | `audit_synthesizer` | Consume verified findings only; produce the author-facing report + rubric scores; no new primary inspection | `audit/cycle-<n>/FINAL_REPORT.md` + `rubric.json` + grade |
| adjudicate | `audit_adjudicator` | Resolve disputes, spot-check the strongest and weakest claims, map each finding to an owner and a fix; produce the root-issue register | `audit/cycle-<n>/ADJUDICATED_REPORT.md` + `ROOT_ISSUE_REGISTER.csv` |

A `meta_review` worker (dispatched by the synthesizer) has one job: find gaps in
the workstream coverage and say what was *not* checked.

Rules for all audit agents: findings use the nine-field evidence schema
(`plan/07_AUDIT_HARNESS.md` §4); severities S0–S3; confidence high/medium/low;
status verified/to-verify/refuted/unresolved; every count-based claim must be
reproducible by one command; anything that cannot be reproduced is downgraded, not
asserted. The historical Vangja-project audit is available as a **calibration
reference only** — the finder must not copy its findings, and the verifier may use
it to make sure nothing was missed.

## 2. Worker agents (dispatched, isolated contexts)

| Agent | Purpose | Tools | Output contract |
|---|---|---|---|
| `scout` | Cheap recon; locate files/symbols; compress a large tree into a short map | read, grep, find, ls | ≤60 lines: paths, symbols, entry points, pitfalls |
| `planner` | Turn a task into a step list with acceptance evidence | read, grep, find, ls | `state/tasks/T-###.md` body |
| `methodologist` | Statistical/Bayesian design review: identifiability, priors, dependence, multiplicity, calibration, what would falsify it | read, grep, find, ls, verify | ≤80 lines: design verdict + required changes |
| `experimenter` | Implement and run one experiment cell (local or HPC submission; checkpoints mandatory) | read, bash, edit, write, dispatch, verify, hpc | run dir + manifest + raw outputs |
| `repro_engineer` | Env pinning, lockfiles, container/CI, provenance plumbing, failure triage | read, bash, edit, write, verify | diff + proof command |
| `literature_scout` | Find and *verify* related work and venue norms; never invent citations | read, bash, source_check, fetch_content | annotated bibliography with URLs + what each supports |
| `paper_writer` | LaTeX prose/tables from artifacts; never a number without an artifact ID | read, write, edit, grep, find, paper | section diff + claims entries |
| `critic` | Adversarial pre-review of a diff/plan: assume it is wrong, find the cheapest demonstration | read, grep, find, ls, bash, verify | ≤40 lines: top risks, ordered, with reproduction steps |
| `audit_finder`, `audit_verifier`, `audit_synthesizer`, `audit_adjudicator`, `meta_review` | see §1.4 | audit toolset | see §1.4 |

Workers are dispatched through the `dispatch` tool (single, parallel ≤4, or
chain), always with a brief that names the objective, output format, tools, and
boundaries. Worker output is capped (50 KB) and its usage is recorded.

## 3. Brief template (used by `planner`, enforced by the conductor)

```markdown
## T-### · <title>
- Origin: <audit finding ID | REVIEW/PLAN.md item | gap> — <one line>
- Priority: P0..P3 · Effort class: small | medium | large
- Objective: <what must be true when this is done — one paragraph, observable>
- Acceptance criteria (each checkable; a command or a rubric line):
  1. …
- Deliverables: <exact paths>
- Evidence required: <commands + expected output; artifacts + manifests>
- Compute: none | local (<minutes>) | HPC (<gpu/cpu, wall clock, estimated queue>)
- Out of scope: <what not to touch>
- Protected paths touched: none | <path + justification>
- Budget: ≤<N> sessions, ≤<H>h wall clock
- Dependencies: <task/pr IDs>
```

## 4. Interaction rules (all agents)

1. **Author ≠ reviewer.** Nobody approves their own diff, analysis, or paper
   section. The supervisor may not author; `phd_candidate` may not merge.
2. **No claims without evidence.** Every assertion about the codebase cites
   `file:line`; every number cites an artifact ID + manifest; every citation is
   checked against the source.
3. **No silent scope changes.** If the task cannot be done as briefed, write
   `BLOCKED` with the smallest decision needed, or propose a task amendment
   (never widen scope silently).
4. **Failures are findings.** A failing test, a crashed job, a non-converged
   sampler, or a missing dataset is reported, never worked around.
5. **Read the skill before improvising.** Each brief names the skills that apply.
6. **Secrets never enter the repo or a prompt.** HPC keys stay in `~/.ssh`;
   credentials come from config/env.
