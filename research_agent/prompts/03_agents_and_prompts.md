# Prompt 03 — Build stage S3: agents, prompt templates, dispatch

Prerequisite: S0–S2 green. Read `research_agent/plan/03_AGENTS.md` (the full
roster and the brief template) and `research_agent/plan/05_EXTENSIONS_AND_TOOLS.md`
§4 (permission matrix) and §6 (`dispatch`).

## Deliverables

1. `.pi/agents/` — one Markdown file per agent, frontmatter
   `name`, `description`, `tools`, `model: opencode-go/deepseek-v4.1-flash`;
   body = system prompt, ≤120 lines, encoding exactly the table in
   `plan/03_AGENTS.md` (purpose, inputs, procedure, output contract,
   prohibitions, stop conditions):
   - primary: `phd_candidate.md`, `collector.md`, `supervisor.md`
   - audit: `audit_finder.md`, `audit_verifier.md`, `audit_synthesizer.md`,
     `audit_adjudicator.md`, `meta_review.md`
   - workers: `scout.md`, `planner.md`, `methodologist.md`, `experimenter.md`,
     `repro_engineer.md`, `literature_scout.md`, `paper_writer.md`, `critic.md`
   Requirements: every agent names the skills it should load; `phd_candidate`
   must refuse to merge and to queue a second HPC job; `collector` must refuse
   to touch code; `supervisor` must run acceptance commands itself; audit agents
   must use the nine-field schema and never copy the historical audit.
2. `.pi/extensions/dispatch/index.ts` — adapted from pi's bundled `subagent`
   example (kept local; record its licence in `.pi/extensions/README.md`):
   - tools `dispatch_agent`, `dispatch_parallel` (≤4 concurrent), `dispatch_chain`
   - spawns `pi --mode json` children with: pinned `--provider/--model/--thinking`,
     `--append-system-prompt <agent body>`, `--tools` from the manifest,
     `--session-dir .pi/state/sessions/<run_id>/<agent>/`, `--name
     <agent>__<task>__<attempt>`, the task text last
   - `PI_LOOP_ROLE` propagated; per-role worker allowlist enforced (e.g. only
     `phd_candidate` may dispatch `experimenter`; only audit roles may dispatch
     `audit_*`)
   - returns final text + usage; caps model-visible output at 50 KB and points at
     the persisted session for the rest; appends one `dispatch` event per child
     (argv, agent file hash, usage, exit state)
   - failures surface stderr + stopReason, never swallowed.
3. `.pi/prompts/` — slash-command templates (thin, argument-driven):
   `research-next-task.md`, `research-implement.md`, `research-review.md`,
   `research-queue-experiment.md`, `research-collect.md`, `research-write-paper.md`,
   `audit-cycle.md`, `audit-workstream.md`, `audit-verify.md`, `audit-synthesize.md`,
   `loop-status.md`. Each ≤30 lines: assemble the brief via `.pi/state/`, name the
   agent, name the acceptance evidence required.
4. `.pi/tests/agents/` — tests that: every agent file parses and matches the
   manifest; `tools` names exist in the loaded tool set (fail on typos); the role
   matrix refuses forbidden calls; a dispatched `scout` on a small fixture repo
   returns a summary and records usage in the ledger.

## Design requirements

- Agent files must not contain project facts that belong in skills; they contain
  procedure and boundaries. Example: `phd_candidate.md` says "load the skill your
  brief names; run the narrowest test first", not the PyMC pin list.
- The dispatch extension must be robust to parallel calls and must never let a
  child inherit a session from another child.
- Missing agent file → explicit error listing available agents.
- `dispatch` results include `structuredContent` (agent, exit state, usage,
  session path) so the conductor can parse them without guessing.

## Acceptance tests (paste output into `REPORT_BUILD.md`)

1. `pi --print "/research-next-task" --dry-run` (or equivalent) resolves a brief
   template without executing it.
2. Dispatch a `scout` from a `phd_candidate` role child on `.pi/skills/` →
   summary returned, usage + session path in `.pi/state/EVENTS.jsonl`.
3. Dispatch `merge_pr` via a `phd_candidate` child → refused by the guard;
   the same from `supervisor` is allowed through to the orchestrator's stub.
4. Dispatch an unknown agent name → explicit error.
5. Kill a child mid-run → parent receives an error, ledger records `aborted`,
   no partial result is presented as final.
6. All 17 agent files validate against the manifest and the tool registry.

## Do not

- Do not implement the state machine (S4) or HPC tools (S5); the orchestrator
  stub can accept `merge_pr` and log it.
- Do not add any model other than the pin.

## Final report

Append `## S3 — agents, prompt templates, dispatch` to
`research_agent/REPORT_BUILD.md`.
