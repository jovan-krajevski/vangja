/**
 * The `understand_loop` curriculum: one small chapter per concept.
 *
 * The extension delivers one chapter at a time and stops so the reader can ask
 * questions. Content is deliberately grounded in versioned files; every chapter
 * lists the paths a curious reader (or the tutor model) can open to verify it.
 */

export interface Chapter {
	/** 1-based position in the tour. */
	id: number;
	slug: string;
	title: string;
	/** One line used in the progress widget. */
	summary: string;
	/** The chunk itself. Kept short by design (roughly 150–250 words). */
	body: string;
	/** Durable files that prove the chapter. */
	refs: string[];
	/** Checkpoint questions the tutor should leave with the reader. */
	checks: string[];
}

export const CURRICULUM: Chapter[] = [
	{
		id: 1,
		slug: "what-it-is",
		title: "What the loop is",
		summary: "A conductor plus specialist agents; the human keeps the gates",
		body: `The loop is a research conductor for the Vangja paper. It is not an autonomous scientist. It turns the frozen build plan into audited, reproducible work: pick a task, dispatch the right specialist, collect evidence, review it, merge it, and only then move on.

Two ideas keep it honest. First, every action is one of a fixed set of loop states, and the conductor performs exactly one action per step, so a crash never leaves half a step behind. Second, the human owns the irreversible decisions: merge, confirmation, audit sign-off, submission. The loop can prepare all of them; it cannot take them.

The plan lives in research_agent/plan/ (read-only). The implementation lives in .pi/. The manuscript lives in paper/ and may only cite artifacts produced on update_pymc. The project rules live in AGENTS.md, and the project policy in .pi/config/project.json.

So when you read "the loop decided X", translate it as: a state-machine rule selected an action, an agent produced evidence, and a gate (mechanical or human) allowed it through.`,
		refs: [
			"AGENTS.md",
			"research_agent/plan/00_GOAL_AND_CONSTRAINTS.md",
			"research_agent/plan/02_ARCHITECTURE.md",
			".pi/README.md",
		],
		checks: [
			"Which decisions does the loop deliberately leave to the human?",
			"What is the difference between research_agent/plan/ and .pi/?",
		],
	},
	{
		id: 2,
		slug: "state-machine",
		title: "The state machine",
		summary: "Sixteen states, one legal transition per step, one mutable state file",
		body: `The conductor is a state machine. The states are UNINITIALIZED, BOOT, SELECT_TASK, WORK, WAITING_HPC, REVIEW, MERGE, POST_MERGE, READINESS, PAPER, PAPER_REVIEW, AUDIT, DECIDE, PAUSED, BLOCKED and DONE. Only transitions declared in .pi/lib/state.ts are legal; an illegal move throws instead of guessing.

LOOP_STATE.json is the one mutable state file. It records the run id, the selected task/PR/experiment, the attempt counter, the next human gate, and the base branch pinned at BOOT. Writes are atomic (temp file, fsync, rename) under a lock.

A "step" computes the next action from the state plus the ledgers, then performs exactly one action: boot, spawn, bookkeep, open a gate, merge, pause, block, or an HPC transition. The action key is the idempotency key. The effect is applied before the completion event, so replaying a killed step is a no-op rather than a duplicate.

PAUSED remembers where it came from (paused_from) and only resumes there. BLOCKED records the smallest decision a human must make. DONE is terminal.`,
		refs: [
			".pi/lib/state.ts",
			".pi/extensions/orchestrator/state-machine.ts",
			".pi/extensions/orchestrator/execute.ts",
			"research_agent/plan/02_ARCHITECTURE.md",
		],
		checks: [
			"What happens if a step is killed after the side effect but before the event?",
			"Why is PAUSED allowed to return only to paused_from?",
		],
	},
	{
		id: 3,
		slug: "ledgers",
		title: "The ledger is the memory",
		summary: "Append-only JSONL records; markdown views are derived, never edited",
		body: `Everything durable is a ledger record. .pi/state/ holds append-only JSONL files: EVENTS, TASKS, PRS, REVIEWS, EXPERIMENTS, FINDINGS, DECISIONS, WAIVERS, AMENDMENTS, BUDGET and GATES. Records are never rewritten; an update appends a new record with the same id, and readers project the latest one.

The ledgers are the loop's memory, not the conversation. A fresh-context session reconstructs the world by reading LOOP_STATE.json plus the ledgers. That is why a handoff must name files and artifact ids instead of saying "continue as discussed".

The markdown views (CONTEXT.md, TASKS.md, EXPERIMENTS.md, DECISIONS.md, REVIEWS.md) are generated from the ledgers. Do not hand-edit them; regenerate or append a record instead.

Schema validation is real: .pi/lib/records.ts declares required fields per kind, and a record with a bad field is rejected before it is written. Ids are assigned by the ledger (T-###, PR-###, RV-###, E-####, F-###, D-###, G-###).

The practical consequence: if it is not in a ledger, it did not happen. If you want the loop to know something, record it.`,
		refs: [
			".pi/extensions/ledger/store.ts",
			".pi/extensions/ledger/handlers.ts",
			".pi/lib/records.ts",
			".pi/lib/views.ts",
		],
		checks: [
			"Why is an update a second record with the same id rather than an edit?",
			"Which files are safe to hand-edit: the JSONL ledgers or the markdown views?",
		],
	},
	{
		id: 4,
		slug: "roles",
		title: "Roles, agents and dispatch",
		summary: "Sixteen specialists, one pinned model, one child at a time",
		body: `The conductor does not do the work itself; it dispatches a specialist. .pi/config/agents.json is the roster of sixteen agents (phd_candidate, supervisor, collector, experimenter, scout, planner, methodologist, repro_engineer, paper_writer, critic, literature_scout, the four audit roles, and meta_review). Each entry names the agent file, allowed tools, skills, effort, parallelism and model.

The model is pinned: .pi/config/model.json is the single source of truth, and the guard refuses model or thinking-level drift. A dispatched child receives the pinned provider/model/thinking, its own session directory, its allowed tools and skills, and the agent body as its system prompt. Children never resume each other's sessions.

Dispatch is bounded: at most four children in parallel (dispatch_parallel), and the conductor starts only one child at a time, protected by a pid lock. A live lock blocks a second spawn; a dead lock is reported and cleared.

Roles are also a permission system. .pi/lib/roles.ts maps tools to capabilities, and the guard blocks a child from using a tool its role does not own. A worker cannot merge, and a collector cannot edit code. That is enforced in two places: the guard and the tool handler.`,
		refs: [
			".pi/config/agents.json",
			".pi/extensions/dispatch/lib.ts",
			".pi/lib/roles.ts",
			".pi/extensions/guards/policy.ts",
		],
		checks: [
			"What exactly does a dispatched child inherit, and what does it not?",
			"Where is the 'only one child at a time' rule enforced?",
		],
	},
	{
		id: 5,
		slug: "tasks",
		title: "Tasks and briefs",
		summary: "An origin, acceptance criteria, deliverables — and no silent scope drift",
		body: `Work starts with a task record, not with a prompt. A task carries an id, title, origin (for example PLAN:P1-2 or AUDIT:F-003), priority, objective, acceptance_criteria, deliverables, optional protected_paths, and a status. The ledger rejects a task without acceptance criteria or deliverables.

The conductor writes a brief under .pi/state/tasks/<id>.md and hands it to the specialist. The brief names the objective, the origin, the worktree and branch, the narrowest test to run first, the PR footer, and the experiment record if compute is needed. The child either produces the deliverable or returns BLOCKED with the smallest decision a human must make.

This is how the loop avoids "while I was there" changes. If a task needs to grow, it gets a new task record or a recorded amendment, not an invisible expansion of the current one.

Task status moves open → in_progress → review → merged (or blocked/rejected/closed). The conductor advances it from ledger evidence; it does not take the child's word for it.`,
		refs: [
			".pi/lib/records.ts",
			".pi/extensions/orchestrator/state-machine.ts",
			".pi/extensions/ledger/handlers.ts",
			"research_agent/plan/03_AGENTS.md",
		],
		checks: [
			"What two fields make a task record valid?",
			"Why is 'while I was there' scope growth a protocol problem?",
		],
	},
	{
		id: 6,
		slug: "git",
		title: "Worktrees, PRs and merges",
		summary: "Work branches, a machine-readable PR footer, merge --no-ff only",
		body: `Work happens in a git worktree under .pi/state/worktrees/<task_id>/ on a branch named work/T-###-<slug>. The task branch is created from the run's base branch and never from another work branch.

When the child is done it calls pr_open with a machine-readable footer: task_id, head_sha, base_sha, branch, summary, artifacts, tests, claims and hpc_jobs. The PR record is the reviewable unit. It is not a GitHub PR; the loop never pushes.

A merge is always git merge --no-ff, run from a clean checkout of the base branch, with the review id in the commit message, and tagged run/<run_id>/merge/<n>. The conductor never rebases and never force-pushes; those commands do not exist in its git module and the guard blocks them in bash.

The base branch is whatever branch the checkout was on when the run booted. project.json sets baseBranch to "@current", and BOOT pins the resolved name in LOOP_STATE.base_branch. That means the loop merges into your working branch (for example vangja_loop_20261018), not into a hard-coded update_pymc. A branch switch mid-run cannot silently retarget the merge.`,
		refs: [
			".pi/extensions/orchestrator/git.ts",
			".pi/extensions/orchestrator/gates.ts",
			".pi/config/project.json",
			"AGENTS.md",
		],
		checks: [
			"Which branch does a run merge into, and when is that decided?",
			"What does the PR footer have to contain before review can start?",
		],
	},
	{
		id: 7,
		slug: "verification",
		title: "Verification before merge",
		summary: "Seven pre-merge gates plus guards on every tool call",
		body: `A PR does not merge because an agent said it is good. Before MERGE the conductor runs the pre-merge gates: tests, protected-paths, test-diff, integrity, provenance, hygiene and claims. Each gate records pass/fail with a detail line in .pi/state/gate-runs/. A red gate pauses the loop with the failed gate named; a supervisor can waive a gate only by recording why.

The verify extension exposes the same checks as tools: verify_tests runs the suite, verify_artifact checks manifest schemas and SHA-256 hashes, verify_reproduce reruns a recorded command and compares numbers within a tolerance, verify_traceability checks that every paper number maps to an ancestor artifact, verify_guards scans a diff for protected paths, weakened tests and gaming patterns.

The guards extension runs on every tool call, not just at merge: it blocks writes to Vangja-project/**, research_agent/plan/** (except recorded exceptions), fixed_case_studies/protocol/** without an amendment, and paper/** outside PAPER. It also blocks secret-like content, destructive bash (force-push, rebase on shared history, rm -rf outside loop state), and role/tool mismatches.

Tests are part of the contract: a change must add or update tests, and never weaken an existing one to make a change pass.`,
		refs: [
			".pi/extensions/verify/index.ts",
			".pi/extensions/verify/gates.ts",
			".pi/extensions/guards/protected.ts",
			".pi/extensions/orchestrator/gates.ts",
		],
		checks: [
			"Name the seven pre-merge gates.",
			"What is the difference between a mechanical gate and a human gate?",
		],
	},
	{
		id: 8,
		slug: "hpc",
		title: "Experiments and HPC",
		summary: "Checkpointed experiment records, one active job, hashed artifacts",
		body: `Compute is an experiment record, not an ad-hoc command. Before a job can be queued, the record must carry the task and branch, commit_sha, data_hash, command, resources, seeds, expectedOutputs, an idempotencyKey, and a checkpoint block: dir, progressFile, resumeCommand and flushOnSignal: true. A local resume smoke test must prove that --resume skips completed units.

The submit gate is strict: no other experiment may be active (single slot), the record must be queued and not already submitted, the local uv.lock must match the record's uv_lock_sha, the container hash must match when a container is configured, the remote checkout must be a clean detached checkout at commit_sha, and wall clock must fit 24 hours or be split into stages.

Once running, the collector polls squeue/sacct. TIMEOUT, PREEMPTION, NODE_FAIL, OOM and exit 99 trigger a checkpoint-only fetch first, then a recorded requeue decision; the collector never resubmits on its own. On success, hpc_fetch copies declared outputs and checkpoints, hashes every file, and writes a manifest with sacct accounting.

If there is no HPC host in .pi/config/hpc.json, the whole path is disabled cleanly: no network calls, one explicit hpc-disabled event.`,
		refs: [
			".pi/extensions/hpc/gate.ts",
			".pi/extensions/hpc/manifest.ts",
			".pi/config/hpc.example.json",
			"research_agent/plan/06_HPC_AND_COLLECTOR.md",
		],
		checks: [
			"What four fields make up the checkpoint block?",
			"What happens first when a job hits TIMEOUT?",
		],
	},
	{
		id: 9,
		slug: "audit",
		title: "The audit harness",
		summary: "Ten workstreams, deterministic gates, canaries, and a graded verdict",
		body: `After the paper is merged, the loop audits its own work. The audit is independent: frozen snapshot tag, separate finder workstreams, separate verifiers, a synthesizer and an adjudicator.

The workstreams are M (methodology), S (statistics and math), P (probabilistic UQ), D (data and leakage), I (implementation), R (reproducibility), H (documentation), E (evaluation integrity), C (claims and traceability) and L (literature). Every finding must fit a nine-field schema and carry an executable command; a claim without a command is not a finding.

Before any finder runs, thirteen deterministic gates run: tests, protocol-hashes, confirmation-frozen, no-test-selection, test-diff, no-failures-hidden, context-excluded, artifacts-complete, reproduce-headline, traceability, env-pinned, gaming-scan and canary. A missing input is a skip with a reason, never a silent pass.

Findings are deduplicated by location plus title and registered per cycle. A seeded-defect canary (injected into a scratch copy) measures recall and false positives, and the archived calibration set keeps the grader honest.

The final grade uses a rubric and blocking rules: a confirmed S0, or a confirmed S1 in M/S/P/C, caps the result at weak reject; a failed deterministic gate also caps it, unless an adjudicator records a waiver.`,
		refs: [
			"research_agent/plan/07_AUDIT_HARNESS.md",
			".pi/extensions/auditkit/index.ts",
			".pi/extensions/verify/gates.ts",
			".pi/rubrics/SCHEMA.md",
		],
		checks: [
			"Why must every finding carry an executable command?",
			"What can an S1 finding in a core workstream do to the final grade?",
		],
	},
	{
		id: 10,
		slug: "protocol",
		title: "The frozen study protocol",
		summary: "Development and confirmation stay separate; no selection on the test horizon",
		body: `The science has its own non-negotiables, and the loop enforces them rather than trusting good intentions.

Development data and confirmation data are kept separate. CONFIRMATION_ORIGINS.csv and its .sha256 are frozen before any confirmation result is viewed. One frozen global configuration is used per study; there is no configuration selection on the test horizon. The context series (for example ^GSPC or the smart-home temperature) never enters target aggregates.

The metric is Relative MAE versus persistence per target×origin unit. The aggregate is the median over units, and a unit whose persistence MAE is below 1e-3 on the scaled target is excluded. Failures, excluded units and missing data go to failure/availability files and are reported; nothing is silently dropped.

Stocks run on trading days only, and load_stock_data uses interpolate=False by default. If interpolation is used it happens within a split after splitting, never across the boundary.

If a rule must change, the change is a protocol amendment: recorded, justified, and human-approved. It is not an edit to a frozen file during a run.`,
		refs: [
			".pi/skills/experiment-protocol-design/SKILL.md",
			".pi/skills/experiment-protocol-design/references/vangja-protocol.md",
			"fixed_case_studies/protocol/CONFIRMATION_ORIGINS.sha256",
			"AGENTS.md",
			".pi/skills/forecasting-evaluation/SKILL.md",
		],
		checks: [
			"What exactly is frozen before confirmation results may be viewed?",
			"When is a change a protocol amendment rather than an ordinary edit?",
		],
	},
	{
		id: 11,
		slug: "safety",
		title: "Budgets, pauses and kill switches",
		summary: "Caps are enforced, pauses are recorded, and destructive actions are refused",
		body: `Autonomy is bounded by numbers. .pi/config/budgets.json sets nine caps, including sessions per day, parallel workers, parallel HPC jobs (hard single slot), task wall clock, HPC wall clock, consecutive failures, no-progress cycles, audit cycles and cost per run. Exceeding a cap pauses the loop with a human-readable reason.

Pausing is not stopping in place. /loop:pause writes PAUSED with the state it came from, so /loop:resume returns exactly there. A running child is left alone; an HPC job keeps running and the collector still watches it. A STOP flag exists for a harder stop, and status shows it.

The guard enforces safety on every tool call: no force-push, no history rewrite, no rebase on shared history, no recursive delete outside .pi/state and .pi/artifacts, no writes to secret-bearing paths, and no secret-shaped content in commands or file writes.

The budget ledger records what was spent; a run that exceeds its cost cap pauses rather than silently continuing. Every one of these behaviours is testable offline, and the test suite asserts them.

So "the loop is autonomous" really means "the loop schedules within caps, and a human can pause, block, or stop it at any point".`,
		refs: [
			".pi/config/budgets.json",
			".pi/lib/state.ts",
			".pi/extensions/orchestrator/budget.ts",
			".pi/extensions/guards/bash.ts",
		],
		checks: [
			"What does pause preserve that a kill would lose?",
			"Which destructive git operations are categorically refused, even with a reason?",
		],
	},
	{
		id: 12,
		slug: "driving",
		title: "Driving the loop and picking up the thread",
		summary: "Verify, start, watch, gate — and leave files for the next session",
		body: `In pi, the loop is driven by a small command surface: /loop:verify-env checks the model pin, configs, schemas, protected paths and tool collisions; /loop:start steps until it pauses, blocks or finishes; /loop:step performs exactly one action; /loop:status shows state, current task/PR/experiment, budget, gates, the last events and the next action; /loop:pause and /loop:resume move in and out of PAUSED; /loop:gate list|approve|reject decides a human gate.

The same commands exist headless via .pi/loop/step.sh, status.sh and collect.sh, which exec the orchestrator CLI. There is no second implementation to drift.

When you return to a run, read .pi/state/CONTEXT.md first, then LOOP_STATE.json, TASKS.md, and the last lines of EVENTS.jsonl, and run /loop:status. The conversation is not the memory; the files are. A handoff should name the file, the command and the expected output, and end with a ledger record (task, PR, experiment, review) or BLOCKED plus the smallest decision a human must make.

If you want to keep learning the loop, this tour is the map; the plan in research_agent/plan/ is the territory.`,
		refs: [
			".pi/loop/README.md",
			".pi/extensions/orchestrator/index.ts",
			".pi/state/CONTEXT.md",
			".pi/skills/research-memory-and-handoff/SKILL.md",
		],
		checks: [
			"Which command shows the next action without performing it?",
			"What belongs in a handoff so the next session does not re-litigate decisions?",
		],
	},
];

export function chapterById(id: number): Chapter | undefined {
	return CURRICULUM.find((chapter) => chapter.id === id);
}

export function chapterTitle(id: number): string {
	return chapterById(id)?.title ?? `chapter ${id}`;
}
