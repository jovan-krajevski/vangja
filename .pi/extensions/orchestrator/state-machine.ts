/**
 * The loop state machine (plan/02_ARCHITECTURE.md §2).
 *
 * `nextAction(state, ledgers, options)` is a **pure function**: it reads a
 * snapshot and returns the single next `Action`. No I/O, no clocks, no spawning —
 * so the whole transition table is testable with hand-built snapshots, and the
 * headless CLI, the TUI commands, and the tests cannot diverge.
 *
 * Contract (plan/05_EXTENSIONS_AND_TOOLS.md §2):
 * - one action per step; the action is keyed by `(state, task_id, attempt)` plus
 *   the entity it belongs to;
 * - the executor records a completion event for a key; a replayed key is a no-op;
 * - actions never "look ahead": a spawn leaves the state unchanged, the *next*
 *   step re-reads the ledgers (which the child just wrote) and transitions.
 *
 * Coverage: BOOT, SELECT_TASK, WORK, WAITING_HPC, REVIEW, MERGE, POST_MERGE,
 * READINESS, PAPER, PAPER_REVIEW, AUDIT, DECIDE, PAUSED, BLOCKED, DONE, budget
 * stops, human gates, and the WAITING_HPC sub-state of WORK.
 */

import { ACTIVE_EXPERIMENT_STATUSES, projectLatest } from "../../lib/records.ts";
import type { BudgetConfig } from "../../lib/config.ts";
import type { LoopState, LoopStateName } from "../../lib/state.ts";
import { decideRetry, isTerminalExperimentStatus, type RetryDecision } from "../hpc/retry.ts";
import { budgetViolationReason, budgetViolations, resumeCommand, type BudgetSnapshot } from "./budget.ts";

// ---------------------------------------------------------------------------
// Snapshot types (the read-only input to the pure function)
// ---------------------------------------------------------------------------

export interface TaskView {
	id: string;
	title: string;
	priority: string;
	status: string;
	origin: string;
}

export interface PrView {
	id: string;
	task_id: string;
	status: string;
	branch: string;
	head_sha: string;
	base_sha: string;
	/** "work" (default) or "paper": decides whether MERGE goes to POST_MERGE or AUDIT. */
	kind: string;
	summary: string;
}

export interface ReviewView {
	id: string;
	pr_id: string;
	decision: string;
	actor: string;
}

export interface ExperimentView {
	id: string;
	status: string;
	attempt: number;
	task_id: string;
	/** Current SLURM job id, or null before submission. */
	job_id?: string | null;
	/** Per-attempt checkpoint progress snapshots appended by the collector. */
	progressHistory?: Array<{ attempt: number; completed: number }>;
	memoryDoubled?: boolean;
	memoryGb?: number | null;
	/** `resources.partitionMemGbLimit`; null means unknown (never guessed). */
	partitionMemGbLimit?: number | null;
	/** Last classified sacct state. */
	lastState?: string | null;
	exitCode?: number | null;
	stderrTail?: string | null;
	missingOutputs?: string[];
	checkpointsFetched?: boolean;
	resourcesKind?: string | null;
}

export interface GateView {
	id: string;
	gate: string;
	status: string;
	note: string;
	pr_id?: string;
	task_id?: string;
}

export interface EventView {
	seq: number;
	ts: string;
	actor: string;
	summary: string;
	action_key?: string;
	action_phase?: "start" | "complete" | string;
	exit_state?: string;
	from?: string;
	to?: string;
	experiment_id?: string;
	heartbeat?: boolean;
	hpc_disabled?: boolean;
	hpc_op?: string;
	hpc_ok?: boolean;
	hpc_missing?: string[];
}

export interface FindingView {
	id: string;
	severity: string;
	status: string;
}

export interface WaiverView {
	id?: string;
	gate: string;
	reason: string;
}

export interface ReadinessView {
	pass: boolean;
	reason: string;
}

export interface DecisionView {
	done: boolean;
	grade?: string;
	reason?: string;
}

export interface AuditView {
	/** 1-based cycle currently being run. */
	cycle: number;
	/** True when the S6 pipeline has finished all four phases for this cycle. */
	complete: boolean;
	/** Current phase when incomplete: find | verify | synthesise | adjudicate. */
	phase: string;
}

export interface LedgerView {
	tasks: TaskView[];
	prs: PrView[];
	reviews: ReviewView[];
	experiments: ExperimentView[];
	gates: GateView[];
	events: EventView[];
	findings: FindingView[];
	waivers: WaiverView[];
	budget: BudgetSnapshot;
	readiness: ReadinessView | null;
	decision: DecisionView | null;
	audit: AuditView | null;
	stopFlag: boolean;
	/** Set during BOOT: the recorded base SHA and whether the checkout is clean. */
	baseSha: string | null;
	repoClean: boolean;
}

export interface LoopOptions {
	requireHumanMerge: boolean;
	budgets: BudgetConfig;
	sandbox?: string;
	/** False when `.pi/config/hpc.json` has an empty host: WAITING_HPC is skipped. */
	hpcEnabled?: boolean;
	/** Set when the HPC config is present but invalid; the loop blocks instead of guessing. */
	hpcError?: string | null;
	/** Collector poll budget (`.pi/config/hpc.json` maxPolls). */
	maxPolls?: number;
	/** Requeue cap (`.pi/config/hpc.json` maxAttemptsPerExperiment). */
	maxAttempts?: number;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

export type ActionKind =
	| "boot"
	| "spawn"
	| "bookkeep"
	| "gate"
	| "merge"
	| "pause"
	| "block"
	| "hpc"
	| "none"
	| "done";

export type HumanGateKind = "merge" | "confirmation" | "audit" | "submission" | "enqueue";

export type BookkeepEffect =
	| { type: "task_status"; task_id: string; status: string; evidence: string }
	| { type: "set_task"; task_id: string | null; attempt: number }
	| { type: "set_experiment"; experiment_id: string | null; attempt: number }
	| { type: "bump_attempt" }
	| { type: "reset_failures" }
	| { type: "clear_pr" }
	| { type: "hpc_disabled"; experiment_id: string; reason: string }
	| { type: "local_fallback"; experiment_id: string; reason: string }
	| { type: "experiment_status"; experiment_id: string; status: string; evidence: string }
	| { type: "enqueue_task"; task: Record<string, unknown> };

export interface ActionBase {
	kind: ActionKind;
	/** The state this action runs in. */
	state: LoopStateName;
	/** Idempotency key; recorded in the completion event. */
	key: string;
	summary: string;
	task_id?: string | null;
	pr_id?: string | null;
	experiment_id?: string | null;
	attempt?: number;
}

export interface BootAction extends ActionBase {
	kind: "boot";
	to: LoopStateName;
}

export interface SpawnAction extends ActionBase {
	kind: "spawn";
	/** Primary agent (or audit agent) from `.pi/config/agents.json`. */
	agent: string;
	/** The brief the child receives as its user message. */
	brief: string;
	/** Files the brief names, echoed into the event for traceability. */
	evidence: string[];
}

export interface BookkeepAction extends ActionBase {
	kind: "bookkeep";
	to: LoopStateName;
	effects: BookkeepEffect[];
	reason: string;
}

export interface GateAction extends ActionBase {
	kind: "gate";
	gate: HumanGateKind;
	note: string;
}

export interface MergeAction extends ActionBase {
	kind: "merge";
	pr_id: string;
	review_id: string;
	/** Where the loop goes after a successful merge. */
	to: LoopStateName;
}

export interface PauseAction extends ActionBase {
	kind: "pause";
	reason: string;
}

export interface BlockAction extends ActionBase {
	kind: "block";
	reason: string;
	/** The smallest decision the human must make. */
	question: string;
	/** Effects applied before the loop enters BLOCKED (e.g. failed-final, task). */
	effects?: BookkeepEffect[];
}

/** One HPC side effect performed by the conductor (never by an LLM). */
export interface HpcAction extends ActionBase {
	kind: "hpc";
	op: "submit" | "requeue";
	experiment_id: string;
	reason: string;
	/** Set for the OOM retry (doubled job memory). */
	memoryGb?: number;
}

export interface NoneAction extends ActionBase {
	kind: "none";
	reason: string;
	/** True when the loop is waiting on a human gate. */
	waiting_on_gate?: string;
}

export interface DoneAction extends ActionBase {
	kind: "done";
	reason: string;
}

export type Action =
	| BootAction
	| SpawnAction
	| BookkeepAction
	| GateAction
	| MergeAction
	| PauseAction
	| BlockAction
	| HpcAction
	| NoneAction
	| DoneAction;

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

export function latestTasks(tasks: Array<Record<string, unknown>>): TaskView[] {
	return projectLatest(tasks as Array<{ id?: string; seq: number }>).map((task) => ({
		id: String(task.id ?? ""),
		title: String(task.title ?? ""),
		priority: String(task.priority ?? "P3"),
		status: String(task.status ?? "open"),
		origin: String(task.origin ?? ""),
	}));
}

export function latestPrs(prs: Array<Record<string, unknown>>): PrView[] {
	return projectLatest(prs as Array<{ id?: string; seq: number }>).map((pr) => ({
		id: String(pr.id ?? ""),
		task_id: String(pr.task_id ?? ""),
		status: String(pr.status ?? "open"),
		branch: String(pr.branch ?? ""),
		head_sha: String(pr.head_sha ?? ""),
		base_sha: String(pr.base_sha ?? ""),
		kind: String(pr.kind ?? "work"),
		summary: String(pr.summary ?? ""),
	}));
}

export function latestReviews(reviews: Array<Record<string, unknown>>): ReviewView[] {
	return projectLatest(reviews as Array<{ id?: string; seq: number }>).map((review) => ({
		id: String(review.id ?? ""),
		pr_id: String(review.pr_id ?? ""),
		decision: String(review.decision ?? ""),
		actor: String(review.actor ?? ""),
	}));
}

export function latestExperiments(experiments: Array<Record<string, unknown>>): ExperimentView[] {
	return projectLatest(experiments as Array<{ id?: string; seq: number }>).map((experiment) => {
		const resources = (experiment.resources ?? {}) as Record<string, unknown>;
		const history = Array.isArray(experiment.progressHistory)
			? (experiment.progressHistory as Array<Record<string, unknown>>).map((entry) => ({
					attempt: Number(entry.attempt ?? 0),
					completed: Number(entry.completed ?? 0),
				}))
			: [];
		return {
			id: String(experiment.id ?? ""),
			status: String(experiment.status ?? "draft"),
			attempt: Number(experiment.attempt ?? 0),
			task_id: String(experiment.task_id ?? ""),
			job_id: typeof experiment.job_id === "string" && experiment.job_id !== "" ? experiment.job_id : null,
			progressHistory: history,
			memoryDoubled: experiment.memoryDoubled === true,
			memoryGb: Number.isFinite(Number(resources.memGb)) ? Number(resources.memGb) : null,
			partitionMemGbLimit: Number.isFinite(Number(resources.partitionMemGbLimit))
				? Number(resources.partitionMemGbLimit)
				: null,
			lastState: typeof experiment.lastState === "string" ? experiment.lastState : null,
			exitCode: typeof experiment.exitCode === "number" ? experiment.exitCode : null,
			stderrTail: typeof experiment.stderrTail === "string" ? experiment.stderrTail : null,
			missingOutputs: Array.isArray(experiment.missingOutputs) ? (experiment.missingOutputs as string[]) : [],
			checkpointsFetched: experiment.checkpointsFetched === true,
			resourcesKind: typeof resources.kind === "string" ? resources.kind : null,
		};
	});
}

/** An experiment the loop still owes an outcome: not collected/failed-final/cancelled. */
export function findUnfinishedExperiment(experiments: ExperimentView[]): ExperimentView | null {
	return experiments.find((experiment) => !isTerminalExperimentStatus(experiment.status)) ?? null;
}

/** Heartbeat events recorded for one experiment (collector stuck-poll budget). */
export function heartbeatCount(events: EventView[], experimentId: string): number {
	return events.filter((event) => event.experiment_id === experimentId && event.heartbeat === true).length;
}

/** True when the conductor already recorded the `hpc-disabled` event for this experiment. */
export function hpcDisabledRecorded(events: EventView[], experimentId: string): boolean {
	return events.some((event) => event.experiment_id === experimentId && event.hpc_disabled === true);
}

/** Build the `decideRetry` input from the experiment view + options. */
export function retryDecisionFor(experiment: ExperimentView, events: EventView[], options: LoopOptions): RetryDecision {
	return decideRetry({
		status: experiment.status,
		attempt: experiment.attempt,
		maxAttempts: options.maxAttempts ?? 6,
		progressHistory: experiment.progressHistory ?? [],
		memoryDoubled: experiment.memoryDoubled === true,
		memoryGb: experiment.memoryGb ?? 0,
		partitionMemGbLimit: experiment.partitionMemGbLimit ?? null,
		sacctState: experiment.lastState ?? null,
		exitCode: experiment.exitCode ?? null,
		stderrTail: experiment.stderrTail ?? null,
		heartbeatCount: heartbeatCount(events, experiment.id),
		maxPolls: options.maxPolls ?? 40,
		missingOutputs: experiment.missingOutputs ?? [],
		taskId: experiment.task_id,
		jobId: experiment.job_id ?? null,
	});
}

export function latestGates(gates: Array<Record<string, unknown>>): GateView[] {
	return projectLatest(gates as Array<{ id?: string; seq: number }>).map((gate) => ({
		id: String(gate.id ?? ""),
		gate: String(gate.gate ?? ""),
		status: String(gate.status ?? "pending"),
		note: String(gate.note ?? ""),
		pr_id: typeof gate.pr_id === "string" ? gate.pr_id : undefined,
		task_id: typeof gate.task_id === "string" ? gate.task_id : undefined,
	}));
}

export function findActiveExperiment(experiments: ExperimentView[]): ExperimentView | null {
	return (
		experiments.find((experiment) =>
			(ACTIVE_EXPERIMENT_STATUSES as readonly string[]).includes(experiment.status),
		) ?? null
	);
}

export function openTasks(tasks: TaskView[]): TaskView[] {
	return tasks.filter((task) => ["open", "in_progress", "blocked"].includes(task.status));
}

export function selectableTasks(tasks: TaskView[]): TaskView[] {
	const rank: Record<string, number> = { P0: 0, P1: 1, P2: 2, P3: 3 };
	return tasks
		.filter((task) => task.status === "open" || task.status === "in_progress")
		.sort((a, b) => (rank[a.priority] ?? 9) - (rank[b.priority] ?? 9) || a.id.localeCompare(b.id));
}

export function prForTask(prs: PrView[], taskId: string | null | undefined): PrView | null {
	if (!taskId) return null;
	return prs.find((pr) => pr.task_id === taskId && ["open", "review"].includes(pr.status)) ?? null;
}

export function latestReviewForPr(reviews: ReviewView[], prId: string | null | undefined): ReviewView | null {
	if (!prId) return null;
	const matching = reviews.filter((review) => review.pr_id === prId);
	return matching.length > 0 ? matching[matching.length - 1] : null;
}

/** Number of completed executions of an action-key prefix (used for counters). */
export function completions(events: EventView[], prefix: string): number {
	return events.filter((event) => event.action_phase === "complete" && (event.action_key ?? "").startsWith(prefix))
		.length;
}

export function pendingGate(gates: GateView[], gate: string, prId?: string | null): GateView | null {
	return (
		gates.find(
			(record) =>
				record.gate === gate &&
				record.status === "pending" &&
				(prId === undefined || prId === null || record.pr_id === prId || record.pr_id === undefined),
		) ?? null
	);
}

export function decidedGate(gates: GateView[], gate: string, prId?: string | null): GateView | null {
	const matching = gates.filter(
		(record) =>
			record.gate === gate &&
			record.status !== "pending" &&
			(prId === undefined || prId === null || record.pr_id === prId || record.pr_id === undefined),
	);
	return matching.length > 0 ? matching[matching.length - 1] : null;
}

/**
 * The mechanical subset of the readiness rubric (plan/02_ARCHITECTURE.md §5.3)
 * the conductor can check without a model: no open P0/P1 task and no open S0/S1
 * finding (a waiver naming `finding:<id>` closes one). The full rubric is the
 * supervisor's job in READINESS; this decides whether to *enter* it.
 */
export function readinessCriteria(ledgers: LedgerView): { green: boolean; blockers: string[] } {
	const blockers: string[] = [];
	for (const task of selectableTasks(ledgers.tasks)) {
		if (task.priority === "P0" || task.priority === "P1") blockers.push(`${task.id} (${task.priority} ${task.status})`);
	}
	const waived = new Set(
		ledgers.waivers
			.map((waiver) => waiver.gate)
			.filter((gate) => gate.startsWith("finding:"))
			.map((gate) => gate.slice("finding:".length)),
	);
	for (const finding of ledgers.findings) {
		if (!["S0", "S1"].includes(finding.severity)) continue;
		if (finding.status === "refuted") continue;
		if (waived.has(finding.id)) continue;
		blockers.push(`${finding.id} (${finding.severity} ${finding.status})`);
	}
	return { green: blockers.length === 0, blockers };
}

function baseAction(state: LoopState, kind: ActionKind, key: string, summary: string): ActionBase {
	return {
		kind,
		state: state.state,
		key,
		summary,
		task_id: state.task_id,
		pr_id: state.pr_id,
		experiment_id: state.experiment_id,
		attempt: state.attempt,
	};
}

function spawn(
	state: LoopState,
	key: string,
	agent: string,
	summary: string,
	brief: string,
	evidence: string[] = [],
): SpawnAction {
	return {
		...baseAction(state, "spawn", key, summary),
		kind: "spawn",
		agent,
		brief,
		evidence,
		attempt: state.attempt + 1,
	};
}

function bookkeep(
	state: LoopState,
	to: LoopStateName,
	effects: BookkeepEffect[],
	reason: string,
	suffix = "",
	overrides: { task_id?: string | null; pr_id?: string | null; experiment_id?: string | null } = {},
): BookkeepAction {
	return {
		...baseAction(state, "bookkeep", `book:${state.state}->${to}:${suffix || state.task_id || "loop"}:a${state.attempt}`, `bookkeeping: ${reason}`),
		kind: "bookkeep",
		to,
		effects,
		reason,
		task_id: overrides.task_id !== undefined ? overrides.task_id : state.task_id,
		pr_id: overrides.pr_id !== undefined ? overrides.pr_id : state.pr_id,
		experiment_id: overrides.experiment_id !== undefined ? overrides.experiment_id : state.experiment_id,
	};
}

function none(state: LoopState, reason: string, waitingOnGate?: string): NoneAction {
	return { ...baseAction(state, "none", `none:${state.state}:${waitingOnGate ?? reason.slice(0, 40)}`, reason), kind: "none", reason, waiting_on_gate: waitingOnGate };
}

function pause(state: LoopState, reason: string): PauseAction {
	return { ...baseAction(state, "pause", `pause:${state.state}:${reason.slice(0, 60)}`, reason), kind: "pause", reason };
}

function block(state: LoopState, reason: string, question: string, effects: BookkeepEffect[] = []): BlockAction {
	return {
		...baseAction(state, "block", `block:${state.state}:${reason.slice(0, 60)}`, reason),
		kind: "block",
		reason,
		question,
		effects,
	};
}

function hpc(
	state: LoopState,
	op: "submit" | "requeue",
	experimentId: string,
	reason: string,
	memoryGb?: number,
): HpcAction {
	return {
		...baseAction(state, "hpc", `hpc:${op}:${experimentId}:a${state.attempt}`, `${op} ${experimentId}: ${reason}`),
		kind: "hpc",
		op,
		experiment_id: experimentId,
		reason,
		memoryGb,
	};
}

// ---------------------------------------------------------------------------
// Briefs (deterministic text; the executor writes the brief into the event)
// ---------------------------------------------------------------------------

export function selectBrief(ledgers: LedgerView): string {
	const open = selectableTasks(ledgers.tasks).map((task) => `${task.id} ${task.priority} ${task.title}`);
	return [
		"## SELECT_TASK",
		"",
		"Objective: choose (or write) exactly one task brief for the next work unit and enqueue it with",
		"`task_enqueue`. Prefer, in order: open S0/S1 findings, REVIEW/PLAN.md P0/P1 items, protocol gaps,",
		"paper-evidence gaps. Do not start work in this session.",
		"",
		`Open tasks already in the ledger: ${open.length > 0 ? open.join("; ") : "none"}.`,
		`Read: .pi/state/CONTEXT.md, .pi/state/TASKS.md, .pi/state/FINDINGS.jsonl, research_agent/REVIEW/PLAN.md (if present).`,
		"",
		"Acceptance: one `task` record with acceptance_criteria and deliverables (or BLOCKED with the smallest decision needed).",
	].join("\n");
}

export function workBrief(state: LoopState, task: TaskView | null): string {
	return [
		`## WORK · ${task?.id ?? state.task_id ?? "unknown"}`,
		"",
		`Objective: complete ${task?.id ?? "the task"} (${task?.title ?? "see .pi/state/tasks/"}) to its acceptance criteria.`,
		`Origin: ${task?.origin ?? "see the task record"}. Attempt: ${state.attempt + 1}.`,
		"",
		"Rules: work in the worktree at .pi/state/worktrees/<task_id>/ on branch work/<task>-<slug>;",
		"read the brief in .pi/state/tasks/<id>.md; run the narrowest test first, then the full suite;",
		"commit atomically; then call `pr_open` with the machine-readable footer (task_id, head_sha, base_sha,",
		"branch, summary, artifacts, tests, claims, hpc_jobs). If compute is needed, write the experiment",
		"record with checkpoint + --resume + local resume smoke test and call `experiment_queue` — one job max.",
		"",
		"Acceptance: a PR record at the committed head SHA, or BLOCKED with the smallest decision needed.",
	].join("\n");
}

export function reviewBrief(pr: PrView | null, attempt: number): string {
	return [
		`## REVIEW · ${pr?.id ?? "unknown PR"}`,
		"",
		`Objective: verify ${pr?.id ?? "the PR"} at head ${pr?.head_sha ?? "?"} against its task's acceptance criteria.`,
		`Read .pi/state/prs/${pr?.id ?? "PR-###"}.md and the diff; run the acceptance commands yourself;`,
		"check the pre-merge gates (tests, protected paths, test diff, integrity, provenance, hygiene, claims).",
		`Attempt: ${attempt}.`,
		"",
		"Write `state/reviews/<RV-id>.md` and call `review_write` with decision, rubric_scores, evidence",
		"(commands you actually ran + observed output), required_changes, residual_risk. Prefer",
		"changes_requested over accepting ambiguity; never accept without running the commands.",
		"",
		"Acceptance: one review record with a decision and evidence, or BLOCKED.",
	].join("\n");
}

export function collectorBrief(experiment: ExperimentView | null, decision?: RetryDecision): string {
	if (!experiment) {
		return [
			"## WAITING_HPC · no active experiment",
			"",
			"No experiment is unfinished. Append one `hpc-disabled` event only if `.pi/config/hpc.json` has no host; otherwise stop.",
		].join("\n");
	}
	return [
		`## WAITING_HPC · ${experiment.id} (${experiment.status}, attempt ${experiment.attempt})`,
		"",
		`Objective: perform ONE collector pass for ${experiment.id}${experiment.job_id ? ` (job ${experiment.job_id})` : ""}.`,
		`Conductor decision: ${decision?.action ?? "collect"} — ${decision?.reason ?? "poll the job and record evidence"}.`,
		"",
		"Procedure (plan/06_HPC_AND_COLLECTOR.md §6):",
		`1. \`hpc_status\` for ${experiment.id} — the raw squeue/sacct output is the evidence. Append at most one`,
		"   `experiment_update` per transition, with that raw evidence in the `evidence` argument.",
		"2. Still queued/running and nothing changed → append ONE `heartbeat` event (ledger_append kind=event,",
		"   `{experiment_id, heartbeat: true, poll: n, evidence}`) and stop. Never invent a transition.",
		"3. Exit 0 and all declared outputs present → transition running→collecting, then `hpc_fetch` (outputs +",
		"   checkpoints), then collecting→collected with the manifest path + SHA-256. The manifest must carry the",
		"   sacct accounting fields and a SHA-256 per fetched file.",
		"4. A declared output is missing after fetch → collecting→collection-failed with the missing list; do not",
		"   repair, interpolate, or fabricate. The conductor opens the task.",
		"5. TIMEOUT / TIME_LIMIT / PREEMPTED / NODE_FAIL / OOM / exit 99 → run `hpc_fetch` with",
		"   `checkpointsOnly: true` FIRST; record the progress delta via `fields: {progressHistory: [...],",
		"   progress: {...}, lastState, exitCode, stderrTail}`; then transition to timed_out/interrupted_clean/",
		"   failed with the raw state, then append requeue_requested. Never resubmit yourself — the conductor",
		"   decides from your evidence.",
		"",
		"Prohibitions: no code/config edits, no resubmission, no cancellation without an instruction record, no",
		"metric interpretation, no second job. Never touch `update_pymc`.",
		"",
		"Acceptance: one collector pass whose ledger records carry raw evidence (or one explicit heartbeat), or",
		"BLOCKED with the smallest decision needed.",
	].join("\n");
}

export function localFallbackBrief(experiment: ExperimentView, task: TaskView | null): string {
	return [
		`## WORK (local fallback) · ${experiment.id}`,
		"",
		`Objective: run the queued experiment ${experiment.id} locally because .pi/config/hpc.json has no host`,
		"(plan/06_HPC_AND_COLLECTOR.md §7). The task is " +
			`${task?.id ?? "?"} (${task?.title ?? "see the task record"}).`,
		"",
		"Rules:",
		"- Use the record's `command` and `checkpoints.resumeCommand` verbatim; the same checkpoint directory,",
		"  `progress.json` after every completed unit, and the same declared `expectedOutputs` apply.",
		"- Write results under `.pi/artifacts/<experiment_id>/` and a manifest with a SHA-256 per file and the exit",
		"  state; the audit must not be able to tell a local run from an HPC one except through the record.",
		"- Record `resources.kind = \"local\"` and the observed progress in the experiment evidence; do not invent",
		"  job ids or sacct fields.",
		"- A failed fit or a missing output is reported, never worked around.",
		"",
		`Acceptance: the experiment's outputs exist with hashes and a manifest, or BLOCKED with the failure and the`,
		"smallest decision needed.",
	].join("\n");
}

export function readinessBrief(blockers: string[]): string {
	return [
		"## READINESS",
		"",
		"Evaluate the readiness rubric (plan/02_ARCHITECTURE.md §5.3) and write `.pi/state/READINESS.json`",
		'as {"pass": true|false, "reason": "...", "evidence": ["<command/output>", ...]}.',
		"",
		`Mechanical blockers currently visible to the conductor: ${blockers.length > 0 ? blockers.join("; ") : "none"}.`,
		"Check the full rubric: all S0/S1 closed or waived; P0/P1 items closed; confirmation artifacts per",
		"PROTOCOL §11 with verified frozen-origin hashes; baselines on identical data; calibration done;",
		"`make verify` green; environment pinned; limitations documented.",
		"",
		"Acceptance: READINESS.json exists with a boolean `pass` and evidence; never pass with an open blocker.",
	].join("\n");
}

export function paperBrief(state: LoopState, pr: PrView | null): string {
	return [
		"## PAPER",
		"",
		pr
			? `Objective: revise paper/** for the accepted review work and open a paper PR (currently ${pr.id} at ${pr.head_sha}).`
			: "Objective: revise paper/** from the artifacts and open a paper PR.",
		"Every number must come from `paper/generated/**` through `paper_build`; update paper/CLAIMS.json;",
		"keep the limitations, AI-use and reproducibility statements current.",
		`Attempt: ${state.attempt + 1}.`,
		"",
		"Acceptance: a PR record with kind=paper, head SHA, artifacts and claims, or BLOCKED.",
	].join("\n");
}

export function auditBrief(cycle: number, phase: string): string {
	return [
		`## AUDIT · cycle ${cycle} · phase ${phase}`,
		"",
		"The four-phase audit pipeline is delivered in stage S6 (`.pi/prompts/audit-cycle.md`). This work unit",
		"starts (or continues) the cycle: freeze the snapshot, write SCOPE.md, and run the finders per workstream.",
		"Never read Vangja-project/audit/** for findings; it is calibration reference only.",
		"",
		"Acceptance: the artifacts the audit-cycle prompt names for this phase; the S6 harness sets",
		"`.pi/state/AUDIT.json` to {cycle, phase, complete} when the phase finishes.",
	].join("\n");
}

export function decideBrief(): string {
	return [
		"## DECIDE",
		"",
		"Apply the audit decision rule (plan/02_ARCHITECTURE.md §5.4) to the final audit report + findings.json.",
		'Write `.pi/state/DECISION.json` as {"done": true|false, "grade": "<grade>", "reason": "..."} where',
		"`done` is true only for accept/strong accept with no blocking fixes.",
		"",
		"Acceptance: DECISION.json with the grade, the rule trace, and the blocking-fix list (empty for done).",
	].join("\n");
}

// ---------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------

interface Rule {
	id: string;
	when: (state: LoopState, ledgers: LedgerView, options: LoopOptions) => boolean;
	action: (state: LoopState, ledgers: LedgerView, options: LoopOptions) => Action;
}

function budgetRule(): Rule {
	return {
		id: "budget",
		when: (_state, ledgers, options) => budgetViolations(ledgers.budget, options.budgets, options.sandbox).length > 0,
		action: (state, ledgers, options) => {
			const violation = budgetViolations(ledgers.budget, options.budgets, options.sandbox)[0];
			return pause(state, budgetViolationReason(violation, options.sandbox));
		},
	};
}

function taskOf(state: LoopState, ledgers: LedgerView): TaskView | null {
	if (state.task_id) return ledgers.tasks.find((task) => task.id === state.task_id) ?? null;
	return selectableTasks(ledgers.tasks)[0] ?? null;
}

export const RULES: Rule[] = [
	{
		id: "stop-flag",
		when: (state, ledgers) => ledgers.stopFlag && state.state !== "DONE",
		action: (state, ledgers, options) =>
			pause(
				state,
				`STOP flag is set (.pi/state/STOP); no further step runs. Remove it and resume with: ` +
					`${resumeCommand(options.sandbox)}`,
			),
	},
	{
		id: "done",
		when: (state) => state.state === "DONE",
		action: (state) => ({ ...baseAction(state, "done", "done", "the run is DONE (human sign-off recorded)"), kind: "done", reason: "the run is DONE (human sign-off recorded)" }),
	},
	{
		id: "paused",
		when: (state) => state.state === "PAUSED",
		action: (state) =>
			none(
				state,
				`loop is PAUSED (${state.reason ?? "no reason recorded"}); resume with the recorded state ` +
					`${state.paused_from ?? "?"} and never repeat completed steps`,
			),
	},
	{
		id: "blocked",
		when: (state) => state.state === "BLOCKED",
		action: (state) => none(state, `loop is BLOCKED: ${state.reason ?? "no reason recorded"}`),
	},
	budgetRule(),
	{
		id: "uninitialized",
		when: (state) => state.state === "UNINITIALIZED",
		action: (state) => ({
			...baseAction(state, "boot", "boot", "verify the environment, create the run id, snapshot the base SHA"),
			kind: "boot",
			to: "SELECT_TASK",
		}),
	},
	{
		id: "boot",
		when: (state) => state.state === "BOOT",
		action: (state) => ({
			...baseAction(state, "boot", "boot", "BOOT did not complete; re-run the environment check and continue"),
			kind: "boot",
			to: "SELECT_TASK",
		}),
	},
	{
		id: "select-task-to-readiness",
		when: (state, ledgers) =>
			state.state === "SELECT_TASK" && selectableTasks(ledgers.tasks).length === 0 && readinessCriteria(ledgers).green,
		action: (state) => bookkeep(state, "READINESS", [], "no work left and the mechanical readiness criteria are green", "loop"),
	},
	{
		id: "select-task-assign",
		when: (state, ledgers) => state.state === "SELECT_TASK" && selectableTasks(ledgers.tasks).length > 0,
		action: (state, ledgers) => {
			const task = selectableTasks(ledgers.tasks)[0];
			return bookkeep(
				state,
				"WORK",
				[
					{ type: "set_task", task_id: task.id, attempt: 0 },
					{ type: "reset_failures" },
				],
				`selected ${task.id} (${task.priority}): ${task.title}`,
				task.id,
				{ task_id: task.id, pr_id: null },
			);
		},
	},
	{
		id: "select-task",
		when: (state) => state.state === "SELECT_TASK",
		action: (state, ledgers) =>
			spawn(
				state,
				`select:${completions(ledgers.events, "select:") + 1}`,
				"phd_candidate",
				"select the next task",
				selectBrief(ledgers),
				[".pi/state/TASKS.md", ".pi/state/CONTEXT.md"],
			),
	},
	{
		id: "work-any",
		when: (state) => state.state === "WORK",
		action: (state, ledgers, options) => {
			const task = taskOf(state, ledgers);
			if (!task) {
				return bookkeep(state, "SELECT_TASK", [], "no task is assigned; return to task selection");
			}
			if (task.status === "blocked") {
				return block(
					state,
					`${task.id} is blocked: ${task.title}`,
					"resolve the blocker named in the task record, then resume the loop",
				);
			}
			const pr = prForTask(ledgers.prs, task.id);
			if (task.status === "review" || pr) {
				return bookkeep(
					state,
					"REVIEW",
					pr ? [] : [{ type: "task_status", task_id: task.id, status: "review", evidence: "task status set to review by the conductor" }],
					`${task.id} has a PR${pr ? ` (${pr.id})` : " and is marked review"}; supervise it`,
					pr?.id ?? task.id,
					{ pr_id: pr?.id ?? null },
				);
			}
			const experiment = findUnfinishedExperiment(ledgers.experiments);
			if (experiment) {
				if (options.hpcError) {
					return block(
						state,
						`.pi/config/hpc.json is invalid: ${options.hpcError}`,
						"fix the HPC config (or set host to \"\" to disable HPC deliberately), then resume",
					);
				}
				if (options.hpcEnabled !== false) {
					return bookkeep(
						state,
						"WAITING_HPC",
						[{ type: "set_experiment", experiment_id: experiment.id, attempt: experiment.attempt }],
						`experiment ${experiment.id} (${experiment.status}, attempt ${experiment.attempt}) is unfinished; one job at a time`,
						experiment.id,
					);
				}
				// HPC disabled: skip WAITING_HPC entirely (plan/06 §1) and run the same
				// command locally with the same checkpoint contract (plan/06 §7).
				if (!hpcDisabledRecorded(ledgers.events, experiment.id)) {
					return bookkeep(
						state,
						"WORK",
						[
							{ type: "hpc_disabled", experiment_id: experiment.id, reason: "host is empty; the HPC path is disabled" },
							{ type: "local_fallback", experiment_id: experiment.id, reason: "no HPC host" },
						],
						`HPC is disabled; experiment ${experiment.id} runs locally with the same checkpoint contract`,
						experiment.id,
						{ experiment_id: experiment.id },
					);
				}
				return spawn(
					state,
					`work-local:${task.id}:${state.attempt + 1}`,
					"phd_candidate",
					`run ${experiment.id} locally (attempt ${state.attempt + 1})`,
					localFallbackBrief(experiment, task),
					[`.pi/state/tasks/${task.id}.md`, ".pi/state/EXPERIMENTS.md"],
				);
			}
			return spawn(
				state,
				`work:${task.id}:${state.attempt + 1}`,
				"phd_candidate",
				`work ${task.id} (attempt ${state.attempt + 1})`,
				workBrief(state, task),
				[`.pi/state/tasks/${task.id}.md`],
			);
		},
	},
	{
		id: "waiting-hpc",
		when: (state) => state.state === "WAITING_HPC",
		action: (state, ledgers, options) => {
			const experiment = findUnfinishedExperiment(ledgers.experiments);
			if (!experiment) {
				return bookkeep(
					state,
					"WORK",
					[{ type: "set_experiment", experiment_id: null, attempt: 0 }],
					"no experiment is unfinished; return to WORK",
					state.task_id ?? "loop",
				);
			}
			if (options.hpcError) {
				return block(
					state,
					`.pi/config/hpc.json is invalid: ${options.hpcError}`,
					"fix the HPC config (or set host to \"\" to disable HPC deliberately), then resume",
				);
			}
			if (options.hpcEnabled === false) {
				return bookkeep(
					state,
					"WORK",
					[{ type: "set_experiment", experiment_id: null, attempt: 0 }],
					"HPC is disabled; the experiment runs locally from WORK (plan/06 §7)",
					state.task_id ?? "loop",
				);
			}
			// First submission of a queued record: the conductor submits (the collector
			// only polls). hpc_submit is gate-checked and idempotent by attempt.
			if (experiment.status === "queued" && (experiment.job_id ?? null) === null) {
				return hpc(state, "submit", experiment.id, `queued attempt ${experiment.attempt} has no job id yet`);
			}
			const decision = retryDecisionFor(experiment, ledgers.events, options);
			switch (decision.action) {
				case "wait":
					return none(state, decision.reason);
				case "collect":
					return spawn(
						state,
						`collect:${experiment.id}:a${experiment.attempt}:${experiment.status}`,
						"collector",
						`collect ${experiment.id} (${experiment.status}, attempt ${experiment.attempt})`,
						collectorBrief(experiment, decision),
						[".pi/state/EXPERIMENTS.md"],
					);
				case "requeue":
					return hpc(state, "requeue", experiment.id, decision.reason, decision.memoryGb);
				case "failed-final":
					return block(state, `experiment ${experiment.id}: ${decision.reason}`, decision.question ?? "resolve the failure", [
						{
							type: "experiment_status",
							experiment_id: experiment.id,
							status: "failed-final",
							evidence: `conductor retry decision: ${decision.reason}`,
						},
					]);
				case "open-task": {
					const origin = `REVIEW:${experiment.id}`;
					const already = ledgers.tasks.some(
						(task) => task.origin === origin && !["merged", "closed", "rejected"].includes(task.status),
					);
					if (already) {
						return block(state, `experiment ${experiment.id}: ${decision.reason}`, decision.question ?? "resolve the missing outputs", []);
					}
					return block(state, `experiment ${experiment.id}: ${decision.reason}`, decision.question ?? "resolve the missing outputs", [
						{
							type: "enqueue_task",
							task: {
								origin,
								priority: "P1",
								title: decision.title ?? `repair experiment ${experiment.id}`,
								objective: decision.reason,
								acceptance_criteria: [
									"the missing outputs are either produced by a fixed runner or the experiment record is corrected",
									"the repair is verified by `hpc_fetch`/`verify_artifact` producing a manifest with no missing entries",
								],
								deliverables: [`.pi/state/EXPERIMENTS.jsonl`],
								status: "open",
								effort: "small",
							},
						},
					]);
				}
				case "blocked":
				default:
					return block(state, `experiment ${experiment.id}: ${decision.reason}`, decision.question ?? "resolve the blocker", []);
			}
		},
	},
	{
		id: "review-to-merge-changes-reject",
		when: (state, ledgers) => {
			if (state.state !== "REVIEW") return false;
			const pr = state.pr_id ? ledgers.prs.find((p) => p.id === state.pr_id) ?? prForTask(ledgers.prs, state.task_id) : prForTask(ledgers.prs, state.task_id);
			const review = latestReviewForPr(ledgers.reviews, pr?.id);
			return pr !== null && review !== null && review.decision !== "accept";
		},
		action: (state, ledgers) => {
			const pr = state.pr_id ? ledgers.prs.find((p) => p.id === state.pr_id) ?? prForTask(ledgers.prs, state.task_id) : prForTask(ledgers.prs, state.task_id);
			const review = latestReviewForPr(ledgers.reviews, pr?.id);
			if (review?.decision === "reject") {
				return bookkeep(
					state,
					"SELECT_TASK",
					[
						...(state.task_id ? [{ type: "task_status" as const, task_id: state.task_id, status: "rejected", evidence: `review ${review.id} rejected` }] : []),
						{ type: "clear_pr" },
					],
					`review ${review?.id} rejected ${pr?.id}; close the task and select the next one`,
					pr?.id ?? "loop",
					{ pr_id: null },
				);
			}
			return bookkeep(
				state,
				"WORK",
				[
					...(state.task_id ? [{ type: "task_status" as const, task_id: state.task_id, status: "in_progress", evidence: `review ${review?.id} requested changes` }] : []),
					{ type: "bump_attempt" },
				],
				`review ${review?.id} requested changes on ${pr?.id}; send it back for another attempt`,
				pr?.id ?? "loop",
				{ pr_id: pr?.id ?? null },
			);
		},
	},
	{
		id: "review-accept",
		when: (state, ledgers) => {
			if (state.state !== "REVIEW") return false;
			const pr = state.pr_id ? ledgers.prs.find((p) => p.id === state.pr_id) ?? prForTask(ledgers.prs, state.task_id) : prForTask(ledgers.prs, state.task_id);
			const review = latestReviewForPr(ledgers.reviews, pr?.id);
			return pr !== null && review?.decision === "accept";
		},
		action: (state, ledgers) => {
			const pr = (state.pr_id ? ledgers.prs.find((p) => p.id === state.pr_id) : undefined) ?? prForTask(ledgers.prs, state.task_id);
			return bookkeep(state, "MERGE", [], `${pr?.id} accepted; run the pre-merge gate and merge`, pr?.id ?? "loop", { pr_id: pr?.id ?? null });
		},
	},
	{
		id: "review-no-pr",
		when: (state, ledgers) => state.state === "REVIEW" && !prForTask(ledgers.prs, state.task_id),
		action: (state) => bookkeep(state, "WORK", [{ type: "bump_attempt" }], "no PR exists for the current task; return to WORK"),
	},
	{
		id: "review-spawn",
		when: (state) => state.state === "REVIEW",
		action: (state, ledgers) => {
			const pr = prForTask(ledgers.prs, state.task_id);
			return spawn(
				state,
				`review:${pr?.id ?? state.task_id ?? "loop"}:${state.attempt + 1}`,
				"supervisor",
				`review ${pr?.id ?? "PR"} (attempt ${state.attempt + 1})`,
				reviewBrief(pr, state.attempt + 1),
				[`.pi/state/prs/${pr?.id ?? "PR-###"}.md`],
			);
		},
	},
	{
		id: "merge-no-review",
		when: (state, ledgers) => {
			if (state.state !== "MERGE") return false;
			const pr = state.pr_id ? ledgers.prs.find((p) => p.id === state.pr_id) : prForTask(ledgers.prs, state.task_id);
			return latestReviewForPr(ledgers.reviews, pr?.id)?.decision !== "accept";
		},
		action: (state) =>
			bookkeep(state, "REVIEW", [], "no accepted review exists for the PR; return to REVIEW", state.pr_id ?? "loop", {
				pr_id: state.pr_id ?? null,
			}),
	},
	{
		id: "merge-waiting-on-gate",
		when: (state, ledgers) =>
			state.state === "MERGE" && pendingGate(ledgers.gates, "merge", state.pr_id) !== null,
		action: (state, ledgers) => {
			const gate = pendingGate(ledgers.gates, "merge", state.pr_id);
			return none(state, `waiting for human gate ${gate?.id} (merge): ${gate?.note ?? ""}`, gate?.id);
		},
	},
	{
		id: "merge-gate-rejected",
		when: (state, ledgers) =>
			state.state === "MERGE" && decidedGate(ledgers.gates, "merge", state.pr_id)?.status === "rejected",
		action: (state, ledgers) => {
			const gate = decidedGate(ledgers.gates, "merge", state.pr_id);
			return bookkeep(
				state,
				"WORK",
				[{ type: "bump_attempt" }],
				`human gate ${gate?.id} rejected the merge: ${gate?.note ?? "no reason recorded"}`,
				state.pr_id ?? "loop",
			);
		},
	},
	{
		id: "merge-open-gate",
		when: (state, ledgers, options) =>
			state.state === "MERGE" && options.requireHumanMerge && decidedGate(ledgers.gates, "merge", state.pr_id) === null,
		action: (state, ledgers) => {
			const pr = state.pr_id ? ledgers.prs.find((p) => p.id === state.pr_id) : null;
			return {
				...baseAction(state, "gate", `gate:merge:${state.pr_id ?? "loop"}`, `open the merge gate for ${state.pr_id ?? "the PR"}`),
				kind: "gate",
				gate: "merge" as HumanGateKind,
				note: `merge ${state.pr_id ?? "the PR"}${pr ? ` (${pr.branch} → update_pymc)` : ""}: approve to let the conductor merge`,
			};
		},
	},
	{
		id: "merge-execute",
		when: (state, ledgers) => {
			if (state.state !== "MERGE") return false;
			const accepted = ["accept"].includes(
				latestReviewForPr(
					ledgers.reviews,
					(state.pr_id ? ledgers.prs.find((p) => p.id === state.pr_id)?.id : undefined) ?? prForTask(ledgers.prs, state.task_id)?.id,
				)?.decision ?? "",
			);
			return accepted;
		},
		action: (state, ledgers) => {
			const pr = state.pr_id ? ledgers.prs.find((p) => p.id === state.pr_id) : null;
			const review = latestReviewForPr(ledgers.reviews, pr?.id);
			return {
				...baseAction(state, "merge", `merge:${pr?.id ?? "loop"}:${review?.id ?? "review"}`, `merge ${pr?.id} (review ${review?.id})`),
				kind: "merge",
				pr_id: pr?.id ?? state.pr_id ?? "",
				review_id: review?.id ?? "",
				to: pr?.kind === "paper" ? "AUDIT" : "POST_MERGE",
			};
		},
	},
	{
		id: "post-merge",
		when: (state) => state.state === "POST_MERGE",
		action: (state, ledgers) => {
			const pr = state.pr_id ? ledgers.prs.find((p) => p.id === state.pr_id) : null;
			const task = state.task_id ? ledgers.tasks.find((t) => t.id === state.task_id) : null;
			const effects: BookkeepEffect[] = [];
			if (task && task.status !== "merged") {
				effects.push({
					type: "task_status",
					task_id: task.id,
					status: "merged",
					evidence: `${pr?.id ?? "PR"} merged into update_pymc${pr?.head_sha ? ` at ${pr.head_sha}` : ""}`,
				});
			}
			effects.push({ type: "clear_pr" });
			return bookkeep(state, "SELECT_TASK", effects, "post-merge bookkeeping: close the task, then pick the next one", "loop");
		},
	},
	{
		id: "readiness-pass",
		when: (state, ledgers) => state.state === "READINESS" && ledgers.readiness?.pass === true,
		action: (state, ledgers) => bookkeep(state, "PAPER", [], `readiness rubric passed: ${ledgers.readiness?.reason ?? ""}`, "loop"),
	},
	{
		id: "readiness-fail",
		when: (state, ledgers) => state.state === "READINESS" && ledgers.readiness?.pass === false,
		action: (state, ledgers) =>
			bookkeep(state, "SELECT_TASK", [], `readiness rubric failed: ${ledgers.readiness?.reason ?? "no reason"}`, "loop"),
	},
	{
		id: "readiness-spawn",
		when: (state) => state.state === "READINESS",
		action: (state, ledgers) =>
			spawn(
				state,
				`readiness:${completions(ledgers.events, "readiness:") + 1}`,
				"supervisor",
				"evaluate the readiness rubric",
				readinessBrief(readinessCriteria(ledgers).blockers),
				[".pi/state/READINESS.json"],
			),
	},
	{
		id: "paper-review",
		when: (state, ledgers) => state.state === "PAPER" && ledgers.prs.some((pr) => pr.kind === "paper" && ["open", "review"].includes(pr.status)),
		action: (state, ledgers) => {
			const pr = ledgers.prs.find((candidate) => candidate.kind === "paper" && ["open", "review"].includes(candidate.status)) ?? null;
			return bookkeep(state, "PAPER_REVIEW", [], `paper PR ${pr?.id} is open; supervise it`, pr?.id ?? "loop", { pr_id: pr?.id ?? null });
		},
	},
	{
		id: "paper-spawn",
		when: (state) => state.state === "PAPER",
		action: (state, ledgers) => {
			const pr = ledgers.prs.find((candidate) => candidate.kind === "paper") ?? null;
			return spawn(state, `paper:${state.attempt + 1}`, "phd_candidate", `write/revise the paper (attempt ${state.attempt + 1})`, paperBrief(state, pr), ["paper/CLAIMS.json"]);
		},
	},
	{
		id: "paper-review-no-pr",
		when: (state, ledgers) =>
			state.state === "PAPER_REVIEW" && !ledgers.prs.some((pr) => pr.kind === "paper" && ["open", "review"].includes(pr.status)),
		action: (state) => bookkeep(state, "PAPER", [], "no paper PR exists; return to PAPER"),
	},
	{
		id: "paper-review-decision",
		when: (state, ledgers) => {
			if (state.state !== "PAPER_REVIEW") return false;
			const pr = ledgers.prs.find((candidate) => candidate.kind === "paper" && ["open", "review", "merged"].includes(candidate.status)) ?? null;
			return pr !== null && latestReviewForPr(ledgers.reviews, pr.id) !== null;
		},
		action: (state, ledgers) => {
			const pr = ledgers.prs.find((candidate) => candidate.kind === "paper") ?? null;
			const review = latestReviewForPr(ledgers.reviews, pr?.id);
			if (review?.decision === "accept") {
				return bookkeep(state, "MERGE", [], `paper review ${review.id} accepted; merge the paper PR`, pr?.id ?? "loop", { pr_id: pr?.id ?? null });
			}
			return bookkeep(state, "PAPER", [{ type: "bump_attempt" }], `paper review ${review?.id} wants changes; revise`, pr?.id ?? "loop", { pr_id: pr?.id ?? null });
		},
	},
	{
		id: "paper-review-spawn",
		when: (state) => state.state === "PAPER_REVIEW",
		action: (state, ledgers) => {
			const pr = ledgers.prs.find((candidate) => candidate.kind === "paper") ?? null;
			return spawn(
				state,
				`paper-review:${pr?.id ?? "loop"}:${state.attempt + 1}`,
				"supervisor",
				`review the paper (attempt ${state.attempt + 1})`,
				reviewBrief(pr, state.attempt + 1),
				[".pi/state/prs/" + (pr?.id ?? "PR-###") + ".md"],
			);
		},
	},
	{
		id: "audit-over-cap",
		when: (_state, ledgers, options) => (ledgers.audit?.cycle ?? 1) > options.budgets.maxAuditCycles,
		action: (state, ledgers, options) =>
			block(
				state,
				`audit cycle ${ledgers.audit?.cycle} exceeds maxAuditCycles=${options.budgets.maxAuditCycles} (plan/02_ARCHITECTURE.md §5.4: max 5 cycles, then human direction)`,
				"decide how to proceed after the audit-cycle cap (accept the current grade, narrow the audit, or stop the run)",
			),
	},
	{
		id: "audit-complete",
		when: (state, ledgers) => state.state === "AUDIT" && ledgers.audit?.complete === true,
		action: (state) => bookkeep(state, "DECIDE", [], "audit cycle complete; apply the decision rule", "loop"),
	},
	{
		id: "audit-spawn",
		when: (state) => state.state === "AUDIT",
		action: (state, ledgers) => {
			const audit = ledgers.audit ?? { cycle: ledgers.budget.auditCycles + 1, phase: "find", complete: false };
			return spawn(
				state,
				`audit:${audit.cycle}:${audit.phase}`,
				"audit_finder",
				`audit cycle ${audit.cycle}, phase ${audit.phase}`,
				auditBrief(audit.cycle, audit.phase),
				[".pi/state/AUDIT.json"],
			);
		},
	},
	{
		id: "decide-done-waiting",
		when: (state, ledgers) =>
			state.state === "DECIDE" && ledgers.decision?.done === true && pendingGate(ledgers.gates, "submission") !== null,
		action: (state, ledgers) => {
			const gate = pendingGate(ledgers.gates, "submission");
			return none(state, `waiting for human gate ${gate?.id} (submission): ${gate?.note ?? ""}`, gate?.id);
		},
	},
	{
		id: "decide-done-approved",
		when: (state, ledgers) =>
			state.state === "DECIDE" && ledgers.decision?.done === true && decidedGate(ledgers.gates, "submission")?.status === "approved",
		action: (state) => ({
			...baseAction(state, "done", "done:submission-approved", "human sign-off recorded; the run is DONE"),
			kind: "done",
			reason: "the submission gate was approved; the run is DONE",
		}),
	},
	{
		id: "decide-done-rejected",
		when: (state, ledgers) =>
			state.state === "DECIDE" && ledgers.decision?.done === true && decidedGate(ledgers.gates, "submission")?.status === "rejected",
		action: (state, ledgers) => {
			const gate = decidedGate(ledgers.gates, "submission");
			return bookkeep(state, "SELECT_TASK", [], `human gate ${gate?.id} rejected the submission: ${gate?.note ?? ""}`, "loop");
		},
	},
	{
		id: "decide-open-gate",
		when: (state, ledgers) =>
			state.state === "DECIDE" && ledgers.decision?.done === true && decidedGate(ledgers.gates, "submission") === null,
		action: (state, ledgers) => ({
			...baseAction(state, "gate", "gate:submission", "open the DONE sign-off gate"),
			kind: "gate",
			gate: "submission" as HumanGateKind,
			note: `the audit decision rule says DONE (${ledgers.decision?.grade ?? "grade unknown"}); approve to finish the run`,
		}),
	},
	{
		id: "decide-more-work",
		when: (state, ledgers) => state.state === "DECIDE" && ledgers.decision?.done === false,
		action: (state, ledgers) =>
			bookkeep(state, "SELECT_TASK", [], `decision rule requires more work (${ledgers.decision?.grade ?? "grade unknown"}): ${ledgers.decision?.reason ?? ""}`, "loop"),
	},
	{
		id: "decide-spawn",
		when: (state) => state.state === "DECIDE",
		action: (state, ledgers) =>
			spawn(
				state,
				`decide:${completions(ledgers.events, "decide:") + 1}`,
				"supervisor",
				"apply the audit decision rule",
				decideBrief(),
				[".pi/state/DECISION.json"],
			),
	},
	{
		id: "fallback",
		when: () => true,
		action: (state) => none(state, `no rule matched state ${state.state}; this is a bug in the state machine — report it with the current LOOP_STATE.json`),
	},
];

/** The pure transition function. First matching rule wins. */
export function nextAction(state: LoopState, ledgers: LedgerView, options: LoopOptions): Action {
	for (const rule of RULES) {
		if (rule.when(state, ledgers, options)) return rule.action(state, ledgers, options);
	}
	/* istanbul ignore next — the fallback rule matches unconditionally. */
	throw new Error("state machine has no fallback rule (see RULES)");
}

/** Stable one-line rendering used by `status`, `--json`, and the event summary. */
export function describeAction(action: Action): string {
	const parts = [`${action.kind}`, `state=${action.state}`, `key=${action.key}`];
	if (action.kind === "spawn") parts.push(`agent=${action.agent}`, `attempt=${action.attempt ?? 1}`);
	if (action.kind === "bookkeep") parts.push(`to=${action.to}`);
	if (action.kind === "merge") parts.push(`pr=${action.pr_id}`);
	if (action.kind === "gate") parts.push(`gate=${action.gate}`);
	if (action.kind === "hpc") parts.push(`op=${action.op}`, `experiment=${action.experiment_id}`);
	return parts.join(" ");
}

export function resumeHint(action: Action): string {
	return `next: ${describeAction(action)} — ${action.summary}`;
}
