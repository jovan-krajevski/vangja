/**
 * Ledger tool handlers — the behaviour behind the `ledger_*`, `task_*`,
 * `pr_open`, `review_write`, `experiment_*`, `*_record`, and `state_*` tools.
 *
 * Storage and validation live in `./store.ts`; this module holds the tool
 * semantics (gates, transitions, role checks) and returns model-facing results
 * with a `details` object. No pi imports, so `node` tests exercise every path.
 */

import { VangjaLoopError } from "../../lib/errors.ts";
import { readJsonFile, writeFileAtomic } from "../../lib/io.ts";
import { ledgerPath, statePath } from "../../lib/paths.ts";
import { EXPERIMENT_TRANSITIONS, isLedgerKind, LEDGER_FILES, projectLatest } from "../../lib/records.ts";
import { renderDecisionsMd } from "../../lib/views.ts";
import { isLoopStateName, loadLoopState, saveLoopState, isStopFlagSet, type LoopStateName } from "../../lib/state.ts";
import {
	activeExperiments,
	appendRecord,
	checkCapability,
	fail,
	failText,
	ok,
	readLedger,
	renderContext,
	requireObject,
	requireString,
	summarise,
	type LedgerContext,
	type ToolTextResult,
} from "./store.ts";

export type { LedgerContext, ToolTextResult } from "./store.ts";
export { appendRecord, failText, readLedger, renderContext, summarise } from "./store.ts";

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

export interface LedgerReadArgs {
	kind: string;
	where?: Record<string, unknown>;
	limit?: number;
	tail?: boolean;
}

export function handleLedgerRead(args: LedgerReadArgs, ctx: LedgerContext): ToolTextResult {
	try {
		if (!isLedgerKind(args.kind)) {
			return fail(
				`unknown ledger kind "${args.kind}"; valid kinds: ${Object.keys(LEDGER_FILES).join(", ")}`,
				{ kind: args.kind },
			);
		}
		const file = ledgerPath(ctx.root, LEDGER_FILES[args.kind]);
		const all = readLedger(ctx.root, args.kind);
		let records = all;
		if (args.where && Object.keys(args.where).length > 0) {
			records = records.filter((record) =>
				Object.entries(args.where as Record<string, unknown>).every(([key, value]) => {
					if (Array.isArray(record[key])) return (record[key] as unknown[]).includes(value);
					return record[key] === value;
				}),
			);
		}
		const limit = Math.min(Math.max(args.limit ?? 50, 1), 500);
		const truncated = records.length > limit;
		const selected = args.tail ? records.slice(-limit) : records.slice(0, limit);
		const hint = truncated
			? `\n(returned ${selected.length} of ${records.length} matching records; full ledger: ${file} — ` +
				`use where/limit or read the file)`
			: "";
		return ok(`${selected.map(summarise).join("\n")}${hint}`, {
			kind: args.kind,
			path: file,
			total: all.length,
			matched: records.length,
			returned: selected.length,
			truncated,
			records: selected,
		});
	} catch (error) {
		return fail((error as Error).message, { kind: args.kind });
	}
}

export function handleLedgerAppend(args: { kind: string; record: Record<string, unknown> }, ctx: LedgerContext): ToolTextResult {
	try {
		if (!isLedgerKind(args.kind)) {
			return fail(`unknown ledger kind "${args.kind}"; valid kinds: ${Object.keys(LEDGER_FILES).join(", ")}`);
		}
		const record = requireObject(args as unknown as Record<string, unknown>, "record", "pass the JSON record to append");
		const appended = appendRecord(ctx.root, args.kind, record, ctx);
		return ok(`appended ${args.kind} record id=${String(appended.id ?? "-")} seq=${String(appended.seq)}`, {
			kind: args.kind,
			record: appended,
		});
	} catch (error) {
		return fail((error as Error).message, { kind: args.kind });
	}
}

const ORIGIN_PREFIXES = ["FINDING", "PLAN", "GAP", "PROTOCOL", "PAPER", "HUMAN", "REVIEW"];

function validateOrigin(root: string, origin: string): string | undefined {
	const prefix = origin.split(":")[0];
	if (ORIGIN_PREFIXES.includes(prefix)) return undefined;
	const findings = projectLatest(readLedger(root, "finding") as Array<{ id?: string; seq: number }>);
	if (findings.some((finding) => finding.id === origin)) return undefined;
	return (
		`origin "${origin}" is not a known finding id and does not use a known prefix ` +
		`(${ORIGIN_PREFIXES.join("|")}:<id>). A brief must be traceable to a finding, a REVIEW/PLAN item, ` +
		`a protocol gap, a paper-evidence gap, or a human request.`
	);
}

export function handleTaskEnqueue(args: { brief: Record<string, unknown> }, ctx: LedgerContext): ToolTextResult {
	try {
		checkCapability(ctx, "task_enqueue");
		const brief = requireObject(args as unknown as Record<string, unknown>, "brief", "see plan/03_AGENTS.md §3");
		if (!Array.isArray(brief.acceptance_criteria) || brief.acceptance_criteria.length === 0) {
			return fail(
				"brief rejected: acceptance_criteria must be a non-empty list; a brief without checkable criteria is not actionable",
				{ brief },
			);
		}
		if (!Array.isArray(brief.deliverables) || brief.deliverables.length === 0) {
			return fail("brief rejected: deliverables must be a non-empty list of paths", { brief });
		}
		const origin = typeof brief.origin === "string" ? brief.origin : "";
		if (origin === "") return fail("brief rejected: origin is required (finding id or <PREFIX>:<id>)", { brief });
		const originProblem = validateOrigin(ctx.root, origin);
		if (originProblem) return fail(`brief rejected: ${originProblem}`, { brief, origin });
		const record = appendRecord(ctx.root, "task", { ...brief, status: brief.status ?? "open" }, ctx);
		return ok(
			`task ${String(record.id)} enqueued: ${String(record.title)} (${String(record.priority)}, ${String(record.status)}). ` +
				`Write the brief to .pi/state/tasks/${String(record.id)}.md before WORK starts.`,
			{ kind: "task", record },
		);
	} catch (error) {
		return fail((error as Error).message, {});
	}
}

export function handleTaskClose(
	args: { task_id: string; status: string; evidence: string },
	ctx: LedgerContext,
): ToolTextResult {
	try {
		checkCapability(ctx, "task_close");
		const taskId = requireString(args as unknown as Record<string, unknown>, "task_id", "e.g. T-012");
		const status = requireString(args as unknown as Record<string, unknown>, "status", "merged|closed|rejected|blocked");
		const evidence = requireString(args as unknown as Record<string, unknown>, "evidence", "name the verifying command/output");
		const tasks = readLedger(ctx.root, "task");
		const latest = projectLatest(tasks as Array<{ id?: string; seq: number }>).find((task) => task.id === taskId);
		if (!latest) return fail(`unknown task ${taskId}; known tasks: ${tasks.map((t) => t.id).join(", ") || "none"}`);
		const record = appendRecord(
			ctx.root,
			"task",
			{ ...latest, id: taskId, status, close_evidence: evidence } as Record<string, unknown>,
			ctx,
		);
		return ok(`task ${taskId} → ${status}; evidence: ${evidence}`, { kind: "task", record });
	} catch (error) {
		return fail((error as Error).message, {});
	}
}

export function handlePrOpen(args: { footer: Record<string, unknown> }, ctx: LedgerContext): ToolTextResult {
	try {
		checkCapability(ctx, "pr_open");
		const footer = requireObject(args as unknown as Record<string, unknown>, "footer", "machine-readable PR footer");
		for (const field of ["task_id", "head_sha", "base_sha", "branch", "summary"]) {
			requireString(footer, field, `required in the PR record (plan/03_AGENTS.md §3)`);
		}
		// NOTE (S4): the clean-worktree check at the head SHA is enforced by the
		// orchestrator's `pr_open` gate; the ledger tool records what it was told.
		const record = appendRecord(ctx.root, "pr", { ...footer, status: footer.status ?? "open" }, ctx);
		return ok(
			`PR ${String(record.id)} recorded for ${String(record.task_id)} at ${String(record.head_sha)} ` +
				`(base ${String(record.base_sha)})`,
			{ kind: "pr", record },
		);
	} catch (error) {
		return fail((error as Error).message, {});
	}
}

export function handleReviewWrite(args: { record: Record<string, unknown> }, ctx: LedgerContext): ToolTextResult {
	try {
		checkCapability(ctx, "review_write");
		const record = requireObject(args as unknown as Record<string, unknown>, "record", "review record");
		const appended = appendRecord(ctx.root, "review", record, ctx);
		return ok(
			`review ${String(appended.id)} for PR ${String(appended.pr_id)}: ${String(appended.decision)}`,
			{ kind: "review", record: appended },
		);
	} catch (error) {
		return fail((error as Error).message, {});
	}
}

export function handleExperimentQueue(
	args: { record: Record<string, unknown> },
	ctx: LedgerContext,
): ToolTextResult {
	try {
		checkCapability(ctx, "experiment_queue");
		const record = requireObject(args as unknown as Record<string, unknown>, "record", "experiment record (plan/06 §4)");
		const active = activeExperiments(readLedger(ctx.root, "experiment"));
		if (active.length > 0) {
			return fail(
				`refused: experiment ${active.map((job) => `${String(job.id)}(${String(job.status)})`).join(", ")} is active. ` +
					`Exactly one job may be active at a time (plan/06_HPC_AND_COLLECTOR.md). Wait for the collector to reach a terminal state.`,
				{ active: active.map((job) => job.id) },
			);
		}
		if (typeof record.resumeSmokeTest !== "string" || record.resumeSmokeTest.trim() === "") {
			return fail(
				`refused: "resumeSmokeTest" is missing. Every job must carry local evidence that --resume skips completed units ` +
					`(plan/06_HPC_AND_COLLECTOR.md §5.4). Run 1–2 units, kill, rerun with --resume, and paste the observed output.`,
				{},
			);
		}
		const checkpoints = record.checkpoints;
		if (typeof checkpoints !== "object" || checkpoints === null || Array.isArray(checkpoints)) {
			return fail(
				`refused: "checkpoints" must be an object with dir, progressFile, resumeCommand and flushOnSignal ` +
					`(plan/06_HPC_AND_COLLECTOR.md §5). A job without a checkpoint block does not run.`,
				{},
			);
		}
		const checkpointRecord = checkpoints as Record<string, unknown>;
		for (const field of ["dir", "progressFile", "resumeCommand"]) {
			const value = checkpointRecord[field];
			if (typeof value !== "string" || value.trim() === "") {
				return fail(
					`refused: checkpoints.${field} is missing; the checkpoint contract requires dir, progressFile, ` +
						`resumeCommand and flushOnSignal=true (plan/06_HPC_AND_COLLECTOR.md §5).`,
					{ missing: `checkpoints.${field}` },
				);
			}
		}
		if (checkpointRecord.flushOnSignal !== true) {
			return fail(
				`refused: checkpoints.flushOnSignal must be true; the USR1 trap flushes the checkpoint and exits 99, ` +
					`which is what makes a requeue safe (plan/06_HPC_AND_COLLECTOR.md §3).`,
				{ missing: "checkpoints.flushOnSignal" },
			);
		}
		const appended = appendRecord(ctx.root, "experiment", { ...record, status: "queued", attempt: record.attempt ?? 1 }, ctx);
		return ok(
			`experiment ${String(appended.id)} queued (idempotency ${String(appended.idempotencyKey)}); ` +
				`single-slot: no further submission until it reaches a terminal state`,
			{ kind: "experiment", record: appended },
		);
	} catch (error) {
		return fail((error as Error).message, {});
	}
}

/** Fields the collector/conductor may attach to an experiment record. */
export const EXPERIMENT_UPDATE_FIELDS = [
	"job_id",
	"previous_job_id",
	"submit",
	"progress",
	"progressHistory",
	"lastState",
	"exitCode",
	"stderrTail",
	"missingOutputs",
	"checkpointsFetched",
	"checkpointListing",
	"collection",
	"memoryDoubled",
	"requeue",
] as const;

export interface ExperimentUpdateArgs {
	experiment_id: string;
	/** Omit for a metadata-only update; when present it must be a legal transition. */
	status?: string;
	evidence: string;
	/** Allowlisted machine-readable fields (progress, job_id, …). */
	fields?: Record<string, unknown>;
}

export function handleExperimentUpdate(args: ExperimentUpdateArgs, ctx: LedgerContext): ToolTextResult {
	try {
		checkCapability(ctx, "experiment_update");
		const experimentId = requireString(args as unknown as Record<string, unknown>, "experiment_id", "e.g. E-0007");
		const evidence = requireString(args as unknown as Record<string, unknown>, "evidence", "raw squeue/sacct/exit evidence");
		const fields = args.fields ?? {};
		for (const key of Object.keys(fields)) {
			if (!(EXPERIMENT_UPDATE_FIELDS as readonly string[]).includes(key)) {
				return fail(
					`field "${key}" is not updatable through experiment_update. Allowed: ${EXPERIMENT_UPDATE_FIELDS.join(", ")}. ` +
						`The experiment contract (commit_sha, command, checkpoints, expectedOutputs, idempotencyKey) is immutable; ` +
						`a change of contract needs a new record.`,
					{ key },
				);
			}
		}
		const experiments = readLedger(ctx.root, "experiment");
		const latest = projectLatest(experiments as Array<{ id?: string; seq: number }>).find(
			(experiment) => experiment.id === experimentId,
		);
		if (!latest) return fail(`unknown experiment ${experimentId}`);
		const from = String(latest.status);
		const status = typeof args.status === "string" && args.status.trim() !== "" ? args.status : from;
		if (status !== from) {
			const allowed = EXPERIMENT_TRANSITIONS[from] ?? [];
			if (!allowed.includes(status)) {
				return fail(
					`illegal experiment transition ${from} → ${status} for ${experimentId}; allowed: ${allowed.join(", ") || "none (terminal)"}. ` +
						`The lifecycle is in plan/06_HPC_AND_COLLECTOR.md §4 — do not invent a transition.`,
					{ from, status },
				);
			}
		}
		const record = appendRecord(
			ctx.root,
			"experiment",
			{
				...latest,
				...fields,
				id: experimentId,
				status,
				evidence: [...(Array.isArray(latest.evidence) ? latest.evidence : []), evidence],
			} as Record<string, unknown>,
			ctx,
		);
		return ok(
			status === from
				? `experiment ${experimentId}: metadata update (status stays ${from}); evidence recorded`
				: `experiment ${experimentId}: ${from} → ${status}; evidence recorded`,
			{ kind: "experiment", record },
		);
	} catch (error) {
		return fail((error as Error).message, {});
	}
}

export function handleDecisionRecord(args: { adr: Record<string, unknown> }, ctx: LedgerContext): ToolTextResult {
	try {
		const adr = requireObject(args as unknown as Record<string, unknown>, "adr", "ADR body (title/context/decision/alternatives/consequences)");
		const record = appendRecord(ctx.root, "decision", adr, ctx);
		writeFileAtomic(statePath(ctx.root, "DECISIONS.md"), renderDecisionsMd(readLedger(ctx.root, "decision")));
		return ok(`decision ${String(record.id)} recorded: ${String(record.title)}`, { kind: "decision", record });
	} catch (error) {
		return fail((error as Error).message, {});
	}
}

export function handleWaiverRecord(args: { waiver: Record<string, unknown> }, ctx: LedgerContext): ToolTextResult {
	try {
		checkCapability(ctx, "waiver_record");
		const waiver = requireObject(args as unknown as Record<string, unknown>, "waiver", "waiver record {gate, reason}");
		const record = appendRecord(ctx.root, "waiver", waiver, ctx);
		return ok(`waiver ${String(record.id)} recorded for gate "${String(record.gate)}"`, { kind: "waiver", record });
	} catch (error) {
		return fail((error as Error).message, {});
	}
}

export function handleAmendmentRecord(
	args: { amendment: Record<string, unknown> },
	ctx: LedgerContext,
): ToolTextResult {
	try {
		checkCapability(ctx, "amendment_record");
		const amendment = requireObject(args as unknown as Record<string, unknown>, "amendment", "amendment record {path, change, justification}");
		const record = appendRecord(ctx.root, "amendment", amendment, ctx);
		const gate = appendRecord(
			ctx.root,
			"gate",
			{
				gate: "enqueue",
				status: "pending",
				note: `protocol amendment ${String(record.id)} touches ${String(record.path)}: ${String(record.change)}`,
			},
			ctx,
		);
		return ok(
			`amendment ${String(record.id)} recorded for ${String(record.path)}; human gate ${String(gate.id)} opened. ` +
				`Set PI_LOOP_AMENDMENT=${String(record.id)} for the session that applies it.`,
			{ kind: "amendment", record, gate },
		);
	} catch (error) {
		return fail((error as Error).message, {});
	}
}

export function handleStateGet(_args: Record<string, unknown>, ctx: LedgerContext): ToolTextResult {
	try {
		const loaded = loadLoopState(ctx.root);
		const stopped = isStopFlagSet(ctx.root);
		return ok(JSON.stringify(loaded.state), {
			state: loaded.state,
			initialized: loaded.initialized,
			path: loaded.path,
			stop_flag: stopped,
		});
	} catch (error) {
		return fail((error as Error).message, {});
	}
}

export interface StateSetArgs {
	state: string;
	run_id?: string | null;
	task_id?: string | null;
	pr_id?: string | null;
	experiment_id?: string | null;
	attempt?: number;
	next_gate?: string | null;
	reason?: string | null;
	force?: boolean;
}

export function handleStateSet(args: StateSetArgs, ctx: LedgerContext): ToolTextResult {
	try {
		checkCapability(ctx, "state_set");
		if (!isLoopStateName(args.state)) {
			return fail(
				`unknown loop state "${args.state}"; valid states: see .pi/lib/state.ts LOOP_STATES ` +
					`(plan/02_ARCHITECTURE.md §2)`,
				{ state: args.state },
			);
		}
		const before = loadLoopState(ctx.root).state;
		const next = saveLoopState(
			ctx.root,
			{
				state: args.state as LoopStateName,
				...(args.run_id !== undefined ? { run_id: args.run_id } : {}),
				...(args.task_id !== undefined ? { task_id: args.task_id } : {}),
				...(args.pr_id !== undefined ? { pr_id: args.pr_id } : {}),
				...(args.experiment_id !== undefined ? { experiment_id: args.experiment_id } : {}),
				...(args.attempt !== undefined ? { attempt: args.attempt } : {}),
				...(args.next_gate !== undefined ? { next_gate: args.next_gate } : {}),
				...(args.reason !== undefined ? { reason: args.reason } : {}),
			},
			{ actor: ctx.actor, force: args.force === true },
		);
		if (before.state !== next.state) {
			appendRecord(
				ctx.root,
				"event",
				{ from: before.state, to: next.state, summary: args.reason ?? `state ${before.state} → ${next.state}` },
				ctx,
			);
		}
		writeFileAtomic(statePath(ctx.root, "CONTEXT.md"), renderContext(ctx.root));
		return ok(`state ${before.state} → ${next.state} (updated_by ${next.updated_by})`, { state: next });
	} catch (error) {
		return fail((error as Error).message, {});
	}
}

