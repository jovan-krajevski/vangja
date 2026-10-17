/**
 * ledger — append-only state tools.
 *
 * Registers the tools from plan/05_EXTENSIONS_AND_TOOLS.md §3. All behaviour
 * lives in `./handlers.ts` (pi-free, tested with `node`); this file declares the
 * model-facing parameter schemas and forwards calls.
 *
 * Tools are sequential: they share the state directory and the lock.
 */

import { Type } from "typebox";
import { defineTool, type ExtensionAPI, type ExtensionToolContext } from "@earendil-works/pi-coding-agent";
import { resolveRepoRoot } from "../../lib/paths.ts";
import { parseRole, type LoopRole } from "../../lib/roles.ts";
import {
	failText,
	handleAmendmentRecord,
	handleDecisionRecord,
	handleExperimentQueue,
	handleExperimentUpdate,
	handleLedgerAppend,
	handleLedgerRead,
	handlePrOpen,
	handleReviewWrite,
	handleStateGet,
	handleStateSet,
	handleTaskClose,
	handleTaskEnqueue,
	handleWaiverRecord,
	type LedgerContext,
	type StateSetArgs,
	type ToolTextResult,
} from "./handlers.ts";

const JSON_OBJECT = Type.Record(Type.String(), Type.Unknown());

function contextFor(ctx: ExtensionToolContext): LedgerContext {
	const root = resolveRepoRoot(process.env.PI_LOOP_ROOT ?? ctx.cwd);
	const role: LoopRole = parseRole(process.env.PI_LOOP_ROLE);
	return { root, role, actor: process.env.PI_AGENT_NAME ?? `agent:${role}` };
}

function guard<T>(
	fn: (params: T, ctx: LedgerContext) => ToolTextResult,
): (params: T, ctx: LedgerContext) => ToolTextResult {
	return (params, ctx) => {
		try {
			return fn(params, ctx);
		} catch (error) {
			return failText((error as Error).message);
		}
	};
}

export default function (pi: ExtensionAPI) {
	pi.registerTool(
		defineTool({
			name: "ledger_read",
			label: "Ledger read",
			description:
				"Read append-only loop ledger records (bounded). Kinds: event, task, pr, review, experiment, finding, decision, waiver, amendment, budget, gate.",
			parameters: Type.Object({
				kind: Type.String({ description: "Ledger kind" }),
				where: Type.Optional(JSON_OBJECT),
				limit: Type.Optional(Type.Number({ description: "Max records (default 50, max 500)" })),
				tail: Type.Optional(Type.Boolean({ description: "Return the last N instead of the first N" })),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleLedgerRead)(
					params as { kind: string; where?: Record<string, unknown>; limit?: number; tail?: boolean },
					contextFor(ctx),
				);
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "ledger_append",
			label: "Ledger append",
			description:
				"Append one validated, locked, fsynced record to a ledger. Records are never rewritten; an update repeats the id.",
			parameters: Type.Object({ kind: Type.String(), record: JSON_OBJECT }),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleLedgerAppend)(
					params as { kind: string; record: Record<string, unknown> },
					contextFor(ctx),
				);
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "task_enqueue",
			label: "Task enqueue",
			description:
				"Enqueue a task brief (plan/03_AGENTS.md §3). Rejected without acceptance_criteria, deliverables, or a known origin.",
			parameters: Type.Object({ brief: JSON_OBJECT }),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleTaskEnqueue)(params as { brief: Record<string, unknown> }, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "task_close",
			label: "Task close",
			description: "Close a task with a status and the evidence that justifies it. Conductor/supervisor only.",
			parameters: Type.Object({
				task_id: Type.String(),
				status: Type.String({ description: "merged | closed | rejected | blocked" }),
				evidence: Type.String({ description: "Verifying command and observed output" }),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleTaskClose)(params as { task_id: string; status: string; evidence: string }, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "pr_open",
			label: "PR open",
			description:
				"Record a PR for a task: task_id, head_sha, base_sha, branch, summary, artifacts, tests, claims, hpc_jobs.",
			parameters: Type.Object({ footer: JSON_OBJECT }),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handlePrOpen)(params as { footer: Record<string, unknown> }, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "review_write",
			label: "Review write",
			description:
				"Record a review decision (accept | changes_requested | reject) with rubric scores, evidence, required changes, and residual risk. Supervisor/conductor only.",
			parameters: Type.Object({ record: JSON_OBJECT }),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleReviewWrite)(params as { record: Record<string, unknown> }, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "experiment_queue",
			label: "Experiment queue",
			description:
				"Queue an HPC/local experiment (plan/06 §4). Refused while any job is active, and without checkpoint metadata plus a local resume smoke test.",
			parameters: Type.Object({ record: JSON_OBJECT }),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleExperimentQueue)(params as { record: Record<string, unknown> }, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "experiment_update",
			label: "Experiment update",
			description:
				"Append an experiment status transition with raw evidence (or a metadata-only update when status is " +
				"omitted: progress, job_id, lastState, exitCode, …). Only legal lifecycle transitions are accepted. " +
				"Collector/conductor only.",
			parameters: Type.Object({
				experiment_id: Type.String(),
				status: Type.Optional(Type.String({ description: "Omit for a metadata-only update" })),
				evidence: Type.String({ description: "Raw squeue/sacct/exit-code evidence" }),
				fields: Type.Optional(JSON_OBJECT),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleExperimentUpdate)(
					params as { experiment_id: string; status?: string; evidence: string; fields?: Record<string, unknown> },
					contextFor(ctx),
				);
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "decision_record",
			label: "Decision record",
			description: "Record an ADR-style decision: title, context, decision, alternatives, consequences.",
			parameters: Type.Object({ adr: JSON_OBJECT }),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleDecisionRecord)(params as { adr: Record<string, unknown> }, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "waiver_record",
			label: "Waiver record",
			description: "Record a waiver for a failed gate, naming the gate and why it does not invalidate the merge. Supervisor/conductor only.",
			parameters: Type.Object({ waiver: JSON_OBJECT }),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleWaiverRecord)(params as { waiver: Record<string, unknown> }, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "amendment_record",
			label: "Amendment record",
			description:
				"Record a protocol amendment (path, change, justification). Opens a human gate. Supervisor/conductor only.",
			parameters: Type.Object({ amendment: JSON_OBJECT }),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleAmendmentRecord)(params as { amendment: Record<string, unknown> }, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "state_get",
			label: "State get",
			description: "Read LOOP_STATE.json plus the STOP flag and derived views (CONTEXT.md, TASKS.md, EXPERIMENTS.md).",
			parameters: Type.Object({}),
			executionMode: "sequential",
			async execute(_id, _params, _signal, _update, ctx) {
				return guard(handleStateGet)({}, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "state_set",
			label: "State set",
			description:
				"Set the loop state (validated against plan/02_ARCHITECTURE.md §2). Conductor only; the conductor is not an LLM session.",
			parameters: Type.Object({
				state: Type.String(),
				run_id: Type.Optional(Type.String()),
				task_id: Type.Optional(Type.String()),
				pr_id: Type.Optional(Type.String()),
				experiment_id: Type.Optional(Type.String()),
				attempt: Type.Optional(Type.Number()),
				next_gate: Type.Optional(Type.String()),
				reason: Type.Optional(Type.String()),
				force: Type.Optional(Type.Boolean()),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleStateSet)(params as StateSetArgs, contextFor(ctx));
			},
		}),
	);
}
