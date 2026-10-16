/**
 * dispatch — isolated worker sessions for the research loop.
 *
 * Registers `dispatch_agent` (one worker), `dispatch_parallel` (≤4 concurrent),
 * and `dispatch_chain` (sequential, `{previous}` placeholder). The spawn engine
 * and its guarantees live in `./lib.ts`; this file declares the model-facing
 * schemas, enforces the per-role worker allowlist, and maps outcomes to the
 * `content` + `structuredContent` contract the conductor parses.
 */

import { defineTool, type ExtensionAPI, type ExtensionToolContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { resolveRepoRoot } from "../../lib/paths.ts";
import { parseRole, roleGate, type LoopRole } from "../../lib/roles.ts";
import { DISPATCH_CONCURRENCY } from "../../lib/agents.ts";
import {
	DEFAULT_OUTPUT_CAP_BYTES,
	isFailedOutcome,
	mapWithConcurrencyLimit,
	runDispatchAgent,
	runDispatchChain,
	visibleText,
	type DispatchOutcome,
	type DispatchRequest,
} from "./lib.ts";

const TaskSchema = Type.Object({
	agent: Type.String({ description: "Agent name from .pi/config/agents.json (e.g. scout, experimenter)" }),
	task: Type.String({ description: "Self-contained brief: objective, output format, boundaries" }),
	task_id: Type.Optional(Type.String({ description: "Task id for the session name/ledger (default: LOOP_STATE.task_id)" })),
	attempt: Type.Optional(Type.Number({ description: "Attempt number (default: 1)" })),
});

const UsageSchema = Type.Object({
	input: Type.Number(),
	output: Type.Number(),
	cacheRead: Type.Number(),
	cacheWrite: Type.Number(),
	cost: Type.Number(),
	contextTokens: Type.Number(),
	turns: Type.Number(),
});

const ResultSchema = Type.Object({
	agent: Type.String(),
	exit_state: Type.String(),
	exit_code: Type.Number(),
	stop_reason: Type.String(),
	session_path: Type.String(),
	truncated: Type.Boolean(),
	usage: UsageSchema,
});

const DispatchOutputSchema = Type.Object({
	tool: Type.String(),
	success_count: Type.Number(),
	results: Type.Array(ResultSchema),
});

const SINGLE_PARAMETERS = TaskSchema;
const PARALLEL_PARAMETERS = Type.Object({
	tasks: Type.Array(TaskSchema, {
		minItems: 1,
		maxItems: DISPATCH_CONCURRENCY,
		description: `Parallel workers, at most ${DISPATCH_CONCURRENCY} (plan/02_ARCHITECTURE.md §6)`,
	}),
});
const CHAIN_PARAMETERS = Type.Object({
	steps: Type.Array(TaskSchema, {
		minItems: 1,
		description: "Sequential workers; each task may use {previous} for the prior step's final text",
	}),
});

interface DispatchContext {
	root: string;
	role: LoopRole;
	parentAgent?: string;
}

function contextFor(ctx: ExtensionToolContext): DispatchContext {
	return {
		root: resolveRepoRoot(process.env.PI_LOOP_ROOT ?? ctx.cwd),
		role: parseRole(process.env.PI_LOOP_ROLE),
		parentAgent: process.env.PI_AGENT_NAME,
	};
}

function usageLine(outcome: DispatchOutcome): string {
	const usage = outcome.usage;
	const parts: string[] = [];
	if (usage.turns > 0) parts.push(`${usage.turns} turn${usage.turns === 1 ? "" : "s"}`);
	if (usage.input > 0) parts.push(`↑${usage.input}`);
	if (usage.output > 0) parts.push(`↓${usage.output}`);
	if (usage.cacheRead > 0) parts.push(`R${usage.cacheRead}`);
	if (usage.cacheWrite > 0) parts.push(`W${usage.cacheWrite}`);
	if (usage.cost > 0) parts.push(`$${usage.cost.toFixed(4)}`);
	return parts.join(" ");
}

function outcomeBlock(outcome: DispatchOutcome): string {
	const status =
		isFailedOutcome(outcome) && outcome.stop_reason
			? `failed (${outcome.stop_reason})`
			: outcome.exit_state;
	const lines: string[] = [`### [${outcome.agent}] ${status}`];
	if (isFailedOutcome(outcome)) {
		if (outcome.error_message) lines.push(`error: ${outcome.error_message}`);
		const tail = outcome.stderr.split("\n").slice(-12).join("\n").trim();
		if (tail !== "") lines.push(`stderr tail:\n${tail}`);
		// A failed/aborted child's partial text is never presented as the result;
		// the complete transcript stays in the session file.
		lines.push(
			`partial output (not a result) is in ${outcome.session_file ?? outcome.session_dir}; ` +
				`no final answer was produced.`,
		);
	} else {
		const text = visibleText(outcome, DEFAULT_OUTPUT_CAP_BYTES);
		if (text.trim() !== "") lines.push(text.trim());
	}
	const usage = usageLine(outcome);
	if (usage !== "") lines.push(`usage: ${usage}`);
	lines.push(`session: ${outcome.session_file ?? outcome.session_dir}`);
	return lines.join("\n");
}

function structured(outcome: DispatchOutcome): Record<string, unknown> {
	return {
		agent: outcome.agent,
		exit_state: outcome.exit_state,
		exit_code: outcome.exit_code,
		stop_reason: outcome.stop_reason ?? "",
		session_path: outcome.session_file ?? outcome.session_dir,
		truncated: outcome.truncated,
		usage: { ...outcome.usage },
	};
}

function toolResult(
	tool: string,
	outcomes: DispatchOutcome[],
	text: string,
	isError = false,
): {
	content: Array<{ type: "text"; text: string }>;
	details: Record<string, unknown>;
	structuredContent: Record<string, unknown>;
	isError?: boolean;
} {
	const successes = outcomes.filter((outcome) => !isFailedOutcome(outcome)).length;
	const structuredContent = {
		tool,
		success_count: successes,
		results: outcomes.map(structured),
	};
	return {
		content: [{ type: "text", text }],
		details: { tool, results: outcomes.map(structured) },
		structuredContent,
		...(isError ? { isError: true } : {}),
	};
}

function refusal(tool: string, message: string) {
	return toolResult(tool, [], `dispatch refused: ${message}`, true);
}

function runOptions(context: DispatchContext, extra: Record<string, unknown>) {
	return {
		root: context.root,
		parentRole: context.role,
		parentAgent: context.parentAgent,
		...extra,
	};
}

/** Map the parent's role to its worker allowlist before spawning anything. */
function checkCapability(role: LoopRole): void {
	const decision = roleGate("dispatch", role);
	if (!decision.allowed) {
		throw new Error(decision.reason ?? `role "${role}" may not dispatch`);
	}
}


export default function (pi: ExtensionAPI) {
	pi.registerTool(
		defineTool({
			name: "dispatch_agent",
			label: "Dispatch agent",
			description:
				"Spawn one worker agent in an isolated pi process (pinned model, its own session) and return its final " +
				"text, usage, and session path. The parent role's worker allowlist is enforced.",
			parameters: SINGLE_PARAMETERS,
			outputSchema: DispatchOutputSchema,
			async execute(_toolCallId, params, signal, onUpdate, ctx) {
				const context = contextFor(ctx);
				try {
					checkCapability(context.role);
					const outcome = await runDispatchAgent(params as DispatchRequest, {
						...runOptions(context, {}),
						signal,
						mode: "agent",
						onUpdate: onUpdate ? (text: string) => onUpdate({ content: [{ type: "text", text }], details: {} }) : undefined,
					});
					return toolResult("dispatch_agent", [outcome], outcomeBlock(outcome), isFailedOutcome(outcome));
				} catch (error) {
					return refusal("dispatch_agent", (error as Error).message);
				}
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "dispatch_parallel",
			label: "Dispatch parallel",
			description:
				`Run up to ${DISPATCH_CONCURRENCY} worker agents concurrently, each isolated, and return a summary per ` +
				"worker plus structured results.",
			parameters: PARALLEL_PARAMETERS,
			outputSchema: DispatchOutputSchema,
			async execute(_toolCallId, params, signal, _onUpdate, ctx) {
				const context = contextFor(ctx);
				try {
					checkCapability(context.role);
					const tasks = (params as { tasks: DispatchRequest[] }).tasks;
					const outcomes = await mapWithConcurrencyLimit(tasks, DISPATCH_CONCURRENCY, (task) =>
						runDispatchAgent(task, {
							...runOptions(context, {}),
							signal,
							mode: "parallel",
						}),
					);
					const successes = outcomes.filter((outcome) => !isFailedOutcome(outcome)).length;
					const text = `parallel: ${successes}/${outcomes.length} succeeded\n\n${outcomes.map(outcomeBlock).join("\n\n---\n\n")}`;
					return toolResult("dispatch_parallel", outcomes, text, successes !== outcomes.length);
				} catch (error) {
					return refusal("dispatch_parallel", (error as Error).message);
				}
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "dispatch_chain",
			label: "Dispatch chain",
			description:
				"Run workers sequentially; each task may embed {previous}, replaced by the prior step's final text. " +
				"The chain stops at the first failure and reports it.",
			parameters: CHAIN_PARAMETERS,
			outputSchema: DispatchOutputSchema,
			async execute(_toolCallId, params, signal, _onUpdate, ctx) {
				const context = contextFor(ctx);
				try {
					checkCapability(context.role);
					const steps = (params as { steps: DispatchRequest[] }).steps;
					const { outcomes, failedAt } = await runDispatchChain(steps, {
						...runOptions(context, {}),
						signal,
					});
					if (failedAt !== undefined) {
						const step = steps[failedAt - 1];
						const text = `chain stopped at step ${failedAt} (${step.agent}):\n\n${outcomeBlock(outcomes[failedAt - 1])}`;
						return toolResult("dispatch_chain", outcomes, text, true);
					}
					return toolResult("dispatch_chain", outcomes, outcomeBlock(outcomes[outcomes.length - 1]));
				} catch (error) {
					return refusal("dispatch_chain", (error as Error).message);
				}
			},
		}),
	);
}
