/**
 * orchestrator — the conductor's pi-facing surface (S4).
 *
 * Registers:
 * - `/loop:start|step|pause|resume|status|verify-env|gate` — thin wrappers around
 *   the same `commands.ts` functions the headless CLI calls, so TUI and
 *   `.pi/loop/*.sh` cannot diverge;
 * - `merge_pr` — the supervisor's merge request; the conductor runs the
 *   pre-merge gate and performs the merge (see `merge-tool.ts`).
 *
 * No loop logic lives here: this file only maps pi's context to a `RunContext`
 * and formats the result for the TUI.
 */

import { defineTool, type ExtensionAPI, type ExtensionCommandContext, type ExtensionToolContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import * as path from "node:path";
import { parseRole } from "../../lib/roles.ts";
import { parseFlags } from "./sandbox.ts";
import { emitResult } from "./emit.ts";
import { contextFromEnvironment, handleMergePr } from "./merge-tool.ts";
import {
	commandGate,
	commandPause,
	commandResume,
	commandStart,
	commandStatus,
	commandStep,
	commandVerifyEnv,
} from "./commands.ts";

function runContextFor(cwd: string) {
	const envRoot = process.env.PI_LOOP_ROOT?.trim() ? process.env.PI_LOOP_ROOT : cwd;
	const realRoot = process.env.PI_LOOP_REAL_ROOT?.trim() ? process.env.PI_LOOP_REAL_ROOT : envRoot;
	return contextFromEnvironment(path.resolve(envRoot), path.resolve(realRoot), process.env.PI_LOOP_COMMAND);
}

/**
 * Show a command result (see emit.ts): TUI toast in interactive mode, plain
 * stdout otherwise, so `pi --print "/loop:verify-env"` is observable.
 */
function emit(ctx: ExtensionCommandContext, text: string, lines?: string[], kind: "info" | "error" = "info"): void {
	emitResult(ctx as never, text, lines, kind);
}

export default function (pi: ExtensionAPI) {
	pi.registerCommand("loop:start", {
		description: "Step the loop until it pauses, blocks, finishes, or hits the cap",
		handler: async (args: string, ctx: ExtensionCommandContext) => {
			const flags = parseFlags(args.split(/\s+/).filter((part) => part !== ""));
			const context = runContextFor(ctx.cwd);
			const result = await commandStart({
				context,
				maxSteps: flags.forever ? Math.max(flags.maxSteps, 1000) : flags.maxSteps,
				forever: flags.forever,
			});
			const lines = result.steps.map((step, index) => `[${index + 1}] ${step.result.message}`);
			lines.push(`loop stopped: ${result.stopped}`);
			emit(ctx, lines.join("\n"), lines.slice(-4));
		},
	});

	pi.registerCommand("loop:step", {
		description: "Perform exactly one loop action",
		handler: async (_args: string, ctx: ExtensionCommandContext) => {
			const context = runContextFor(ctx.cwd);
			const result = await commandStep({ context });
			emit(ctx, `${result.skipped ? `skipped(${result.skipped}) ` : ""}${result.message}`);
		},
	});

	pi.registerCommand("loop:pause", {
		description: "Write PAUSED with a reason; the loop stops scheduling (HPC jobs keep running)",
		handler: async (args: string, ctx: ExtensionCommandContext) => {
			const context = runContextFor(ctx.cwd);
			const result = commandPause(context, args.trim() || "paused by human (no reason given)");
			emit(ctx, result.message);
		},
	});

	pi.registerCommand("loop:resume", {
		description: "Return to the state recorded before the pause",
		handler: async (_args: string, ctx: ExtensionCommandContext) => {
			const context = runContextFor(ctx.cwd);
			const result = commandResume(context);
			emit(ctx, result.message);
		},
	});

	pi.registerCommand("loop:status", {
		description: "Show state, current task/PR/experiment, budget, gates, last events, and nextAction",
		handler: async (args: string, ctx: ExtensionCommandContext) => {
			const context = runContextFor(ctx.cwd);
			const output = commandStatus(context);
			if (args.trim() === "--json") {
				emit(ctx, JSON.stringify(output.report, null, 2));
				return;
			}
			emit(ctx, output.lines.join("\n"), output.lines.slice(-4));
		},
	});

	pi.registerCommand("loop:verify-env", {
		description: "Model pin, resource load, schemas, protected paths, tool collisions, drift",
		handler: async (_args: string, ctx: ExtensionCommandContext) => {
			const context = runContextFor(ctx.cwd);
			const { checks, ok } = commandVerifyEnv(context);
			const lines = checks.map((check) => `${check.ok ? "PASS" : "FAIL"}  ${check.name}: ${check.detail}`);
			lines.push(`${checks.filter((check) => check.ok).length}/${checks.length} orchestrator checks passed`);
			if (!ok) lines.push("loop:verify-env FAILED — see the lines above");
			emit(ctx, lines.join("\n"), lines.slice(-3), ok ? "info" : "error");
		},
	});

	pi.registerCommand("loop:gate", {
		description: "Human gates: list | approve <id> <note> | reject <id> <why>",
		handler: async (args: string, ctx: ExtensionCommandContext) => {
			const context = runContextFor(ctx.cwd);
			const result = commandGate(context, args.trim() === "" ? ["list"] : args.trim().split(/\s+/));
			emit(ctx, result.message, undefined, result.ok ? "info" : "error");
		},
	});

	pi.registerTool(
		defineTool({
			name: "merge_pr",
			label: "Merge PR (conductor)",
			description:
				"Supervisor → conductor merge request. Requires an accepted review. With requireHumanMerge=true the " +
				"conductor opens the merge human gate instead of merging; once approved the loop lands the PR with " +
				"`git merge --no-ff` and tags run/<run_id>/merge/<n>.",
			parameters: Type.Object({
				pr_id: Type.String({ description: "PR record id (PR-###)" }),
				task_id: Type.Optional(Type.String({ description: "Task id for bookkeeping" })),
				reason: Type.Optional(Type.String({ description: "Why the merge is recommended" })),
			}),
			outputSchema: Type.Object({
				stub: Type.Boolean(),
				merged: Type.Boolean(),
				pr_id: Type.String(),
				status: Type.Optional(Type.String()),
			}),
			executionMode: "sequential",
			async execute(_toolCallId, params, _signal, _update, ctx: ExtensionToolContext) {
				const root = path.resolve(process.env.PI_LOOP_ROOT ?? ctx.cwd);
				const role = parseRole(process.env.PI_LOOP_ROLE);
				const actor = process.env.PI_AGENT_NAME ?? `agent:${role}`;
				return handleMergePr(params as Record<string, unknown>, { root, role, actor });
			},
		}),
	);
}
