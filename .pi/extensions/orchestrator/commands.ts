/**
 * Command layer shared by the headless CLI and the `/loop:*` pi commands, so
 * the two entry points cannot diverge (plan/05_EXTENSIONS_AND_TOOLS.md §2).
 *
 * Every function takes a resolved `RunContext`; nothing here reads argv or
 * environment state on its own.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { appendRecord } from "../ledger/store.ts";
import { loadBudgetConfig, loadProjectConfig } from "../../lib/config.ts";
import { loadLoopState, saveLoopState, type LoopState } from "../../lib/state.ts";
import { refreshViews, composeStatus, renderStatus, renderStatusLine, type StatusReport } from "./context.ts";
import { executeAction, type ExecutionResult } from "./execute.ts";
import { approveGate, listGates, rejectGate, runCommand, hpcGate, type HpcGateResult } from "./gates.ts";
import { loadLedgerView } from "./ledgers.ts";
import { nextAction, type Action, type LedgerView, type LoopOptions } from "./state-machine.ts";
import type { RunContext } from "./sandbox.ts";

export function loadLoopOptions(context: RunContext): LoopOptions {
	const project = loadProjectConfig(context.root);
	return {
		requireHumanMerge: project.requireHumanMerge,
		budgets: loadBudgetConfig(context.root),
		sandbox: context.sandbox ?? undefined,
	};
}

export interface StepDeps {
	context: RunContext;
	options?: LoopOptions;
	timeoutMs?: number;
}

/** Compute the next action and perform it (exactly one). */
export async function commandStep(deps: StepDeps): Promise<ExecutionResult> {
	const options = deps.options ?? loadLoopOptions(deps.context);
	const state = loadLoopState(deps.context.root).state;
	const ledgers: LedgerView = loadLedgerView(deps.context.root);
	const action = nextAction(state, ledgers, options);
	return executeAction({ context: deps.context, action, options, timeoutMs: deps.timeoutMs });
}

export interface StartResult {
	steps: Array<{ action: Action; result: ExecutionResult }>;
	stopped: string;
}

export interface StartDeps extends StepDeps {
	maxSteps: number;
	forever: boolean;
	onStep?: (line: string) => void;
}

/** Step until the loop stops making progress, pauses, blocks, or hits the cap. */
export async function commandStart(deps: StartDeps): Promise<StartResult> {
	const steps: Array<{ action: Action; result: ExecutionResult }> = [];
	let stopped = `step cap reached (${deps.maxSteps})`;
	for (let i = 0; i < deps.maxSteps; i += 1) {
		const result = await commandStep(deps);
		steps.push({ action: result.action, result });
		deps.onStep?.(`[${i + 1}] ${result.action.kind} ${result.action.state} — ${result.message}`);
		if (result.skipped === "no-action") {
			stopped = `no action: ${result.action.summary}`;
			break;
		}
		if (!result.performed && result.skipped !== "dry-run") {
			stopped = `step not performed (${result.skipped}): ${result.message}`;
			break;
		}
		const state = loadLoopState(deps.context.root).state.state;
		if (["PAUSED", "BLOCKED", "DONE"].includes(state)) {
			stopped = `state ${state}`;
			break;
		}
	}
	return { steps, stopped };
}

/** `pause`: finish nothing, schedule nothing; HPC jobs keep running remotely. */
export function commandPause(context: RunContext, reason: string): { state: LoopState; message: string } {
	const loaded = loadLoopState(context.root).state;
	if (loaded.state === "PAUSED") return { state: loaded, message: `already PAUSED: ${loaded.reason ?? ""}` };
	const next = saveLoopState(context.root, { state: "PAUSED", reason }, { actor: "conductor" });
	appendRecord(
		context.root,
		"event",
		{ summary: `paused by human: ${reason}`, from: loaded.state, to: "PAUSED", reason },
		{ root: context.root, role: "conductor", actor: "human" },
	);
	refreshViews(context.root);
	return { state: next, message: `paused from ${loaded.state}: ${reason}` };
}

/** `resume`: return to the recorded state; completed steps are never repeated. */
export function commandResume(context: RunContext): { state: LoopState; action: Action; message: string } {
	const loaded = loadLoopState(context.root).state;
	if (loaded.state !== "PAUSED") {
		return {
			state: loaded,
			action: nextAction(loaded, loadLedgerView(context.root), loadLoopOptions(context)),
			message: `loop is not PAUSED (state ${loaded.state}); nothing to resume`,
		};
	}
	const target = loaded.paused_from;
	if (!target) {
		throw new Error(
			"LOOP_STATE.json is PAUSED without paused_from; fix the state file (or re-run the transition that paused it) before resuming",
		);
	}
	const next = saveLoopState(context.root, { state: target, reason: `resumed to ${target}` }, { actor: "conductor" });
	appendRecord(
		context.root,
		"event",
		{ summary: `resumed from PAUSED to ${target}`, from: "PAUSED", to: target },
		{ root: context.root, role: "conductor", actor: "human" },
	);
	refreshViews(context.root);
	const action = nextAction(next, loadLedgerView(context.root), loadLoopOptions(context));
	return { state: next, action, message: `resumed to ${target}; next: ${action.kind} — ${action.summary}` };
}

export interface StatusOutput {
	report: StatusReport;
	lines: string[];
	line: string;
}

export function commandStatus(context: RunContext): StatusOutput {
	const { report } = composeStatus(context.root, loadLoopOptions(context));
	return { report, lines: renderStatus(report), line: renderStatusLine(report) };
}

export interface EnvCheck {
	name: string;
	ok: boolean;
	detail: string;
}

/** `/loop:verify-env`: the environment script plus the orchestrator's own checks. */
export function commandVerifyEnv(context: RunContext): { checks: EnvCheck[]; ok: boolean } {
	const checks: EnvCheck[] = [];
	const envScript = path.join(context.root, ".pi", "tests", "check_env.ts");
	if (envScript.startsWith(`${context.root}${path.sep}`) === false) {
		checks.push({ name: "environment script path", ok: false, detail: `${envScript} is outside ${context.root}` });
	} else if (!fs.existsSync(envScript)) {
		checks.push({ name: "environment script", ok: false, detail: `${envScript} does not exist; the sandbox is not a checkout` });
	} else {
		const result = runCommand(["node", envScript], context.root, 120_000);
		const lines = `${result.stdout}\n${result.stderr}`.trim().split("\n");
		const failed = lines.filter((line) => line.startsWith("FAIL"));
		checks.push({
			name: "environment checks",
			ok: result.status === 0 && failed.length === 0,
			detail:
				result.status === 0
					? `${lines.filter((line) => line.startsWith("PASS")).length} checks passed`
					: failed.join("; ") || `exit ${result.status}`,
		});
	}
	const hpc: HpcGateResult = hpcGate(context.root);
	checks.push({ name: "hpc gate", ok: hpc.status !== "blocked", detail: `${hpc.status}: ${hpc.reason}` });
	const project = loadProjectConfig(context.root);
	checks.push({
		name: "human merge gate",
		ok: true,
		detail: `requireHumanMerge=${project.requireHumanMerge} base=${project.baseBranch}`,
	});
	const pending = listGates(context.root).filter((gate) => gate.status === "pending");
	checks.push({ name: "open human gates", ok: true, detail: `${pending.length} pending` });
	return { checks, ok: checks.every((check) => check.ok) };
}

export interface GateCommandResult {
	ok: boolean;
	message: string;
	gates?: Array<Record<string, unknown>>;
}

export function commandGate(context: RunContext, args: string[]): GateCommandResult {
	const [action, id, ...rest] = args;
	if (!action || action === "list") {
		const gates = listGates(context.root);
		const pending = gates.filter((gate) => gate.status === "pending");
		return {
			ok: true,
			message:
				gates.length === 0
					? "no human gates recorded"
					: `${pending.length} pending of ${gates.length}: ` +
						pending.map((gate) => `${gate.id} ${gate.gate} — ${gate.note}`).join(" · "),
			gates: gates as unknown as Array<Record<string, unknown>>,
		};
	}
	if (action === "approve" || action === "reject") {
		if (!id) return { ok: false, message: `gate ${action} needs a gate id (see \`gate list\`)` };
		const note = rest.join(" ").trim();
		if (action === "approve" && note === "") {
			return { ok: false, message: "gate approve needs a note stating what you verified (the note is the audit trail)" };
		}
		if (action === "reject" && note === "") {
			return { ok: false, message: "gate reject needs a reason; the loop sends the note back with the task" };
		}
		try {
			const record = action === "approve" ? approveGate(context.root, id, note, "human") : rejectGate(context.root, id, note, "human");
			refreshViews(context.root);
			return { ok: true, message: `gate ${String(record.id)} (${String(record.gate)}) → ${String(record.status)}: ${note}` };
		} catch (error) {
			return { ok: false, message: (error as Error).message };
		}
	}
	return { ok: false, message: `unknown gate action "${action}"; use list|approve|reject` };
}

/** `--dry-run` convenience: the action `step` would perform, without side effects. */
export function peekAction(context: RunContext): Action {
	const state = loadLoopState(context.root).state;
	return nextAction(state, loadLedgerView(context.root), loadLoopOptions(context));
}
