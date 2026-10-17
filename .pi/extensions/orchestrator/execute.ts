/**
 * `execute(action)` — performs exactly one loop action.
 *
 * Guarantees, all asserted by `.pi/tests/orchestrator/`:
 * - the child argv is fixed *before* the spawn and recorded in a `start` event
 *   together with the agent-file hash and the prompt hash;
 * - one completion event per action key; a replayed key is a no-op, and a killed
 *   child is detected (stale pid lock) and reported instead of silently doubled;
 * - no second child starts while one is alive (pid lock under `state/locks/`);
 * - state updates go through `saveLoopState` (temp + fsync + rename under a lock);
 * - `--dry-run` writes only `state/DRYRUN.jsonl`.
 *
 * Ordering rule: the side effect happens first, the completion event second. If
 * the process dies between them the state/ledger already reflects the effect and
 * the next `step` recomputes from the state — never a duplicate.
 */

import { spawn } from "node:child_process";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { appendRecord } from "../ledger/store.ts";
import { handleHpcRequeue, handleHpcSubmit } from "../hpc/handlers.ts";
import { emptyUsage, parseStreamLine, type DispatchUsage } from "../dispatch/lib.ts";
import { readJsonlFile, removeFileBestEffort, writeJsonAtomic } from "../../lib/io.ts";
import { VangjaLoopError } from "../../lib/errors.ts";
import { loadAgentsManifest, loadBudgetConfig, loadModelPin, loadProjectConfig, type AgentManifestEntry } from "../../lib/config.ts";
import { EXPERIMENT_TRANSITIONS, projectLatest } from "../../lib/records.ts";
import { assertNoUnknownTools, loadToolRegistry } from "../../lib/toolset.ts";
import { dispatchSessionName, sanitizeSegment } from "../../lib/agents.ts";
import { loadAgentDefinitions } from "../../lib/agents.ts";
import { loadLoopState, saveLoopState, type LoopState, type LoopStateName } from "../../lib/state.ts";
import { statePath } from "../../lib/paths.ts";
import { ledgerDigest, recordOutcome, recordSession } from "./budget.ts";
import {
	childLockPath,
	isCompleted,
	readChildLock,
} from "./ledgers.ts";
import {
	mergeNoFf,
	assertCleanCheckout,
	headSha,
} from "./git.ts";
import { runPreMergeGates, openGate, type GateRun } from "./gates.ts";
import { childEnvironment, logDryRun, type RunContext } from "./sandbox.ts";
import { refreshViews } from "./context.ts";
import type { Action, LoopOptions } from "./state-machine.ts";

export interface ChildToolCall {
	tool: string;
	ok: boolean;
	duration_ms: number | null;
}

export interface ChildOutcome {
	agent: string;
	role: string;
	exit_code: number;
	exit_state: "ok" | "failed" | "aborted";
	stop_reason?: string;
	error_message?: string;
	final_text: string;
	usage: DispatchUsage;
	tool_calls: ChildToolCall[];
	argv: string[];
	started_at: string;
	ended_at: string;
	duration_ms: number;
	session_dir: string;
	session_file: string | null;
	session_id: string | null;
	agent_file: string;
	agent_file_hash: string;
	prompt_hash: string;
}

export type ExecutionSkip =
	| "completed"
	| "dry-run"
	| "child-alive"
	| "no-action"
	| "waiting";

export interface ExecutionResult {
	performed: boolean;
	skipped: ExecutionSkip | null;
	action: Action;
	child?: ChildOutcome;
	gate_run?: GateRun;
	transition?: { from: LoopStateName; to: LoopStateName };
	message: string;
}

interface StreamState {
	sessionId: string | null;
	finalText: string;
	usage: DispatchUsage;
	model: string | null;
	stopReason?: string;
	errorMessage?: string;
	assistantMessages: number;
	toolCalls: ChildToolCall[];
	toolFailure: string | null;
}

function emptyStreamState(): StreamState {
	return { sessionId: null, finalText: "", usage: emptyUsage(), model: null, assistantMessages: 0, toolCalls: [], toolFailure: null };
}

/** Extend the shared dispatch parser with the tool-execution events. */
export function parseChildStreamLine(line: string, state: StreamState): void {
	const trimmed = line.trim();
	if (trimmed === "") return;
	parseStreamLine(trimmed, state as never);
	let event: Record<string, unknown>;
	try {
		event = JSON.parse(trimmed) as Record<string, unknown>;
	} catch {
		return;
	}
	if (event.type === "tool_execution_end" && typeof event.toolName === "string") {
		const failed = event.isError === true;
		state.toolCalls.push({
			tool: event.toolName,
			ok: !failed,
			duration_ms: typeof event.durationMs === "number" ? event.durationMs : null,
		});
		if (failed) state.toolFailure = `tool ${event.toolName} returned an error`;
	}
}

// ---------------------------------------------------------------------------
// Child invocation
// ---------------------------------------------------------------------------

/** Agents the conductor itself spawns (workers are dispatched by agents). */
export const CONDUCTOR_AGENTS: readonly string[] = [
	"phd_candidate",
	"collector",
	"supervisor",
	"audit_finder",
	"audit_verifier",
	"audit_synthesizer",
	"audit_adjudicator",
];

export interface PrimaryInvocation {
	command: string;
	args: string[];
	argv: string[];
	env: NodeJS.ProcessEnv;
	sessionDir: string;
	sessionName: string;
	agentFile: string;
	agentFileHash: string;
	promptHash: string;
	role: string;
	taskId: string;
	attempt: number;
	systemPromptFile: string;
	cleanup: () => void;
}

export function hashText(text: string): string {
	return crypto.createHash("sha256").update(text, "utf8").digest("hex");
}

function manifestEntry(root: string, agent: string): AgentManifestEntry {
	const manifest = loadAgentsManifest(root);
	const entry = manifest.agents.find((candidate) => candidate.name === agent);
	if (!entry) {
		throw new VangjaLoopError(
			"AGENT_NOT_FOUND",
			`the conductor tried to spawn "${agent}", which is not in .pi/config/agents.json. ` +
				`Roster: ${manifest.agents.map((candidate) => candidate.name).join(", ")}`,
		);
	}
	if (!CONDUCTOR_AGENTS.includes(agent)) {
		throw new VangjaLoopError(
			"ROLE_DENIED",
			`the conductor may not spawn "${agent}" directly; worker agents are dispatched by an agent ` +
				`(plan/02_ARCHITECTURE.md §1). Conductor-spawnable: ${CONDUCTOR_AGENTS.join(", ")}.`,
		);
	}
	return entry;
}

export function buildPrimaryInvocation(context: RunContext, action: Action): PrimaryInvocation {
	if (action.kind !== "spawn") throw new VangjaLoopError("STATE_INVALID", "buildPrimaryInvocation needs a spawn action");
	const root = context.root;
	const entry = manifestEntry(root, action.agent);
	const definition = loadAgentDefinitions(root).find((candidate) => candidate.name === action.agent);
	if (!definition) {
		throw new VangjaLoopError(
			"AGENT_NOT_FOUND",
			`roster names "${action.agent}" but .pi/agents/${action.agent}.md does not exist or does not parse`,
		);
	}
	const pin = loadModelPin(root);
	const pinnedId = `${pin.provider}/${pin.model}`;
	if (entry.model !== pinnedId || definition.model !== pinnedId) {
		throw new VangjaLoopError(
			"AGENT_INVALID",
			`agent "${action.agent}" pins model "${entry.model}" (frontmatter "${definition.model}") but ` +
				`.pi/config/model.json pins "${pinnedId}" (plan/02_ARCHITECTURE.md §7)`,
		);
	}
	const registry = loadToolRegistry(root);
	const tools = assertNoUnknownTools(registry, entry.tools, `agent "${action.agent}"`);

	const loaded = loadLoopState(root);
	const runId = loaded.state.run_id ?? "no-run";
	const entity = sanitizeSegment(action.task_id ?? action.pr_id ?? action.experiment_id ?? "loop");
	const attempt = action.attempt ?? 1;
	const sessionName = dispatchSessionName(action.agent, entity, attempt);
	const sessionDir = path.join(statePath(root, "sessions"), sanitizeSegment(runId), sanitizeSegment(action.agent), entity, `a${attempt}`);
	fs.mkdirSync(sessionDir, { recursive: true });

	const skills = entry.skills.map((skill) => path.join(root, ".pi", "skills", skill));
	const missingSkills = skills.filter((dir) => !fs.existsSync(dir));
	if (missingSkills.length > 0) {
		throw new VangjaLoopError(
			"AGENT_INVALID",
			`agent "${action.agent}" names skills that do not exist: ${missingSkills.join(", ")}. Skills live under .pi/skills/.`,
		);
	}

	const prompt = `TASK: ${action.brief}`;
	const promptHash = hashText(prompt);
	const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-conductor-"));
	const systemPromptFile = path.join(tmpDir, `${sanitizeSegment(action.agent)}.md`);
	fs.writeFileSync(systemPromptFile, `${definition.body}\n`, { encoding: "utf8", mode: 0o600 });

	const extensions = ["guards", "ledger", "dispatch", "orchestrator"].map((name) =>
		path.join(root, ".pi", "extensions", name, "index.ts"),
	);
	const env = childEnvironment(context, {
		PI_LOOP_ROLE: entry.role,
		PI_AGENT_NAME: sessionName,
		PI_LOOP_STATE: action.state,
		PI_LOOP_RUN: runId,
		PI_DISPATCH_DEPTH: "1",
		PI_DISPATCH_PARENT: "agent:conductor",
		...(action.task_id ? { PI_LOOP_BRIEF: `.pi/state/tasks/${action.task_id}.md` } : {}),
	});
	const args: string[] = [
		"--mode",
		"json",
		"-p",
		"-na",
		"--provider",
		pin.provider,
		"--model",
		pinnedId,
		"--thinking",
		pin.thinking,
	];
	for (const extension of extensions) args.push("-e", extension);
	for (const skill of skills) args.push("--skill", skill);
	args.push(
		"--append-system-prompt",
		systemPromptFile,
		"--tools",
		tools.join(","),
		"--session-dir",
		sessionDir,
		"--name",
		sessionName,
		prompt,
	);
	const argv = [context.command, ...args];
	return {
		command: context.command,
		args,
		argv,
		env,
		sessionDir,
		sessionName,
		agentFile: definition.file,
		agentFileHash: definition.sha256,
		promptHash,
		role: entry.role,
		taskId: action.task_id ?? entity,
		attempt,
		systemPromptFile,
		cleanup: () => {
			try {
				fs.rmSync(tmpDir, { recursive: true, force: true });
			} catch {
				// Temp cleanup is best effort; the run must not fail because of it.
			}
		},
	};
}

function findSessionFile(dir: string, sessionId: string | null): string | null {
	try {
		const files = fs
			.readdirSync(dir)
			.filter((file) => file.endsWith(".jsonl"))
			.map((file) => path.join(dir, file));
		if (files.length === 0) return null;
		if (sessionId) {
			const matching = files.find((file) => file.includes(sessionId));
			if (matching) return matching;
		}
		return files.map((file) => ({ file, mtime: fs.statSync(file).mtimeMs })).sort((a, b) => b.mtime - a.mtime)[0].file;
	} catch {
		return null;
	}
}

export interface SpawnChildOptions {
	timeoutMs: number;
	onUpdate?: (text: string) => void;
	/** Called with the child pid immediately after spawn (pid lock metadata). */
	onSpawn?: (pid: number | undefined) => void;
}

/** Spawn the child, parse its JSON stream, and return the outcome (never throws for a child that ran). */
export async function spawnChild(
	invocation: PrimaryInvocation,
	root: string,
	options: SpawnChildOptions,
): Promise<ChildOutcome> {
	const startedAt = new Date().toISOString();
	const startedMs = Date.now();
	const state = emptyStreamState();
	let stderr = "";
	let aborted = false;
	let timedOut = false;
	const result = await new Promise<{ code: number | null; spawnError?: string }>((resolve) => {
		const child = spawn(invocation.command, invocation.args, {
			cwd: root,
			env: invocation.env,
			shell: false,
			stdio: ["ignore", "pipe", "pipe"],
		});
		options.onSpawn?.(child.pid);
		let buffer = "";
		child.stdout.on("data", (data: Buffer) => {
			buffer += data.toString("utf8");
			const lines = buffer.split("\n");
			buffer = lines.pop() ?? "";
			for (const line of lines) {
				parseChildStreamLine(line, state);
				options.onUpdate?.(state.finalText || "(running...)");
			}
		});
		child.stderr.on("data", (data: Buffer) => {
			stderr += data.toString("utf8");
		});
		let settled = false;
		let killTimer: NodeJS.Timeout | undefined;
		let timeoutTimer: NodeJS.Timeout | undefined;
		const finish = (code: number | null, spawnError?: string): void => {
			if (settled) return;
			settled = true;
			if (buffer.trim() !== "") parseChildStreamLine(buffer, state);
			if (killTimer) clearTimeout(killTimer);
			if (timeoutTimer) clearTimeout(timeoutTimer);
			resolve({ code, spawnError });
		};
		child.on("close", (code) => finish(code));
		child.on("error", (error) => finish(null, (error as Error).message));
		const killProc = (): void => {
			child.kill("SIGTERM");
			killTimer = setTimeout(() => {
				if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
			}, 5000);
			killTimer.unref?.();
		};
		if (options.timeoutMs > 0) {
			timeoutTimer = setTimeout(() => {
				timedOut = true;
				killProc();
			}, options.timeoutMs);
		}
	});
	const exitCode = result.code ?? 1;
	const stopReason = state.stopReason;
	const failed = result.spawnError !== undefined || exitCode !== 0 || stopReason === "error" || stopReason === "aborted";
	const exitState: ChildOutcome["exit_state"] = aborted || stopReason === "aborted" ? "aborted" : failed || timedOut ? "failed" : "ok";
	const sessionFile = findSessionFile(invocation.sessionDir, state.sessionId);
	return {
		agent: invocation.sessionName.split("__")[0],
		role: invocation.role,
		slip: undefined as never,
		exit_code: exitCode,
		exit_state: exitState,
		stop_reason: stopReason,
		error_message: timedOut
			? `child timed out after ${options.timeoutMs}ms and was killed`
			: (result.spawnError ?? state.errorMessage ?? state.toolFailure ?? undefined),
		final_text: state.finalText,
		usage: state.usage,
		tool_calls: state.toolCalls,
		argv: invocation.argv,
		started_at: startedAt,
		ended_at: new Date().toISOString(),
		duration_ms: Math.max(0, Date.now() - startedMs),
		session_dir: invocation.sessionDir,
		session_file: sessionFile,
		session_id: state.sessionId,
		agent_file: invocation.agentFile,
		agent_file_hash: invocation.agentFileHash,
		prompt_hash: invocation.promptHash,
	};
}

// ---------------------------------------------------------------------------
// The executor
// ---------------------------------------------------------------------------

export interface ExecuteOptions {
	context: RunContext;
	action: Action;
	options: LoopOptions;
	/** Child wall-clock limit; defaults to the task budget cap. */
	timeoutMs?: number;
	actor?: string;
}

function firstLine(text: string, max = 160): string {
	const line = text.split("\n").find((candidate) => candidate.trim() !== "")?.trim() ?? "";
	return Buffer.byteLength(line) > max ? `${line.slice(0, max - 1)}…` : line;
}

function appendConductorEvent(
	context: RunContext,
	action: Action,
	fields: Record<string, unknown>,
	actor: string,
): Record<string, unknown> {
	return appendRecord(
		context.root,
		"event",
		{
			summary: action.summary,
			...fields,
			action_key: action.key,
			action_kind: action.kind,
			action_state: action.state,
		},
		{ root: context.root, role: "conductor", actor },
	);
}

function transitionState(
	context: RunContext,
	to: LoopStateName,
	actor: string,
	reason: string,
	extra: Partial<LoopState> = {},
): LoopState {
	return saveLoopState(context.root, { state: to, reason, ...extra }, { actor });
}

export function dryRun(context: RunContext, action: Action, detail: Record<string, unknown> = {}): void {
	logDryRun(context.root, { action_key: action.key, kind: action.kind, state: action.state, summary: action.summary, ...detail });
}

/**
 * True when the completion event for `action` describes an effect that is
 * actually visible in the current loop state. Every executor writes the effect
 * before the completion event, so this is the correct replay test:
 * a completed key whose state the loop has already left is a no-op, while a
 * completed key whose state the loop has returned to is new work.
 */
export function effectApplied(action: Action, currentState: string): boolean {
	switch (action.kind) {
		case "bookkeep":
			return currentState !== action.state;
		case "pause":
			return currentState === "PAUSED";
		case "block":
			return currentState === "BLOCKED";
		case "merge":
			return currentState !== "MERGE";
		case "boot":
			return currentState !== "UNINITIALIZED" && currentState !== "BOOT";
		case "gate":
		case "spawn":
			return true;
		case "hpc":
			// The effect lands in the experiment ledger, not in LOOP_STATE; a
			// completed key is a no-op and the submit marker/adoption path covers a
			// crash between the effect and the event.
			return true;
		default:
			return true;
	}
}

export async function executeAction(options: ExecuteOptions): Promise<ExecutionResult> {
	const { context, action } = options;
	const actor = options.actor ?? "conductor";
	if (context.dryRun) {
		dryRun(context, action, { agent: action.kind === "spawn" ? action.agent : undefined });
		return { performed: false, skipped: "dry-run", action, message: `dry-run: would ${action.kind} — ${action.summary}` };
	}
	if (action.kind === "none" || action.kind === "done") {
		return { performed: false, skipped: "no-action", action, message: action.kind === "done" ? "the run is already DONE" : action.summary };
	}
	// Idempotency: a completed key is never executed twice — unless the effect was
	// never applied, which for the state transitions is visible as "the loop is
	// still where the action started". A legitimate loop back into the same state
	// (a second pause after a resume, a second WORK cycle) must still run.
	const events = readJsonlFile<Record<string, unknown>>(statePath(context.root, "EVENTS.jsonl"));
	const view = events.map((event) => ({
		seq: Number(event.seq ?? 0),
		ts: String(event.ts ?? ""),
		actor: String(event.actor ?? ""),
		summary: String(event.summary ?? ""),
		action_key: typeof event.action_key === "string" ? event.action_key : undefined,
		action_phase: typeof event.action_phase === "string" ? event.action_phase : undefined,
	}));
	const currentState = loadLoopState(context.root).state.state;
	if (isCompleted(view, action.key) && effectApplied(action, currentState)) {
		return { performed: false, skipped: "completed", action, message: `action ${action.key} already completed; no-op` };
	}
	switch (action.kind) {
		case "boot":
			return executeBoot(context, action, actor);
		case "spawn":
			return executeSpawn(context, action, options);
		case "bookkeep":
			return executeBookkeep(context, action, actor);
		case "gate":
			return executeGate(context, action, actor);
		case "merge":
			return executeMerge(context, action, options);
		case "pause":
			return executePause(context, action, actor);
		case "block":
			return executeBlock(context, action, actor);
		case "hpc":
			return executeHpc(context, action, actor);
		default:
			throw new VangjaLoopError("STATE_INVALID", `executeAction cannot handle kind "${(action as Action).kind}"`);
	}
}

async function executeBoot(context: RunContext, action: Action, actor: string): Promise<ExecutionResult> {
	const root = context.root;
	// Environment verification: anything that cannot load must stop BOOT loudly.
	const pin = loadModelPin(root);
	const manifest = loadAgentsManifest(root);
	const budgets = loadBudgetConfig(root);
	const project = loadProjectConfig(root);
	assertCleanCheckout(root, "start a loop run (BOOT requires a clean checkout)");
	const baseSha = fs.existsSync(path.join(root, ".git")) ? headSha(root, project.baseBranch) : "no-git";
	const stamp = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
	const runId = `run-${stamp}-${baseSha.slice(0, 8)}`;
	for (const dir of ["sessions", "worktrees", "gate-runs", "locks"]) {
		fs.mkdirSync(statePath(root, dir), { recursive: true });
	}
	const loaded = loadLoopState(root);
	const from = loaded.state.state;
	saveLoopState(
		root,
		{ state: "BOOT", run_id: runId, reason: `BOOT: ${pin.provider}/${pin.model} thinking=${pin.thinking}; base ${project.baseBranch}@${baseSha.slice(0, 12)}` },
		{ actor, force: true },
	);
	appendConductorEvent(
		context,
		action,
		{
			action_phase: "complete",
			from,
			to: "SELECT_TASK",
			run_id: runId,
			base_sha: baseSha,
			model: `${pin.provider}/${pin.model}`,
			thinking: pin.thinking,
			agents: manifest.agents.length,
			budgets,
			summary: `BOOT: run ${runId} on base ${project.baseBranch}@${baseSha.slice(0, 12)} (${manifest.agents.length} agents, budget caps loaded)`,
		},
		actor,
	);
	if (action.kind === "boot") {
		const to = (action as { to: LoopStateName }).to;
		transitionState(context, to, actor, `BOOT complete: ${runId}`, { run_id: runId });
		refreshViews(root);
		return { performed: true, skipped: null, action, transition: { from, to }, message: `booted run ${runId} on base ${baseSha.slice(0, 12)}` };
	}
	return { performed: true, skipped: null, action, message: "booted" };
}

async function executeSpawn(context: RunContext, action: Action, options: ExecuteOptions): Promise<ExecutionResult> {
	const root = context.root;
	if (action.kind !== "spawn") throw new VangjaLoopError("STATE_INVALID", "executeSpawn needs a spawn action");
	// Refuse a second child while one is alive; report (and clear) a stale lock.
	const existingLock = readChildLock(root);
	if (existingLock?.alive) {
		return {
			performed: false,
			skipped: "child-alive",
			action,
			message:
				`refusing to start ${action.agent}: child pid ${existingLock.pid} is still alive ` +
				`(key ${existingLock.action_key}, started ${existingLock.started_at}). Wait for it or stop it deliberately.`,
		};
	}
	if (existingLock && !existingLock.alive) {
		appendConductorEvent(
			context,
			action,
			{
				action_phase: "stale",
				summary: `stale child lock (pid ${existingLock.pid}, key ${existingLock.action_key}, started ${existingLock.started_at}) — the process is dead; resuming ${action.key}`,
			},
			"conductor",
		);
		removeFileBestEffort(childLockPath(root));
	}
	const invocation = buildPrimaryInvocation(context, action);
	const budgetCaps = loadBudgetConfig(root);
	const timeoutMs = options.timeoutMs ?? budgetCaps.maxTaskWallClockMinutes * 60_000;
	const from = loadLoopState(root).state.state;
	appendConductorEvent(
		context,
		action,
		{
			action_phase: "start",
			agent: action.agent,
			role: invocation.role,
			attempt: invocation.attempt,
			session_dir: invocation.sessionDir,
			agent_file: invocation.agentFile,
			agent_file_hash: invocation.agentFileHash,
			prompt_hash: invocation.promptHash,
			argv: invocation.argv,
			evidence: (action as { evidence?: string[] }).evidence ?? [],
			summary: `start ${action.agent} (${action.key})`,
		},
		"conductor",
	);
	// Record the started attempt so a killed run resumes with a different key.
	saveLoopState(root, { attempt: invocation.attempt, reason: `${action.agent} running (${action.key})` }, { actor: "conductor" });
	// The lock is written only after the start event: a crash while acquiring it
	// can therefore never leave a lock that the start event does not explain
	// (the pause/resume test waits for this lock before killing the conductor).
	writeJsonAtomic(childLockPath(root), {
		pid: process.pid,
		action_key: action.key,
		agent: action.agent,
		argv: invocation.argv,
		started_at: new Date().toISOString(),
		note: "provisional: the child pid replaces this as soon as the process exists",
	});
	let child: ChildOutcome;
	try {
		child = await spawnChild(invocation, root, {
			timeoutMs,
			onSpawn: (pid) => {
				if (typeof pid === "number" && pid > 0) {
					writeJsonAtomic(childLockPath(root), {
						pid,
						action_key: action.key,
						agent: action.agent,
						argv: invocation.argv,
						started_at: new Date().toISOString(),
					});
				}
			},
		});
	} finally {
		invocation.cleanup();
		removeFileBestEffort(childLockPath(root));
	}
	const transcript = child.final_text;
	appendConductorEvent(
		context,
		action,
		{
			action_phase: "complete",
			agent: action.agent,
			role: invocation.role,
			attempt: invocation.attempt,
			exit_state: child.exit_state,
			exit_code: child.exit_code,
			stop_reason: child.stop_reason,
			error: child.error_message,
			usage: { ...child.usage },
			tool_calls: child.tool_calls,
			session_path: child.session_file ?? child.session_dir,
			session_dir: child.session_dir,
			session_id: child.session_id,
			agent_file: child.agent_file,
			agent_file_hash: child.agent_file_hash,
			prompt_hash: child.prompt_hash,
			argv: child.argv,
			duration_ms: child.duration_ms,
			from,
			to: from,
			summary: `${action.agent} ${child.exit_state}: ${firstLine(transcript) || child.error_message || "(no output)"}`,
		},
		"conductor",
	);
	// Budget: sessions/wall clock/cost/progress, then the failure streak.
	recordSession(
		root,
		{ usage: child.usage, durationMs: child.duration_ms, ledgerDigest: ledgerDigest(root), reason: `${action.agent} ${action.key}` },
		"conductor",
	);
	if (child.exit_state !== "ok") {
		recordOutcome(root, false, `${action.agent} ${action.key} failed (exit ${child.exit_code})`, "conductor");
	} else {
		recordOutcome(root, true, `${action.agent} ${action.key} ok`, "conductor");
	}
	refreshViews(root);
	return {
		performed: true,
		skipped: null,
		action,
		child,
		message:
			`${action.agent} ${child.exit_state} in ${(child.duration_ms / 1000).toFixed(1)}s ` +
			`(${child.tool_calls.length} tool calls, $${child.usage.cost.toFixed(4)})`,
	};
}

function applyEffects(context: RunContext, action: Action, actor: string): void {
	if (action.kind !== "bookkeep" && action.kind !== "block") return;
	for (const effect of action.effects ?? []) {
		switch (effect.type) {
			case "task_status": {
				const events = readJsonlFile<Record<string, unknown>>(statePath(context.root, "TASKS.jsonl"));
				const latest = events.filter((record) => record.id === effect.task_id).pop();
				if (!latest) {
					appendRecord(
						context.root,
						"task",
						{
							id: effect.task_id,
							title: `task ${effect.task_id}`,
							origin: "HUMAN:conductor",
							priority: "P2",
							objective: "recorded by the conductor before the brief was written",
							acceptance_criteria: ["the task brief exists"],
							deliverables: [`.pi/state/tasks/${effect.task_id}.md`],
							status: effect.status,
							close_evidence: effect.evidence,
						},
						{ root: context.root, role: "conductor", actor },
					);
					break;
				}
				appendRecord(
					context.root,
					"task",
					{ ...latest, id: effect.task_id, status: effect.status, close_evidence: effect.evidence },
					{ root: context.root, role: "conductor", actor },
				);
				break;
			}
			case "set_task":
				saveLoopState(context.root, { task_id: effect.task_id, pr_id: null, experiment_id: null, attempt: effect.attempt, next_gate: null }, { actor });
				break;
			case "set_experiment":
				saveLoopState(
					context.root,
					{ experiment_id: effect.experiment_id, attempt: effect.attempt, next_gate: null },
					{ actor },
				);
				break;
			case "bump_attempt": {
				const current = loadLoopState(context.root).state;
				saveLoopState(context.root, { attempt: current.attempt + 1 }, { actor });
				break;
			}
			case "reset_failures":
				// Budget reset is part of the effect so a new task starts with a clean streak.
				recordOutcome(context.root, true, `new work unit: ${effect.type}`, actor);
				break;
			case "clear_pr":
				saveLoopState(context.root, { pr_id: null, next_gate: null }, { actor });
				break;
			case "hpc_disabled": {
				const events = readJsonlFile<Record<string, unknown>>(statePath(context.root, "EVENTS.jsonl"));
				const already = events.some(
					(event) => event.experiment_id === effect.experiment_id && event.hpc_disabled === true,
				);
				if (already) break;
				appendRecord(
					context.root,
					"event",
					{
						summary: `hpc-disabled: ${effect.experiment_id} runs locally — ${effect.reason}`,
						experiment_id: effect.experiment_id,
						hpc_disabled: true,
					},
					{ root: context.root, role: "conductor", actor },
				);
				break;
			}
			case "local_fallback": {
				const latest = latestExperimentRecord(context.root, effect.experiment_id);
				if (!latest) break;
				const resources = (latest.resources ?? {}) as Record<string, unknown>;
				const status = String(latest.status ?? "");
				appendRecord(
					context.root,
					"experiment",
					{
						...latest,
						id: effect.experiment_id,
						status: status === "queued" ? "running" : status,
						resources: { ...resources, kind: "local" },
						evidence: [
							...(Array.isArray(latest.evidence) ? latest.evidence : []),
							`HPC disabled (${effect.reason}); the conductor marked this experiment as a local run (plan/06 §7)`,
						],
					},
					{ root: context.root, role: "conductor", actor },
				);
				break;
			}
			case "experiment_status": {
				const latest = latestExperimentRecord(context.root, effect.experiment_id);
				if (!latest) break;
				const from = String(latest.status ?? "");
				if (from === effect.status) break;
				const allowed = EXPERIMENT_TRANSITIONS[from] ?? [];
				if (!allowed.includes(effect.status)) {
					throw new VangjaLoopError(
						"STATE_INVALID",
						`conductor effect would make an illegal experiment transition ${from} → ${effect.status} for ${effect.experiment_id}`,
					);
				}
				appendRecord(
					context.root,
					"experiment",
					{
						...latest,
						id: effect.experiment_id,
						status: effect.status,
						evidence: [...(Array.isArray(latest.evidence) ? latest.evidence : []), effect.evidence],
					},
					{ root: context.root, role: "conductor", actor },
				);
				break;
			}
			case "enqueue_task": {
				const task = effect.task ?? {};
				appendRecord(context.root, "task", { ...task, status: task.status ?? "open" }, { root: context.root, role: "conductor", actor });
				break;
			}
		}
	}
}

function latestExperimentRecord(root: string, experimentId: string): Record<string, unknown> | null {
	const records = readJsonlFile<Record<string, unknown>>(statePath(root, "EXPERIMENTS.jsonl"));
	const latest = projectLatest(records as Array<{ id?: string; seq: number }>).find(
		(record) => record.id === experimentId,
	);
	return (latest as Record<string, unknown> | undefined) ?? null;
}

function executeBookkeep(context: RunContext, action: Action, actor: string): ExecutionResult {
	if (action.kind !== "bookkeep") throw new VangjaLoopError("STATE_INVALID", "executeBookkeep needs a bookkeep action");
	const from = loadLoopState(context.root).state.state;
	applyEffects(context, action, actor);
	const extra: Partial<LoopState> = { next_gate: null };
	const clearsPr = action.effects.some((effect) => effect.type === "clear_pr");
	if (action.pr_id && !clearsPr) extra.pr_id = action.pr_id;
	saveLoopState(context.root, { state: action.to, reason: action.reason, ...extra }, { actor });
	appendConductorEvent(
		context,
		action,
		{ action_phase: "complete", from, to: action.to, reason: action.reason, effects: action.effects },
		actor,
	);
	refreshViews(context.root);
	return {
		performed: true,
		skipped: null,
		action,
		transition: { from, to: action.to },
		message: `${from} → ${action.to}: ${action.reason}`,
	};
}

function executeGate(context: RunContext, action: Action, actor: string): ExecutionResult {
	if (action.kind !== "gate") throw new VangjaLoopError("STATE_INVALID", "executeGate needs a gate action");
	const loaded = loadLoopState(context.root).state;
	const gate = openGate(
		context.root,
		{
			gate: action.gate,
			note: action.note,
			pr_id: action.pr_id ?? undefined,
			task_id: action.task_id ?? undefined,
			run_id: loaded.run_id,
		},
		actor,
	);
	saveLoopState(context.root, { next_gate: action.gate, reason: `human gate ${String(gate.id)} opened: ${action.note}` }, { actor });
	appendConductorEvent(
		context,
		action,
		{ action_phase: "complete", gate: action.gate, gate_id: gate.id, note: action.note, to: loaded.state },
		actor,
	);
	refreshViews(context.root);
	return {
		performed: true,
		skipped: null,
		action,
		message: `human gate ${String(gate.id)} (${action.gate}) is pending: ${action.note}`,
	};
}

async function executeMerge(
	context: RunContext,
	action: Action,
	options: ExecuteOptions,
): Promise<ExecutionResult> {
	if (action.kind !== "merge") throw new VangjaLoopError("STATE_INVALID", "executeMerge needs a merge action");
	const root = context.root;
	const loaded = loadLoopState(root).state;
	const project = loadProjectConfig(root);
	const events = readJsonlFile<Record<string, unknown>>(statePath(root, "PRS.jsonl"));
	const prRecord = events.filter((record) => record.id === action.pr_id).pop();
	const reviews = readJsonlFile<Record<string, unknown>>(statePath(root, "REVIEWS.jsonl")).filter(
		(record) => record.pr_id === action.pr_id,
	);
	const review = reviews[reviews.length - 1];
	if (!prRecord || !review) {
		throw new VangjaLoopError(
			"STATE_INVALID",
			`merge ${action.pr_id} needs a PR record and a review; found pr=${Boolean(prRecord)} review=${Boolean(review)}`,
		);
	}
	if (!["accept", "changes_requested", "reject"].includes(String(review.decision)) || review.decision !== "accept") {
		throw new VangjaLoopError("STATE_INVALID", `merge ${action.pr_id} refused: review ${String(review.id)} decided "${String(review.decision)}"`);
	}
	const reviewActor = String(review.actor ?? "");
	const artifacts = Array.isArray(prRecord.artifacts) ? (prRecord.artifacts as string[]) : [];
	const protectedPaths = Array.isArray(prRecord.protected_paths) ? (prRecord.protected_paths as string[]) : [];
	const gateRun = runPreMergeGates({
		root,
		prId: action.pr_id,
		taskId: String(prRecord.task_id ?? ""),
		baseSha: String(prRecord.base_sha ?? project.baseBranch),
		headSha: String(prRecord.head_sha ?? ""),
		branch: String(prRecord.branch ?? ""),
		kind: String(prRecord.kind ?? "work"),
		state: action.state,
		artifacts,
		protectedPaths,
		reviewActor,
		runId: loaded.run_id,
		actor: "conductor",
	});
	if (!gateRun.ok) {
		const failed = gateRun.checks.filter((check) => check.status === "fail" || check.status === "error");
		const reason =
			`pre-merge gate RED for ${action.pr_id}: ${failed.map((check) => `${check.id} (${check.detail})`).join("; ")}. ` +
			`Evidence: ${gateRun.path}. Fix the branch and resume, or record a supervisor waiver naming pre-merge:<check>.`;
		// Effect first, event second: a `failed` (not `complete`) event means the
		// merge action is retried after the human fixes the branch and resumes.
		saveLoopState(root, { state: "PAUSED", reason, next_gate: null }, { actor: "conductor" });
		appendConductorEvent(
			context,
			action,
			{
				action_phase: "failed",
				pr_id: action.pr_id,
				gate_ok: false,
				gate_run: path.relative(root, gateRun.path),
				to: "PAUSED",
				summary: reason,
			},
			"conductor",
		);
		refreshViews(root);
		return { performed: true, skipped: null, action, gate_run: gateRun, transition: { from: "MERGE", to: "PAUSED" }, message: reason };
	}
	const merge = mergeNoFf(root, {
		prId: action.pr_id,
		reviewId: action.review_id,
		branch: String(prRecord.branch ?? ""),
		baseBranch: project.baseBranch,
		summary: String(prRecord.summary ?? ""),
		runId: loaded.run_id,
	});
	// Effect first, completion event second; the merge itself is idempotent
	// (`alreadyMerged` when the head is an ancestor of the base branch).
	appendRecord(
		root,
		"pr",
		{
			...prRecord,
			id: action.pr_id,
			status: "merged",
			merge_sha: merge.sha,
			merge_tag: merge.tag,
			exit_state: "merged",
		},
		{ root, role: "conductor", actor: "conductor" },
	);
	saveLoopState(
		root,
		{ state: action.to, reason: `merged ${action.pr_id} (${merge.sha.slice(0, 12)})`, next_gate: null, pr_id: action.pr_id },
		{ actor: "conductor" },
	);
	appendConductorEvent(
		context,
		action,
		{
			action_phase: "complete",
			pr_id: action.pr_id,
			review_id: action.review_id,
			merge_sha: merge.sha,
			merge_tag: merge.tag,
			already_merged: merge.alreadyMerged,
			gate_run: path.relative(root, gateRun.path),
			gate_ok: true,
			to: action.to,
			summary: `merged ${action.pr_id} (review ${action.review_id}) into ${project.baseBranch}${merge.alreadyMerged ? " (already up to date)" : ` as ${merge.sha.slice(0, 12)}`}${merge.tag ? `; tag ${merge.tag}` : ""}`,
		},
		"conductor",
	);
	refreshViews(root);
	return {
		performed: true,
		skipped: null,
		action,
		gate_run: gateRun,
		transition: { from: "MERGE", to: action.to },
		message: `merged ${action.pr_id} → ${action.to} at ${merge.sha.slice(0, 12)}${merge.tag ? ` (tag ${merge.tag})` : ""}`,
	};
}

function executePause(context: RunContext, action: Action, actor: string): ExecutionResult {
	if (action.kind !== "pause") throw new VangjaLoopError("STATE_INVALID", "executePause needs a pause action");
	const from = loadLoopState(context.root).state.state;
	const next = saveLoopState(context.root, { state: "PAUSED", reason: action.reason }, { actor });
	appendConductorEvent(context, action, { action_phase: "complete", from, to: "PAUSED", reason: action.reason }, actor);
	refreshViews(context.root);
	return { performed: true, skipped: null, action, transition: { from, to: "PAUSED" }, message: `paused: ${action.reason}` };
}

function executeBlock(context: RunContext, action: Action, actor: string): ExecutionResult {
	if (action.kind !== "block") throw new VangjaLoopError("STATE_INVALID", "executeBlock needs a block action");
	const from = loadLoopState(context.root).state.state;
	applyEffects(context, action, actor);
	const next = saveLoopState(context.root, { state: "BLOCKED", reason: action.reason }, { actor });
	appendConductorEvent(
		context,
		action,
		{ action_phase: "complete", from, to: "BLOCKED", reason: action.reason, question: action.question, effects: action.effects ?? [] },
		actor,
	);
	refreshViews(context.root);
	return { performed: true, skipped: null, action, transition: { from, to: "BLOCKED" }, message: `blocked: ${action.reason} — decision needed: ${action.question}` };
}

/**
 * One HPC side effect: `hpc_submit` or `hpc_requeue`, executed by the conductor
 * through the same handlers the tools use. On refusal the loop goes BLOCKED with
 * the handler's message (which names every missing item) — it never retries
 * silently, and never issues a second job.
 */
function executeHpc(context: RunContext, action: Action, actor: string): ExecutionResult {
	if (action.kind !== "hpc") throw new VangjaLoopError("STATE_INVALID", "executeHpc needs an hpc action");
	const from = loadLoopState(context.root).state.state;
	const ctx = { root: context.root, role: "conductor" as const, actor };
	const result =
		action.op === "submit"
			? handleHpcSubmit({ experiment_id: action.experiment_id }, ctx)
			: handleHpcRequeue(
					{ experiment_id: action.experiment_id, reason: action.reason, memoryGb: action.memoryGb },
					ctx,
				);
	const message = result.content[0]?.text ?? "(no output)";
	if (result.isError) {
		saveLoopState(context.root, { state: "BLOCKED", reason: message }, { actor });
		appendConductorEvent(
			context,
			action,
			{
				action_phase: "complete",
				from,
				to: "BLOCKED",
				hpc_op: action.op,
				hpc_ok: false,
				experiment_id: action.experiment_id,
				hpc_missing: Array.isArray(result.details.missing) ? result.details.missing : [message],
				question: `fix the refusal above, then resume ${action.experiment_id}`,
			},
			actor,
		);
		refreshViews(context.root);
		return {
			performed: true,
			skipped: null,
			action,
			transition: { from, to: "BLOCKED" },
			message: `blocked: ${action.op} ${action.experiment_id} refused — ${message}`,
		};
	}
	appendConductorEvent(
		context,
		action,
		{
			action_phase: "complete",
			from,
			to: from,
			hpc_op: action.op,
			hpc_ok: true,
			experiment_id: action.experiment_id,
			job_id: typeof result.details.job_id === "string" ? result.details.job_id : undefined,
			attempt: typeof result.details.attempt === "number" ? result.details.attempt : action.attempt,
			reason: action.reason,
		},
		actor,
	);
	refreshViews(context.root);
	return {
		performed: true,
		skipped: null,
		action,
		transition: { from, to: from },
		message: `${action.op} ${action.experiment_id}: ${message}`,
	};
}
