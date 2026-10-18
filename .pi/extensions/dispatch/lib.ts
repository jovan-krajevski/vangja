/**
 * Dispatch engine: spawn one `pi --mode json` child per worker with an isolated
 * context, pinned model, and its own persisted session.
 *
 * All logic is pi-free so `node --test` exercises the real spawn path with a
 * stub command through `PI_DISPATCH_COMMAND`. The extension in `./index.ts`
 * only declares the three tools and maps their results.
 *
 * Guarantees (research_agent/prompts/03_agents_and_prompts.md):
 * - the argv is built here, once, and recorded verbatim;
 * - the child always gets `--provider/--model/--thinking` from
 *   `.pi/config/model.json`, never from the agent frontmatter;
 * - the child's session directory is unique (`<run>/<agent>/<task>/<attempt>/`),
 *   so no child can inherit another child's session;
 * - the agent file hash, usage, exit state, and session path are appended to
 *   `EVENTS.jsonl` exactly once per spawned child;
 * - output is capped for the model; the full text stays in the session file.
 */

import { spawn } from "node:child_process";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { appendRecord } from "../ledger/store.ts";
import { VangjaLoopError } from "../../lib/errors.ts";
import { loadModelPin } from "../../lib/config.ts";
import { loadToolRegistry, assertNoUnknownTools } from "../../lib/toolset.ts";
import { resolveDispatchTarget, sanitizeSegment, dispatchSessionName, MAX_DISPATCH_DEPTH } from "../../lib/agents.ts";
import { statePath } from "../../lib/paths.ts";
import type { LoopRole } from "../../lib/roles.ts";

export const DEFAULT_OUTPUT_CAP_BYTES = 50 * 1024;

export interface DispatchRequest {
	agent: string;
	task: string;
	/** Defaults to the current task from LOOP_STATE.json, else "adhoc". */
	taskId?: string;
	/** Defaults to 1. */
	attempt?: number;
}

export interface DispatchRunOptions {
	root: string;
	/** Role of the *parent* session (from PI_LOOP_ROLE); gates the allowlist. */
	parentRole: LoopRole;
	/** Parent agent name for the ledger actor; defaults to `agent:<role>`. */
	parentAgent?: string;
	runId?: string | null;
	/** Override the pi executable (tests use a stub; production uses "pi"). */
	command?: string;
	extraEnv?: NodeJS.ProcessEnv;
	onUpdate?: (text: string) => void;
	/** Hard wall-clock limit for a child (tests only; the orchestrator owns real budgets). */
	timeoutMs?: number;
	outputCapBytes?: number;
	/** How this child was reached; recorded in the ledger event. */
	mode?: "agent" | "parallel" | "chain";
	/** 1-based chain step, when mode is "chain". */
	step?: number;
}

export interface DispatchUsage {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	cost: number;
	contextTokens: number;
	turns: number;
}

export type ExitState = "ok" | "failed" | "aborted";

export interface DispatchOutcome {
	agent: string;
	role: string;
	task: string;
	task_id: string;
	attempt: number;
	name: string;
	model: string;
	exit_code: number;
	exit_state: ExitState;
	stop_reason?: string;
	error_message?: string;
	stderr: string;
	final_text: string;
	truncated: boolean;
	output_bytes: number;
	usage: DispatchUsage;
	session_dir: string;
	session_file: string | null;
	session_id: string | null;
	argv: string[];
	agent_file: string;
	agent_file_hash: string;
	started_at: string;
	ended_at: string;
	duration_ms: number;
}

export function emptyUsage(): DispatchUsage {
	return { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, cost: 0, contextTokens: 0, turns: 0 };
}

function byteLength(text: string): number {
	return Buffer.byteLength(text, "utf8");
}

/** Full text for the session, capped text for the model, with the pointer note. */
export function visibleText(outcome: DispatchOutcome, capBytes = DEFAULT_OUTPUT_CAP_BYTES): string {
	if (!outcome.truncated) return outcome.final_text;
	let cut = outcome.final_text.slice(0, capBytes);
	while (byteLength(cut) > capBytes) cut = cut.slice(0, -1);
	const omitted = outcome.output_bytes - byteLength(cut);
	const where = outcome.session_file ?? outcome.session_dir;
	return `${cut}\n\n[dispatch output truncated: ${omitted} bytes omitted. The complete transcript is in ${where}.]`;
}

export function isFailedOutcome(outcome: DispatchOutcome): boolean {
	return outcome.exit_state !== "ok";
}

/** Bounded worker pool: at most `concurrency` calls in flight (plan/02 §6). */
export async function mapWithConcurrencyLimit<T, R>(
	items: T[],
	concurrency: number,
	fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
	if (items.length === 0) return [];
	const limit = Math.max(1, Math.min(concurrency, items.length));
	const results: R[] = new Array(items.length);
	let next = 0;
	const workers = new Array(limit).fill(null).map(async () => {
		for (;;) {
			const index = next;
			next += 1;
			if (index >= items.length) return;
			results[index] = await fn(items[index], index);
		}
	});
	await Promise.all(workers);
	return results;
}

export interface ChainResult {
	outcomes: DispatchOutcome[];
	/** 1-based index of the step that failed, when the chain stopped early. */
	failedAt?: number;
}

/** Run steps sequentially, substituting `{previous}` with the prior step's final text. */
export async function runDispatchChain(
	steps: DispatchRequest[],
	options: DispatchRunOptions,
): Promise<ChainResult> {
	const outcomes: DispatchOutcome[] = [];
	let previous = "";
	for (let i = 0; i < steps.length; i += 1) {
		const step = steps[i];
		const task = step.task.replace(/\{previous\}/g, previous);
		const outcome = await runDispatchAgent(
			{ ...step, task },
			{ ...options, mode: "chain", step: i + 1 },
		);
		outcomes.push(outcome);
		if (isFailedOutcome(outcome)) return { outcomes, failedAt: i + 1 };
		previous = outcome.final_text;
	}
	return { outcomes };
}

function readRunId(root: string): string | null {
	try {
		const raw = JSON.parse(fs.readFileSync(statePath(root, "LOOP_STATE.json"), "utf8")) as { run_id?: unknown };
		return typeof raw.run_id === "string" && raw.run_id !== "" ? raw.run_id : null;
	} catch {
		return null;
	}
}

function readStateTaskId(root: string): string | null {
	try {
		const raw = JSON.parse(fs.readFileSync(statePath(root, "LOOP_STATE.json"), "utf8")) as { task_id?: unknown };
		return typeof raw.task_id === "string" && raw.task_id !== "" ? raw.task_id : null;
	} catch {
		return null;
	}
}

export interface ChildInvocation {
	command: string;
	args: string[];
	env: NodeJS.ProcessEnv;
	sessionDir: string;
	sessionName: string;
	systemPromptFile: string;
	cleanup: () => void;
	argv: string[];
}

interface ParsedStream {
	sessionId: string | null;
	finalText: string;
	usage: DispatchUsage;
	model: string | null;
	stopReason?: string;
	errorMessage?: string;
	assistantMessages: number;
}

interface MessageLike {
	role?: string;
	model?: unknown;
	stopReason?: unknown;
	errorMessage?: unknown;
	usage?: {
		input?: unknown;
		output?: unknown;
		cacheRead?: unknown;
		cacheWrite?: unknown;
		totalTokens?: unknown;
		cost?: { total?: unknown } | number;
	};
	content?: unknown;
}

function textOf(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	const parts: string[] = [];
	for (const part of content) {
		if (typeof part === "object" && part !== null && (part as { type?: unknown }).type === "text") {
			const text = (part as { text?: unknown }).text;
			if (typeof text === "string") parts.push(text);
		}
	}
	return parts.join("");
}

function numberOrZero(value: unknown): number {
	return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/** Parse one JSONL record from the child's `--mode json` stream. */
export function parseStreamLine(line: string, state: ParsedStream): void {
	const trimmed = line.trim();
	if (trimmed === "") return;
	let event: Record<string, unknown>;
	try {
		event = JSON.parse(trimmed) as Record<string, unknown>;
	} catch {
		return;
	}
	if (event.type === "session" && typeof event.id === "string") {
		state.sessionId = event.id;
		return;
	}
	if (event.type !== "message_end" || typeof event.message !== "object" || event.message === null) return;
	const message = event.message as MessageLike;
	if (message.role !== "assistant") return;
	state.assistantMessages += 1;
	const text = textOf(message.content);
	if (text !== "") state.finalText = text;
	if (typeof message.model === "string" && message.model !== "") state.model = message.model;
	if (typeof message.stopReason === "string") state.stopReason = message.stopReason;
	if (typeof message.errorMessage === "string" && message.errorMessage !== "") state.errorMessage = message.errorMessage;
	const usage = message.usage;
	if (usage) {
		state.usage.turns += 1;
		state.usage.input += numberOrZero(usage.input);
		state.usage.output += numberOrZero(usage.output);
		state.usage.cacheRead += numberOrZero(usage.cacheRead);
		state.usage.cacheWrite += numberOrZero(usage.cacheWrite);
		state.usage.contextTokens = numberOrZero(usage.totalTokens) || state.usage.contextTokens;
		const cost = typeof usage.cost === "number" ? usage.cost : numberOrZero(usage.cost?.total);
		state.usage.cost += cost;
	}
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
		return files
			.map((file) => ({ file, mtime: fs.statSync(file).mtimeMs }))
			.sort((a, b) => b.mtime - a.mtime)[0].file;
	} catch {
		return null;
	}
}

/** Build the exact child argv + environment. Exported for tests and the report. */
export function buildChildInvocation(
	request: DispatchRequest,
	options: DispatchRunOptions,
): { invocation: ChildInvocation; agentFile: string; agentFileHash: string; role: string; taskId: string; attempt: number } {
	if (request.task.trim() === "") {
		throw new VangjaLoopError("AGENT_INVALID", `dispatch task for "${request.agent}" must not be empty`);
	}
	const depth = Number(process.env.PI_DISPATCH_DEPTH ?? "0");
	if (Number.isFinite(depth) && depth >= MAX_DISPATCH_DEPTH) {
		throw new VangjaLoopError(
			"ROLE_DENIED",
			`dispatch depth ${depth} reached the cap (${MAX_DISPATCH_DEPTH}); a worker may not keep spawning ` +
				`sub-workers. Return a BLOCKED result to the parent instead.`,
		);
	}

	const { definition, manifest } = resolveDispatchTarget(options.root, options.parentRole, request.agent);
	const pin = loadModelPin(options.root);
	const pinnedId = `${pin.provider}/${pin.model}`;
	if (manifest.model !== pinnedId || definition.model !== pinnedId) {
		throw new VangjaLoopError(
			"AGENT_INVALID",
			`agent "${request.agent}" pins model "${manifest.model}" (frontmatter "${definition.model}") ` +
				`but .pi/config/model.json pins "${pinnedId}"; every agent runs the pinned model ` +
				`(plan/02_ARCHITECTURE.md §7).`,
		);
	}

	const registry = loadToolRegistry(options.root);
	const tools = assertNoUnknownTools(registry, manifest.tools, `agent "${request.agent}"`);

	const taskId = (request.taskId ?? readStateTaskId(options.root) ?? "adhoc").trim() || "adhoc";
	const attempt = Number.isFinite(request.attempt) && (request.attempt ?? 0) >= 1 ? (request.attempt as number) : 1;
	const sessionName = dispatchSessionName(request.agent, taskId, attempt);
	const runId = options.runId === undefined ? readRunId(options.root) : options.runId;
	const sessionDir = path.join(
		statePath(options.root, "sessions"),
		sanitizeSegment(runId ?? "no-run"),
		sanitizeSegment(request.agent),
		sanitizeSegment(taskId),
		`a${attempt}`,
	);
	fs.mkdirSync(sessionDir, { recursive: true });

	// The system prompt is the agent body only; the frontmatter never reaches the model.
	const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "pi-dispatch-"));
	const systemPromptFile = path.join(tmpDir, `${sanitizeSegment(request.agent)}.md`);
	fs.writeFileSync(systemPromptFile, `${definition.body}\n`, { encoding: "utf8", mode: 0o600 });

	const extensions = ["guards", "ledger", "dispatch", "orchestrator"].map((name) =>
		path.join(options.root, ".pi", "extensions", name, "index.ts"),
	);
	const skills = manifest.skills.map((skill) => path.join(options.root, ".pi", "skills", skill));
	const missingSkills = skills.filter((dir) => !fs.existsSync(dir));
	const cleanup = (): void => {
		try {
			fs.rmSync(tmpDir, { recursive: true, force: true });
		} catch {
			// Temp cleanup is best effort; the run must not fail because of it.
		}
	};
	if (missingSkills.length > 0) {
		cleanup();
		throw new VangjaLoopError(
			"AGENT_INVALID",
			`agent "${request.agent}" names skills that do not exist: ${missingSkills.join(", ")}. ` +
				`Skills live under .pi/skills/ (plan/04_SKILLS.md).`,
		);
	}

	const env: NodeJS.ProcessEnv = {
		...process.env,
		...options.extraEnv,
		PI_LOOP_ROOT: options.root,
		PI_LOOP_ROLE: manifest.role,
		PI_AGENT_NAME: sessionName,
		PI_DISPATCH_DEPTH: String(depth + 1),
		PI_DISPATCH_PARENT: options.parentAgent ?? `agent:${options.parentRole}`,
	};

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
		`TASK: ${request.task}`,
	);

	const argv = [options.command ?? process.env.PI_DISPATCH_COMMAND ?? "pi", ...args];
	return {
		invocation: {
			command: argv[0],
			args,
			env,
			sessionDir,
			sessionName,
			systemPromptFile,
			cleanup,
			argv,
		},
		agentFile: definition.file,
		agentFileHash: definition.sha256,
		role: manifest.role,
		taskId,
		attempt,
	};
}

/** Spawn one worker and resolve to its outcome. Never throws for a child that ran and failed. */
export async function runDispatchAgent(
	request: DispatchRequest,
	options: DispatchRunOptions,
): Promise<DispatchOutcome> {
	const { invocation, agentFile, agentFileHash, role, taskId, attempt } = buildChildInvocation(request, options);
	const startedAt = new Date().toISOString();
	const startedMs = Date.now();
	const state: ParsedStream = {
		sessionId: null,
		finalText: "",
		usage: emptyUsage(),
		model: null,
		assistantMessages: 0,
	};
	const outputCap = options.outputCapBytes ?? DEFAULT_OUTPUT_CAP_BYTES;
	let stderr = "";
	let aborted = false;
	let timedOut = false;

	try {
		const result = await new Promise<{ code: number | null; spawnError?: string }>((resolve) => {
			const child = spawn(invocation.command, invocation.args, {
				cwd: options.root,
				env: invocation.env,
				shell: false,
				stdio: ["ignore", "pipe", "pipe"],
			});
			let buffer = "";

			const flush = (line: string): void => {
				parseStreamLine(line, state);
				if (options.onUpdate) {
					options.onUpdate(state.finalText || "(running...)");
				}
			};

			child.stdout.on("data", (data: Buffer) => {
				buffer += data.toString("utf8");
				const lines = buffer.split("\n");
				buffer = lines.pop() ?? "";
				for (const line of lines) flush(line);
			});
			child.stderr.on("data", (data: Buffer) => {
				stderr += data.toString("utf8");
			});

			let settled = false;
			const finish = (code: number | null, spawnError?: string): void => {
				if (settled) return;
				settled = true;
				if (buffer.trim() !== "") flush(buffer);
				if (killTimer) clearTimeout(killTimer);
				if (timeoutTimer) clearTimeout(timeoutTimer);
				resolve({ code, spawnError });
			};

			child.on("close", (code) => finish(code));
			child.on("error", (error) => finish(null, (error as Error).message));

			let killTimer: NodeJS.Timeout | undefined;
			const killProc = (): void => {
				child.kill("SIGTERM");
				killTimer = setTimeout(() => {
					// `child.killed` only says a signal was sent; check the real exit state.
					if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL");
				}, 5000);
				killTimer.unref?.();
			};
			let timeoutTimer: NodeJS.Timeout | undefined;
			if (options.timeoutMs) {
				timeoutTimer = setTimeout(() => {
					timedOut = true;
					killProc();
				}, options.timeoutMs);
			}
			if (options.signal) {
				if (options.signal.aborted) {
					aborted = true;
					killProc();
				} else {
					options.signal.addEventListener(
						"abort",
						() => {
							aborted = true;
							killProc();
						},
						{ once: true },
					);
				}
			}
		});

		const exitCode = result.code ?? 1;
		const stopReason = state.stopReason;
		const failed =
			result.spawnError !== undefined ||
			exitCode !== 0 ||
			stopReason === "error" ||
			stopReason === "aborted";
		const exitState: ExitState =
			aborted || stopReason === "aborted" ? "aborted" : failed || timedOut ? "failed" : "ok";

		const sessionFile = findSessionFile(invocation.sessionDir, state.sessionId);
		const outputBytes = byteLength(state.finalText);
		const outcome: DispatchOutcome = {
			agent: request.agent,
			role,
			task: request.task,
			task_id: taskId,
			attempt,
			name: invocation.sessionName,
			model: state.model ?? loadModelPin(options.root).model,
			exit_code: exitCode,
			exit_state: exitState,
			stop_reason: stopReason,
			error_message: timedOut
				? `child timed out after ${options.timeoutMs}ms and was killed`
				: (result.spawnError ?? state.errorMessage),
			stderr,
			final_text: state.finalText,
			truncated: outputBytes > outputCap,
			output_bytes: outputBytes,
			usage: state.usage,
			session_dir: invocation.sessionDir,
			session_file: sessionFile,
			session_id: state.sessionId,
			argv: invocation.argv,
			agent_file: agentFile,
			agent_file_hash: agentFileHash,
			started_at: startedAt,
			ended_at: new Date().toISOString(),
			duration_ms: Math.max(0, Date.now() - startedMs),
		};
		appendDispatchEvent(options.root, outcome, options.parentRole, options.parentAgent, {
			mode: options.mode ?? "agent",
			step: options.step,
		});
		return outcome;
	} finally {
		invocation.cleanup();
	}
}

function firstLine(text: string, max = 160): string {
	const line = text.split("\n").find((candidate) => candidate.trim() !== "")?.trim() ?? "";
	return byteLength(line) > max ? `${line.slice(0, max - 1)}…` : line;
}

const EXIT_STATES = new Set(["ok", "failed", "aborted"]);

/** Append exactly one `dispatch` event for a spawned child. */
export function appendDispatchEvent(
	root: string,
	outcome: DispatchOutcome,
	parentRole: LoopRole,
	parentAgent?: string,
	extras: { mode?: "agent" | "parallel" | "chain"; step?: number } = {},
): Record<string, unknown> {
	if (!EXIT_STATES.has(outcome.exit_state)) {
		throw new VangjaLoopError(
			"LEDGER_INVALID",
			`refusing to record dispatch with unknown exit state "${outcome.exit_state}"`,
		);
	}
	const status = outcome.exit_state === "ok" ? "completed" : outcome.exit_state;
	return appendRecord(
		root,
		"event",
		{
			summary: `dispatch ${outcome.agent} ${status}: ${firstLine(outcome.final_text) || outcome.error_message || "(no output)"}`,
			task_id: outcome.task_id,
			agent: outcome.agent,
			dispatch_mode: extras.mode ?? "agent",
			dispatch_step: extras.step,
			exit_state: outcome.exit_state,
			exit_code: outcome.exit_code,
			stop_reason: outcome.stop_reason,
			usage: { ...outcome.usage },
			session_path: outcome.session_file ?? outcome.session_dir,
			session_dir: outcome.session_dir,
			agent_file: outcome.agent_file,
			agent_file_hash: outcome.agent_file_hash,
			attempt: outcome.attempt,
			duration_ms: outcome.duration_ms,
			argv: outcome.argv,
		},
		{ root, role: parentRole, actor: parentAgent ?? `agent:${parentRole}` },
	);
}
