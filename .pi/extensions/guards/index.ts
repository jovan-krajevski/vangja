/**
 * guards — hard invariants for every pi session in this repository.
 *
 * On load it wires:
 * - model/thinking pin enforcement (`session_start`, `model_select`,
 *   `thinking_level_select`, `before_agent_start`, `input`) — a drifting session
 *   is blocked and never reaches the model, and the process exits non-zero;
 * - the 5-line `<loop-role>` preamble in the system prompt;
 * - protected-path, secret, destructive-bash, and role gates on tool calls;
 * - secret redaction of bash tool results.
 *
 * The decisions live in `./policy.ts` and `.pi/lib/*` so they can be tested
 * with `node` directly (this file imports pi types only).
 */

import { execFileSync } from "node:child_process";
import type {
	BeforeAgentStartEvent,
	ExtensionAPI,
	ExtensionContext,
	InputEvent,
	ToolCallEvent,
	ToolResultEvent,
} from "@earendil-works/pi-coding-agent";
import { loadModelPin, type ModelPin } from "../../lib/config.ts";
import { readJsonlFile } from "../../lib/io.ts";
import { resolveRepoRoot, statePath } from "../../lib/paths.ts";
import { describeFindings, scanForSecrets } from "../../lib/secrets.ts";
import { loadLoopState } from "../../lib/state.ts";
import {
	amendmentCovers as amendmentCoversRecords,
	buildRolePreamble,
	checkModelPin,
	formatDenial,
	roleFromEnv,
	toolCallDecision,
	type ActiveModel,
	type AmendmentRecordLike,
	type ToolCallInputs,
} from "./policy.ts";
import type { LoopRole } from "../../lib/roles.ts";

export const GUARD_EXIT_CODE = 3;

interface GitInfo {
	branch: string | null;
	dirty: boolean;
	at: number;
}

let gitCache: GitInfo | null = null;

function gitInfo(root: string): { branch: string | null; dirty: boolean } {
	if (gitCache && Date.now() - gitCache.at < 2000) return gitCache;
	let branch: string | null = null;
	let dirty = false;
	try {
		branch = execFileSync("git", ["-C", root, "rev-parse", "--abbrev-ref", "HEAD"], {
			encoding: "utf8",
			stdio: ["ignore", "pipe", "ignore"],
		}).trim();
		const status = execFileSync("git", ["-C", root, "status", "--porcelain", "--untracked-files=no"], {
			encoding: "utf8",
			stdio: ["ignore", "pipe", "ignore"],
		});
		dirty = status.trim() !== "";
	} catch {
		// Not a git repository, or git is unavailable; bash policy treats it as unknown.
	}
	gitCache = { branch, dirty, at: Date.now() };
	return { branch, dirty };
}

function activeModelOf(ctx: ExtensionContext): ActiveModel {
	return {
		provider: ctx.model?.provider ?? "unknown",
		model: ctx.model?.id ?? "unknown",
		thinking: ctx.thinkingLevel,
	};
}

export function registerGuard(pi: ExtensionAPI, env: NodeJS.ProcessEnv = process.env) {
	const role: LoopRole = roleFromEnv(env);
	let pinCache: { root: string; pin: ModelPin | null; error: string } | null = null;
	let drifted: string | null = null;

	const rootFor = (ctx: ExtensionContext): string => {
		try {
			return resolveRepoRoot(env.PI_LOOP_ROOT ?? ctx.cwd);
		} catch {
			return ctx.cwd;
		}
	};

	const pinFor = (root: string): { pin: ModelPin | null; error: string } => {
		if (pinCache && pinCache.root === root) return pinCache;
		try {
			pinCache = { root, pin: loadModelPin(root), error: "" };
		} catch (error) {
			pinCache = { root, pin: null, error: (error as Error).message };
		}
		return pinCache;
	};

	const loopStateFor = (root: string): string | null => {
		if (env.PI_LOOP_STATE && env.PI_LOOP_STATE.trim() !== "") return env.PI_LOOP_STATE.trim();
		try {
			const loaded = loadLoopState(root);
			return loaded.initialized ? loaded.state.state : null;
		} catch {
			return null;
		}
	};

	const amendmentCovers = (root: string, rel: string): boolean => {
		try {
			const records = readJsonlFile<AmendmentRecordLike>(statePath(root, "AMENDMENTS.jsonl"));
			return amendmentCoversRecords(rel, records, env.PI_LOOP_AMENDMENT);
		} catch {
			return false;
		}
	};

	const reportDrift = (ctx: ExtensionContext, message: string): void => {
		if (drifted === message) return;
		drifted = message;
		process.stderr.write(`${message}\n`);
		try {
			ctx.ui.notify(message, "error");
		} catch {
			// No UI available (print/json/rpc without notifications); stderr already has it.
		}
		process.exitCode = GUARD_EXIT_CODE;
	};

	const evaluatePin = (ctx: ExtensionContext): void => {
		if (drifted) return;
		const root = rootFor(ctx);
		const { pin, error } = pinFor(root);
		if (!pin) {
			reportDrift(
				ctx,
				`[guard:MODEL_PIN] cannot load the model pin: ${error}. ` +
					`Every session must run with the pin recorded in .pi/config/model.json (plan/02_ARCHITECTURE.md §7).`,
			);
			return;
		}
		const check = checkModelPin(activeModelOf(ctx), pin);
		if (check.ok) return;
		if (!pin.enforce) {
			process.stderr.write(
				`[guard:MODEL_PIN] warning: ${check.message} (enforce=false in .pi/config/model.json; not blocking)\n`,
			);
			return;
		}
		reportDrift(ctx, check.message);
	};

	pi.on("session_start", async (_event, ctx) => {
		evaluatePin(ctx);
	});
	pi.on("model_select", async (_event, ctx) => {
		evaluatePin(ctx);
	});
	pi.on("thinking_level_select", async (_event, ctx) => {
		evaluatePin(ctx);
	});

	pi.on("before_agent_start", async (event: BeforeAgentStartEvent, ctx: ExtensionContext) => {
		evaluatePin(ctx);
		if (drifted) {
			ctx.abort();
			return undefined;
		}
		const root = rootFor(ctx);
		const loaded = loadLoopState(root);
		// PI_LOOP_STATE is authoritative for a child process (the conductor sets it
		// per action); the state file is the fallback for interactive sessions.
		const stateName = loopStateFor(root) ?? (loaded.initialized ? loaded.state.state : "UNINITIALIZED");
		const sections = event.systemPromptOptions.sections;
		sections["loop-role"] = buildRolePreamble({
			role,
			state: stateName,
			taskId: loaded.state.task_id,
			experimentId: loaded.state.experiment_id,
			attempt: loaded.state.attempt,
			nextGate: loaded.state.next_gate,
			briefPath: env.PI_LOOP_BRIEF ?? null,
		});
		return undefined;
	});

	pi.on("input", async (_event: InputEvent, ctx: ExtensionContext) => {
		if (drifted) {
			ctx.ui.notify("Input dropped: the model/thinking pin check failed (see .pi/config/model.json)", "error");
			return { action: "handled" as const };
		}
		return undefined;
	});

	pi.on("tool_call", async (event: ToolCallEvent, ctx: ExtensionContext) => {
		if (drifted) {
			return { block: true, reason: `${drifted} Tool call "${event.toolName}" was not executed.`, terminate: true };
		}
		const root = rootFor(ctx);
		const git = gitInfo(root);
		const input = event.input as Record<string, unknown>;
		const call: ToolCallInputs = { toolName: event.toolName };
		if (typeof input.path === "string") call.path = input.path;
		if (typeof input.command === "string") call.command = input.command;
		if (typeof input.content === "string") call.contentToScan = input.content;
		if (Array.isArray(input.edits)) {
			call.contentToScan = (input.edits as Array<Record<string, unknown>>)
				.map((edit) => `${String(edit.oldText ?? "")}\n${String(edit.newText ?? "")}`)
				.join("\n");
		}
		const decision = toolCallDecision(call, {
			role,
			loopState: loopStateFor(root),
			root,
			cwd: ctx.cwd,
			currentBranch: git.branch,
			worktreeDirty: git.dirty,
			amendmentCovers: (rel) => amendmentCovers(root, rel),
		});
		if (!decision.allowed) {
			return { block: true, reason: formatDenial(decision) };
		}
		return undefined;
	});

	pi.on("tool_result", async (event: ToolResultEvent, _ctx: ExtensionContext) => {
		if (event.toolName !== "bash") return undefined;
		const text = event.content
			.filter((part): part is { type: "text"; text: string } => part.type === "text")
			.map((part) => part.text)
			.join("\n");
		if (text === "") return undefined;
		const scan = scanForSecrets(text);
		if (scan.findings.length === 0) return undefined;
		const warning =
			`\n[guard:SECRET] redacted ${scan.findings.length} secret-like value(s) from this output ` +
			`(${describeFindings(scan.findings)}); the values were replaced with [REDACTED:<kind>] and are not shown.`;
		return {
			content: [{ type: "text" as const, text: scan.redacted + warning }],
			isError: event.isError,
		};
	});
}

export default function (pi: ExtensionAPI) {
	registerGuard(pi, process.env);
}
