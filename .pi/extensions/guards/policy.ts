/**
 * Guard policy — model pin, tool-call decisions, and the role preamble.
 *
 * Destructive-bash and shell-write policy lives in `./bash.ts`; protected-path
 * rules in `./protected.ts`; the denial shape in `./decision.ts`. This module
 * composes them for the `tool_call` handler and is re-exported for tests.
 */

import { bashSecretDecision, bashWriteTargetDecision, destructiveBashDecision, type BashPolicyContext } from "./bash.ts";
import { ALLOWED, deny, type GuardDecision } from "./decision.ts";
import { protectedPathDecision, type ProtectionContext } from "./protected.ts";
import { parseRole, roleGateForTool, type LoopRole } from "../../lib/roles.ts";
import { secretPathDecision, scanForSecrets, describeFindings } from "../../lib/secrets.ts";
import { toRepoRelative, underPath } from "../../lib/paths.ts";

export type { GuardDecision, GuardDenial } from "./decision.ts";
export { deny, formatDenial } from "./decision.ts";
export { protectedPathDecision, type ProtectionContext } from "./protected.ts";
export {
	destructiveBashDecision,
	bashWriteTargetDecision,
	bashSecretDecision,
	extractWriteTargets,
	splitSegments,
	tokenize,
	type BashPolicyContext,
} from "./bash.ts";
// ---------------------------------------------------------------------------
// Model pin
// ---------------------------------------------------------------------------

export interface ActiveModel {
	provider: string;
	model: string;
	thinking: string | undefined;
}

export interface ModelCheck {
	ok: boolean;
	message: string;
}

export function checkModelPin(active: ActiveModel, pin: { provider: string; model: string; thinking: string }): ModelCheck {
	const problems: string[] = [];
	if (active.provider !== pin.provider) {
		problems.push(`provider is "${active.provider}" but the pin requires "${pin.provider}"`);
	}
	if (active.model !== pin.model) {
		problems.push(`model is "${active.model}" but the pin requires "${pin.model}"`);
	}
	if (active.thinking !== pin.thinking) {
		problems.push(
			`thinking level is "${active.thinking ?? "unknown"}" but the pin requires "${pin.thinking}"`,
		);
	}
	if (problems.length === 0) return { ok: true, message: "" };
	return {
		ok: false,
		message:
			`[guard:MODEL_PIN] model/thinking drift: ${problems.join("; ")}. ` +
			`The pin is enforced for every session (plan/02_ARCHITECTURE.md §7). ` +
			`Fix .pi/config/model.json, or relaunch with ` +
			`"--provider ${pin.provider} --model ${pin.provider}/${pin.model} --thinking ${pin.thinking}".`,
	};
}

// ---------------------------------------------------------------------------
// Tool-call level decisions
// ---------------------------------------------------------------------------

export interface ToolCallInputs {
	toolName: string;
	path?: string;
	contentToScan?: string;
	command?: string;
}

export interface ToolCallContext extends ProtectionContext {
	root: string;
	cwd: string;
	currentBranch: string | null;
	worktreeDirty: boolean;
	sharedBranches?: string[];
	/** Caller reads the amendment ledger; receives the repo-relative path. */
	amendmentCovers?: (rel: string) => boolean;
}

export function toolCallDecision(input: ToolCallInputs, ctx: ToolCallContext): GuardDecision {
	// 1. Role matrix for gated tool names.
	const roleDecision = roleGateForTool(input.toolName, ctx.role);
	if (!roleDecision.allowed) {
		return deny("ROLE", roleDecision.reason ?? `role ${ctx.role} may not call ${input.toolName}`, roleDenialAlternative(input.toolName));
	}

	if (input.toolName === "write" || input.toolName === "edit") {
		if (!input.path) {
			return deny(
				"PROTECTED_PATH",
				`${input.toolName} call has no "path" argument`,
				"pass the file path explicitly so the guard can check it",
			);
		}
		const relative = toRepoRelative(input.path, ctx.cwd, ctx.root);
		if (!relative.outsideRoot) {
			const pathDecision = protectedPathDecision(relative.rel, {
				role: ctx.role,
				loopState: ctx.loopState,
				amendmentCovers: ctx.amendmentCovers?.(relative.rel) === true,
			});
			if (!pathDecision.allowed) return pathDecision;
			const secretPath = secretPathDecision(relative.rel);
			if (secretPath.blocked) {
				return deny("SECRET", secretPath.reason ?? `refusing to write ${relative.rel}`, "keep credentials outside the repository; commit only examples with placeholders");
			}
		} else {
			const secretPath = secretPathDecision(relative.abs);
			if (secretPath.blocked) {
				return deny("SECRET", secretPath.reason ?? `refusing to write ${relative.abs}`, "keep credentials outside the repository");
			}
		}
		if (input.contentToScan) {
			const scan = scanForSecrets(input.contentToScan);
			if (scan.findings.length > 0) {
				return deny(
					"SECRET",
					`content for "${input.path}" contains secret-like material (${describeFindings(scan.findings)}); the values are not echoed`,
					"remove the credential and reference it as an environment variable or a path outside the repository",
				);
			}
		}
	}

	if (input.toolName === "bash" && input.command) {
		const bashCtx: BashPolicyContext = {
			root: ctx.root,
			cwd: ctx.cwd,
			currentBranch: ctx.currentBranch,
			worktreeDirty: ctx.worktreeDirty,
			sharedBranches: ctx.sharedBranches,
			role: ctx.role,
			loopState: ctx.loopState,
		};
		const destructive = destructiveBashDecision(input.command, bashCtx);
		if (!destructive.allowed) return destructive;
		const writeTarget = bashWriteTargetDecision(input.command, bashCtx);
		if (!writeTarget.allowed) return writeTarget;
		const secrets = bashSecretDecision(input.command, bashCtx);
		if (!secrets.allowed) return secrets;
	}

	return ALLOWED;
}

function roleDenialAlternative(toolName: string): string {
	const capability = toolName;
	return (
		`ask the conductor to set PI_LOOP_ROLE for this session (allowed roles are recorded in ` +
		`.pi/config/agents.json and research_agent/plan/05_EXTENSIONS_AND_TOOLS.md §7 for "${capability}"); ` +
		`if the operation is genuinely needed, record a decision_record explaining why`
	);
}

// Role preamble
// ---------------------------------------------------------------------------

export interface PreambleInput {
	role: LoopRole;
	state: string;
	taskId?: string | null;
	experimentId?: string | null;
	attempt?: number;
	nextGate?: string | null;
	briefPath?: string | null;
}

/** Exactly five lines injected as the `<loop-role>` system-prompt section. */
export function buildRolePreamble(input: PreambleInput): string {
	const task = input.taskId || "none";
	const experiment = input.experimentId ? `${input.experimentId} attempt ${input.attempt ?? 0}` : "none";
	const gate = input.nextGate || "none";
	const brief = input.briefPath || ".pi/state/CONTEXT.md";
	return [
		`LOOP ROLE: ${input.role} | STATE: ${input.state} | TASK: ${task} | EXPERIMENT: ${experiment} | NEXT GATE: ${gate}`,
		`Read your brief first: ${brief} — this session starts with fresh context; files are the memory, the conversation is not.`,
		`Stay inside the brief: Vangja-project/**, research_agent/plan/**, fixed_case_studies/protocol/**, paper/** (outside PAPER), AGENTS.md, .git/**, and .pi/state/** are enforced by the guard extension.`,
		`No claim without an artifact ID: every number cites an artifact path plus its manifest SHA; every codebase statement cites file:line.`,
		`End with a ledger record (task/PR/experiment/review) or BLOCKED plus the smallest decision the human must make.`,
	].join("\n");
}

// Helpers shared by the extension wiring
// ---------------------------------------------------------------------------

export interface AmendmentRecordLike {
	id?: string;
	path?: string;
}

/** True when a recorded amendment covers `rel` (latest record per id wins). */
export function amendmentCovers(rel: string, records: AmendmentRecordLike[], amendmentId?: string): boolean {
	const latest = new Map<string, AmendmentRecordLike>();
	for (const record of records) {
		if (record.id) latest.set(record.id, record);
	}
	const candidates = amendmentId ? [latest.get(amendmentId)].filter(Boolean) : [...latest.values()];
	return (candidates as AmendmentRecordLike[]).some((record) => record.path !== undefined && underPath(rel, record.path));
}

export function roleFromEnv(env: NodeJS.ProcessEnv): LoopRole {
	return parseRole(env.PI_LOOP_ROLE);
}

