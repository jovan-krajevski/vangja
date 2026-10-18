/**
 * Protected-path rules (plan/05_EXTENSIONS_AND_TOOLS.md §4).
 *
 * `Vangja-project/**` and `research_agent/plan/**` are always read-only;
 * `fixed_case_studies/protocol/**` opens only with a recorded amendment;
 * `paper/**` only in the PAPER state; `AGENTS.md` only for the
 * supervisor/conductor; `.git/**` and `.pi/state/**` for nobody via file tools.
 */

import { underPath } from "../../lib/paths.ts";
import { ALLOWED, deny, type GuardDecision } from "./decision.ts";
import type { LoopRole } from "../../lib/roles.ts";

// ---------------------------------------------------------------------------
// Protected paths
// ---------------------------------------------------------------------------

export interface ProtectionContext {
	role: LoopRole;
	loopState: string | null;
	/** True when a recorded protocol amendment covers this path (caller reads the ledger). */
	amendmentCovers?: boolean;
	/** True when an S7 decision names this plan path (human-only exception, see below). */
	planEditCovers?: boolean;
}

interface ProtectionRule {
	match: (rel: string) => boolean;
	code: GuardDenial["code"];
	reason: (rel: string) => string;
	alternative: string;
	unless?: (ctx: ProtectionContext) => boolean;
}

const PROTECTION_RULES: ProtectionRule[] = [
	{
		match: (rel) => underPath(rel, "Vangja-project"),
		code: "PROTECTED_PATH",
		reason: (rel) => `"${rel}" is inside Vangja-project/, the read-only historical audit archive`,
		alternative:
			"read Vangja-project/** as evidence, and write your own analysis under .pi/artifacts/ or a path named by your task brief",
	},
	{
		match: (rel) => underPath(rel, "research_agent/plan"),
		code: "PROTECTED_PATH",
		reason: (rel) => `"${rel}" is inside research_agent/plan/, the frozen build plan`,
		alternative:
			"record an S7 decision naming the exact path (decision_record, title starting \"S7\") and edit it from the " +
			"human S7 build session; every other role and every uncovered path stays read-only",
		// research_agent/prompts/07_verify_smoke_and_docs.md makes S7 the one build
		// stage allowed to correct the plan. The exception is deliberately narrow:
		// human role only, and only for a path named by a recorded S7 decision.
		unless: (ctx) => ctx.role === "human" && ctx.planEditCovers === true,
	},
	{
		match: (rel) => underPath(rel, "fixed_case_studies/protocol"),
		code: "PROTECTED_PATH",
		reason: (rel) =>
			`"${rel}" is a frozen protocol file (dev/confirmation separation depends on it)`,
		alternative:
			"amend the protocol through a supervisor-recorded amendment (amendment_record) and set PI_LOOP_AMENDMENT to that id; otherwise treat the file as read-only",
		unless: (ctx) => ctx.amendmentCovers === true,
	},
	{
		match: (rel) => underPath(rel, "paper"),
		code: "PROTECTED_PATH",
		reason: (rel) => `"${rel}" may only be written in the PAPER state (current state: ${"see guard context"})`,
		alternative:
			"wait for the conductor to enter PAPER (PI_LOOP_STATE=PAPER) before editing paper/**; until then put results in artifacts/ or a task branch file named by your brief",
		unless: (ctx) => ctx.loopState === "PAPER",
	},
	{
		match: (rel) => rel === "AGENTS.md",
		code: "PROTECTED_PATH",
		reason: (rel) => `"${rel}" is the shared project instruction file`,
		alternative:
			"supervisor/conductor may edit AGENTS.md; other roles propose the edit in the PR body or a task brief instead",
		unless: (ctx) => ctx.role === "supervisor" || ctx.role === "conductor",
	},
	{
		match: (rel) => underPath(rel, ".git"),
		code: "PROTECTED_PATH",
		reason: (rel) => `"${rel}" is the git object/store directory`,
		alternative: "use git commands through bash (they are audited); never write .git/** directly",
	},
	{
		match: (rel) => underPath(rel, ".pi/state"),
		code: "PROTECTED_PATH",
		reason: (rel) => `"${rel}" is loop state, owned by the ledger tools`,
		alternative:
			"use the ledger_* tools (ledger_append, task_enqueue, experiment_update, state_set, …) so every change is append-only and locked",
	},
];

/** Rewrite the paper-rule reason with the real state (the factory above cannot see it). */
function reasonWithState(rule: ProtectionRule, rel: string, ctx: ProtectionContext): string {
	if (underPath(rel, "paper")) {
		return `"${rel}" may only be written in the PAPER state (current state: ${ctx.loopState ?? "unknown"})`;
	}
	return rule.reason(rel);
}

export function protectedPathDecision(rel: string, ctx: ProtectionContext): GuardDecision {
	for (const rule of PROTECTION_RULES) {
		if (!rule.match(rel)) continue;
		if (rule.unless?.(ctx)) continue;
		return deny(rule.code, reasonWithState(rule, rel, ctx), rule.alternative);
	}
	return ALLOWED;
}
