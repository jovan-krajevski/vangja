/**
 * The denial shape shared by every guard rule.
 *
 * A denial always states what was refused, why, and which alternative is
 * allowed — "no silent fallbacks" applies to refusals too.
 */

import type { LoopRole } from "../../lib/roles.ts";

export interface GuardDenial {
	allowed: false;
	reason: string;
	alternative: string;
	code: "PROTECTED_PATH" | "SECRET" | "DESTRUCTIVE_BASH" | "ROLE" | "MODEL_PIN";
}

export type GuardDecision = { allowed: true } | GuardDenial;

export const ALLOWED: GuardDecision = { allowed: true };

export function deny(code: GuardDenial["code"], reason: string, alternative: string): GuardDenial {
	return { allowed: false, code, reason, alternative };
}

export function formatDenial(denial: GuardDenial): string {
	return `[guard:${denial.code}] ${denial.reason} Allowed alternative: ${denial.alternative}`;
}
