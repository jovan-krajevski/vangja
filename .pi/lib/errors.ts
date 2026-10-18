/**
 * Typed errors for the Vangja research-agent environment.
 *
 * Every error message names the file/path/value involved. There are no silent
 * fallbacks in this environment: a missing or malformed input throws.
 */

export type ErrorCode =
	| "CONFIG_MISSING"
	| "CONFIG_INVALID"
	| "LEDGER_INVALID"
	| "LEDGER_LOCK_TIMEOUT"
	| "GUARD_DENIED"
	| "ROLE_DENIED"
	| "STATE_INVALID"
	| "PATH_INVALID"
	| "AGENT_INVALID"
	| "AGENT_NOT_FOUND";

export class VangjaLoopError extends Error {
	readonly code: ErrorCode;

	constructor(code: ErrorCode, message: string) {
		super(message);
		this.name = code;
		this.code = code;
	}
}

export function configMissing(path: string): VangjaLoopError {
	return new VangjaLoopError(
		"CONFIG_MISSING",
		`required config file is missing: ${path}. Create it from the schema in .pi/config/README.md.`,
	);
}

export function configInvalid(path: string, detail: string): VangjaLoopError {
	return new VangjaLoopError("CONFIG_INVALID", `invalid config ${path}: ${detail}`);
}

export function ledgerInvalid(path: string, detail: string): VangjaLoopError {
	return new VangjaLoopError("LEDGER_INVALID", `invalid record for ${path}: ${detail}`);
}

export function lockTimeout(path: string, ms: number): VangjaLoopError {
	return new VangjaLoopError(
		"LEDGER_LOCK_TIMEOUT",
		`could not acquire lock ${path} within ${ms}ms. Another process is holding it, or a stale ` +
			`lock directory was left behind: inspect it, then remove the directory if its owner is dead.`,
	);
}

export function guardDenied(reason: string): VangjaLoopError {
	return new VangjaLoopError("GUARD_DENIED", reason);
}

export function roleDenied(reason: string): VangjaLoopError {
	return new VangjaLoopError("ROLE_DENIED", reason);
}

export function agentInvalid(path: string, detail: string): VangjaLoopError {
	return new VangjaLoopError("AGENT_INVALID", `invalid agent ${path}: ${detail}`);
}

export function agentNotFound(detail: string): VangjaLoopError {
	return new VangjaLoopError("AGENT_NOT_FOUND", detail);
}
