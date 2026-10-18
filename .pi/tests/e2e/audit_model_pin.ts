#!/usr/bin/env node
/**
 * Model-pin audit for recorded conductor→child invocations (S7).
 *
 * `plan/02_ARCHITECTURE.md` §7 requires every child process to run with the
 * pinned provider/model and thinking level. The guard extension refuses drift at
 * spawn time; this audit proves the *recorded* history agrees: every event in
 * `.pi/state/EVENTS.jsonl` that carries an `argv` array must name
 * `--model <provider>/<model>` and `--thinking <level>` from
 * `.pi/config/model.json`, and `--provider <provider>` when it is present.
 *
 * Usage:
 *   node .pi/tests/e2e/audit_model_pin.ts                 # the repository's EVENTS.jsonl
 *   node .pi/tests/e2e/audit_model_pin.ts <file.jsonl>... # explicit files (tests)
 *
 * Exit 0: every recorded invocation is pinned (or none are recorded yet).
 * Exit 1: at least one violation; each violation is printed with its event.
 *
 * Read-only: it never writes state and never calls a model.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { loadModelPin } from "../../lib/config.ts";
import { resolveRepoRoot, statePath } from "../../lib/paths.ts";

export interface PinAuditViolation {
	file: string;
	seq: number | null;
	actor: string;
	reason: string;
	argv: string[];
}

export interface PinAuditFile {
	path: string;
	present: boolean;
	invocations: number;
	violations: PinAuditViolation[];
}

export interface PinAuditResult {
	files: PinAuditFile[];
	invocations: number;
	violations: PinAuditViolation[];
}

export interface ModelPin {
	provider: string;
	model: string;
	thinking: string;
}

/** Index of `flag` in argv; the value follows the flag. */
function flagValue(argv: string[], flag: string): string | null {
	const index = argv.indexOf(flag);
	if (index < 0) return null;
	const value = argv[index + 1];
	return typeof value === "string" ? value : null;
}

export function auditArgv(root: string, file: string, pin: ModelPin): PinAuditViolation[] {
	if (!fs.existsSync(file)) return [];
	const violations: PinAuditViolation[] = [];
	const lines = fs.readFileSync(file, "utf8").split("\n");
	lines.forEach((line, lineIndex) => {
		if (line.trim() === "") return;
		let record: Record<string, unknown>;
		try {
			record = JSON.parse(line) as Record<string, unknown>;
		} catch (error) {
			violations.push({
				file: path.relative(root, file),
				seq: null,
				actor: "",
				reason: `line ${lineIndex + 1} is not valid JSON: ${(error as Error).message}`,
				argv: [],
			});
			return;
		}
		const argv = record.argv;
		if (!Array.isArray(argv) || argv.length === 0) return;
		const normalized = argv.map(String);
		const pinnedId = `${pin.provider}/${pin.model}`;
		const model = flagValue(normalized, "--model");
		const thinking = flagValue(normalized, "--thinking");
		const provider = flagValue(normalized, "--provider");
		const reasons: string[] = [];
		if (model === null) reasons.push(`no --model flag (expected ${pinnedId})`);
		else if (model !== pinnedId) reasons.push(`--model is "${model}", expected "${pinnedId}"`);
		if (thinking === null) reasons.push(`no --thinking flag (expected "${pin.thinking}")`);
		else if (thinking !== pin.thinking) reasons.push(`--thinking is "${thinking}", expected "${pin.thinking}"`);
		if (provider !== null && provider !== pin.provider) {
			reasons.push(`--provider is "${provider}", expected "${pin.provider}"`);
		}
		if (reasons.length > 0) {
			violations.push({
				file: path.relative(root, file),
				seq: typeof record.seq === "number" ? record.seq : null,
				actor: String(record.actor ?? ""),
				reason: reasons.join("; "),
				argv: normalized,
			});
		}
	});
	return violations;
}

export function auditFiles(root: string, files: string[]): PinAuditResult {
	const pin = loadModelPin(root);
	const results: PinAuditFile[] = [];
	for (const file of files) {
		const present = fs.existsSync(file);
		let invocations = 0;
		if (present) {
			for (const line of fs.readFileSync(file, "utf8").split("\n")) {
				if (line.trim() === "") continue;
				try {
					const record = JSON.parse(line) as Record<string, unknown>;
					if (Array.isArray(record.argv) && record.argv.length > 0) invocations += 1;
				} catch {
					// A malformed line is reported by auditArgv below.
				}
			}
		}
		results.push({ path: file, present, invocations, violations: auditArgv(root, file, pin) });
	}
	const violations = results.flatMap((file) => file.violations);
	return { files: results, invocations: results.reduce((sum, file) => sum + file.invocations, 0), violations };
}

export function renderPinAudit(root: string, result: PinAuditResult): string[] {
	const lines: string[] = [];
	for (const file of result.files) {
		const rel = path.relative(root, file.path) || file.path;
		lines.push(
			file.present
				? `model-pin audit: ${rel}: ${file.invocations} recorded child invocation(s)`
				: `model-pin audit: ${rel}: file absent — nothing recorded yet`,
		);
	}
	for (const violation of result.violations) {
		lines.push(
			`VIOLATION  ${violation.file}${violation.seq === null ? "" : `#seq=${violation.seq}`} (${violation.actor || "unknown actor"}): ${violation.reason}`,
		);
		lines.push(`           argv: ${JSON.stringify(violation.argv)}`);
	}
	lines.push(
		result.violations.length === 0
			? `model-pin audit: 0 violation(s) — ${result.invocations}/${result.invocations} recorded invocation(s) use the pin`
			: `model-pin audit: ${result.violations.length} violation(s)`,
	);
	return lines;
}

function main(argv: string[]): number {
	const root = resolveRepoRoot(process.env.PI_LOOP_REAL_ROOT ?? undefined);
	const explicit = argv.filter((arg) => !arg.startsWith("-"));
	const files = explicit.length > 0 ? explicit.map((file) => path.resolve(file)) : [statePath(root, "EVENTS.jsonl")];
	const result = auditFiles(root, files);
	for (const line of renderPinAudit(root, result)) process.stdout.write(`${line}\n`);
	return result.violations.length === 0 ? 0 : 1;
}

const invokedDirectly =
	process.argv[1] !== undefined && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url));
if (invokedDirectly) {
	process.exitCode = main(process.argv.slice(2));
}
