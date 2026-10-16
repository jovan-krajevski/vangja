/**
 * Ledger storage: locked, validated, appended records plus the derived views.
 *
 * `appendRecord` is the single write path: under the file lock it assigns
 * `seq` (and `id` when the kind has one), validates against the record schema,
 * appends one fsynced line, and rewrites the derived Markdown views.
 *
 * No pi imports, so tests can drive the whole store with `node`.
 */

import { VangjaLoopError } from "../../lib/errors.ts";
import { appendLineDurable, readJsonFile, readJsonlFile, withLock, writeFileAtomic } from "../../lib/io.ts";
import { ledgerPath, statePath } from "../../lib/paths.ts";
import {
	activeExperiments,
	isLedgerKind,
	LEDGER_FILES,
	LEDGER_SCHEMA,
	nextId,
	nextSeq,
	projectLatest,
	validateRecord,
	type LedgerKind,
} from "../../lib/records.ts";
import { renderContextMd, renderExperimentsMd, renderPrsMd, renderReviewsMd, renderTasksMd } from "../../lib/views.ts";
import { roleGate, type Capability, type LoopRole } from "../../lib/roles.ts";
import { loadLoopState } from "../../lib/state.ts";

export interface LedgerContext {
	root: string;
	role: LoopRole;
	actor: string;
}

export interface ToolTextResult {
	content: Array<{ type: "text"; text: string }>;
	details: Record<string, unknown>;
	isError?: boolean;
}

export function ok(text: string, details: Record<string, unknown> = {}): ToolTextResult {
	return { content: [{ type: "text", text }], details };
}

export function fail(message: string, details: Record<string, unknown> = {}): ToolTextResult {
	return { content: [{ type: "text", text: message }], details: { ...details, error: message }, isError: true };
}

export function failText(message: string, details: Record<string, unknown> = {}): ToolTextResult {
	return fail(message, details);
}

export function summarise(record: Record<string, unknown>): string {
	return JSON.stringify(record);
}

export function requireString(args: Record<string, unknown>, name: string, hint: string): string {
	const value = args[name];
	if (typeof value !== "string" || value.trim() === "") {
		throw new VangjaLoopError("LEDGER_INVALID", `argument "${name}" must be a non-empty string; ${hint}`);
	}
	return value;
}

export function requireObject(args: Record<string, unknown>, name: string, hint: string): Record<string, unknown> {
	const value = args[name];
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		throw new VangjaLoopError("LEDGER_INVALID", `argument "${name}" must be a JSON object; ${hint}`);
	}
	return value as Record<string, unknown>;
}

export function checkCapability(ctx: LedgerContext, capability: Capability): void {
	const decision = roleGate(capability, ctx.role);
	if (!decision.allowed) throw new VangjaLoopError("ROLE_DENIED", decision.reason ?? `role ${ctx.role} may not ${capability}`);
}

export function readLedger(root: string, kind: LedgerKind): Array<Record<string, unknown>> {
	return readJsonlFile<Record<string, unknown>>(ledgerPath(root, LEDGER_FILES[kind]));
}

/**
 * Append one validated record. Under the lock: assign seq (and id when the kind
 * has one and the caller did not supply it), validate, append+fsync, then
 * rewrite the derived views.
 */
export function appendRecord(
	root: string,
	kind: LedgerKind,
	raw: Record<string, unknown>,
	ctx: LedgerContext,
): Record<string, unknown> {
	const file = ledgerPath(root, LEDGER_FILES[kind]);
	const lockPath = `${file}.lock`;
	return withLock(lockPath, () => {
		const existing = readJsonlFile<Record<string, unknown>>(file);
		const id = typeof raw.id === "string" && raw.id !== "" ? raw.id : nextId(kind, existing);
		const record: Record<string, unknown> = {
			...raw,
			schema: LEDGER_SCHEMA,
			kind,
			seq: nextSeq(existing),
			ts: new Date().toISOString(),
			actor: ctx.actor,
		};
		if (id !== undefined) record.id = id;
		validateRecord(kind, record, file);
		appendLineDurable(file, JSON.stringify(record));
		updateViews(root, kind);
		return record;
	});
}

function updateViews(root: string, kind: LedgerKind): void {
	const write = (name: string, content: string) => writeFileAtomic(statePath(root, name), content);
	switch (kind) {
		case "task":
			write("TASKS.md", renderTasksMd(readLedger(root, "task")));
			break;
		case "experiment":
			write("EXPERIMENTS.md", renderExperimentsMd(readLedger(root, "experiment")));
			break;
		case "review":
			write("REVIEWS.md", renderReviewsMd(readLedger(root, "review")));
			break;
		case "pr":
			write("PRS.md", renderPrsMd(readLedger(root, "pr")));
			break;
		default:
			break;
	}
	write("CONTEXT.md", renderContext(root));
}

export function renderContext(root: string): string {
	const loaded = loadLoopState(root);
	const budget = readJsonFile<Record<string, unknown>>(statePath(root, "BUDGET.json")) ?? {};
	return renderContextMd({
		state: loaded.state as unknown as Record<string, unknown>,
		budget,
		tasks: readLedger(root, "task"),
		experiments: readLedger(root, "experiment"),
		reviews: readLedger(root, "review"),
		prs: readLedger(root, "pr"),
		gates: readLedger(root, "gate"),
		events: readLedger(root, "event"),
	});
}

export { activeExperiments };
