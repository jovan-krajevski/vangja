/**
 * Ledger record schemas, validation, ids, and projections.
 *
 * Append-only rule: a record is never rewritten. An update appends a new line
 * that repeats the `id`; the current state of an entity is the record with the
 * highest `seq` for that id (`projectLatest`).
 *
 * The Markdown views live in `./views.ts`. No pi/host imports here so `node`
 * can import this module in tests.
 */

import { ledgerInvalid } from "./errors.ts";

export const LEDGER_SCHEMA = "pi-ledger/1";

export const LEDGER_KINDS = [
	"event",
	"task",
	"pr",
	"review",
	"experiment",
	"finding",
	"decision",
	"waiver",
	"amendment",
	"budget",
	"gate",
] as const;

export type LedgerKind = (typeof LEDGER_KINDS)[number];

export const LEDGER_FILES: Record<LedgerKind, string> = {
	event: "EVENTS.jsonl",
	task: "TASKS.jsonl",
	pr: "PRS.jsonl",
	review: "REVIEWS.jsonl",
	experiment: "EXPERIMENTS.jsonl",
	finding: "FINDINGS.jsonl",
	decision: "DECISIONS.jsonl",
	waiver: "WAIVERS.jsonl",
	amendment: "AMENDMENTS.jsonl",
	budget: "BUDGET.jsonl",
	gate: "GATES.jsonl",
};

export const ID_FORMAT: Record<LedgerKind, { prefix: string; width: number } | null> = {
	event: { prefix: "EV", width: 6 },
	task: { prefix: "T", width: 3 },
	pr: { prefix: "PR", width: 3 },
	review: { prefix: "RV", width: 3 },
	experiment: { prefix: "E", width: 4 },
	finding: { prefix: "F", width: 3 },
	decision: { prefix: "D", width: 3 },
	waiver: { prefix: "W", width: 3 },
	amendment: { prefix: "A", width: 3 },
	budget: null,
	gate: { prefix: "G", width: 3 },
};

export const TASK_PRIORITIES = ["P0", "P1", "P2", "P3"] as const;
export const TASK_STATUSES = ["open", "in_progress", "blocked", "review", "merged", "closed", "rejected"] as const;
export const REVIEW_DECISIONS = ["accept", "changes_requested", "reject"] as const;
export const EXPERIMENT_STATUSES = [
	"draft",
	"queued",
	"running",
	"collecting",
	"collected",
	"collection-failed",
	"timed_out",
	"requeue_requested",
	"interrupted_clean",
	"failed",
	"failed-final",
	"cancelled",
] as const;
export const ACTIVE_EXPERIMENT_STATUSES = ["queued", "running", "collecting", "requeue_requested"] as const;

/** Legal experiment lifecycle transitions (plan/06_HPC_AND_COLLECTOR.md §4). */
export const EXPERIMENT_TRANSITIONS: Record<string, readonly string[]> = {
	draft: ["queued"],
	queued: ["running", "cancelled", "failed", "failed-final"],
	running: ["collecting", "timed_out", "interrupted_clean", "failed", "cancelled"],
	collecting: ["collected", "collection-failed"],
	timed_out: ["requeue_requested", "failed-final"],
	interrupted_clean: ["requeue_requested", "failed-final"],
	requeue_requested: ["queued", "cancelled", "failed-final"],
	failed: ["requeue_requested", "failed-final"],
	collected: [],
	"collection-failed": ["requeue_requested", "failed-final"],
	"failed-final": [],
	cancelled: [],
};
export const GATE_KINDS = ["merge", "confirmation", "audit", "submission", "enqueue"] as const;
export const GATE_STATUSES = ["pending", "approved", "rejected"] as const;
export const FINDING_SEVERITIES = ["S0", "S1", "S2", "S3"] as const;
export const FINDING_CONFIDENCES = ["high", "medium", "low"] as const;
export const FINDING_STATUSES = ["verified", "to-verify", "refuted", "unresolved"] as const;

export interface LedgerRecord {
	schema: string;
	kind: LedgerKind;
	id?: string;
	seq: number;
	ts: string;
	actor: string;
	[key: string]: unknown;
}

type FieldType = "string" | "boolean" | "number" | "string[]" | "number[]" | "object";

interface FieldSpec {
	name: string;
	type: FieldType;
	optional?: boolean;
	values?: readonly string[];
	nonEmpty?: boolean;
}

const COMMON_FIELDS: FieldSpec[] = [
	{ name: "schema", type: "string" },
	{ name: "kind", type: "string" },
	{ name: "seq", type: "number" },
	{ name: "ts", type: "string", nonEmpty: true },
	{ name: "actor", type: "string", nonEmpty: true },
];

const RECORD_FIELDS: Record<LedgerKind, FieldSpec[]> = {
	event: [
		{ name: "summary", type: "string", nonEmpty: true },
		{ name: "from", type: "string", optional: true },
		{ name: "to", type: "string", optional: true },
		{ name: "task_id", type: "string", optional: true },
		{ name: "pr_id", type: "string", optional: true },
		{ name: "session_path", type: "string", optional: true },
		{ name: "argv", type: "string[]", optional: true },
	],
	task: [
		{ name: "id", type: "string", nonEmpty: true },
		{ name: "title", type: "string", nonEmpty: true },
		{ name: "origin", type: "string", nonEmpty: true },
		{ name: "priority", type: "string", values: TASK_PRIORITIES },
		{ name: "effort", type: "string", optional: true, values: ["small", "medium", "large"] },
		{ name: "objective", type: "string", nonEmpty: true },
		{ name: "acceptance_criteria", type: "string[]", nonEmpty: true },
		{ name: "deliverables", type: "string[]", nonEmpty: true },
		{ name: "out_of_scope", type: "string", optional: true },
		{ name: "evidence_required", type: "string[]", optional: true },
		{ name: "compute", type: "string", optional: true },
		{ name: "protected_paths", type: "string[]", optional: true },
		{ name: "budget", type: "string", optional: true },
		{ name: "dependencies", type: "string[]", optional: true },
		{ name: "status", type: "string", values: TASK_STATUSES },
	],
	pr: [
		{ name: "id", type: "string", nonEmpty: true },
		{ name: "task_id", type: "string", nonEmpty: true },
		{ name: "head_sha", type: "string", nonEmpty: true },
		{ name: "base_sha", type: "string", nonEmpty: true },
		{ name: "branch", type: "string", nonEmpty: true },
		{ name: "summary", type: "string", nonEmpty: true },
		{ name: "artifacts", type: "string[]" },
		{ name: "tests", type: "string[]" },
		{ name: "claims", type: "string[]" },
		{ name: "hpc_jobs", type: "string[]" },
		{ name: "risks", type: "string", optional: true },
		{ name: "status", type: "string", values: ["open", "review", "merged", "closed"] },
	],
	review: [
		{ name: "id", type: "string", nonEmpty: true },
		{ name: "pr_id", type: "string", nonEmpty: true },
		{ name: "decision", type: "string", values: REVIEW_DECISIONS },
		{ name: "rubric_scores", type: "object", optional: true },
		{ name: "evidence", type: "string[]", nonEmpty: true },
		{ name: "required_changes", type: "string[]" },
		{ name: "residual_risk", type: "string", nonEmpty: true },
	],
	experiment: [
		{ name: "id", type: "string", nonEmpty: true },
		{ name: "task_id", type: "string", nonEmpty: true },
		{ name: "branch", type: "string", nonEmpty: true },
		{ name: "commit_sha", type: "string", nonEmpty: true },
		{ name: "data_hash", type: "string", nonEmpty: true },
		{ name: "command", type: "string", nonEmpty: true },
		{ name: "resources", type: "object" },
		{ name: "checkpoints", type: "object" },
		{ name: "expectedOutputs", type: "string[]", nonEmpty: true },
		{ name: "seeds", type: "number[]", nonEmpty: true },
		{ name: "idempotencyKey", type: "string", nonEmpty: true },
		{ name: "status", type: "string", values: EXPERIMENT_STATUSES },
		{ name: "attempt", type: "number" },
	],
	finding: [
		{ name: "id", type: "string", nonEmpty: true },
		{ name: "severity", type: "string", values: FINDING_SEVERITIES },
		{ name: "confidence", type: "string", values: FINDING_CONFIDENCES },
		{ name: "status", type: "string", values: FINDING_STATUSES },
		{ name: "location", type: "string", nonEmpty: true },
		{ name: "claim", type: "string", nonEmpty: true },
		{ name: "command", type: "string", optional: true },
		{ name: "expected", type: "string", optional: true },
		{ name: "observed", type: "string", optional: true },
		{ name: "consequence", type: "string", nonEmpty: true },
	],
	decision: [
		{ name: "id", type: "string", nonEmpty: true },
		{ name: "title", type: "string", nonEmpty: true },
		{ name: "context", type: "string", nonEmpty: true },
		{ name: "decision", type: "string", nonEmpty: true },
		{ name: "alternatives", type: "string[]", nonEmpty: true },
		{ name: "consequences", type: "string[]", nonEmpty: true },
	],
	waiver: [
		{ name: "id", type: "string", nonEmpty: true },
		{ name: "gate", type: "string", nonEmpty: true },
		{ name: "reason", type: "string", nonEmpty: true },
	],
	amendment: [
		{ name: "id", type: "string", nonEmpty: true },
		{ name: "path", type: "string", nonEmpty: true },
		{ name: "change", type: "string", nonEmpty: true },
		{ name: "justification", type: "string", nonEmpty: true },
	],
	budget: [
		{ name: "sessions", type: "number" },
		{ name: "wallClockMinutes", type: "number" },
		{ name: "costUsd", type: "number" },
	],
	gate: [
		{ name: "id", type: "string", nonEmpty: true },
		{ name: "gate", type: "string", values: GATE_KINDS },
		{ name: "status", type: "string", values: GATE_STATUSES },
		{ name: "note", type: "string" },
	],
};

export function isLedgerKind(value: string): value is LedgerKind {
	return (LEDGER_KINDS as readonly string[]).includes(value);
}

function describeValue(value: unknown): string {
	return Array.isArray(value) ? `array(${value.length})` : typeof value;
}

/** Validate `record` against the schema for `kind`. Throws LEDGER_INVALID. */
export function validateRecord(kind: LedgerKind, record: Record<string, unknown>, file = `${kind}.jsonl`): void {
	const specs = [...COMMON_FIELDS, ...RECORD_FIELDS[kind]];
	for (const spec of specs) {
		const value = record[spec.name];
		if (value === undefined || value === null) {
			if (spec.optional) continue;
			throw ledgerInvalid(file, `missing required field "${spec.name}" for kind "${kind}"`);
		}
		switch (spec.type) {
			case "string":
				if (typeof value !== "string") {
					throw ledgerInvalid(file, `field "${spec.name}" must be a string, got ${describeValue(value)}`);
				}
				break;
			case "boolean":
				if (typeof value !== "boolean") {
					throw ledgerInvalid(file, `field "${spec.name}" must be a boolean, got ${describeValue(value)}`);
				}
				break;
			case "number":
				if (typeof value !== "number" || !Number.isFinite(value)) {
					throw ledgerInvalid(file, `field "${spec.name}" must be a finite number, got ${describeValue(value)}`);
				}
				break;
			case "string[]":
				if (!Array.isArray(value) || value.some((v) => typeof v !== "string")) {
					throw ledgerInvalid(file, `field "${spec.name}" must be an array of strings`);
				}
				break;
			case "number[]":
				if (!Array.isArray(value) || value.some((v) => typeof v !== "number")) {
					throw ledgerInvalid(file, `field "${spec.name}" must be an array of numbers`);
				}
				break;
			case "object":
				if (typeof value !== "object" || value === null || Array.isArray(value)) {
					throw ledgerInvalid(file, `field "${spec.name}" must be a JSON object`);
				}
				break;
		}
		if (spec.nonEmpty) {
			if (typeof value === "string" && value.trim() === "") {
				throw ledgerInvalid(file, `field "${spec.name}" must not be empty`);
			}
			if (Array.isArray(value) && value.length === 0) {
				throw ledgerInvalid(file, `field "${spec.name}" must not be empty`);
			}
		}
		if (spec.values && typeof value === "string" && !spec.values.includes(value)) {
			throw ledgerInvalid(file, `field "${spec.name}" must be one of ${spec.values.join(", ")}, got "${value}"`);
		}
	}
	if (record.kind !== kind) {
		throw ledgerInvalid(file, `field "kind" is "${String(record.kind)}" but the target ledger is "${kind}"`);
	}
	if (kind === "experiment") {
		validateExperimentCheckpoints(record.checkpoints as Record<string, unknown>, file);
	}
}

export function nextId(kind: LedgerKind, existing: Array<{ id?: string }>): string | undefined {
	const format = ID_FORMAT[kind];
	if (!format) return undefined;
	const pattern = new RegExp(`^${format.prefix}-(\\d+)$`);
	let max = 0;
	for (const record of existing) {
		const match = record.id ? pattern.exec(record.id) : null;
		if (match) max = Math.max(max, Number(match[1]));
	}
	return `${format.prefix}-${String(max + 1).padStart(format.width, "0")}`;
}

export function nextSeq(existing: Array<{ seq?: number }>): number {
	let max = 0;
	for (const record of existing) {
		if (typeof record.seq === "number") max = Math.max(max, record.seq);
	}
	return max + 1;
}

/** The checkpoint contract from plan/06_HPC_AND_COLLECTOR.md §5. */
export interface ExperimentCheckpoints {
	dir: string;
	progressFile: string;
	resumeCommand: string;
	flushOnSignal?: boolean;
}

export function validateExperimentCheckpoints(
	checkpoints: Record<string, unknown>,
	file: string,
): ExperimentCheckpoints {
	const required = ["dir", "progressFile", "resumeCommand"];
	for (const field of required) {
		const value = checkpoints[field];
		if (typeof value !== "string" || value.trim() === "") {
			throw ledgerInvalid(
				file,
				`experiment.checkpoints.${field} must be a non-empty string; a job without a checkpoint directory, ` +
					`progress marker, and --resume command is refused (plan/06_HPC_AND_COLLECTOR.md §5)`,
			);
		}
	}
	return checkpoints as unknown as ExperimentCheckpoints;
}

/** Latest record per `id`, ordered by that latest record's `seq`. */
export function projectLatest<T extends { id?: string; seq: number }>(records: T[]): T[] {
	const latest = new Map<string, T>();
	for (const record of records) {
		const key = record.id ?? `seq:${record.seq}`;
		const previous = latest.get(key);
		if (!previous || record.seq >= previous.seq) latest.set(key, record);
	}
	return [...latest.values()].sort((a, b) => a.seq - b.seq);
}

export function activeExperiments(records: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
	return projectLatest(records as Array<{ id?: string; seq: number }>).filter((record) =>
		(ACTIVE_EXPERIMENT_STATUSES as readonly string[]).includes(String(record.status)),
	) as Array<Record<string, unknown>>;
}
