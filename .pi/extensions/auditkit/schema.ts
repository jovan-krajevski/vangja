/**
 * The machine-checkable finding schema (plan/07_AUDIT_HARNESS.md §4).
 *
 * A finding is accepted only when it carries the nine evidence fields *and* a
 * command that reproduces the observation. Prose is not a finding: a record
 * whose command is "static reading" / "none" / "manual inspection" is rejected
 * here, because the verification phase must be able to re-run it.
 *
 * No pi imports: usable from `node --test`, from the tools, and from the
 * calibration scorer.
 */

import * as crypto from "node:crypto";

export const SEVERITIES = ["S0", "S1", "S2", "S3"] as const;
export type Severity = (typeof SEVERITIES)[number];

export const CONFIDENCES = ["high", "medium", "low"] as const;
export type Confidence = (typeof CONFIDENCES)[number];

export const STATUSES = ["verified", "to-verify", "refuted", "unresolved"] as const;
export type Status = (typeof STATUSES)[number];

export const OWNERS = ["code", "paper", "protocol", "docs", "env"] as const;
export type Owner = (typeof OWNERS)[number];

export const WORKSTREAMS = ["M", "S", "P", "D", "I", "R", "H", "E", "C", "L"] as const;
export type Workstream = (typeof WORKSTREAMS)[number];

export const PHASES = ["finder", "verify", "synth", "adjudicate", "canary"] as const;
export type Phase = (typeof PHASES)[number];

/** Workstreams whose S1 findings are claim-defining (the §7 blocking rule). */
export const CORE_S1_WORKSTREAMS: readonly Workstream[] = ["M", "S", "P", "C"];

export interface Finding {
	id: string;
	title: string;
	location: string;
	snapshot: string;
	command: string;
	expected: string;
	observed: string;
	consequence: string;
	severity: Severity;
	confidence: Confidence;
	status: Status;
	reproducible: string;
	evidence: string[];
	cross_refs: string[];
	owner: Owner;
	found_in_cycle: number;
	/** Register provenance. Synthesis may not add primary findings. */
	phase?: Phase;
	author_agent?: string;
	/** Set by the verifier when a finding has been fixed. */
	fixed?: boolean;
	/** Required for `unresolved`: the author question tag. */
	question_tag?: "blocking" | "non-blocking";
	[key: string]: unknown;
}

export interface ValidationResult {
	ok: boolean;
	errors: string[];
	warnings: string[];
	normalized?: Finding;
}

const ID_RE = /^(?:WS-[A-Z]|[A-Z])-\d{3}$/;
const WS_FROM_ID_RE = /^(?:WS-)?([A-Z])-\d{3}$/;
const SEED_RE = /^(yes\b.*|no\b.+)$/i;

/**
 * Commands that are not commands. The historical audit allowed "static reading
 * of lines X–Y"; this harness explicitly does not (plan/07 §1 principle 2).
 */
const NON_COMMAND_RE =
	/^(static(\s+read(ing)?)?|reading|manual(\s+inspection)?|inspection|none|n\/?a|tbd|todo|see\s+above|as\s+discussed)\b/i;

const SHELL_FIRST_TOKENS = new Set([
	"uv",
	"pytest",
	"python",
	"python3",
	"node",
	"bash",
	"sh",
	"git",
	"grep",
	"rg",
	"find",
	"ls",
	"cat",
	"head",
	"tail",
	"sed",
	"awk",
	"jq",
	"diff",
	"cmp",
	"sha256sum",
	"shasum",
	"md5sum",
	"make",
	"npm",
	"npx",
	"pip",
	"uvx",
	"sqlite3",
	"test",
	"true",
	"false",
	"wc",
	"sort",
	"uniq",
	"xargs",
	"cp",
	"mv",
	"rm",
	"mkdir",
	"touch",
	"curl",
	"wget",
	"tar",
	"unzip",
	"zip",
]);

/**
 * A command must start with an executable-ish token (or a path), and it must
 * look like an invocation rather than a sentence. This keeps `grep -n x file`,
 * `./run.sh`, `uv run pytest …` and `/usr/bin/time …` valid while rejecting
 * "the sign is wrong".
 */
export function isExecutableCommand(command: string): boolean {
	const trimmed = command.trim();
	if (trimmed.length < 4) return false;
	if (NON_COMMAND_RE.test(trimmed)) return false;
	const first = trimmed.split(/\s+/)[0];
	const base = first.split("/").pop() ?? first;
	if (first.startsWith("./") || first.startsWith("/") || first.startsWith("../")) return true;
	if (SHELL_FIRST_TOKENS.has(base)) return true;
	// Allow a bare script name with an extension (`check.py`, `run.sh`).
	if (/^[\w.-]+\.(py|sh|js|ts|mjs|pl|rb)$/.test(base)) return true;
	return false;
}

export function normalizeLocation(location: string): string {
	return location
		.trim()
		.toLowerCase()
		.replace(/\\/g, "/")
		.replace(/^\.\//, "")
		.replace(/:\d+(?:[-–]\d+)?(?::\d+)?$/, "")
		.replace(/\s+/g, " ")
		.replace(/[`'"]/g, "");
}

export function normalizeTitle(title: string): string {
	return title
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, " ")
		.trim()
		.replace(/\s+/g, " ");
}

/** Dedupe key: normalized location + title (plan/07 §4 / prompt deliverable 4). */
export function dedupeKey(finding: Pick<Finding, "location" | "title">): string {
	return crypto
		.createHash("sha256")
		.update(`${normalizeLocation(finding.location)}|${normalizeTitle(finding.title)}`)
		.digest("hex")
		.slice(0, 16);
}

export function workstreamOfId(id: string): Workstream | null {
	const match = WS_FROM_ID_RE.exec(id);
	return match && (WORKSTREAMS as readonly string[]).includes(match[1]) ? (match[1] as Workstream) : null;
}

function requireString(
	record: Record<string, unknown>,
	field: string,
	errors: string[],
	minLength = 1,
): string | undefined {
	const value = record[field];
	if (typeof value !== "string" || value.trim().length < minLength) {
		errors.push(`${field}: required non-empty string${minLength > 1 ? ` (min ${minLength} chars)` : ""}`);
		return undefined;
	}
	return value;
}

/** Validate one finding against the nine-field schema plus enums. */
export function validateFinding(input: unknown, options: { expectedWorkstream?: Workstream } = {}): ValidationResult {
	const errors: string[] = [];
	const warnings: string[] = [];
	if (typeof input !== "object" || input === null || Array.isArray(input)) {
		return { ok: false, errors: ["finding: expected an object"], warnings };
	}
	const record = input as Record<string, unknown>;

	const id = requireString(record, "id", errors);
	const title = requireString(record, "title", errors);
	const location = requireString(record, "location", errors, 2);
	const snapshot = requireString(record, "snapshot", errors);
	const command = requireString(record, "command", errors, 3);
	const expected = requireString(record, "expected", errors);
	const observed = requireString(record, "observed", errors);
	const consequence = requireString(record, "consequence", errors);
	const reproducible = requireString(record, "reproducible", errors, 2);

	if (id) {
		if (!ID_RE.test(id)) errors.push(`id: "${id}" is not <WS>-NNN (e.g. S-001 or WS-S-001)`);
		const ws = workstreamOfId(id);
		if (!ws) errors.push(`id: "${id}" does not name a known workstream (${WORKSTREAMS.join("/")})`);
		if (options.expectedWorkstream && ws && ws !== options.expectedWorkstream) {
			errors.push(`id: "${id}" belongs to workstream ${ws}, not ${options.expectedWorkstream}`);
		}
	}

	const severity = record.severity;
	if (typeof severity !== "string" || !(SEVERITIES as readonly string[]).includes(severity)) {
		errors.push(`severity: must be one of ${SEVERITIES.join("|")}`);
	}
	const confidence = record.confidence;
	if (typeof confidence !== "string" || !(CONFIDENCES as readonly string[]).includes(confidence)) {
		errors.push(`confidence: must be one of ${CONFIDENCES.join("|")}`);
	}
	const status = record.status;
	if (typeof status !== "string" || !(STATUSES as readonly string[]).includes(status)) {
		errors.push(`status: must be one of ${STATUSES.join("|")}`);
	}
	if (record.owner !== undefined) {
		if (typeof record.owner !== "string" || !(OWNERS as readonly string[]).includes(record.owner)) {
			errors.push(`owner: must be one of ${OWNERS.join("|")} when present`);
		}
	}
	if (record.phase !== undefined) {
		if (typeof record.phase !== "string" || !(PHASES as readonly string[]).includes(record.phase)) {
			errors.push(`phase: must be one of ${PHASES.join("|")} when present`);
		}
	}

	if (command && !isExecutableCommand(command)) {
		errors.push(
			`command: "${command.slice(0, 60)}" is not an executable command — no command → not a finding (plan/07 §4)`,
		);
	}
	if (reproducible && (/\.\s*|\.$/.test(reproducible) || !SEED_RE.test(reproducible))) {
		errors.push('reproducible: must be "yes" (optionally explained) or "no: <why>"');
	}

	// A verified finding must be reproducible and not a low-confidence hypothesis.
	if (status === "verified") {
		if (confidence === "low") {
			errors.push("status: verified cannot carry confidence low (low = hypothesis needing data/run)");
		}
		if (reproducible && !/^yes\b/i.test(reproducible)) {
			errors.push("status: verified requires reproducible=yes (a non-reproducible observation is to-verify)");
		}
	}
	if (status === "unresolved" && record.question_tag !== "blocking" && record.question_tag !== "non-blocking") {
		errors.push("question_tag: unresolved findings need [blocking] or [non-blocking] in QUESTIONS_FOR_AUTHORS.md");
	}

	const evidence = record.evidence;
	if (!Array.isArray(evidence) || evidence.length === 0 || evidence.some((item) => typeof item !== "string")) {
		errors.push("evidence: required array of artifact ids / file:line / output hashes (at least one)");
	}
	const crossRefs = record.cross_refs;
	if (crossRefs !== undefined) {
		if (!Array.isArray(crossRefs) || crossRefs.some((item) => typeof item !== "string")) {
			errors.push("cross_refs: must be an array of finding ids when present");
		} else {
			for (const ref of crossRefs) {
				if (!ID_RE.test(ref)) warnings.push(`cross_refs: "${ref}" is not a finding id`);
			}
		}
	}
	if (record.found_in_cycle !== undefined && (typeof record.found_in_cycle !== "number" || record.found_in_cycle < 0)) {
		errors.push("found_in_cycle: must be a non-negative number when present");
	}
	if (title && title.length > 200) warnings.push("title: longer than 200 characters; keep it to one line");

	if (errors.length > 0) return { ok: false, errors, warnings };

	const normalized: Finding = {
		id: id as string,
		title: title as string,
		location: location as string,
		snapshot: snapshot as string,
		command: command as string,
		expected: expected as string,
		observed: observed as string,
		consequence: consequence as string,
		severity: severity as Severity,
		confidence: confidence as Confidence,
		status: status as Status,
		reproducible: reproducible as string,
		evidence: evidence as string[],
		cross_refs: (crossRefs as string[] | undefined) ?? [],
		owner: (record.owner as Owner | undefined) ?? inferOwner(location as string, consequence as string),
		found_in_cycle: (record.found_in_cycle as number | undefined) ?? 0,
		phase: (record.phase as Phase | undefined) ?? "finder",
		author_agent: typeof record.author_agent === "string" ? record.author_agent : undefined,
		fixed: record.fixed === true,
		question_tag:
			record.question_tag === "blocking" || record.question_tag === "non-blocking" ? record.question_tag : undefined,
	};
	// Provenance fields the report layer uses; they are not part of the nine-field
	// schema but must survive validation (a synthesizer record's `cites`, a
	// finding's `root_issue`, the canary's `probe`).
	for (const key of ["cites", "root_issue", "blocking", "probe", "verified_id"] as const) {
		if (record[key] !== undefined) normalized[key] = record[key];
	}
	return { ok: true, errors, warnings, normalized };
}

/** Owner inference used only when the finder did not state one. */
export function inferOwner(location: string, consequence = ""): Owner {
	const text = `${location} ${consequence}`.toLowerCase();
	if (/paper\/|\.tex\b|manuscript/.test(text)) return "paper";
	if (/protocol|confirmation_origins|dev_origins|amendment/.test(text)) return "protocol";
	if (/readme|docs\/|\.md\b|docstring|notebook/.test(text)) return "docs";
	if (/pyproject|uv\.lock|singularity|environment|requirements|\.github/.test(text)) return "env";
	return "code";
}

export function severityRank(severity: string): number {
	return SEVERITIES.indexOf(severity as Severity);
}

export function isConfirmed(finding: Finding): boolean {
	return finding.status === "verified" && finding.confidence !== "low";
}

// ---------------------------------------------------------------------------
// Register: dedupe by normalized location+title, never duplicate a body
// ---------------------------------------------------------------------------

export interface RegisteredFinding extends Finding {
	/** Key used to detect duplicates across workstreams. */
	dedupe_key?: string;
	/** Incoming duplicate ids folded into this finding. */
	duplicates?: string[];
	history?: Array<{ at: string; status?: Status; confidence?: Confidence; note?: string }>;
}

export interface RegisterInput {
	cycle: number;
	workstream: Workstream;
	findings: unknown[];
	phase?: Phase;
	author_agent?: string;
	now?: string;
}

export interface RegisterResult {
	ok: boolean;
	added: string[];
	updated: string[];
	duplicates: Array<{ id: string; duplicate_of: string; key: string }>;
	rejected: Array<{ index: number; errors: string[] }>;
	register: RegisteredFinding[];
}

/**
 * Merge incoming findings into an existing register.
 *
 * - Same id → the later record wins for status/confidence/observed/evidence and
 *   the previous values are recorded in `history` (verdicts are append-only in
 *   spirit: nothing is silently erased).
 * - Different id, same normalized location+title → the incoming finding is NOT
 *   added; it becomes a `cross_ref` on the canonical finding and is reported in
 *   `duplicates`.
 */
export function registerFindings(existing: RegisteredFinding[], input: RegisterInput): RegisterResult {
	const now = input.now ?? new Date().toISOString();
	const register = existing.map((finding) => ({ ...finding, cross_refs: [...(finding.cross_refs ?? [])] }));
	const byId = new Map(register.map((finding) => [finding.id, finding]));
	const byKey = new Map<string, RegisteredFinding>();
	for (const finding of register) {
		const key = finding.dedupe_key ?? dedupeKey(finding);
		finding.dedupe_key = key;
		if (!byKey.has(key)) byKey.set(key, finding);
	}

	const result: RegisterResult = {
		ok: true,
		added: [],
		updated: [],
		duplicates: [],
		rejected: [],
		register,
	};

	input.findings.forEach((raw, index) => {
		const validation = validateFinding(raw, { expectedWorkstream: input.workstream });
		if (!validation.ok || !validation.normalized) {
			result.rejected.push({ index, errors: validation.errors });
			result.ok = false;
			return;
		}
		const finding = validation.normalized;
		if (input.phase && !finding.phase) finding.phase = input.phase;
		if (input.author_agent) finding.author_agent = input.author_agent;
		if (!finding.found_in_cycle) finding.found_in_cycle = input.cycle;
		const key = dedupeKey(finding);
		finding.dedupe_key = key;

		const sameId = byId.get(finding.id);
		if (sameId) {
			const history = sameId.history ?? [];
			history.push({
				at: now,
				status: sameId.status,
				confidence: sameId.confidence,
				note: `superseded by ${finding.status}/${finding.confidence}`,
			});
			const merged: RegisteredFinding = {
				...sameId,
				...finding,
				history,
				// Preserve phase provenance of the primary record.
				phase: sameId.phase ?? finding.phase,
			};
			const at = register.findIndex((item) => item.id === sameId.id);
			register[at] = merged;
			byId.set(merged.id, merged);
			byKey.set(key, merged);
			result.updated.push(merged.id);
			return;
		}

		const canonical = byKey.get(key);
		if (canonical) {
			if (!canonical.cross_refs.includes(finding.id)) canonical.cross_refs.push(finding.id);
			canonical.duplicates = [...(canonical.duplicates ?? []), finding.id];
			result.duplicates.push({ id: finding.id, duplicate_of: canonical.id, key });
			return;
		}

		register.push(finding);
		byId.set(finding.id, finding);
		byKey.set(key, finding);
		result.added.push(finding.id);
	});
	return result;
}

export function summarizeFindings(findings: readonly Finding[]) {
	const bySeverity: Record<string, number> = { S0: 0, S1: 0, S2: 0, S3: 0 };
	const byStatus: Record<string, number> = { verified: 0, "to-verify": 0, refuted: 0, unresolved: 0 };
	const byWorkstream: Record<string, number> = {};
	for (const finding of findings) {
		bySeverity[finding.severity] = (bySeverity[finding.severity] ?? 0) + 1;
		byStatus[finding.status] = (byStatus[finding.status] ?? 0) + 1;
		const ws = workstreamOfId(finding.id) ?? "?";
		byWorkstream[ws] = (byWorkstream[ws] ?? 0) + 1;
	}
	return { count: findings.length, bySeverity, byStatus, byWorkstream };
}
