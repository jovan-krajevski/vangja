/**
 * Rubric trees: load, validate, autofill deterministic leaves from gate results,
 * score LLM leaves with a mandatory evidence pointer, and render.
 *
 * Contract (plan/05_EXTENSIONS_AND_TOOLS.md §6):
 * - a tree is branches → leaves; leaf `type` is `binary`, `scale`, or `auto`;
 * - weights sum to 1 for the branches and inside every branch;
 * - every leaf carries a one-line grading criterion;
 * - `auto` names a deterministic result (`tests_pass`, `hash_ok`, …);
 * - a scored leaf without an evidence pointer is rejected.
 *
 * No pi imports: usable from `node --test`.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { GRADES, type DecisionMapping } from "../auditkit/grade.ts";

export const RUBRIC_KINDS = ["rubric", "decision"] as const;
export const LEAF_TYPES = ["binary", "scale", "auto"] as const;
export type LeafType = (typeof LEAF_TYPES)[number];

/** Deterministic leaves and where their value comes from. */
export const AUTO_SOURCES: Record<string, string> = {
	tests_pass: "gate:tests",
	hash_ok: "gate:artifacts-complete#hash_ok",
	protocol_hashes_ok: "gate:protocol-hashes",
	confirmation_frozen: "gate:confirmation-frozen",
	no_test_selection: "gate:no-test-selection",
	test_diff_clean: "gate:test-diff",
	no_failures_hidden: "gate:no-failures-hidden",
	context_excluded: "gate:context-excluded",
	artifacts_complete: "gate:artifacts-complete",
	reproduce_headline: "gate:reproduce-headline",
	traceability_ok: "gate:traceability",
	env_pinned: "gate:env-pinned",
	gaming_scan_clean: "gate:gaming-scan",
	canary_detected: "gate:canary",
	calibration_ok: "calibration:ok",
	guards_clean: "guards:ok",
	artifact_hashes_ok: "artifact:hash_ok",
};

export interface RubricLeaf {
	id: string;
	text: string;
	criterion: string;
	type: LeafType;
	auto?: string;
	weight: number;
	levels?: number[];
	/** Filled by rubric_autofill / rubric_score. */
	score?: number | null;
	evidence?: string;
	note?: string;
}

export interface RubricBranch {
	id: string;
	title: string;
	weight: number;
	children: RubricLeaf[];
}

export interface RubricTree {
	schema: string;
	id: string;
	title: string;
	kind: "rubric";
	scale?: string[];
	branches: RubricBranch[];
}

export interface DecisionRubric {
	schema: string;
	id: string;
	title: string;
	kind: "decision";
	scale: string[];
	max_cycles: number;
	mapping: DecisionMapping["mapping"];
	blocking_rules?: Record<string, string>;
}

export type AnyRubric = RubricTree | DecisionRubric;

export interface RubricValidation {
	ok: boolean;
	kind: "rubric" | "decision" | null;
	errors: string[];
	warnings: string[];
	ids: string[];
	branches: number;
	leaves: number;
}

const WEIGHT_TOLERANCE = 1e-9;

function isWeight(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function checkWeights(items: Array<{ id: string; weight: number }>, where: string, errors: string[]): void {
	const sum = items.reduce((total, item) => total + item.weight, 0);
	if (Math.abs(sum - 1) > WEIGHT_TOLERANCE) {
		errors.push(`${where}: weights sum to ${sum.toFixed(12)}, expected 1 (±${WEIGHT_TOLERANCE})`);
	}
}

function validateLeaf(leaf: unknown, prefix: string, seen: Set<string>, errors: string[], warnings: string[]): void {
	if (typeof leaf !== "object" || leaf === null) {
		errors.push(`${prefix}: leaf must be an object`);
		return;
	}
	const record = leaf as Record<string, unknown>;
	const id = typeof record.id === "string" ? record.id : "";
	if (!id) errors.push(`${prefix}: leaf id is required`);
	if (id && seen.has(id)) errors.push(`${prefix}: duplicate leaf id "${id}"`);
	if (id) seen.add(id);
	if (typeof record.text !== "string" || record.text.trim() === "") errors.push(`${prefix}${id}: text is required`);
	if (typeof record.criterion !== "string" || record.criterion.trim() === "") {
		errors.push(`${prefix}${id}: criterion is required (one line describing how to grade it)`);
	} else if ((record.criterion as string).length > 240) {
		errors.push(`${prefix}${id}: criterion must be one line (<=240 chars)`);
	}
	const type = record.type;
	if (typeof type !== "string" || !(LEAF_TYPES as readonly string[]).includes(type)) {
		errors.push(`${prefix}${id}: type must be one of ${LEAF_TYPES.join("|")}`);
	}
	if (!isWeight(record.weight)) errors.push(`${prefix}${id}: weight must be a positive number`);
	if (record.levels !== undefined) {
		if (!Array.isArray(record.levels) || record.levels.some((level) => typeof level !== "number")) {
			errors.push(`${prefix}${id}: levels must be an array of numbers when present`);
		}
	}
	if (type === "auto") {
		if (typeof record.auto !== "string" || !(record.auto in AUTO_SOURCES)) {
			errors.push(
				`${prefix}${id}: auto leaf needs one of ${Object.keys(AUTO_SOURCES).join(", ")}`,
			);
		}
	} else if (record.auto !== undefined) {
		warnings.push(`${prefix}${id}: auto is set but type is ${type}; it will be ignored`);
	}
}

export function validateRubric(input: unknown): RubricValidation {
	const errors: string[] = [];
	const warnings: string[] = [];
	const ids: string[] = [];
	const result: RubricValidation = {
		ok: false,
		kind: null,
		errors,
		warnings,
		ids,
		branches: 0,
		leaves: 0,
	};
	if (typeof input !== "object" || input === null) {
		errors.push("rubric: expected an object");
		return result;
	}
	const record = input as Record<string, unknown>;
	if (record.schema !== "vangja-rubric/1") errors.push('rubric: schema must be "vangja-rubric/1"');
	if (typeof record.id !== "string" || record.id === "") errors.push("rubric: id is required");
	if (typeof record.title !== "string" || record.title === "") errors.push("rubric: title is required");
	const kind = record.kind;
	if (kind !== "rubric" && kind !== "decision") {
		errors.push('rubric: kind must be "rubric" or "decision"');
		return result;
	}
	result.kind = kind;

	if (kind === "decision") {
		const scale = record.scale;
		if (!Array.isArray(scale) || scale.length !== GRADES.length || scale.some((value, index) => value !== GRADES[index])) {
			errors.push(`decision: scale must be exactly "${GRADES.join(" · ")}"`);
		}
		if (record.max_cycles !== 5) errors.push("decision: max_cycles must be 5 (plan/02 §5.4)");
		const mapping = record.mapping;
		if (typeof mapping !== "object" || mapping === null) {
			errors.push("decision: mapping is required");
			return result;
		}
		const table = mapping as Record<string, unknown>;
		for (const grade of GRADES) {
			const entry = table[grade] as Record<string, unknown> | undefined;
			if (!entry || typeof entry !== "object") {
				errors.push(`decision: mapping is missing grade "${grade}"`);
				continue;
			}
			if (typeof entry.action !== "string" || entry.action === "") errors.push(`decision: ${grade}.action is required`);
			if (typeof entry.reaudit !== "boolean") errors.push(`decision: ${grade}.reaudit must be boolean`);
			if (typeof entry.done !== "boolean") errors.push(`decision: ${grade}.done must be boolean`);
		}
		const expected: Record<string, string> = {
			"strong reject": "reopen-p0",
			reject: "reopen-p0",
			"weak reject": "fix-s0-s1",
			borderline: "fix-s0-s1",
			"weak accept": "fix-feasible",
			accept: "fix-blocking-or-done",
			"strong accept": "fix-blocking-or-done",
		};
		for (const [grade, action] of Object.entries(expected)) {
			const entry = table[grade] as Record<string, unknown> | undefined;
			if (entry && entry.action !== action) {
				errors.push(`decision: ${grade}.action must be "${action}" (plan/02 §5.4), found "${String(entry.action)}"`);
			}
		}
		result.ok = errors.length === 0;
		return result;
	}

	if (!Array.isArray(record.branches) || record.branches.length === 0) {
		errors.push("rubric: branches must be a non-empty array");
		return result;
	}
	const branches = record.branches as unknown[];
	const branchWeights: Array<{ id: string; weight: number }> = [];
	const branchIds = new Set<string>();
	const seenLeaves = new Set<string>();

	branches.forEach((rawBranch, index) => {
		if (typeof rawBranch !== "object" || rawBranch === null) {
			errors.push(`branches[${index}]: expected an object`);
			return;
		}
		const branch = rawBranch as Record<string, unknown>;
		const bid = typeof branch.id === "string" ? branch.id : `branch-${index}`;
		if (typeof branch.id !== "string" || branch.id === "") errors.push(`branches[${index}]: id is required`);
		if (branchIds.has(bid)) errors.push(`branches[${index}]: duplicate branch id "${bid}"`);
		branchIds.add(bid);
		ids.push(bid);
		if (typeof branch.title !== "string" || branch.title === "") errors.push(`branch ${bid}: title is required`);
		if (!isWeight(branch.weight)) errors.push(`branch ${bid}: weight must be a positive number`);
		else branchWeights.push({ id: bid, weight: branch.weight });
		const children = branch.children;
		if (!Array.isArray(children) || children.length === 0) {
			errors.push(`branch ${bid}: children must be a non-empty array`);
			return;
		}
		const childWeights: Array<{ id: string; weight: number }> = [];
		for (const child of children) {
			validateLeaf(child, `branch ${bid}: `, seenLeaves, errors, warnings);
			const leaf = child as Record<string, unknown>;
			if (typeof leaf.id === "string") ids.push(leaf.id);
			if (isWeight(leaf.weight) && typeof leaf.id === "string") childWeights.push({ id: leaf.id, weight: leaf.weight });
		}
		result.leaves += children.length;
		if (childWeights.length === children.length) checkWeights(childWeights, `branch ${bid}`, errors);
	});
	result.branches = branches.length;
	if (branchWeights.length === branches.length) checkWeights(branchWeights, "top-level branches", errors);
	result.ok = errors.length === 0;
	return result;
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

export function rubricPath(root: string, nameOrPath: string): string {
	if (nameOrPath.includes("/") || nameOrPath.endsWith(".json")) {
		return path.isAbsolute(nameOrPath) ? nameOrPath : path.resolve(root, nameOrPath);
	}
	return path.join(root, ".pi", "rubrics", `${nameOrPath}.json`);
}

export function loadRubric(root: string, nameOrPath: string): AnyRubric {
	const file = rubricPath(root, nameOrPath);
	if (!fs.existsSync(file)) throw new Error(`rubric not found: ${file}`);
	let parsed: unknown;
	try {
		parsed = JSON.parse(fs.readFileSync(file, "utf8"));
	} catch (error) {
		throw new Error(`cannot parse rubric ${file}: ${(error as Error).message}`);
	}
	const validation = validateRubric(parsed);
	if (!validation.ok) throw new Error(`invalid rubric ${file}:\n- ${validation.errors.join("\n- ")}`);
	return parsed as AnyRubric;
}

// ---------------------------------------------------------------------------
// Autofill
// ---------------------------------------------------------------------------

export interface GateView {
	id: string;
	status: "pass" | "fail" | "skip" | "error";
	detail?: string;
	hash_ok?: boolean;
}

export interface AutofillInput {
	gates?: { gates: GateView[] } | null;
	calibration?: { calibrated: boolean; recall: number; rejected_confirmed: number } | null;
	guards?: { ok: boolean } | null;
	artifacts?: { hash_ok: boolean } | null;
}

export interface AutofillResult {
	ok: boolean;
	tree: RubricTree;
	filled: string[];
	unresolved: Array<{ id: string; reason: string }>;
	warnings: string[];
}

function leafSource(leaf: RubricLeaf, input: AutofillInput): { score: number | null; evidence: string; reason?: string } {
	const source = AUTO_SOURCES[leaf.auto ?? ""];
	if (!source) return { score: null, evidence: "", reason: `unknown auto name ${leaf.auto}` };
	if (source.startsWith("gate:")) {
		const [gateId, sub] = source.slice("gate:".length).split("#");
		const gate = input.gates?.gates.find((entry) => entry.id === gateId);
		if (!gate) return { score: null, evidence: "", reason: `gate ${gateId} did not run` };
		if (sub === "hash_ok") {
			const hashOk = gate.hash_ok ?? gate.status === "pass";
			return { score: hashOk ? 1 : 0, evidence: `gate:${gateId}#hash_ok=${hashOk}` };
		}
		if (gate.status === "pass") return { score: 1, evidence: `gate:${gateId}:pass` };
		if (gate.status === "fail" || gate.status === "error") return { score: 0, evidence: `gate:${gateId}:${gate.status}` };
		return { score: null, evidence: `gate:${gateId}:skip`, reason: `gate ${gateId} skipped — needs adjudication` };
	}
	if (source === "calibration:ok") {
		if (!input.calibration) return { score: null, evidence: "", reason: "no calibration result" };
		return {
			score: input.calibration.calibrated ? 1 : 0,
			evidence: `calibration:recall=${input.calibration.recall.toFixed(3)},rejected=${input.calibration.rejected_confirmed}`,
		};
	}
	if (source === "guards:ok") {
		if (!input.guards) return { score: null, evidence: "", reason: "no guards result" };
		return { score: input.guards.ok ? 1 : 0, evidence: `guards:ok=${input.guards.ok}` };
	}
	if (source === "artifact:hash_ok") {
		if (!input.artifacts) return { score: null, evidence: "", reason: "no artifact verification result" };
		return { score: input.artifacts.hash_ok ? 1 : 0, evidence: `artifact:hash_ok=${input.artifacts.hash_ok}` };
	}
	return { score: null, evidence: "", reason: `unsupported source ${source}` };
}

export function autofillRubric(tree: RubricTree, input: AutofillInput): AutofillResult {
	const copy: RubricTree = JSON.parse(JSON.stringify(tree)) as RubricTree;
	const filled: string[] = [];
	const unresolved: AutofillResult["unresolved"] = [];
	const warnings: string[] = [];
	for (const branch of copy.branches) {
		for (const leaf of branch.children) {
			if (leaf.type !== "auto") continue;
			const result = leafSource(leaf, input);
			leaf.score = result.score;
			leaf.evidence = result.evidence;
			if (result.score === null) unresolved.push({ id: leaf.id, reason: result.reason ?? "unresolved" });
			else filled.push(leaf.id);
			if (result.reason && result.score === null) warnings.push(`${leaf.id}: ${result.reason}`);
		}
	}
	return { ok: unresolved.length === 0, tree: copy, filled, unresolved, warnings };
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

export interface ScoreEntry {
	leaf_id: string;
	score: number;
	evidence: string;
	note?: string;
}

export interface BranchScore {
	id: string;
	title: string;
	weight: number;
	score: number | null;
	leaves: Array<{ id: string; type: string; weight: number; score: number | null; evidence: string }>;
}

export interface RubricScoreResult {
	ok: boolean;
	errors: string[];
	total: number | null;
	complete: boolean;
	branches: BranchScore[];
	unresolved: string[];
	scored: string[];
}

function round(value: number): number {
	return Math.round(value * 1e12) / 1e12;
}

export function scoreRubric(tree: RubricTree, scores: readonly ScoreEntry[]): RubricScoreResult {
	const errors: string[] = [];
	const copy: RubricTree = JSON.parse(JSON.stringify(tree)) as RubricTree;
	const leaves = new Map<string, { leaf: RubricLeaf; branch: RubricBranch }>();
	for (const branch of copy.branches) {
		for (const leaf of branch.children) leaves.set(leaf.id, { leaf, branch });
	}

	const seen = new Set<string>();
	for (const entry of scores) {
		if (!entry || typeof entry.leaf_id !== "string") {
			errors.push("score entry: leaf_id is required");
			continue;
		}
		if (seen.has(entry.leaf_id)) errors.push(`${entry.leaf_id}: scored more than once`);
		seen.add(entry.leaf_id);
		const target = leaves.get(entry.leaf_id);
		if (!target) {
			errors.push(`${entry.leaf_id}: unknown leaf`);
			continue;
		}
		if (target.leaf.type === "auto") {
			errors.push(`${entry.leaf_id}: auto leaf is filled by rubric_autofill, not scored by an LLM`);
			continue;
		}
		if (typeof entry.evidence !== "string" || entry.evidence.trim() === "") {
			errors.push(`${entry.leaf_id}: an LLM score requires an evidence pointer (finding id, gate, or artifact)`);
			continue;
		}
		const value = entry.score;
		if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
			errors.push(`${entry.leaf_id}: score must be a number in [0,1]`);
			continue;
		}
		if (target.leaf.type === "binary" && value !== 0 && value !== 1) {
			errors.push(`${entry.leaf_id}: binary leaf accepts only 0 or 1, found ${value}`);
			continue;
		}
		if (target.leaf.levels && target.leaf.levels.length > 0 && !target.leaf.levels.includes(value)) {
			errors.push(`${entry.leaf_id}: score ${value} is not one of the declared levels ${target.leaf.levels.join(",")}`);
			continue;
		}
		target.leaf.score = value;
		target.leaf.evidence = entry.evidence;
		if (entry.note) target.leaf.note = entry.note;
	}

	const unresolved: string[] = [];
	const scored: string[] = [];
	const branches: BranchScore[] = [];
	for (const branch of copy.branches) {
		let branchSum = 0;
		let complete = true;
		for (const leaf of branch.children) {
			if (leaf.score === null || leaf.score === undefined) {
				complete = false;
				unresolved.push(leaf.id);
				continue;
			}
			branchSum += leaf.weight * leaf.score;
			scored.push(leaf.id);
		}
		branches.push({
			id: branch.id,
			title: branch.title,
			weight: branch.weight,
			score: complete ? round(branchSum) : null,
			leaves: branch.children.map((leaf) => ({
				id: leaf.id,
				type: leaf.type,
				weight: leaf.weight,
				score: leaf.score ?? null,
				evidence: leaf.evidence ?? "",
			})),
		});
	}

	let total: number | null = null;
	if (unresolved.length === 0) {
		total = round(branches.reduce((sum, branch) => sum + branch.weight * (branch.score ?? 0), 0));
	}
	return {
		ok: errors.length === 0,
		errors,
		total,
		complete: unresolved.length === 0,
		branches,
		unresolved,
		scored,
	};
}

/** Render the filled tree as the markdown table FINAL_REPORT/GRADE cite. */
export function renderRubric(tree: RubricTree, result?: RubricScoreResult): string {
	const lines: string[] = [`### Rubric: ${tree.title}`, ""];
	lines.push("| Branch | Leaf | Type | Weight | Score | Evidence |");
	lines.push("|---|---|---|---:|---:|---|");
	for (const branch of tree.branches) {
		for (const leaf of branch.children) {
			const score = leaf.score === null || leaf.score === undefined ? "—" : leaf.score.toFixed(2);
			const evidence = (leaf.evidence ?? "").replace(/\|/g, "\\|");
			lines.push(
				`| ${branch.title} | ${leaf.text} | ${leaf.type}${leaf.auto ? ` (${leaf.auto})` : ""} | ${leaf.weight.toFixed(2)} | ${score} | ${evidence} |`,
			);
		}
	}
	if (result) {
		lines.push("");
		lines.push(`**Total:** ${result.total === null ? "incomplete" : result.total.toFixed(3)}`);
		if (result.unresolved.length > 0) lines.push(`**Unresolved leaves:** ${result.unresolved.join(", ")}`);
	}
	return `${lines.join("\n")}\n`;
}

/** Count leaves by type, used by tests and the readiness view. */
export function rubricStats(tree: RubricTree): { branches: number; leaves: number; auto: number; binary: number; scale: number } {
	const stats = { branches: tree.branches.length, leaves: 0, auto: 0, binary: 0, scale: 0 };
	for (const branch of tree.branches) {
		for (const leaf of branch.children) {
			stats.leaves += 1;
			stats[leaf.type] += 1;
		}
	}
	return stats;
}
