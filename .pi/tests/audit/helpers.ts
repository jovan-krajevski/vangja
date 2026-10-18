/**
 * Audit test helpers: fixture copies, a valid finding, synthetic gate results,
 * and score entries for a rubric tree.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type { GateResults, RubricSummary } from "../../extensions/auditkit/grade.ts";
import type { Finding } from "../../extensions/auditkit/schema.ts";
import type { RubricTree } from "../../extensions/rubric/lib.ts";

export const AUDIT_TESTS_DIR = path.dirname(fileURLToPath(import.meta.url));
export const PI_DIR = path.dirname(path.dirname(AUDIT_TESTS_DIR));
export const REPO_ROOT = path.dirname(PI_DIR);
export const SEEDED_FIXTURE = path.join(AUDIT_TESTS_DIR, "fixtures", "seeded");
export const CALIBRATION_FIXTURES = path.join(AUDIT_TESTS_DIR, "fixtures", "calibration");

export function copyDir(source: string, target: string): void {
	fs.cpSync(source, target, { recursive: true });
}

/** A disposable copy of the seeded fixture (tests may write into it). */
export function scratchFixture(prefix = "vangja-seeded-"): string {
	const root = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
	const target = path.join(root, "seeded");
	copyDir(SEEDED_FIXTURE, target);
	return target;
}

export function baseFinding(overrides: Partial<Finding> = {}): Finding {
	const base: Finding = {
		id: "S-001",
		title: "Transfer potential has the wrong sign in complete pooling",
		location: "src/vangja/components/linear_trend.py:410",
		snapshot: "audit-cycle/1@deadbeef",
		command: "grep -n 'loss_factor' src/vangja/components/linear_trend.py",
		expected: "the potential is minus the squared deviation from the transferred slope",
		observed: "the complete branch adds +abs(slope - transferred), rewarding deviation",
		consequence: "every pooled transfer result moves away from the context posterior",
		severity: "S1",
		confidence: "high",
		status: "verified",
		reproducible: "yes — reproduced with the cited grep",
		evidence: ["src/vangja/components/linear_trend.py:410"],
		cross_refs: [],
		owner: "code",
		found_in_cycle: 1,
		phase: "finder",
	};
	return { ...base, ...overrides } as Finding;
}

export function gateResults(overrides: Partial<GateResults> = {}): GateResults {
	const gates = [
		{ id: "tests", status: "pass" },
		{ id: "protocol-hashes", status: "pass" },
		{ id: "confirmation-frozen", status: "pass" },
		{ id: "no-test-selection", status: "pass" },
		{ id: "test-diff", status: "pass" },
		{ id: "no-failures-hidden", status: "pass" },
		{ id: "context-excluded", status: "pass" },
		{ id: "artifacts-complete", status: "pass", hash_ok: true },
		{ id: "reproduce-headline", status: "pass" },
		{ id: "traceability", status: "pass" },
		{ id: "env-pinned", status: "pass" },
		{ id: "gaming-scan", status: "pass" },
		{ id: "canary", status: "pass" },
	] as GateResults["gates"];
	const result: GateResults = { ok: true, gates, cycle: 1, snapshot: "fixture@sha", ...overrides };
	return result;
}

export const CLEAN_CALIBRATION = {
	recall: 1,
	rejected_confirmed: 0,
	calibrated: true,
};

export const CLEAN_RUBRIC: RubricSummary = { total: 0.95, complete: true };

/** Score every non-auto leaf 1 with an evidence pointer. */
export function fullScores(tree: RubricTree): Array<{ leaf_id: string; score: number; evidence: string }> {
	const scores: Array<{ leaf_id: string; score: number; evidence: string }> = [];
	for (const branch of tree.branches) {
		for (const leaf of branch.children) {
			if (leaf.type === "auto") continue;
			scores.push({ leaf_id: leaf.id, score: 1, evidence: `test:${leaf.id}` });
		}
	}
	return scores;
}

export function readText(file: string): string {
	return fs.readFileSync(file, "utf8");
}
