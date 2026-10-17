/**
 * Seeded-defect canary and the end-to-end seeded audit: ≥80% of seeded S0/S1
 * detected, grade capped at ≤ weak reject, and GRADE.md cites the failing gates.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { auditGrade, type GateResults } from "../../extensions/auditkit/grade.ts";
import { gradeCitesFailingGates, writeGrade } from "../../extensions/auditkit/report.ts";
import { probeWeakenedTest, runSeededAudit, renderCanaryMatrix, type CanaryResult } from "../../extensions/auditkit/canary.ts";
import { registerForCycle } from "../../extensions/auditkit/store.ts";
import { runGates } from "../../extensions/verify/gates.ts";
import { CLEAN_CALIBRATION, CLEAN_RUBRIC, readText, scratchFixture, SEEDED_FIXTURE } from "./helpers.ts";

function asGateResults(gates: ReturnType<typeof runGates>): GateResults {
	return {
		ok: gates.ok,
		cycle: gates.cycle,
		snapshot: gates.snapshot,
		gates: gates.gates.map((gate) => ({
			id: gate.id,
			status: gate.status,
			detail: gate.detail,
			hash_ok: gate.hash_ok,
		})),
	};
}

function canaryOnFixture(): CanaryResult {
	return runSeededAudit(SEEDED_FIXTURE);
}

test("the canary detects every seeded S0/S1 and reports no negative control", () => {
	const result = canaryOnFixture();
	assert.equal(result.expected_s0s1, 8);
	assert.equal(result.detected_s0s1, 8);
	assert.equal(result.recall, 1);
	assert.ok(result.recall >= 0.8);
	assert.deepEqual(result.misses, []);
	assert.deepEqual(result.false_positives, []);
	assert.deepEqual(result.severity_mismatches, []);
	assert.equal(result.ok, true, result.detail);
});

test("the seeded severity/workstream mapping is the expected one", () => {
	const result = canaryOnFixture();
	const byProbe = new Map(result.findings.map((finding) => [String(finding.probe), finding]));
	assert.equal(byProbe.get("wrong_potential_sign")?.severity, "S0");
	assert.equal(byProbe.get("retrospective_confirmation")?.severity, "S0");
	assert.equal(byProbe.get("test_selection")?.severity, "S1");
	assert.equal(byProbe.get("weakened_test")?.severity, "S1");
	assert.equal(byProbe.get("missing_artifact")?.severity, "S1");
	assert.equal(byProbe.get("mislabelled_baseline")?.severity, "S1");
	assert.equal(byProbe.get("favourable_seed")?.severity, "S1");
	assert.equal(byProbe.get("uncited_paper_number")?.severity, "S1");
	for (const finding of result.findings) {
		assert.equal(finding.status, "verified");
		assert.equal(finding.confidence, "high");
		assert.match(finding.command, /\S/);
		assert.equal(finding.phase, "canary");
	}
});

test("the canary matrix names every seeded defect", () => {
	const matrix = renderCanaryMatrix(canaryOnFixture());
	for (const id of ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]) {
		assert.match(matrix, new RegExp(`\\| ${id} \\|`));
	}
	assert.match(matrix, /\| N1 \|/);
	assert.match(matrix, /not reported \(correct\)/);
});

test("the seeded audit caps the grade at weak reject and GRADE.md names the failing gates", () => {
	const root = scratchFixture();
	const canary = runSeededAudit(root);
	const registration = registerForCycle(root, 2, "S", canary.findings.filter((finding) => finding.id.startsWith("S-")));
	assert.ok(registration.added.length > 0);
	for (const workstream of ["M", "E", "I", "C", "D"]) {
		registerForCycle(
			root,
			2,
			workstream as "M",
			canary.findings.filter((finding) => finding.id.startsWith(`${workstream}-`)),
		);
	}
	const gates = runGates(root, { cycle: 2, snapshot: "seeded@fixture" });
	assert.ok(gates.failed.length > 0);
	for (const gate of ["confirmation-frozen", "no-test-selection", "test-diff", "context-excluded"]) {
		assert.ok(gates.failed.includes(gate), `${gate} should fail on the seeded fixture`);
	}

	const result = auditGrade({
		findings: canary.findings,
		gates: asGateResults(gates),
		rubric: CLEAN_RUBRIC,
		calibration: CLEAN_CALIBRATION,
		traceability: { ok: false },
		limitations_disclosed: true,
		cycle: 2,
		max_cycles: 5,
	});
	assert.ok(
		["strong reject", "reject", "weak reject"].includes(result.grade),
		`expected ≤ weak reject, got ${result.grade}`,
	);
	assert.equal(result.grade, "weak reject");
	assert.ok(result.confirmed_counts.S0 >= 2);
	assert.ok(result.confirmed_counts.S1 >= 6);
	assert.ok(result.gate_failures.length > 0);

	const gradeFile = writeGrade(root, 2, result);
	const gradeMd = readText(gradeFile);
	assert.match(gradeMd, /\*\*Grade:\*\* weak reject/);
	assert.match(gradeMd, /## Rule trace/);
	assert.match(gradeMd, /failed-gate/);
	assert.match(gradeMd, /## Failing gates/);
	for (const gate of result.gate_failures) {
		assert.match(gradeMd, new RegExp(gate.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")), `GRADE.md must cite ${gate}`);
	}
	assert.equal(gradeCitesFailingGates(result), true);
	assert.ok(path.isAbsolute(gradeFile));
});

test("a canary with a missing defect is under the threshold and fails", () => {
	const root = scratchFixture();
	// Remove two seeded S0 defects so recall falls below 0.8.
	fsRemove(path.join(root, "src"));
	fsRemove(path.join(root, "data"));
	const result = runSeededAudit(root);
	assert.ok(result.recall < 1);
	assert.ok(result.misses.some((miss) => miss.expected === "D1"));
	assert.ok(result.misses.some((miss) => miss.expected === "D5"));
	assert.equal(result.ok, false);
});

test("the weakened-test probe counts unittest-style assertions and commented ones", () => {
	const root = fs.mkdtempSync(path.join(process.env.TMPDIR ?? "/tmp", "vangja-canary-unittest-"));
	fs.mkdirSync(path.join(root, "tests_baseline"), { recursive: true });
	fs.mkdirSync(path.join(root, "tests"), { recursive: true });
	fs.writeFileSync(
		path.join(root, "tests_baseline", "test_metrics.py"),
		"import unittest\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(1, 1)\n        self.assertTrue(True)\n",
	);
	fs.writeFileSync(
		path.join(root, "tests", "test_metrics.py"),
		"import unittest\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        # self.assertEqual(1, 1)\n        self.assertTrue(True)\n",
	);
	const commented = probeWeakenedTest(root);
	assert.equal(commented.length, 1, JSON.stringify(commented));
	assert.match(commented[0].observed, /1 assertion\(s\) remain/);
	assert.match(commented[0].observed, /commented out/);

	fs.writeFileSync(
		path.join(root, "tests", "test_metrics.py"),
		"import unittest\n\n\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertTrue(True)\n",
	);
	const removed = probeWeakenedTest(root);
	assert.equal(removed.length, 1, JSON.stringify(removed));
	assert.match(removed[0].observed, /1 assertion\(s\) remain/);
	assert.ok(!/commented out/.test(removed[0].observed));
});

function fsRemove(target: string): void {
	fs.rmSync(target, { recursive: true, force: true });
}
