/**
 * Grade derivation and the decision rule: every blocking rule, every grade
 * boundary, and the accept precondition set (plan/07_AUDIT_HARNESS.md §7,
 * plan/02_ARCHITECTURE.md §5.4).
 */

import assert from "node:assert/strict";
import * as path from "node:path";
import { test } from "node:test";
import {
	applyDecision,
	auditGrade,
	buildFixList,
	gradeFromRubric,
	GRADES,
	RUBRIC_BOUNDARIES,
	type CalibrationSummary,
	type GradeInput,
} from "../../extensions/auditkit/grade.ts";
import { validateRubric } from "../../extensions/rubric/lib.ts";
import { baseFinding, CLEAN_CALIBRATION, CLEAN_RUBRIC, gateResults, readText, REPO_ROOT } from "./helpers.ts";

function findingFor(workstream: string, severity: "S0" | "S1" | "S2" | "S3", overrides = {}) {
	return baseFinding({
		id: `${workstream}-001`,
		title: `${severity} finding in ${workstream}`,
		location: `${workstream === "S" ? "src/vangja/components/linear_trend.py" : "case_studies/run.py"}:10`,
		severity,
		...overrides,
	});
}

function grade(overrides: Partial<GradeInput> = {}) {
	return auditGrade({
		findings: [],
		gates: gateResults(),
		rubric: CLEAN_RUBRIC,
		calibration: CLEAN_CALIBRATION,
		traceability: { ok: true },
		limitations_disclosed: true,
		cycle: 1,
		max_cycles: 5,
		...overrides,
	});
}

test("rubric boundaries map to the seven grades", () => {
	const expected: Array<[number, string]> = [
		[0.95, "strong accept"],
		[0.9, "strong accept"],
		[0.899, "accept"],
		[0.8, "accept"],
		[0.799, "weak accept"],
		[0.68, "weak accept"],
		[0.679, "borderline"],
		[0.55, "borderline"],
		[0.549, "weak reject"],
		[0.4, "weak reject"],
		[0.399, "reject"],
		[0.25, "reject"],
		[0.249, "strong reject"],
		[0, "strong reject"],
	];
	for (const [total, value] of expected) {
		assert.equal(gradeFromRubric(total), value, `total ${total}`);
	}
	assert.equal(gradeFromRubric(null), "borderline");
	assert.equal(RUBRIC_BOUNDARIES.length, 7);
});

test("a clean audit with no findings and all conditions is a strong accept", () => {
	const result = grade();
	assert.equal(result.usable, true);
	assert.equal(result.grade, "strong accept");
	assert.equal(result.journal, "accept");
	assert.equal(result.capped, false);
	assert.deepEqual(result.blocking_fixes, []);
});

test("any confirmed S0 caps the grade at weak reject", () => {
	const result = grade({ findings: [findingFor("S", "S0")] });
	assert.equal(result.grade, "weak reject");
	assert.ok(result.trace.some((line) => line.rule === "blocking-s0" && line.effect === "cap"));
	assert.equal(result.confirmed_counts.S0, 1);
});

test("a confirmed S1 in M/S/P/C caps the grade at weak reject", () => {
	for (const workstream of ["M", "S", "P", "C"]) {
		const result = grade({ findings: [findingFor(workstream, "S1")] });
		assert.equal(result.grade, "weak reject", `WS-${workstream}`);
		assert.ok(
			result.trace.some((line) => line.rule === "blocking-core-s1"),
			`WS-${workstream} should hit blocking-core-s1`,
		);
	}
});

test("an unfixed confirmed S1 outside M/S/P/C caps at borderline; fixed does not", () => {
	const unfixed = grade({ findings: [findingFor("R", "S1")] });
	assert.equal(unfixed.grade, "borderline");
	assert.ok(unfixed.trace.some((line) => line.rule === "blocking-other-s1"));

	const fixed = grade({ findings: [findingFor("R", "S1", { fixed: true })] });
	assert.notEqual(fixed.grade, "borderline");
	assert.equal(fixed.grade, "strong accept");
});

test("a failed deterministic gate caps at weak reject and names the gate", () => {
	const gates = gateResults({ ok: false });
	gates.gates = gates.gates.map((gate) =>
		gate.id === "test-diff" ? { ...gate, status: "fail", detail: "assertion removed" } : gate,
	);
	const result = grade({ gates });
	assert.equal(result.grade, "weak reject");
	assert.deepEqual(result.gate_failures, ["test-diff"]);
	const line = result.trace.find((entry) => entry.rule === "failed-gate");
	assert.equal(line?.effect, "cap");
	assert.match(line?.evidence ?? "", /test-diff/);
});

test("a waived or adjudicated-inapplicable gate does not cap", () => {
	const gates = gateResults({ ok: true });
	gates.gates = gates.gates.map((gate) =>
		gate.id === "traceability" ? { ...gate, status: "fail" } : gate,
	);
	assert.equal(grade({ gates }).grade, "weak reject");
	assert.equal(grade({ gates, waived_gates: ["traceability"] }).grade, "strong accept");
	assert.equal(grade({ gates, inapplicable_gates: ["traceability"] }).grade, "strong accept");
});

test("accept requires green gates, traceability, and limitation disclosures", () => {
	assert.equal(grade({ traceability: { ok: false } }).grade, "weak accept");
	assert.equal(grade({ limitations_disclosed: false }).grade, "weak accept");
	const gates = gateResults({ ok: false });
	gates.gates = gates.gates.map((gate) => (gate.id === "env-pinned" ? { ...gate, status: "fail" } : gate));
	assert.equal(grade({ gates }).grade, "weak reject");
});

test("calibration below 0.6 refuses to grade", () => {
	const calibration: CalibrationSummary = { recall: 0.5, rejected_confirmed: 0, calibrated: false };
	const result = grade({ calibration });
	assert.equal(result.usable, false);
	assert.equal(result.grade, "uncalibrated");
	assert.equal(result.journal, "none");
	assert.match(result.reason, /UNCALIBRATED/);
});

test("a confirmed rejected Phase-A lead refuses to grade even at recall 1.0", () => {
	const calibration: CalibrationSummary = { recall: 1, rejected_confirmed: 1, calibrated: false };
	const result = grade({ calibration });
	assert.equal(result.usable, false);
	assert.equal(result.grade, "uncalibrated");
});

test("more than five audit cycles is BLOCKED for human direction", () => {
	const result = grade({ cycle: 6, max_cycles: 5 });
	assert.equal(result.usable, false);
	assert.equal(result.grade, "uncalibrated");
	assert.ok(result.trace.some((line) => line.rule === "max-audit-cycles"));
});

test("the fix list marks S0/S1 blocking and carries owner, effort, feasibility", () => {
	const findings = [
		findingFor("S", "S0", { owner: "code" }),
		findingFor("M", "S1", { owner: "paper", location: "paper/paper.tex:10" }),
		findingFor("H", "S2", { owner: "docs", location: "README.md:4" }),
		findingFor("L", "S3", { owner: "paper", location: "paper/refs.bib:2" }),
	];
	const fixList = buildFixList(findings);
	assert.equal(fixList.length, 4);
	assert.equal(fixList[0].severity, "S0");
	assert.equal(fixList[0].effort, "large");
	assert.equal(fixList[0].blocking, true);
	assert.equal(fixList[1].effort, "medium");
	assert.equal(fixList[1].feasible, "feasible");
	assert.equal(fixList[2].blocking, false);
	assert.equal(fixList[3].effort, "small");
});

test("the decision mapping applies each grade as plan/02 §5.4 says", () => {
	const blocking = buildFixList([findingFor("S", "S0")]);
	const s1 = buildFixList([findingFor("S", "S1")]);
	const strongReject = applyDecision("strong reject", blocking);
	assert.equal(strongReject.action, "reopen-p0");
	assert.equal(strongReject.reaudit, true);
	assert.equal(strongReject.status, "WORK");

	const reject = applyDecision("reject", blocking);
	assert.equal(reject.action, "reopen-p0");

	for (const value of ["weak reject", "borderline"] as const) {
		const result = applyDecision(value, s1);
		assert.equal(result.action, "fix-s0-s1", value);
		assert.equal(result.reaudit, true, value);
	}

	const weakAccept = applyDecision("weak accept", s1);
	assert.equal(weakAccept.action, "fix-feasible");
	assert.equal(weakAccept.reaudit, true);

	const accept = applyDecision("accept", []);
	assert.equal(accept.status, "DONE");
	assert.equal(accept.done, true);
	assert.equal(accept.reaudit, false);

	const acceptFix = applyDecision("accept", s1);
	assert.equal(acceptFix.status, "WORK");
	assert.equal(acceptFix.reaudit, true);
	assert.equal(acceptFix.done, false);
});

test("weak accept with purely editorial fixes skips the re-audit", () => {
	const editorial = buildFixList([findingFor("M", "S1", { owner: "paper", location: "paper/paper.tex:10" })]);
	const result = applyDecision("weak accept", editorial);
	assert.equal(result.reaudit, false);
	assert.ok(result.trace.some((line) => line.rule === "weak-accept-editorial"));
});

test("uncalibrated and over-the-cap decisions are BLOCKED", () => {
	const blocked = applyDecision("uncalibrated", []);
	assert.equal(blocked.status, "BLOCKED");
	assert.equal(blocked.action, "blocked-audit-failure");

	const capped = applyDecision("accept", [], undefined, 6);
	assert.equal(capped.status, "BLOCKED");
	assert.equal(capped.action, "blocked-cycle-cap");
});

test("decision.json validates and encodes the fixed action mapping", () => {
	const decision = JSON.parse(readText(path.join(REPO_ROOT, ".pi", "rubrics", "decision.json")));
	const validation = validateRubric(decision);
	assert.equal(validation.ok, true, validation.errors.join("; "));
	assert.deepEqual(decision.scale, [...GRADES]);
	assert.equal(decision.max_cycles, 5);
	assert.equal(decision.mapping["weak reject"].action, "fix-s0-s1");
	assert.equal(decision.mapping.accept.done, true);
});
