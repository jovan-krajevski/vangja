#!/usr/bin/env node
/**
 * Offline acceptance probe for the S6 audit harness.
 *
 * Usage: node --experimental-strip-types .pi/tests/audit/acceptance_probe.ts <mode> <fixture-root>
 *
 * Modes:
 *   gates      run the 13 gates on a fixture copy; assert every gate reports
 *   tamper     tamper the seeded artifact; assert hash_ok=false and numeric drift
 *   grade      canary + gates + auditGrade; assert ≥80% S0/S1 and grade ≤ weak reject
 *   findings   assert the schema rejects the four invalid shapes and dedupes
 *
 * Exit 0 = all assertions passed. Prints one PASS/FAIL line per assertion.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { auditGrade, type GateResults } from "../../extensions/auditkit/grade.ts";
import { gradeCitesFailingGates, writeGrade } from "../../extensions/auditkit/report.ts";
import { runSeededAudit } from "../../extensions/auditkit/canary.ts";
import { registerFindings, validateFinding, type Finding } from "../../extensions/auditkit/schema.ts";
import { verifyArtifact, verifyReproduce } from "../../extensions/verify/lib.ts";
import { GATE_ORDER, runGates } from "../../extensions/verify/gates.ts";

let failures = 0;

function check(name: string, condition: boolean, detail: string): void {
	if (condition) {
		console.log(`PASS  ${name}: ${detail}`);
	} else {
		failures += 1;
		console.log(`FAIL  ${name}: ${detail}`);
	}
}

function asGateResults(gates: ReturnType<typeof runGates>): GateResults {
	return {
		ok: gates.ok,
		cycle: gates.cycle,
		snapshot: gates.snapshot,
		gates: gates.gates.map((gate) => ({ id: gate.id, status: gate.status, detail: gate.detail, hash_ok: gate.hash_ok })),
	};
}

function baseFinding(overrides: Partial<Finding> = {}): Finding {
	return {
		id: "S-001",
		title: "A finding",
		location: "src/a.py:1",
		snapshot: "fixture@sha",
		command: "grep -n a src/a.py",
		expected: "a holds",
		observed: "a does not hold",
		consequence: "a result is wrong",
		severity: "S1",
		confidence: "high",
		status: "verified",
		reproducible: "yes",
		evidence: ["src/a.py:1"],
		cross_refs: [],
		owner: "code",
		found_in_cycle: 1,
		phase: "finder",
		...overrides,
	} as Finding;
}

function main(): number {
	const mode = process.argv[2];
	const root = process.argv[3];
	if (!mode || !root || !fs.existsSync(root)) {
		console.error("usage: acceptance_probe.ts <gates|tamper|grade|findings> <fixture-root>");
		return 2;
	}

	if (mode === "gates") {
		const gates = runGates(root, { cycle: 1, snapshot: "seeded@fixture" });
		check("all gates report", gates.gates.length === GATE_ORDER.length, `${gates.gates.length}/${GATE_ORDER.length}`);
		check(
			"every status is set",
			gates.gates.every((gate) => ["pass", "fail", "skip", "error"].includes(gate.status) && gate.detail.length > 0),
			gates.gates.map((gate) => `${gate.id}=${gate.status}`).join(" "),
		);
		check("artifacts hash_ok", gates.hash_ok === true, `hash_ok=${gates.hash_ok}`);
		check(
			"failed gates are reported",
			["confirmation-frozen", "no-test-selection", "test-diff", "no-failures-hidden", "context-excluded"].every((id) =>
				gates.failed.includes(id),
			),
			`failed=${gates.failed.join(",")}`,
		);
	} else if (mode === "tamper") {
		const dir = path.join(root, "artifacts", "E-0001");
		fs.writeFileSync(path.join(dir, "numbers.json"), `${JSON.stringify({ rel_mae: 0.0907 })}\n`);
		fs.writeFileSync(path.join(dir, "metrics.csv"), "series,rel_mae\nFridge,0.0907\n");
		const artifact = verifyArtifact(root, dir);
		check("tampered artifact fails hash_ok", artifact.hash_ok === false && artifact.ok === false, artifact.detail);
		const reproduce = verifyReproduce(root, dir);
		check("reproduction prints numeric drift", reproduce.ok === false && /delta=3\.000e-2/.test(reproduce.drift), reproduce.drift.replace(/\n/g, " | "));
	} else if (mode === "grade") {
		const canary = runSeededAudit(root);
		check(
			"seeded S0/S1 detection ≥ 80%",
			canary.recall >= 0.8,
			`${canary.detected_s0s1}/${canary.expected_s0s1} = ${canary.recall.toFixed(3)}`,
		);
		check("negative controls clean", canary.false_positives.length === 0, `${canary.false_positives.length} false positive(s)`);
		const gates = runGates(root, { cycle: 2, snapshot: "seeded@fixture" });
		const result = auditGrade({
			findings: canary.findings,
			gates: asGateResults(gates),
			rubric: { total: 0.95, complete: true },
			calibration: { recall: 1, rejected_confirmed: 0, calibrated: true },
			traceability: { ok: false },
			limitations_disclosed: true,
			cycle: 2,
			max_cycles: 5,
		});
		check("grade ≤ weak reject", ["strong reject", "reject", "weak reject"].includes(result.grade), `grade=${result.grade}`);
		const gradeFile = writeGrade(root, 2, result);
		const text = fs.readFileSync(gradeFile, "utf8");
		check(
			"GRADE.md cites the failing gates",
			gradeCitesFailingGates(result) && result.gate_failures.every((gate) => text.includes(gate)),
			`failed=${result.gate_failures.join(",")}`,
		);
	} else if (mode === "findings") {
		const missing = validateFinding(baseFinding({ command: "" }));
		check("missing command rejected", missing.ok === false, missing.errors.join("; "));
		const severity = validateFinding(baseFinding({ severity: "S9" as never }));
		check("unknown severity rejected", severity.ok === false, severity.errors.join("; "));
		const lowVerified = validateFinding(baseFinding({ confidence: "low" }));
		check("verified + low confidence rejected", lowVerified.ok === false, lowVerified.errors.join("; "));
		const first = baseFinding({ id: "S-001", location: "src/a.py:10" });
		const second = baseFinding({ id: "S-002", location: "src/a.py:25" });
		const register = registerFindings([], { cycle: 1, workstream: "S", findings: [first, second] });
		check(
			"duplicate deduped with a cross-ref",
			register.register.length === 1 && register.duplicates[0]?.duplicate_of === "S-001" && register.register[0].cross_refs.includes("S-002"),
			register.register.length === 1 ? `canonical=${register.register[0].id} cross_refs=${register.register[0].cross_refs.join(",")}` : "duplicate kept",
		);
	} else {
		console.error(`unknown mode ${mode}`);
		return 2;
	}

	return failures === 0 ? 0 : 1;
}

process.exit(main());
