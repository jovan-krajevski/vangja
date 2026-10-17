/**
 * AUDIT/DECIDE wiring: phase order, finder parallelism, verified-only synthesis
 * input, and the decision rule applied to GRADE.md + findings.json.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import {
	AUDIT_PHASE_PLAN,
	AUDIT_WORKSTREAMS,
	auditAgentForPhase,
	applyAuditDecision,
	assertAuditPhaseOrder,
	auditPhaseState,
	finderBatches,
	finderInputPaths,
	MAX_PARALLEL_FINDERS,
	prepareAuditCycle,
	runAuditGates,
	syncAuditState,
	verifiedIds,
} from "../../extensions/orchestrator/audit.ts";
import { auditBrief, decideBrief } from "../../extensions/orchestrator/state-machine.ts";
import { GATE_ORDER } from "../../extensions/verify/gates.ts";
import { registerForCycle } from "../../extensions/auditkit/store.ts";
import { baseFinding, readText, scratchFixture } from "./helpers.ts";

function allPassGates(cycle: number) {
	return {
		schema: "vangja-gates/1",
		cycle,
		snapshot: "test@sha",
		generated_at: new Date().toISOString(),
		ok: true,
		hash_ok: true,
		gates: GATE_ORDER.map((id) => ({ id, label: id, status: "pass", detail: `${id} passed` })),
		failed: [],
		skipped: [],
		config_source: "test",
		detail: {},
	};
}

function writeGates(root: string, cycle: number): void {
	const file = path.join(root, ".pi", "audit", `cycle-${cycle}`, "gates.json");
	fs.mkdirSync(path.dirname(file), { recursive: true });
	fs.writeFileSync(file, `${JSON.stringify(allPassGates(cycle), null, 2)}\n`);
}

function writeRubric(root: string, cycle: number, total = 0.95): void {
	const file = path.join(root, ".pi", "audit", `cycle-${cycle}`, "rubric.json");
	fs.mkdirSync(path.dirname(file), { recursive: true });
	fs.writeFileSync(file, `${JSON.stringify({ total, complete: true, unresolved: [] }, null, 2)}\n`);
}

function registerAll(root: string, cycle: number): void {
	for (const ws of AUDIT_WORKSTREAMS) {
		registerForCycle(root, cycle, ws, [
			baseFinding({
				id: `${ws}-001`,
				title: `${ws} finding`,
				location: ws === "S" ? "src/vangja/components/linear_trend.py:10" : "paper/paper.tex:10",
				severity: "S2",
			}),
		]);
	}
}

test("each phase spawns the agent that owns it (gates are supervised, not a finder)", () => {
	assert.equal(auditAgentForPhase("gates"), "supervisor");
	assert.equal(auditAgentForPhase("find"), "audit_finder");
	assert.equal(auditAgentForPhase("find:M"), "audit_finder");
	assert.equal(auditAgentForPhase("verify"), "audit_verifier");
	assert.equal(auditAgentForPhase("synthesise"), "audit_synthesizer");
	assert.equal(auditAgentForPhase("adjudicate"), "audit_adjudicator");
	assert.equal(auditAgentForPhase("unknown"), "supervisor");
});

test("the phase plan is gates → find → verify → synthesise → adjudicate with ≤8 finders", () => {
	assert.deepEqual(
		AUDIT_PHASE_PLAN.map((phase) => phase.phase),
		["gates", "find", "verify", "synthesise", "adjudicate"],
	);
	const find = AUDIT_PHASE_PLAN.find((phase) => phase.phase === "find");
	assert.equal(find?.parallel, MAX_PARALLEL_FINDERS);
	assert.equal(find?.agents[0], "audit_finder");
	const batches = finderBatches();
	assert.ok(batches.every((batch) => batch.length <= MAX_PARALLEL_FINDERS));
	assert.equal(batches.flat().length, AUDIT_WORKSTREAMS.length);
	assert.deepEqual(new Set(batches.flat()), new Set(AUDIT_WORKSTREAMS));
});

test("phase state advances only when the phase artifacts exist", () => {
	const root = scratchFixture();
	assert.equal(auditPhaseState(root, 1).next, "gates");
	assert.equal(auditPhaseState(root, 1).step, "gates");
	runAuditGates(root, 1, "test@sha");
	const afterGates = auditPhaseState(root, 1);
	assert.equal(afterGates.next, "find");
	assert.equal(afterGates.step, "find:M");
	assert.equal(afterGates.next_finder, "M");
	assert.equal(assertAuditPhaseOrder(root, 1, "verify").ok, false);

	// One finder at a time: the step names the first missing workstream.
	registerForCycle(root, 1, "M", [baseFinding({ id: "M-001" })]);
	assert.equal(auditPhaseState(root, 1).step, "find:S");
	const synced = syncAuditState(root, 1);
	assert.equal(synced.step, "find:S");
	const auditState = JSON.parse(readText(path.join(root, ".pi", "state", "AUDIT.json")));
	assert.equal(auditState.phase, "find:S");
	assert.equal(auditState.complete, false);

	registerAll(root, 1);
	const afterFind = auditPhaseState(root, 1);
	assert.equal(afterFind.finders_complete, true);
	assert.equal(afterFind.next, "verify");
	assert.equal(afterFind.step, "verify");
	assert.equal(afterFind.next_finder, null);
	assert.equal(assertAuditPhaseOrder(root, 1, "verify").ok, true);

	fs.writeFileSync(path.join(root, ".pi", "audit", "cycle-1", "VERIFY.md"), "# verified\n");
	assert.equal(auditPhaseState(root, 1).next, "synthesise");
	assert.equal(auditPhaseState(root, 1).step, "synthesise");

	for (const file of ["FINAL_REPORT.md", "GRADE.md", "rubric.json"]) {
		fs.writeFileSync(path.join(root, ".pi", "audit", "cycle-1", file), "x\n");
	}
	assert.equal(auditPhaseState(root, 1).next, "adjudicate");
	assert.equal(auditPhaseState(root, 1).step, "adjudicate");
	assert.equal(assertAuditPhaseOrder(root, 1, "synthesise").ok, true);

	fs.writeFileSync(path.join(root, ".pi", "audit", "cycle-1", "ADJUDICATED_REPORT.md"), "x\n");
	fs.writeFileSync(path.join(root, ".pi", "audit", "cycle-1", "ROOT_ISSUE_REGISTER.csv"), "root_issue\n");
	assert.equal(auditPhaseState(root, 1).next, "complete");
	assert.equal(auditPhaseState(root, 1).step, "complete");
	const complete = syncAuditState(root, 1);
	assert.equal(complete.step, "complete");
	const finalState = JSON.parse(readText(path.join(root, ".pi", "state", "AUDIT.json")));
	assert.equal(finalState.complete, true);
});

test("a finder's input list never names another workstream's findings", () => {
	const root = scratchFixture();
	const inputs = finderInputPaths(root, 2, "S");
	assert.ok(inputs.some((entry) => entry.endsWith("s/BRIEF.md")));
	assert.ok(!inputs.some((entry) => /\/[^s]\/findings\.json$/.test(entry) && entry.includes("findings.json")));
	assert.ok(!inputs.some((entry) => entry.includes("VERIFY")));
});

test("verifiedIds exposes only verified finder-phase findings", () => {
	const root = scratchFixture();
	registerForCycle(root, 3, "S", [
		baseFinding({ id: "S-001", title: "verified", status: "verified" }),
		baseFinding({ id: "S-002", title: "hypothesis", status: "to-verify", confidence: "medium" }),
		baseFinding({ id: "S-003", title: "synth record", status: "verified", phase: "synth", cites: ["S-001"] }),
	]);
	assert.deepEqual(verifiedIds(root, 3), ["S-001"]);
});

test("prepareAuditCycle seeds SCOPE, the protocol, and all ten briefs without overwriting", () => {
	const root = scratchFixture();
	const first = prepareAuditCycle(root, 5, "abc1234");
	assert.ok(first.created.length >= 13);
	for (const rel of ["_PROTOCOL.md", "SCOPE.md", "M/BRIEF.md", "S/BRIEF.md", "L/BRIEF.md"]) {
		assert.ok(fs.existsSync(path.join(root, ".pi", "audit", "cycle-5", rel)), `missing ${rel}`);
	}
	const brief = readText(path.join(root, ".pi", "audit", "cycle-5", "M", "BRIEF.md"));
	assert.match(brief, /cycle 5/);
	assert.doesNotMatch(brief, /\{\{cycle\}\}/);
	const scope = readText(path.join(root, ".pi", "audit", "cycle-5", "SCOPE.md"));
	assert.match(scope, /abc1234/);
	fs.writeFileSync(path.join(root, ".pi", "audit", "cycle-5", "SCOPE.md"), "edited by the supervisor\n");
	prepareAuditCycle(root, 5, "abc1234");
	assert.equal(readText(path.join(root, ".pi", "audit", "cycle-5", "SCOPE.md")), "edited by the supervisor\n");
});

test("audit-gates writes the deterministic gates before finders", () => {
	const root = scratchFixture();
	const gates = runAuditGates(root, 4, "test@sha");
	assert.equal(gates.gates.length, GATE_ORDER.length);
	assert.ok(fs.existsSync(path.join(root, ".pi", "audit", "cycle-4", "gates.json")));
	assert.equal(auditPhaseState(root, 4).gates, true);
});

test("DECIDE applies the decision rule to findings + gates + rubric and writes DECISION.json", () => {
	const root = scratchFixture();
	registerForCycle(root, 1, "S", [baseFinding({ id: "S-001", severity: "S1" })]);
	writeGates(root, 1);
	writeRubric(root, 1, 0.95);
	const result = applyAuditDecision(root, 1, {
		calibration: { recall: 1, rejected_confirmed: 0, calibrated: true },
		limitationsDisclosed: true,
	});
	assert.equal(result.grade, "weak reject");
	assert.equal(result.status, "WORK");
	assert.equal(result.decision.action, "fix-s0-s1");
	assert.equal(result.decision.reaudit, true);
	assert.equal(result.decision.done, false);
	const file = path.join(root, ".pi", "state", "DECISION.json");
	assert.ok(fs.existsSync(file));
	const decision = JSON.parse(readText(file));
	assert.equal(decision.grade, "weak reject");
	assert.equal(decision.done, false);
	assert.equal(decision.audit_usable, true);
	assert.ok(decision.blocking_fixes.length >= 1);
	assert.ok(Array.isArray(decision.rule_trace));
});

test("a clean audit reaches DONE through the same rule", () => {
	const root = scratchFixture();
	writeGates(root, 1);
	writeRubric(root, 1, 0.95);
	const result = applyAuditDecision(root, 1, {
		calibration: { recall: 1, rejected_confirmed: 0, calibrated: true },
		limitationsDisclosed: true,
	});
	assert.equal(result.grade, "strong accept");
	assert.equal(result.status, "DONE");
	assert.equal(result.decision.done, true);
	assert.equal(result.decision.reaudit, false);
	const decision = JSON.parse(readText(path.join(root, ".pi", "state", "DECISION.json")));
	assert.equal(decision.done, true);
});

test("the UNCALIBRATED marker is authoritative over a stale calibrated recall.json", () => {
	const root = scratchFixture();
	writeGates(root, 1);
	writeRubric(root, 1, 0.95);
	const latest = path.join(root, ".pi", "audit", "calibration", "latest");
	fs.mkdirSync(latest, { recursive: true });
	fs.writeFileSync(
		path.join(latest, "recall.json"),
		`${JSON.stringify({ calibrated: true, recall: 1, rejected_confirmed: 0 }, null, 2)}\n`,
	);
	fs.writeFileSync(path.join(latest, "UNCALIBRATED"), "stale marker\n");
	const result = applyAuditDecision(root, 1, { limitationsDisclosed: true });
	assert.equal(result.grade, "uncalibrated");
	assert.equal(result.status, "BLOCKED");
});

test("a missing calibration result blocks the grade", () => {
	const root = scratchFixture();
	writeGates(root, 1);
	writeRubric(root, 1, 0.95);
	const result = applyAuditDecision(root, 1, { calibration: null, limitationsDisclosed: true });
	assert.equal(result.grade, "uncalibrated");
	assert.equal(result.status, "BLOCKED");
	assert.equal(result.decision.action, "blocked-audit-failure");
	const decision = JSON.parse(readText(path.join(root, ".pi", "state", "DECISION.json")));
	assert.equal(decision.audit_usable, false);
});

test("the sixth cycle is blocked regardless of grade", () => {
	const root = scratchFixture();
	writeGates(root, 6);
	writeRubric(root, 6, 0.95);
	const result = applyAuditDecision(root, 6, {
		calibration: { recall: 1, rejected_confirmed: 0, calibrated: true },
		limitationsDisclosed: true,
	});
	assert.equal(result.status, "BLOCKED");
	assert.equal(result.decision.action, "blocked-cycle-cap");
});

test("the AUDIT and DECIDE briefs name the phases and the deterministic decision", () => {
	const gates = auditBrief(3, "gates");
	assert.match(gates, /run_gates/);
	assert.match(gates, /gates\.json/);
	const find = auditBrief(3, "find");
	assert.match(find, /each finder/);
	assert.match(find, /ten workstreams/);
	assert.match(find, /may read only/);
	const verify = auditBrief(3, "verify");
	assert.match(verify, /re-runs its command/);
	const findOne = auditBrief(3, "find:S");
	assert.match(findOne, /WS-S/);
	assert.match(findOne, /finding_register\(cycle=3, workstream="S"\)/);
	const synth = auditBrief(3, "synthesise");
	assert.match(synth, /verified ids only/);
	const adjudicate = auditBrief(3, "adjudicate");
	assert.match(adjudicate, /ROOT_ISSUE_REGISTER/);
	const decide = decideBrief();
	assert.match(decide, /rubrics\/decision\.json/);
	assert.match(decide, /audit-decide/);
	assert.match(decide, /uncalibrated/);
});
