/**
 * Deterministic AUDIT/DECIDE wiring (plan/02_ARCHITECTURE.md §2, §5.4;
 * plan/07_AUDIT_HARNESS.md §8).
 *
 * The conductor runs the gates before any finder, spawns finders in parallel
 * (≤8), then the verifier, then the synthesizer + meta_review, then the
 * adjudicator. It writes `.pi/audit/cycle-<n>/` and applies
 * `rubrics/decision.json` in DECIDE. This module owns the phase order, the gate
 * step, and the decision application; the agent briefs come from the templates.
 *
 * No pi imports: usable from `node --test` and from the CLI.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { writeJsonAtomic } from "../../lib/io.ts";
import { auditGrade, applyDecision, type DecisionMapping, type DecisionResult, type GradeResult, type GradeOrUncalibrated, type FixItem } from "../auditkit/grade.ts";
import { cycleDir as auditkitCycleDir, readRegister, readRubricScore } from "../auditkit/store.ts";
import { readWorkstreamRegister } from "../auditkit/store.ts";
import { loadGatesFile, runGates, type GatesFile, type RunGatesOptions } from "../verify/gates.ts";
import type { CalibrationSummary } from "../auditkit/grade.ts";
import type { RegisteredFinding } from "../auditkit/schema.ts";

export const AUDIT_WORKSTREAMS = ["M", "S", "P", "D", "I", "R", "H", "E", "C", "L"] as const;
export type AuditWorkstream = (typeof AUDIT_WORKSTREAMS)[number];

export const AUDIT_PHASES = ["gates", "find", "verify", "synthesise", "adjudicate", "complete"] as const;
export type AuditPhase = (typeof AUDIT_PHASES)[number];

/** Parallel finder cap (plan/02 §4: up to 8 in parallel). */
export const MAX_PARALLEL_FINDERS = 8;

export interface AuditPhaseInfo {
	phase: AuditPhase;
	agents: string[];
	parallel: number;
	reads: string[];
	writes: string[];
}

export const AUDIT_PHASE_PLAN: ReadonlyArray<AuditPhaseInfo> = [
	{
		phase: "gates",
		agents: [],
		parallel: 1,
		reads: ["the frozen snapshot", ".pi/config/audit.json"],
		writes: [".pi/audit/cycle-<n>/gates.json"],
	},
	{
		phase: "find",
		agents: ["audit_finder"],
		parallel: MAX_PARALLEL_FINDERS,
		reads: ["SCOPE.md", "_PROTOCOL.md", "<ws>/BRIEF.md", "gates.json", "the snapshot"],
		writes: ["<ws>/FINDINGS.md", "<ws>/findings.json"],
	},
	{
		phase: "verify",
		agents: ["audit_verifier"],
		parallel: 4,
		reads: ["every <ws>/findings.json", "the snapshot"],
		writes: ["VERIFY.md", "verify/VERIFY_<ws>.md"],
	},
	{
		phase: "synthesise",
		agents: ["audit_synthesizer", "meta_review"],
		parallel: 2,
		reads: ["VERIFY.md", "gates.json", "the rubric", "META coverage"],
		writes: ["FINAL_REPORT.md", "rubric.json", "GRADE.md", "META_COVERAGE.md"],
	},
	{
		phase: "adjudicate",
		agents: ["audit_adjudicator"],
		parallel: 1,
		reads: ["FINAL_REPORT.md", "GRADE.md", "VERIFY.md", "QUESTIONS_FOR_AUTHORS.md"],
		writes: ["ADJUDICATED_REPORT.md", "ROOT_ISSUE_REGISTER.csv"],
	},
];

/** The agent that owns each phase. Gates are deterministic and supervised. */
export function auditAgentForPhase(phase: string): string {
	const base = phase.split(":")[0];
	switch (base) {
		case "find":
			return "audit_finder";
		case "verify":
			return "audit_verifier";
		case "synthesise":
			return "audit_synthesizer";
		case "adjudicate":
			return "audit_adjudicator";
		case "gates":
			return "supervisor";
		default:
			return "supervisor";
	}
}

export function cyclePath(root: string, cycle: number): string {
	return auditkitCycleDir(root, cycle);
}

/**
 * Seed a cycle directory from `.pi/audit/_templates/`: SCOPE.md, _PROTOCOL.md,
 * the ten per-workstream briefs, and the report templates. Existing files are
 * never overwritten, so a re-run keeps the agent's SCOPE annotations.
 */
export function prepareAuditCycle(root: string, cycle: number, snapshot: string): { dir: string; created: string[] } {
	const local = path.join(root, ".pi", "audit", "_templates");
	const harness = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "audit", "_templates");
	const source = fs.existsSync(local) ? local : harness;
	const dir = cyclePath(root, cycle);
	const created: string[] = [];
	const copy = (from: string, to: string): void => {
		for (const entry of fs.readdirSync(from, { withFileTypes: true })) {
			const src = path.join(from, entry.name);
			const dest = path.join(to, entry.name);
			if (entry.isDirectory()) {
				fs.mkdirSync(dest, { recursive: true });
				copy(src, dest);
				continue;
			}
			if (fs.existsSync(dest)) continue;
			const text = fs.readFileSync(src, "utf8");
			const rendered = text
				.replaceAll("{{cycle}}", String(cycle))
				.replaceAll("{{sha}}", snapshot)
				.replaceAll("{{started_at}}", new Date().toISOString());
			fs.mkdirSync(path.dirname(dest), { recursive: true });
			fs.writeFileSync(dest, rendered);
			created.push(path.relative(root, dest));
		}
	};
	fs.mkdirSync(dir, { recursive: true });
	if (fs.existsSync(source)) copy(source, dir);
	return { dir, created };
}

export function scopePath(root: string, cycle: number): string {
	return path.join(cyclePath(root, cycle), "SCOPE.md");
}

/** Deterministic gate step; writes gates.json and returns it. */
export function runAuditGates(
	root: string,
	cycle: number,
	snapshot: string,
	options: Omit<RunGatesOptions, "cycle" | "snapshot"> = {},
): GatesFile {
	return runGates(root, { ...options, cycle, snapshot });
}

export interface PhaseState {
	cycle: number;
	gates: boolean;
	finders: Record<AuditWorkstream, boolean>;
	finders_complete: boolean;
	verify: boolean;
	synthesise: boolean;
	adjudicate: boolean;
	next: AuditPhase;
	/** The next concrete phase string, e.g. `find:M`, `verify`, or `complete`. */
	step: string;
	next_finder: AuditWorkstream | null;
}

/** Which phase artifacts exist (files only; no interpretation). */
export function auditPhaseState(root: string, cycle: number): PhaseState {
	const dir = cyclePath(root, cycle);
	const finders = {} as Record<AuditWorkstream, boolean>;
	for (const ws of AUDIT_WORKSTREAMS) {
		finders[ws] = fs.existsSync(path.join(dir, ws.toLowerCase(), "findings.json"));
	}
	const state: PhaseState = {
		cycle,
		gates: fs.existsSync(path.join(dir, "gates.json")),
		finders,
		finders_complete: AUDIT_WORKSTREAMS.every((ws) => finders[ws]),
		verify: fs.existsSync(path.join(dir, "VERIFY.md")),
		synthesise:
			fs.existsSync(path.join(dir, "FINAL_REPORT.md")) &&
			fs.existsSync(path.join(dir, "GRADE.md")) &&
			fs.existsSync(path.join(dir, "rubric.json")),
		adjudicate:
			fs.existsSync(path.join(dir, "ADJUDICATED_REPORT.md")) &&
			fs.existsSync(path.join(dir, "ROOT_ISSUE_REGISTER.csv")),
		next: "complete",
		step: "complete",
		next_finder: null,
	};
	const nextFinder = AUDIT_WORKSTREAMS.find((ws) => !state.finders[ws]) ?? null;
	state.next_finder = nextFinder;
	if (!state.gates) {
		state.next = "gates";
		state.step = "gates";
	} else if (nextFinder) {
		state.next = "find";
		state.step = `find:${nextFinder}`;
	} else if (!state.verify) {
		state.next = "verify";
		state.step = "verify";
	} else if (!state.synthesise) {
		state.next = "synthesise";
		state.step = "synthesise";
	} else if (!state.adjudicate) {
		state.next = "adjudicate";
		state.step = "adjudicate";
	}
	return state;
}

/**
 * Write `.pi/state/AUDIT.json` from the phase artifacts. The conductor calls
 * this after every AUDIT child, so the next step sees the concrete phase
 * (`find:M`, `verify`, …) without an agent writing loop state.
 */
export function syncAuditState(root: string, cycle: number): PhaseState {
	const state = auditPhaseState(root, cycle);
	const file = path.join(root, ".pi", "state", "AUDIT.json");
	writeJsonAtomic(file, {
		schema: "pi-audit-state/1",
		cycle,
		phase: state.step,
		complete: state.step === "complete",
		next: state.next,
		finders: state.finders,
		updated_at: new Date().toISOString(),
	});
	return state;
}

export interface PhaseOrderCheck {
	ok: boolean;
	reason: string;
	missing: string[];
}

/** Refuse a phase whose prerequisite artifacts are absent. */
export function assertAuditPhaseOrder(root: string, cycle: number, phase: AuditPhase): PhaseOrderCheck {
	const state = auditPhaseState(root, cycle);
	const missing: string[] = [];
	if (phase === "gates") return { ok: true, reason: "gates first", missing };
	if (!state.gates) missing.push("gates.json (run_gates)");
	if (phase === "find" && missing.length === 0) return { ok: true, reason: "gates present; finders may run", missing };
	if (phase === "verify") {
		const absent = AUDIT_WORKSTREAMS.filter((ws) => !state.finders[ws]);
		if (absent.length > 0) missing.push(`findings.json for ${absent.join(", ")}`);
	}
	if (phase === "synthesise" && !state.verify) missing.push("VERIFY.md (verifier)");
	if (phase === "adjudicate" && !state.synthesise) missing.push("FINAL_REPORT.md + GRADE.md + rubric.json (synthesizer)");
	if (phase === "complete" && !state.adjudicate) missing.push("ADJUDICATED_REPORT.md + ROOT_ISSUE_REGISTER.csv (adjudicator)");
	return missing.length === 0
		? { ok: true, reason: `${phase}: prerequisites present`, missing }
		: { ok: false, reason: `${phase}: missing ${missing.join("; ")}`, missing };
}

export interface AuditDecisionOptions {
	mapping?: DecisionMapping;
	cycle?: number;
	write?: boolean;
	limitationsDisclosed?: boolean;
	traceabilityOk?: boolean;
	inapplicableGates?: string[];
	waivedGates?: string[];
	calibration?: CalibrationSummary | null;
}

export interface AuditDecisionResult {
	status: DecisionResult["status"];
	grade: GradeOrUncalibrated;
	decision: DecisionResult;
	grade_result: GradeResult;
	blocking_fixes: FixItem[];
	path: string | null;
}

function readJson<T>(file: string): T | null {
	if (!fs.existsSync(file)) return null;
	try {
		return JSON.parse(fs.readFileSync(file, "utf8")) as T;
	} catch {
		return null;
	}
}

function loadCalibration(root: string, explicit?: CalibrationSummary | null): CalibrationSummary | null {
	if (explicit !== undefined) return explicit;
	const file = path.join(root, ".pi", "audit", "calibration", "latest", "recall.json");
	// The UNCALIBRATED marker is authoritative even if a stale recall.json says
	// otherwise: the orchestrator must refuse to grade.
	if (fs.existsSync(path.join(path.dirname(file), "UNCALIBRATED"))) {
		return { recall: 0, rejected_confirmed: 1, calibrated: false };
	}
	const parsed = readJson<{ recall?: number; rejected_confirmed?: number; calibrated?: boolean }>(file);
	if (!parsed || typeof parsed.recall !== "number") return null;
	return {
		recall: parsed.recall,
		rejected_confirmed: parsed.rejected_confirmed ?? 0,
		calibrated: parsed.calibrated === true,
	};
}

function loadDecisionMapping(root: string): DecisionMapping | undefined {
	return readJson<DecisionMapping>(path.join(root, ".pi", "rubrics", "decision.json")) ?? undefined;
}

/**
 * Apply the audit decision rule to the cycle's findings + gates + rubric +
 * calibration, and write `.pi/state/DECISION.json`.
 */
export function applyAuditDecision(root: string, cycle: number, options: AuditDecisionOptions = {}): AuditDecisionResult {
	const dir = cyclePath(root, cycle);
	const findings: RegisteredFinding[] = readRegister(root, cycle);
	const gates = readJson<GatesFile>(path.join(dir, "gates.json"));
	const rubric = readRubricScore(root, cycle);
	const calibration = loadCalibration(root, options.calibration) ?? {
		// Calibration is mandatory once per cycle (plan/07 §7). Without a
		// calibration result the run is uncalibrated and may not grade.
		recall: 0,
		rejected_confirmed: 0,
		calibrated: false,
	};
	const mapping = options.mapping ?? loadDecisionMapping(root);
	const traceabilityGate = gates?.gates.find((gate) => gate.id === "traceability");
	const traceability =
		options.traceabilityOk !== undefined
			? { ok: options.traceabilityOk }
			: traceabilityGate
				? traceabilityGate.status === "pass"
					? { ok: true }
					: traceabilityGate.status === "fail"
						? { ok: false }
						: null
				: null;
	const gradeResult = auditGrade({
		findings,
		gates: gates
			? { ok: gates.ok, cycle: gates.cycle, snapshot: gates.snapshot, gates: gates.gates.map((gate) => ({ id: gate.id, status: gate.status, detail: gate.detail, hash_ok: gate.hash_ok })) }
			: null,
		rubric,
		calibration,
		traceability,
		limitations_disclosed: options.limitationsDisclosed === true,
		inapplicable_gates: options.inapplicableGates,
		waived_gates: options.waivedGates,
		cycle,
		max_cycles: mapping?.max_cycles ?? 5,
	});
	const decision = applyDecision(gradeResult.grade, gradeResult.blocking_fixes, mapping, cycle);
	let file: string | null = null;
	if (options.write !== false) {
		file = path.join(root, ".pi", "state", "DECISION.json");
		writeJsonAtomic(file, {
			schema: "pi-decision/1",
			cycle,
			done: decision.done,
			status: decision.status,
			grade: gradeResult.grade,
			journal: gradeResult.journal,
			reason: decision.reason,
			action: decision.action,
			reaudit: decision.reaudit,
			audit_usable: gradeResult.usable,
			blocking_fixes: gradeResult.blocking_fixes,
			rule_trace: gradeResult.trace,
			decision_trace: decision.trace,
			source: {
				findings: path.relative(root, path.join(dir, "findings.json")),
				gates: gates ? path.relative(root, path.join(dir, "gates.json")) : null,
				grade: path.relative(root, path.join(dir, "GRADE.md")),
				rubric: path.relative(root, path.join(dir, "rubric.json")),
				calibration: calibration ? ".pi/audit/calibration/latest/recall.json" : null,
			},
		});
	}
	return {
		status: decision.status,
		grade: gradeResult.grade,
		decision,
		grade_result: gradeResult,
		blocking_fixes: gradeResult.blocking_fixes,
		path: file,
	};
}

/**
 * Finders run in parallel in batches of at most 8. The list is deterministic so
 * a replay spawns the same batches.
 */
export function finderBatches(): AuditWorkstream[][] {
	const batches: AuditWorkstream[][] = [];
	for (let index = 0; index < AUDIT_WORKSTREAMS.length; index += MAX_PARALLEL_FINDERS) {
		batches.push(AUDIT_WORKSTREAMS.slice(index, index + MAX_PARALLEL_FINDERS) as AuditWorkstream[]);
	}
	return batches;
}

/** A finder may not see another workstream's register (phase independence). */
export function finderInputPaths(root: string, cycle: number, workstream: AuditWorkstream): string[] {
	const dir = cyclePath(root, cycle);
	return [
		path.relative(root, scopePath(root, cycle)),
		path.relative(root, path.join(dir, "_PROTOCOL.md")),
		path.relative(root, path.join(dir, workstream.toLowerCase(), "BRIEF.md")),
		path.relative(root, path.join(dir, "gates.json")),
	];
}

/** Verified ids the synthesizer may cite (phase-2 input). */
export function verifiedIds(root: string, cycle: number): string[] {
	return readRegister(root, cycle)
		.filter((finding) => finding.status === "verified" && finding.confidence !== "low" && finding.phase !== "synth" && finding.phase !== "adjudicate")
		.map((finding) => finding.id);
}

/** Evidence that all ten finder registers exist before verification. */
export function finderCoverage(root: string, cycle: number): { complete: boolean; missing: AuditWorkstream[] } {
	const missing = AUDIT_WORKSTREAMS.filter((ws) => readWorkstreamRegister(root, cycle, ws).length === 0);
	return { complete: missing.length === 0, missing };
}
