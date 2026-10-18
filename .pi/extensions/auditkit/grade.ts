/**
 * Grade derivation and the decision rule (plan/07_AUDIT_HARNESS.md §7,
 * plan/02_ARCHITECTURE.md §5.4).
 *
 * The grade is an application of written rules to verified findings, gate
 * results, a rubric score, and the calibration verdict — never a judgement call.
 * Every cap records a trace line naming the evidence that produced it, so
 * `GRADE.md` can cite the failing gate that capped the grade.
 *
 * No pi imports: usable from `node --test` and from the tools.
 */

import {
	CORE_S1_WORKSTREAMS,
	isConfirmed,
	severityRank,
	workstreamOfId,
	type Finding,
	type Owner,
	type Severity,
	type Workstream,
} from "./schema.ts";

export const GRADES = [
	"strong reject",
	"reject",
	"weak reject",
	"borderline",
	"weak accept",
	"accept",
	"strong accept",
] as const;
export type Grade = (typeof GRADES)[number];

export const BLOCKED_GRADE = "uncalibrated";
export type GradeOrUncalibrated = Grade | typeof BLOCKED_GRADE;

export interface GateEntry {
	id: string;
	status: "pass" | "fail" | "skip" | "error";
	detail?: string;
	hash_ok?: boolean;
}

export interface GateResults {
	ok: boolean;
	gates: GateEntry[];
	cycle?: number;
	snapshot?: string;
}

export interface RubricSummary {
	total: number | null;
	complete?: boolean;
	unresolved?: string[];
}

export interface CalibrationSummary {
	recall: number;
	rejected_confirmed: number;
	calibrated: boolean;
}

export interface GradeInput {
	findings: readonly Finding[];
	gates?: GateResults | null;
	rubric?: RubricSummary | null;
	calibration?: CalibrationSummary | null;
	traceability?: { ok: boolean; dangling?: string[]; uncited?: string[] } | null;
	limitations_disclosed?: boolean;
	/** Gates the adjudicator declared inapplicable, with a recorded reason. */
	inapplicable_gates?: string[];
	/** Gates the supervisor waived with a ledger-visible record. */
	waived_gates?: string[];
	cycle?: number;
	max_cycles?: number;
}

export interface TraceLine {
	rule: string;
	effect: "base" | "cap" | "require" | "refuse" | "none";
	grade: GradeOrUncalibrated;
	evidence: string;
}

export interface FixItem {
	finding_id: string;
	title: string;
	severity: Severity;
	owner: Owner;
	effort: "small" | "medium" | "large";
	feasible: "feasible" | "needs-human";
	blocking: boolean;
	reason: string;
}

export interface GradeResult {
	usable: boolean;
	grade: GradeOrUncalibrated;
	base_grade: Grade | null;
	journal: "reject" | "major revision" | "minor revision" | "accept" | "none";
	capped: boolean;
	trace: TraceLine[];
	fix_list: FixItem[];
	blocking_fixes: FixItem[];
	gate_failures: string[];
	confirmed_counts: { S0: number; S1: number; S2: number; S3: number };
	reason: string;
}

const JOURNAL: Record<Grade, GradeResult["journal"]> = {
	"strong reject": "reject",
	reject: "reject",
	"weak reject": "major revision",
	borderline: "minor revision",
	"weak accept": "minor revision",
	accept: "accept",
	"strong accept": "accept",
};

/** Rubric total (0..1) → grade. Boundaries are inclusive lower bounds. */
export const RUBRIC_BOUNDARIES: ReadonlyArray<[number, Grade]> = [
	[0.9, "strong accept"],
	[0.8, "accept"],
	[0.68, "weak accept"],
	[0.55, "borderline"],
	[0.4, "weak reject"],
	[0.25, "reject"],
	[-Infinity, "strong reject"],
];

export function gradeFromRubric(total: number | null): Grade {
	if (total === null || Number.isNaN(total)) return "borderline";
	for (const [boundary, grade] of RUBRIC_BOUNDARIES) {
		if (total >= boundary) return grade;
	}
	return "strong reject";
}

function worse(a: Grade, b: Grade): Grade {
	return GRADES.indexOf(a) <= GRADES.indexOf(b) ? a : b;
}

function effortFor(finding: Finding): FixItem["effort"] {
	if (finding.severity === "S0") return "large";
	if (finding.severity === "S1") return "medium";
	return "small";
}

function feasibilityFor(finding: Finding): { feasible: FixItem["feasible"]; reason: string } {
	const text = `${finding.consequence} ${finding.observed}`.toLowerCase();
	if (finding.owner === "protocol") {
		return {
			feasible: "needs-human",
			reason: "protocol changes need a recorded amendment and a human gate",
		};
	}
	if (finding.owner === "env" && /cluster|container|singularity|gpu/.test(text)) {
		return { feasible: "needs-human", reason: "environment fix needs cluster/container access" };
	}
	if (/data snapshot|missing data|deferred|hardware|author intent/.test(text)) {
		return { feasible: "needs-human", reason: "needs data, hardware, or author input" };
	}
	if (finding.owner === "paper") {
		return { feasible: "feasible", reason: "paper wording/claims can be corrected in PAPER state" };
	}
	return { feasible: "feasible", reason: "code/docs change with a regression test" };
}

export function buildFixList(findings: readonly Finding[]): FixItem[] {
	return findings
		.filter(isConfirmed)
		.slice()
		.sort((a, b) => severityRank(a.severity) - severityRank(b.severity) || a.id.localeCompare(b.id))
		.map((finding) => {
			const { feasible, reason } = feasibilityFor(finding);
			const blocking = finding.severity === "S0" || finding.severity === "S1" || finding.blocking === true;
			return {
				finding_id: finding.id,
				title: finding.title,
				severity: finding.severity,
				owner: finding.owner,
				effort: effortFor(finding),
				feasible,
				blocking,
				reason,
			};
		});
}

function countsBySeverity(findings: readonly Finding[]): GradeResult["confirmed_counts"] {
	const counts = { S0: 0, S1: 0, S2: 0, S3: 0 };
	for (const finding of findings) {
		if (isConfirmed(finding)) counts[finding.severity] += 1;
	}
	return counts;
}

function failedGates(gates: GateResults | null | undefined, waived: ReadonlySet<string>, inapplicable: ReadonlySet<string>): GateEntry[] {
	if (!gates) return [];
	return gates.gates.filter(
		(gate) =>
			(gate.status === "fail" || gate.status === "error") &&
			!waived.has(gate.id) &&
			!inapplicable.has(gate.id),
	);
}

/**
 * Compute the grade from the rules. The order matters only for the trace; the
 * grade itself is the worst of the base and every applicable cap.
 */
export function auditGrade(input: GradeInput): GradeResult {
	const findings = input.findings;
	const confirmed = findings.filter(isConfirmed);
	const confirmedCounts = countsBySeverity(findings);
	const waived = new Set(input.waived_gates ?? []);
	const inapplicable = new Set(input.inapplicable_gates ?? []);
	const trace: TraceLine[] = [];
	const fixList = buildFixList(findings);
	const blockingFixes = fixList.filter((item) => item.blocking);

	// Calibration dominates: an uncalibrated run may not grade the paper.
	if (input.calibration && (input.calibration.recall < 0.6 || input.calibration.rejected_confirmed > 0)) {
		trace.push({
			rule: "calibration",
			effect: "refuse",
			grade: BLOCKED_GRADE,
			evidence: `recall=${input.calibration.recall.toFixed(3)} (threshold 0.6), rejected-leads-confirmed=${input.calibration.rejected_confirmed}`,
		});
		return {
			usable: false,
			grade: BLOCKED_GRADE,
			base_grade: null,
			journal: "none",
			capped: true,
			trace,
			fix_list: fixList,
			blocking_fixes: blockingFixes,
			gate_failures: failedGates(input.gates, waived, inapplicable).map((gate) => gate.id),
			confirmed_counts: confirmedCounts,
			reason:
				"the audit run is UNCALIBRATED (plan/07 §7): recall below 0.6 or a rejected Phase-A lead was confirmed — the orchestrator refuses to grade",
		};
	}

	const rubricTotal = input.rubric?.total ?? null;
	let grade = gradeFromRubric(rubricTotal);
	const base = grade;
	trace.push({
		rule: "rubric-base",
		effect: "base",
		grade,
		evidence:
			rubricTotal === null
				? "rubric incomplete: no total; base = borderline until every leaf has evidence"
				: `rubric total=${rubricTotal.toFixed(3)} → ${grade}`,
	});

	const cap = (rule: string, capGrade: Grade, evidence: string): void => {
		const next = worse(grade, capGrade);
		const changed = next !== grade;
		grade = next;
		trace.push({ rule, effect: "cap", grade: next, evidence: changed ? `${evidence} → capped at ${capGrade}` : `${evidence} → already at/below ${capGrade}` });
	};

	// Rule 1: any confirmed S0.
	const s0 = confirmed.filter((finding) => finding.severity === "S0");
	if (s0.length > 0) {
		cap("blocking-s0", "weak reject", `confirmed S0: ${s0.map((finding) => finding.id).join(", ")}`);
	}

	// Rule 2: any confirmed S1 in M/S/P/C.
	const coreS1 = confirmed.filter(
		(finding) =>
			finding.severity === "S1" &&
			CORE_S1_WORKSTREAMS.includes((workstreamOfId(finding.id) ?? "M") as Workstream),
	);
	if (coreS1.length > 0) {
		cap(
			"blocking-core-s1",
			"weak reject",
			`confirmed S1 in ${[...CORE_S1_WORKSTREAMS].join("/")}: ${coreS1.map((finding) => finding.id).join(", ")}`,
		);
	}

	// Rule 3: an unfixed confirmed S1 outside the core workstreams.
	const otherS1 = confirmed.filter(
		(finding) =>
			finding.severity === "S1" &&
			!CORE_S1_WORKSTREAMS.includes((workstreamOfId(finding.id) ?? "M") as Workstream) &&
			finding.fixed !== true,
	);
	if (otherS1.length > 0) {
		cap(
			"blocking-other-s1",
			"borderline",
			`unfixed confirmed S1 outside M/S/P/C: ${otherS1.map((finding) => finding.id).join(", ")}`,
		);
	}

	// Rule 4: a failed deterministic gate, unless waived or declared inapplicable.
	const gateFailures = failedGates(input.gates, waived, inapplicable);
	if (gateFailures.length > 0) {
		cap(
			"failed-gate",
			"weak reject",
			`failed deterministic gate(s): ${gateFailures.map((gate) => `${gate.id}${gate.detail ? ` (${gate.detail.slice(0, 80)})` : ""}`).join("; ")}`,
		);
	} else {
		trace.push({
			rule: "failed-gate",
			effect: "none",
			grade,
			evidence: input.gates
				? `all ${input.gates.gates.length} deterministic gates pass/skip${inapplicable.size ? ` (inapplicable: ${[...inapplicable].join(", ")})` : ""}`
				: "no gate results supplied to the grade computation",
		});
	}

	// Rule 5: accept/strong accept preconditions.
	if (GRADES.indexOf(grade) > GRADES.indexOf("weak accept")) {
		const problems: string[] = [];
		const unfixed = confirmed.filter((finding) => finding.fixed !== true);
		const s1 = unfixed.filter((finding) => finding.severity === "S1");
		const s0 = unfixed.filter((finding) => finding.severity === "S0");
		if (s0.length > 0 || s1.length > 0) {
			problems.push(`${s0.length} unfixed confirmed S0 and ${s1.length} unfixed confirmed S1`);
		}
		if (!input.gates || !input.gates.ok || failedGates(input.gates, waived, inapplicable).length > 0) {
			problems.push("gates are not all green");
		}
		if (!input.traceability || !input.traceability.ok) {
			problems.push("traceability is not complete");
		}
		if (input.limitations_disclosed !== true) {
			problems.push("limitation disclosures are missing");
		}
		if (problems.length > 0) {
			cap("accept-preconditions", "weak accept", `accept requires all conditions: ${problems.join("; ")}`);
		} else {
			trace.push({
				rule: "accept-preconditions",
				effect: "require",
				grade,
				evidence: "no confirmed S0/S1, all gates green, traceability complete, limitations disclosed",
			});
		}
	}

	// Rule 6: cycle cap (plan/02 §5.4: max 5 cycles, then BLOCKED for a human).
	if (input.cycle !== undefined && input.max_cycles !== undefined && input.cycle > input.max_cycles) {
		trace.push({
			rule: "max-audit-cycles",
			effect: "refuse",
			grade: BLOCKED_GRADE,
			evidence: `cycle ${input.cycle} exceeds max_cycles ${input.max_cycles} — human direction required`,
		});
		return {
			usable: false,
			grade: BLOCKED_GRADE,
			base_grade: base,
			journal: "none",
			capped: true,
			trace,
			fix_list: fixList,
			blocking_fixes: blockingFixes,
			gate_failures: gateFailures.map((gate) => gate.id),
			confirmed_counts: confirmedCounts,
			reason: `audit cycle ${input.cycle} exceeds the maximum of ${input.max_cycles}`,
		};
	}

	const capped = grade !== base;
	return {
		usable: true,
		grade,
		base_grade: base,
		journal: JOURNAL[grade],
		capped,
		trace,
		fix_list: fixList,
		blocking_fixes: blockingFixes,
		gate_failures: gateFailures.map((gate) => gate.id),
		confirmed_counts: confirmedCounts,
		reason: capped
			? `grade capped from ${base} to ${grade} by ${trace.filter((line) => line.effect === "cap").map((line) => line.rule).join(", ")}`
			: `grade ${grade} from rubric base with no applicable cap`,
	};
}

// ---------------------------------------------------------------------------
// Decision rule (plan/02_ARCHITECTURE.md §5.4)
// ---------------------------------------------------------------------------

export interface DecisionEntry {
	action: string;
	reaudit: boolean;
	done: boolean;
	description?: string;
}

export interface DecisionMapping {
	schema?: string;
	kind?: string;
	mapping: Record<string, DecisionEntry>;
	max_cycles?: number;
}

export const DEFAULT_DECISION: DecisionMapping = {
	schema: "vangja-rubric/1",
	kind: "decision",
	max_cycles: 5,
	mapping: {
		"strong reject": { action: "reopen-p0", reaudit: true, done: false, description: "reopen P0 tasks from the findings; new cycle after fixes" },
		reject: { action: "reopen-p0", reaudit: true, done: false, description: "reopen P0 tasks from the findings; new cycle after fixes" },
		"weak reject": { action: "fix-s0-s1", reaudit: true, done: false, description: "fix all S0/S1 and the S2s named as blocking; re-audit" },
		borderline: { action: "fix-s0-s1", reaudit: true, done: false, description: "fix all S0/S1 and the S2s named as blocking; re-audit" },
		"weak accept": { action: "fix-feasible", reaudit: true, done: false, description: "fix the feasible fixes; re-audit unless purely editorial" },
		accept: { action: "fix-blocking-or-done", reaudit: false, done: true, description: "no blocking fixes → DONE (human sign-off)" },
		"strong accept": { action: "fix-blocking-or-done", reaudit: false, done: true, description: "no blocking fixes → DONE (human sign-off)" },
	},
};

export interface DecisionResult {
	status: "DONE" | "WORK" | "BLOCKED";
	action: string;
	reaudit: boolean;
	done: boolean;
	grade: GradeOrUncalibrated;
	reason: string;
	trace: Array<{ rule: string; evidence: string }>;
	blocking_fixes: FixItem[];
}

/** Apply `rubrics/decision.json` to a grade + fix list. */
export function applyDecision(
	grade: GradeOrUncalibrated,
	blockingFixes: readonly FixItem[],
	mapping: DecisionMapping = DEFAULT_DECISION,
	cycle = 1,
): DecisionResult {
	const maxCycles = mapping.max_cycles ?? 5;
	if (cycle > maxCycles) {
		return {
			status: "BLOCKED",
			action: "blocked-cycle-cap",
			reaudit: false,
			done: false,
			grade,
			reason: `audit cycle ${cycle} > max ${maxCycles}: BLOCKED for human direction regardless of grade`,
			trace: [{ rule: "max-audit-cycles", evidence: `${cycle} > ${maxCycles}` }],
			blocking_fixes: [...blockingFixes],
		};
	}
	if (grade === BLOCKED_GRADE) {
		return {
			status: "BLOCKED",
			action: "blocked-audit-failure",
			reaudit: false,
			done: false,
			grade,
			reason: "the audit did not produce a usable grade (uncalibrated); a human decides",
			trace: [{ rule: "audit-usable", evidence: `grade=${grade}` }],
			blocking_fixes: [...blockingFixes],
		};
	}
	const entry = mapping.mapping[grade];
	if (!entry) {
		return {
			status: "BLOCKED",
			action: "blocked-unknown-grade",
			reaudit: false,
			done: false,
			grade,
			reason: `grade "${grade}" has no entry in the decision mapping`,
			trace: [{ rule: "decision-mapping", evidence: `missing grade ${grade}` }],
			blocking_fixes: [...blockingFixes],
		};
	}

	const trace: DecisionResult["trace"] = [{ rule: "decision-mapping", evidence: `${grade} → ${entry.action}` }];
	let { reaudit, done } = entry;

	if (GRADES.indexOf(grade) === GRADES.indexOf("weak accept") && blockingFixes.length > 0) {
		const purelyEditorial = blockingFixes.every((fix) => fix.owner === "paper" || fix.owner === "docs");
		if (purelyEditorial) {
			reaudit = false;
			trace.push({ rule: "weak-accept-editorial", evidence: "all blocking fixes are paper/docs edits → no re-audit" });
		}
	}

	if (grade === "accept" || grade === "strong accept") {
		if (blockingFixes.length === 0) {
			done = true;
			reaudit = false;
			trace.push({ rule: "accept-no-blocking", evidence: "no blocking fixes → DONE (human sign-off still required)" });
		} else {
			done = false;
			reaudit = true;
			trace.push({
				rule: "accept-with-blocking",
				evidence: `${blockingFixes.length} blocking fix(es) remain → fix and re-audit`,
			});
		}
	}

	if (grade === "strong reject" || grade === "reject") {
		trace.push({ rule: "reject-reopen", evidence: "reopen P0 tasks from the findings; new cycle after fixes" });
	}

	return {
		status: done ? "DONE" : "WORK",
		action: entry.action,
		reaudit,
		done,
		grade,
		reason: entry.description ?? entry.action,
		trace,
		blocking_fixes: [...blockingFixes],
	};
}
