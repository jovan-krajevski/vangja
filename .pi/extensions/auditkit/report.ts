/**
 * Report assembly: findings markdown, the author-facing FINAL_REPORT.md, GRADE.md,
 * ROOT_ISSUE_REGISTER.csv, and QUESTIONS_FOR_AUTHORS.md.
 *
 * The synthesis layer may only consume verified finder-phase findings. A record
 * created by the synthesizer is rejected unless it cites a verified id; a
 * finding that is `to-verify`, `refuted`, or `unresolved` never appears as a
 * confirmed report item.
 *
 * No pi imports: usable from `node --test`.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { writeFileAtomic } from "../../lib/io.ts";
import type { GatesFile } from "../verify/gates.ts";
import type { FixItem, GradeResult } from "./grade.ts";
import { GRADES } from "./grade.ts";
import {
	isConfirmed,
	severityRank,
	SEVERITIES,
	summarizeFindings,
	workstreamOfId,
	type Finding,
	type Severity,
} from "./schema.ts";

export interface AssembleInput {
	root: string;
	cycle: number;
	snapshot: string;
	findings: readonly Finding[];
	gates: GatesFile | null;
	grade?: GradeResult | null;
	fixList?: FixItem[];
	metaCoveragePath?: string | null;
	scopePath?: string | null;
	now?: string;
}

export interface AssembleResult {
	final_report: string;
	rejected: Array<{ id: string; reason: string }>;
	reported: string[];
	unverified: string[];
	refuted: string[];
	synth_added: string[];
}

const WS_NAMES: Record<string, string> = {
	M: "Methodology & experimental design",
	S: "Statistics & math",
	P: "Probabilistic UQ & calibration",
	D: "Data handling & leakage",
	I: "Implementation & software quality",
	R: "Reproducibility & provenance",
	H: "Documentation & hygiene",
	E: "Evaluation integrity & anti-gaming",
	C: "Claims ↔ artifacts traceability",
	L: "Literature & positioning",
};

function cell(text: string): string {
	return String(text ?? "").replace(/\|/g, "\\|").replace(/\n+/g, " ");
}

function sorted(findings: readonly Finding[]): Finding[] {
	return findings
		.slice()
		.sort(
			(a, b) =>
				severityRank(a.severity) - severityRank(b.severity) ||
				(workstreamOfId(a.id) ?? "Z").localeCompare(workstreamOfId(b.id) ?? "Z") ||
				a.id.localeCompare(b.id),
		);
}

export function renderFindingsMarkdown(cycle: number, findings: readonly Finding[], workstream?: string): string {
	const lines: string[] = [];
	lines.push(`# Findings${workstream ? ` — WS-${workstream} (${WS_NAMES[workstream] ?? workstream})` : ""} — cycle ${cycle}`, "");
	const summary = summarizeFindings(findings);
	lines.push(
		`${summary.count} finding(s): ` +
			SEVERITIES.map((severity) => `${severity}=${summary.bySeverity[severity] ?? 0}`).join(", "),
		"",
	);
	for (const severity of SEVERITIES) {
		const group = sorted(findings.filter((finding) => finding.severity === severity));
		if (group.length === 0) continue;
		lines.push(`## ${severity} (${group.length})`, "");
		for (const finding of group) {
			lines.push(`### ${finding.id} · ${finding.title}`, "");
			lines.push(`- **Location:** ${finding.location}`);
			lines.push(`- **Snapshot:** ${finding.snapshot}`);
			lines.push(`- **Command / test:** \`${finding.command}\``);
			lines.push(`- **Expected:** ${finding.expected}`);
			lines.push(`- **Observed:** ${finding.observed}`);
			lines.push(`- **Consequence:** ${finding.consequence}`);
			lines.push(`- **Severity:** ${finding.severity}`);
			lines.push(`- **Confidence:** ${finding.confidence}`);
			lines.push(`- **Status:** ${finding.status}`);
			lines.push(`- **Reproducible:** ${finding.reproducible}`);
			lines.push(`- **Owner:** ${finding.owner}`);
			if (finding.evidence.length > 0) lines.push(`- **Evidence:** ${finding.evidence.join(" · ")}`);
			if ((finding.cross_refs ?? []).length > 0) lines.push(`- **Cross-refs:** ${finding.cross_refs.join(", ")}`);
			if (finding.question_tag) lines.push(`- **Question tag:** [${finding.question_tag}]`);
			lines.push("");
		}
	}
	if (findings.length === 0) lines.push("_No findings reported within this boundary._", "");
	return `${lines.join("\n")}\n`;
}

export function writeFindings(root: string, cycle: number, findings: readonly Finding[], workstream?: string): string {
	const dir = workstream
		? path.join(root, ".pi", "audit", `cycle-${cycle}`, workstream.toLowerCase())
		: path.join(root, ".pi", "audit", `cycle-${cycle}`);
	const file = path.join(dir, "FINDINGS.md");
	writeFileAtomic(file, renderFindingsMarkdown(cycle, findings, workstream));
	return file;
}

export function writeFindingsJson(root: string, cycle: number, findings: readonly Finding[], workstream?: string): string {
	const dir = workstream
		? path.join(root, ".pi", "audit", `cycle-${cycle}`, workstream.toLowerCase())
		: path.join(root, ".pi", "audit", `cycle-${cycle}`);
	const file = path.join(dir, "findings.json");
	const summary = summarizeFindings(findings);
	writeFileAtomic(
		file,
		`${JSON.stringify({ schema: "vangja-findings/1", cycle, workstream: workstream ?? null, summary, findings }, null, 2)}\n`,
	);
	return file;
}

function gatesTable(gates: GatesFile | null): string[] {
	if (!gates) return ["_No deterministic gate results were supplied._", ""];
	const lines = ["| Gate | Status | Detail |", "|---|---|---|"];
	for (const gate of gates.gates) {
		lines.push(`| ${gate.id} | ${gate.status} | ${cell(gate.detail ?? "")} |`);
	}
	lines.push("");
	return lines;
}

function gradeSection(grade: GradeResult): string[] {
	const lines: string[] = [];
	lines.push("## Grade", "");
	lines.push(`**Grade:** ${grade.grade}${grade.usable ? "" : " (not usable — the orchestrator refuses to grade)"}`);
	if (grade.base_grade) lines.push(`**Rubric base:** ${grade.base_grade}`);
	lines.push(`**Journal mapping:** ${grade.journal}`);
	lines.push(`**Reason:** ${grade.reason}`);
	lines.push("");
	lines.push("| Rule | Effect | Grade | Evidence |", "|---|---|---|---|");
	for (const trace of grade.trace) {
		lines.push(`| ${trace.rule} | ${trace.effect} | ${trace.grade} | ${cell(trace.evidence)} |`);
	}
	lines.push("");
	if (grade.gate_failures.length > 0) {
		lines.push(`**Failing gates that capped the grade:** ${grade.gate_failures.join(", ")}`);
		lines.push("");
	}
	return lines;
}

function fixSection(fixList: readonly FixItem[]): string[] {
	const lines = ["## Fix list", ""];
	if (fixList.length === 0) {
		lines.push("_No confirmed findings: no fixes required._", "");
		return lines;
	}
	lines.push("| Finding | Severity | Owner | Effort | Feasible | Blocking | Fix |");
	lines.push("|---|---|---|---|---|---|---|");
	for (const fix of fixList) {
		lines.push(
			`| ${fix.finding_id} | ${fix.severity} | ${fix.owner} | ${fix.effort} | ${fix.feasible} | ${fix.blocking ? "yes" : "no"} | ${cell(fix.reason)} |`,
		);
	}
	lines.push("");
	return lines;
}

/**
 * Assemble FINAL_REPORT.md. Verified finder-phase findings only; synthesizer
 * records need a `cites` pointer to a verified id.
 */
export function assembleReport(input: AssembleInput): AssembleResult {
	const roots = path.join(input.root, ".pi", "audit", `cycle-${input.cycle}`);
	fs.mkdirSync(roots, { recursive: true });
	const rejected: AssembleResult["rejected"] = [];
	const reported: Finding[] = [];
	const unverified: string[] = [];
	const refuted: string[] = [];
	const synthAdded: string[] = [];
	const verifiedIds = new Set(
		input.findings
			.filter((finding) => finding.phase !== "synth" && finding.phase !== "adjudicate" && finding.status === "verified")
			.map((finding) => finding.id),
	);

	for (const finding of input.findings) {
		if (finding.phase === "synth" || finding.phase === "adjudicate") {
			const cites = Array.isArray(finding.cites) ? (finding.cites as string[]) : [];
			const allowed = cites.some((id) => verifiedIds.has(id));
			if (!allowed) {
				rejected.push({
					id: finding.id,
					reason: "synthesis/adjudication may not add a primary finding; no verified id is cited",
				});
				synthAdded.push(finding.id);
				continue;
			}
			reported.push(finding);
			continue;
		}
		if (finding.status === "verified" && finding.confidence !== "low") {
			reported.push(finding);
		} else if (finding.status === "refuted") {
			refuted.push(finding.id);
		} else {
			unverified.push(finding.id);
		}
	}

	const lines: string[] = [];
	lines.push(`# Audit report — cycle ${input.cycle}`, "");
	lines.push(`**Snapshot:** ${input.snapshot}`);
	lines.push(`**Generated:** ${input.now ?? new Date().toISOString()}`);
	lines.push(`**Reported findings:** ${reported.length} verified (of ${input.findings.length} registered)`);
	lines.push("");
	lines.push("## Deterministic gates", "");
	lines.push(...gatesTable(input.gates));
	if (input.gates) {
		lines.push(
			`Gate summary: ${input.gates.ok ? "all green" : `failed ${input.gates.failed.join(", ")}`}${
				input.gates.skipped.length > 0 ? `; skipped ${input.gates.skipped.join(", ")}` : ""
			}`,
			"",
		);
	}

	lines.push("## Confirmed findings", "");
	if (reported.length === 0) {
		lines.push("_No confirmed findings._", "");
	}
	for (const severity of SEVERITIES) {
		const group = sorted(reported.filter((finding) => finding.severity === severity));
		if (group.length === 0) continue;
		lines.push(`### ${severity} (${group.length})`, "");
		for (const finding of group) {
			lines.push(`#### ${finding.id} · ${finding.title}`, "");
			lines.push(`- **Location:** ${finding.location}`);
			lines.push(`- **Command:** \`${finding.command}\``);
			lines.push(`- **Expected:** ${finding.expected}`);
			lines.push(`- **Observed:** ${finding.observed}`);
			lines.push(`- **Consequence:** ${finding.consequence}`);
			lines.push(`- **Evidence:** ${finding.evidence.join(" · ")}`);
			lines.push(`- **Owner:** ${finding.owner}`);
			lines.push("");
		}
	}

	if (input.grade) lines.push(...gradeSection(input.grade));
	if (input.fixList) lines.push(...fixSection(input.fixList));

	const metaFile = input.metaCoveragePath
		? path.resolve(input.root, input.metaCoveragePath)
		: path.join(roots, "META_COVERAGE.md");
	lines.push("## Coverage and limitations", "");
	if (fs.existsSync(metaFile)) {
		lines.push(fs.readFileSync(metaFile, "utf8").trim(), "");
	} else {
		lines.push(
			"_No meta-review coverage file was supplied. The audit does not claim complete coverage; the per-workstream coverage statements in `<ws>/FINDINGS.md` are the record of what was inspected._",
			"",
		);
	}

	lines.push("## Unverified leads and refutations", "");
	lines.push(
		`Unverified / to-verify: ${unverified.length > 0 ? unverified.join(", ") : "none"}. Refuted: ${refuted.length > 0 ? refuted.join(", ") : "none"}.`,
		"",
	);
	if (rejected.length > 0) {
		lines.push("## Rejected synthesis records", "");
		for (const entry of rejected) lines.push(`- **${entry.id}**: ${entry.reason}`);
		lines.push("");
	}
	lines.push("## Residual risk", "");
	lines.push(
		"- The audit inspects a frozen snapshot; anything merged after the snapshot tag is outside scope.",
	);
	lines.push("- Findings are limited to what the briefs asked the workstreams to check; see the coverage section.");
	if (input.gates?.skipped.length) {
		lines.push(`- Skipped gates (${input.gates.skipped.join(", ")}) prove nothing unless the adjudicator records why they are inapplicable.`);
	}
	lines.push("");

	const finalReport = path.join(roots, "FINAL_REPORT.md");
	writeFileAtomic(finalReport, `${lines.join("\n")}\n`);
	return { final_report: finalReport, rejected, reported: reported.map((finding) => finding.id), unverified, refuted, synth_added: synthAdded };
}

/** Write GRADE.md with the rule trace, the failing gates, and the fix list. */
export function writeGrade(root: string, cycle: number, grade: GradeResult): string {
	const lines: string[] = [];
	lines.push(`# Grade — cycle ${cycle}`, "");
	lines.push(`**Grade:** ${grade.grade}${grade.usable ? "" : " — UNCALIBRATED / not usable"}`);
	lines.push(`**Base grade:** ${grade.base_grade ?? "none"}`);
	lines.push(`**Journal mapping:** ${grade.journal}`);
	lines.push(`**Capped:** ${grade.capped ? "yes" : "no"}`);
	lines.push(`**Reason:** ${grade.reason}`);
	lines.push("");
	lines.push("## Rule trace", "");
	lines.push("| # | Rule | Effect | Result | Evidence |");
	lines.push("|---:|---|---|---|---|");
	grade.trace.forEach((trace, index) => {
		lines.push(`| ${index + 1} | ${trace.rule} | ${trace.effect} | ${trace.grade} | ${cell(trace.evidence)} |`);
	});
	lines.push("");
	lines.push("## Failing gates", "");
	if (grade.gate_failures.length === 0) {
		lines.push("_None._", "");
	} else {
		for (const gate of grade.gate_failures) lines.push(`- \`${gate}\` — a deterministic gate failed; it caps the grade at weak reject unless the adjudicator records inapplicability.`);
		lines.push("");
	}
	lines.push("## Confirmed findings by severity", "");
	lines.push(
		`S0=${grade.confirmed_counts.S0}, S1=${grade.confirmed_counts.S1}, S2=${grade.confirmed_counts.S2}, S3=${grade.confirmed_counts.S3}`,
		"",
	);
	lines.push("## Blocking fixes", "");
	if (grade.blocking_fixes.length === 0) {
		lines.push("_None._", "");
	} else {
		for (const fix of grade.blocking_fixes) {
			lines.push(`- **${fix.finding_id}** (${fix.severity}, ${fix.owner}, ${fix.effort}, ${fix.feasible}): ${fix.title}`);
		}
		lines.push("");
	}
	lines.push("## Fix list", "");
	lines.push(...fixSection(grade.fix_list));
	const file = path.join(root, ".pi", "audit", `cycle-${cycle}`, "GRADE.md");
	writeFileAtomic(file, `${lines.join("\n")}\n`);
	return file;
}

// ---------------------------------------------------------------------------
// Root issue register + questions
// ---------------------------------------------------------------------------

export interface RootIssue {
	id: string;
	title: string;
	finding_ids: string[];
	owner: string;
	fix: string;
	status: string;
	dependency: string;
	severity: Severity;
}

/** Group confirmed findings into root issues (explicit `root_issue` or one per finding). */
export function buildRootIssues(findings: readonly Finding[]): RootIssue[] {
	const groups = new Map<string, Finding[]>();
	for (const finding of findings.filter(isConfirmed)) {
		const key =
			typeof finding.root_issue === "string" && finding.root_issue.trim() !== ""
				? finding.root_issue.trim()
				: finding.id;
		const list = groups.get(key) ?? [];
		list.push(finding);
		groups.set(key, list);
	}
	const issues: RootIssue[] = [];
	let index = 1;
	for (const [key, group] of [...groups.entries()].sort((a, b) => {
		const severity = Math.min(...a[1].map((finding) => severityRank(finding.severity)));
		const other = Math.min(...b[1].map((finding) => severityRank(finding.severity)));
		return severity - other || key.localeCompare(key);
	})) {
		const worst = group.slice().sort((a, b) => severityRank(a.severity) - severityRank(b.severity))[0];
		issues.push({
			id: `RI-${String(index).padStart(3, "0")}`,
			title: key === worst.id ? worst.title : key,
			finding_ids: group.map((finding) => finding.id).sort(),
			owner: worst.owner,
			fix: worst.expected,
			status: worst.fixed ? "fixed" : "open",
			dependency:
				worst.owner === "protocol" ? "human amendment + re-run" : worst.owner === "paper" ? "settled artifacts" : "code fix + tests",
			severity: worst.severity,
		});
		index += 1;
	}
	return issues;
}

export function writeRootIssueRegister(root: string, cycle: number, findings: readonly Finding[]): string {
	const issues = buildRootIssues(findings);
	const lines = ["root_issue,title,finding_ids,owner,severity,fix,status,dependency"];
	for (const issue of issues) {
		const quote = (value: string): string => `"${String(value).replace(/"/g, '""')}"`;
		lines.push(
			[
				issue.id,
				quote(issue.title),
				quote(issue.finding_ids.join(" ")),
				issue.owner,
				issue.severity,
				quote(issue.fix),
				issue.status,
				quote(issue.dependency),
			].join(","),
		);
	}
	const file = path.join(root, ".pi", "audit", `cycle-${cycle}`, "ROOT_ISSUE_REGISTER.csv");
	writeFileAtomic(file, `${lines.join("\n")}\n`);
	return file;
}

export function writeQuestions(root: string, cycle: number, findings: readonly Finding[]): string {
	const unresolved = findings.filter((finding) => finding.status === "unresolved");
	const lines = ["# Questions for authors", ""];
	if (unresolved.length === 0) {
		lines.push("_No unresolved findings._", "");
	} else {
		for (const finding of unresolved) {
			const tag = finding.question_tag === "blocking" ? "[blocking]" : "[non-blocking]";
			lines.push(`- ${tag} **${finding.id}** — ${finding.title}`);
			lines.push(`  Location: ${finding.location}. Evidence: ${finding.observed}`);
		}
		lines.push("");
	}
	const file = path.join(root, ".pi", "audit", `cycle-${cycle}`, "QUESTIONS_FOR_AUTHORS.md");
	writeFileAtomic(file, `${lines.join("\n")}\n`);
	return file;
}

/** Validate that GRADE.md names at least one failing gate when the grade was capped by one. */
export function gradeCitesFailingGates(grade: GradeResult): boolean {
	const gateCaps = grade.trace.filter((trace) => trace.rule === "failed-gate" && trace.effect === "cap");
	if (gateCaps.length === 0) return grade.gate_failures.length === 0;
	return gateCaps.every((trace) => grade.gate_failures.some((gate) => trace.evidence.includes(gate)));
}

export function gradeOrder(): readonly string[] {
	return GRADES;
}
