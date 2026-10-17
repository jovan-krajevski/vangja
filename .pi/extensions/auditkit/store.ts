/**
 * Audit register storage: read/merge the global and per-workstream findings
 * files for a cycle. Writes are atomic; the per-workstream file and the global
 * register are derived from the same records.
 *
 * No pi imports: usable from `node --test`.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { withLock, writeFileAtomic, writeJsonAtomic } from "../../lib/io.ts";
import type { Finding, Phase, Workstream } from "./schema.ts";
import { registerFindings, summarizeFindings, type RegisteredFinding, type RegisterResult } from "./schema.ts";

export function auditRoot(root: string): string {
	return path.join(root, ".pi", "audit");
}

export function cycleDir(root: string, cycle: number): string {
	return path.join(auditRoot(root), `cycle-${cycle}`);
}

export interface FindingsEnvelope {
	schema: string;
	cycle: number;
	workstream: string | null;
	summary?: Record<string, unknown>;
	findings: RegisteredFinding[];
}

export function readFindingsFile(file: string): RegisteredFinding[] {
	if (!fs.existsSync(file)) return [];
	try {
		const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as FindingsEnvelope | RegisteredFinding[];
		if (Array.isArray(parsed)) return parsed;
		return Array.isArray(parsed.findings) ? parsed.findings : [];
	} catch (error) {
		throw new Error(`cannot parse findings file ${file}: ${(error as Error).message}`);
	}
}

/** Global register for a cycle (merged across workstreams). */
export function readRegister(root: string, cycle: number): RegisteredFinding[] {
	return readFindingsFile(path.join(cycleDir(root, cycle), "findings.json"));
}

export function readWorkstreamRegister(root: string, cycle: number, workstream: string): RegisteredFinding[] {
	return readFindingsFile(path.join(cycleDir(root, cycle), workstream.toLowerCase(), "findings.json"));
}

function renderFindingsMd(cycle: number, findings: readonly Finding[], workstream?: string): string {
	const lines: string[] = [];
	const summary = summarizeFindings(findings);
	lines.push(`# Findings${workstream ? ` — WS-${workstream}` : ""} — cycle ${cycle}`, "");
	lines.push(
		`${summary.count} finding(s): ` +
			Object.entries(summary.bySeverity)
				.map(([severity, count]) => `${severity}=${count}`)
				.join(", "),
		"",
	);
	for (const severity of ["S0", "S1", "S2", "S3"]) {
		const group = findings.filter((finding) => finding.severity === severity);
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
			lines.push(`- **Evidence:** ${finding.evidence.join(" · ")}`);
			if ((finding.cross_refs ?? []).length > 0) lines.push(`- **Cross-refs:** ${finding.cross_refs.join(", ")}`);
			if ((finding.duplicates ?? []).length > 0) lines.push(`- **Folded duplicates:** ${(finding.duplicates ?? []).join(", ")}`);
			lines.push("");
		}
	}
	if (findings.length === 0) lines.push("_No findings registered._", "");
	return `${lines.join("\n")}\n`;
}

export interface RegisterWriteResult extends RegisterResult {
	files: { workstream: string; register: string; markdown: string };
}

/**
 * Merge findings into the cycle register and write both files. The global
 * register is the union (deduped by normalized location+title).
 */
export function registerForCycle(
	root: string,
	cycle: number,
	workstream: Workstream,
	findings: unknown[],
	options: { phase?: Phase; author_agent?: string; now?: string } = {},
): RegisterWriteResult {
	const dir = cycleDir(root, cycle);
	fs.mkdirSync(dir, { recursive: true });
	// Finders run in parallel; the register is the shared mutable file. The
	// lock makes each read-merge-write atomic so no workstream's findings are
	// lost to a last-writer-wins race.
	return withLock(path.join(dir, ".register.lock"), () => {
		const existing = readRegister(root, cycle);
		const outcome = registerFindings(existing, {
			cycle,
			workstream,
			findings,
			phase: options.phase,
			author_agent: options.author_agent,
			now: options.now,
		});

		fs.mkdirSync(path.join(dir, workstream.toLowerCase()), { recursive: true });
		const perWorkstream = outcome.register.filter((finding) => (finding.id.replace(/^WS-/, "").charAt(0) || "") === workstream);
		const wsFile = path.join(dir, workstream.toLowerCase(), "findings.json");
		writeJsonAtomic(wsFile, {
			schema: "vangja-findings/1",
			cycle,
			workstream,
			summary: summarizeFindings(perWorkstream),
			findings: perWorkstream,
		});
		const wsMd = path.join(dir, workstream.toLowerCase(), "FINDINGS.md");
		writeFileAtomic(wsMd, renderFindingsMd(cycle, perWorkstream, workstream));

		const registerFile = path.join(dir, "findings.json");
		writeJsonAtomic(registerFile, {
			schema: "vangja-findings/1",
			cycle,
			workstream: null,
			summary: summarizeFindings(outcome.register),
			findings: outcome.register,
		});
		const registerMd = path.join(dir, "FINDINGS.md");
		writeFileAtomic(registerMd, renderFindingsMd(cycle, outcome.register));

		return {
			...outcome,
			files: { workstream: wsFile, register: registerFile, markdown: registerMd },
		};
	});
}

/** Read the latest rubric score written by rubric_score. */
export function readRubricScore(root: string, cycle: number): { total: number | null; complete?: boolean; unresolved?: string[] } | null {
	const file = path.join(cycleDir(root, cycle), "rubric.json");
	if (!fs.existsSync(file)) return null;
	try {
		const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as { total?: number | null; complete?: boolean; unresolved?: string[] };
		return { total: parsed.total ?? null, complete: parsed.complete, unresolved: parsed.unresolved ?? [] };
	} catch {
		return null;
	}
}
