/**
 * Calibration against the historical audit (plan/07_AUDIT_HARNESS.md §7).
 *
 * The scorer — not the finders — holds the historical expectations. A run is
 * calibrated only when it rediscovers the expected historical S1 findings and
 * confirms none of the four rejected Phase-A leads. Recall below 0.6, or any
 * confirmed rejected lead, writes `UNCALIBRATED`; the orchestrator then refuses
 * to grade and may only re-run finders with narrowed briefs.
 *
 * No pi imports: usable from `node --test` and from `calibrate.sh`.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { writeJsonAtomic } from "../../lib/io.ts";
import { isConfirmed, type Finding, type Severity, type Workstream } from "./schema.ts";

export interface CalibrationCase {
	id: string;
	workstream: Workstream;
	severity: Severity;
	title: string;
	/** Regex on the finding location (normalized, repo-relative). */
	location: string;
	/** Regex on title + observed + expected (the claim the finding makes). */
	claim: string;
	/** Additional pattern that must also match (optional). */
	also?: string;
	historical_file?: string;
}

export interface RejectedLead {
	id: string;
	workstream: Workstream;
	title: string;
	claim: string;
	/** If a finding matches `claim` but also matches `unless`, it is the corrected finding, not the rejected lead. */
	unless?: string;
	historical_note?: string;
}

export interface CalibrationExpected {
	schema: string;
	snapshot: string;
	threshold: number;
	workstreams: Workstream[];
	cases: CalibrationCase[];
	rejected_leads: RejectedLead[];
}

export const CALIBRATION_EXPECTED: CalibrationExpected = {
	schema: "vangja-calibration/1",
	snapshot: "8028c5108d926c58ec4e9b29f72842f2b9d8d095",
	threshold: 0.6,
	workstreams: ["S", "M", "R"],
	cases: [
		{
			id: "WS-S-001",
			workstream: "S",
			severity: "S1",
			title: '"Parametric" transfer uses the posterior mean, not the MAP the paper argues for',
			location: "linear_trend\\.py|fourier_seasonality\\.py|flat_trend\\.py",
			claim: "posterior (mean|MEAN)|\\bmean\\(\\)|\\.mean\\(|mean, not (the )?map|not the mode",
			also: "\\bmap\\b|\\bmode\\b",
			historical_file: "src/vangja/components/linear_trend.py:204",
		},
		{
			id: "WS-S-002",
			workstream: "S",
			severity: "S1",
			title: "LinearTrend transfer potential has the wrong sign/form in 2 of 3 pooling modes",
			location: "linear_trend\\.py",
			claim: "sign|anti-?regulari[sz]|rewards? deviation|\\+\\s*abs|\\+\\s*sqr|wrong (sign|form)",
			historical_file: "src/vangja/components/linear_trend.py complete/individual definition",
		},
		{
			id: "WS-S-003",
			workstream: "S",
			severity: "S1",
			title: "FourierSeasonality regularization scale lam is inconsistent across pooling modes",
			location: "fourier_seasonality\\.py",
			claim: "\\blam\\b|\\blambda\\b|regularization (scale|strength)|22\\s*[×x]|inconsistent|instead of the documented",
			historical_file: "src/vangja/components/fourier_seasonality.py:261-263 vs :356-363",
		},
		{
			id: "WS-M-001",
			workstream: "M",
			severity: "S1",
			title: "Test-set configuration selection with no validation split",
			location: "run_vangja\\.py|metrics\\.csv|case_studies|generate_tables\\.py",
			claim: "test[- ]set|test horizon|selection|selected|grid|lowest mape",
			historical_file: "my_paper.tex:659; metrics.csv 2,592 vs 2 configs",
		},
		{
			id: "WS-M-002",
			workstream: "M",
			severity: "S1",
			title: 'The "Prophet"/"TimeSeers" baselines are Vangja components, not the packages',
			location: "case_studies|baselines|pyproject\\.toml|runner",
			claim: "not prophet|is vangja|vangja (component|ablation)|never imports? prophet|no prophet (import|dependency)|mislabell?ed|baselines? (are )?absent|no (runner )?imports? (either|prophet|timeseers)|(prophet|timeseers).{0,40}(absent|not imported)",
			historical_file: "case_studies baseline runners; pyproject.toml has no prophet dependency",
		},
		{
			id: "WS-R-001",
			workstream: "R",
			severity: "S1",
			title: "PyMC is not a declared dependency; the sampler stack floats",
			location: "pyproject\\.toml|requirements|environment|uv\\.lock",
			claim: "pymc.*(not|missing|unpinned|undeclared|transitive)|declared dependency|floats",
			historical_file: "pyproject.toml",
		},
		{
			id: "WS-R-004",
			workstream: "R",
			severity: "S1",
			title: "No random seed anywhere in fit(), runners, SLURM, or the container",
			location: "src|case_studies|pyproject\\.toml|slurm|singularity",
			claim: "seed|random_seed|unseeded|reproducib",
			historical_file: "git grep seed -> only synthetic generators; pm.sample has no random_seed",
		},
	],
	rejected_leads: [
		{
			id: "F-012",
			workstream: "I",
			title: "mapx vs map/nuts naming inconsistency",
			claim: "mapx.*(inconsisten|mismatch|wrong|does not exist|not a real|typo)|(inconsisten|mismatch).*mapx",
			historical_note: "Refuted on main: mapx is a real dispatched method and the README uses it consistently.",
		},
		{
			id: "F-014",
			workstream: "D",
			title: "Temperature context: paper says Boston, code says NYC",
			claim: "boston.{0,120}nyc|nyc.{0,120}boston|(paper|manuscript).{0,120}(boston|nyc).{0,120}(code|runner|loader|data).{0,120}(nyc|boston)|(temperature|context).{0,80}(paper|code).{0,80}(mismatch|disagree|discrepan|instead of)",
			historical_note: "Refuted: paper and code agree on Boston/Kaggle.",
		},
		{
			id: "F-015",
			workstream: "M",
			title: "Prophet baseline default vs Vangja heavily tuned (fairness framing)",
			claim: "(prophet.*(under-?tuned|default|untuned).*(vangja|tuned))|(unfair.*(tuned|baseline))|(heavily tuned.*prophet)|(prophet baseline.*tuned)",
			unless: "not prophet|is vangja|vangja (component|ablation)|never imports? prophet|no prophet (import|dependency)",
			historical_note: "Superseded by WS-M-002: the baseline is not Prophet at all; the fairness framing is the wrong claim.",
		},
		{
			id: "F-016",
			workstream: "M",
			title: "TimeSeers baseline on unmaintained PyMC3",
			claim: "timeseers.*pymc3|pymc3.*timeseers|timeseers.*(unmaintained|outdated)",
			historical_note: "Moot: TimeSeers is never imported or run; the TimeSeers baseline is a Vangja ablation.",
		},
	],
};

export interface CalibrationHit {
	expected: string;
	finding: string;
	workstream: Workstream;
	severity: Severity;
	title: string;
}

export interface CalibrationMiss {
	expected: string;
	workstream: Workstream;
	severity: Severity;
	title: string;
}

export interface RejectedLeadHit {
	lead: string;
	finding: string;
	title: string;
	evidence: string;
}

export interface CalibrationResult {
	calibrated: boolean;
	threshold: number;
	recall: number;
	cases: number;
	hit_count: number;
	miss_count: number;
	hits: CalibrationHit[];
	misses: CalibrationMiss[];
	rejected_confirmed: number;
	rejected_hits: RejectedLeadHit[];
	false_positives: CalibrationMiss[];
	by_workstream: Record<string, { expected: number; hit: number }>;
	narrow_briefs: Record<string, string[]>;
	detail: string;
}

function normalized(text: string): string {
	return text.toLowerCase().replace(/\s+/g, " ");
}

function findingText(finding: Finding): string {
	return normalized([finding.title, finding.observed, finding.expected, finding.consequence].join(" "));
}

function findingLocation(finding: Finding): string {
	return finding.location.toLowerCase().replace(/\\/g, "/");
}

/** Score confirmed findings against the historical expectation list. */
export function scoreCalibration(
	findings: readonly Finding[],
	expected: CalibrationExpected = CALIBRATION_EXPECTED,
): CalibrationResult {
	const confirmed = findings.filter(isConfirmed);
	const hits: CalibrationHit[] = [];
	const misses: CalibrationMiss[] = [];
	const used = new Set<string>();

	for (const testCase of expected.cases) {
		const locationRe = new RegExp(testCase.location, "i");
		const claimRe = new RegExp(testCase.claim, "i");
		const alsoRe = testCase.also ? new RegExp(testCase.also, "i") : null;
		const match = confirmed.find((finding) => {
			if (used.has(finding.id)) return false;
			const ws = finding.id.replace(/^WS-/, "").charAt(0);
			if (ws !== testCase.workstream) return false;
			if (!locationRe.test(findingLocation(finding))) return false;
			const text = findingText(finding);
			if (!claimRe.test(text)) return false;
			if (alsoRe && !alsoRe.test(text)) return false;
			return true;
		});
		if (match) {
			used.add(match.id);
			hits.push({
				expected: testCase.id,
				finding: match.id,
				workstream: testCase.workstream,
				severity: match.severity,
				title: match.title,
			});
		} else {
			misses.push({
				expected: testCase.id,
				workstream: testCase.workstream,
				severity: testCase.severity,
				title: testCase.title,
			});
		}
	}

	const rejectedHits: RejectedLeadHit[] = [];
	const rejectedByWorkstream: Record<string, string[]> = {};
	for (const lead of expected.rejected_leads) {
		const claimRe = new RegExp(lead.claim, "i");
		const unlessRe = lead.unless ? new RegExp(lead.unless, "i") : null;
		for (const finding of confirmed) {
			const text = findingText(finding);
			if (!claimRe.test(text)) continue;
			if (unlessRe && unlessRe.test(text)) continue;
			rejectedHits.push({
				lead: lead.id,
				finding: finding.id,
				title: finding.title,
				evidence: `${finding.location}: ${finding.observed.slice(0, 200)}`,
			});
			rejectedByWorkstream[lead.workstream] = [
				...(rejectedByWorkstream[lead.workstream] ?? []),
				`${lead.id} (superseded framing; report the stronger identity finding, not this one)`,
			];
		}
	}

	const hitCount = hits.length;
	const recall = expected.cases.length === 0 ? 1 : hitCount / expected.cases.length;
	const byWorkstream: CalibrationResult["by_workstream"] = {};
	for (const workstream of expected.workstreams) {
		const total = expected.cases.filter((testCase) => testCase.workstream === workstream).length;
		const hit = hits.filter((entry) => entry.workstream === workstream).length;
		byWorkstream[workstream] = { expected: total, hit };
	}
	const narrowBriefs: Record<string, string[]> = {};
	for (const miss of misses) {
		narrowBriefs[miss.workstream] = [...(narrowBriefs[miss.workstream] ?? []), miss.expected];
	}
	for (const [workstream, leads] of Object.entries(rejectedByWorkstream)) {
		narrowBriefs[workstream] = [...(narrowBriefs[workstream] ?? []), ...leads];
	}
	const calibrated = recall >= expected.threshold && rejectedHits.length === 0;
	return {
		calibrated,
		threshold: expected.threshold,
		recall,
		cases: expected.cases.length,
		hit_count: hitCount,
		miss_count: misses.length,
		hits,
		misses,
		rejected_confirmed: rejectedHits.length,
		rejected_hits: rejectedHits,
		false_positives: [],
		by_workstream: byWorkstream,
		narrow_briefs: narrowBriefs,
		detail: `recall ${hitCount}/${expected.cases.length} = ${recall.toFixed(3)} (threshold ${expected.threshold}); rejected leads confirmed ${rejectedHits.length}; ${calibrated ? "CALIBRATED" : "UNCALIBRATED"}`,
	};
}

export function loadCalibrationExpected(root: string): CalibrationExpected {
	const file = path.join(root, ".pi", "audit", "calibration", "expected.json");
	if (!fs.existsSync(file)) return CALIBRATION_EXPECTED;
	return JSON.parse(fs.readFileSync(file, "utf8")) as CalibrationExpected;
}

export function loadFindingsFile(file: string): Finding[] {
	const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as unknown;
	if (Array.isArray(parsed)) return parsed as Finding[];
	const container = parsed as { findings?: Finding[] };
	if (!Array.isArray(container.findings)) throw new Error(`${file} carries no findings array`);
	return container.findings;
}

/**
 * Collect findings from a directory.
 *
 * A cycle directory carries a global `findings.json` (workstream: null) and one
 * per-workstream file each; loading both would duplicate every finding. When a
 * global register is present, use it; otherwise recurse into per-workstream
 * files. Ids are de-duplicated defensively.
 */
export function loadFindingsDir(dir: string): Finding[] {
	if (!fs.existsSync(dir)) return [];
	const rootFile = path.join(dir, "findings.json");
	if (fs.existsSync(rootFile)) {
		try {
			const parsed = JSON.parse(fs.readFileSync(rootFile, "utf8")) as {
				workstream?: string | null;
				findings?: Finding[];
			};
			if (parsed && !Array.isArray(parsed) && parsed.workstream === null && Array.isArray(parsed.findings)) {
				return parsed.findings;
			}
		} catch {
			// fall through to the recursive path
		}
	}
	const findings: Finding[] = [];
	const seen = new Set<string>();
	for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
		const nested = entry.isDirectory()
			? loadFindingsDir(path.join(dir, entry.name))
			: entry.isFile() && entry.name === "findings.json"
				? loadFindingsFile(path.join(dir, entry.name))
				: [];
		for (const finding of nested) {
			if (seen.has(finding.id)) continue;
			seen.add(finding.id);
			findings.push(finding);
		}
	}
	return findings;
}

export interface CalibrationReportInput {
	outDir: string;
	result: CalibrationResult;
	mode: "fixture" | "finder" | "collect";
	snapshot: string;
	snapshot_readonly: boolean;
	findings_source: string;
	expected_source: string;
	note?: string;
	now?: string;
}

export interface CalibrationReportFiles {
	report: string;
	recall: string;
	uncalibrated: string | null;
}

/** Write REPORT.md + recall.json and, when uncalibrated, the UNCALIBRATED marker. */
export function writeCalibrationReport(input: CalibrationReportInput): CalibrationReportFiles {
	fs.mkdirSync(input.outDir, { recursive: true });
	const result = input.result;
	const recallJson = {
		schema: "vangja-calibration-report/1",
		generated_at: input.now ?? new Date().toISOString(),
		mode: input.mode,
		snapshot: input.snapshot,
		snapshot_readonly: input.snapshot_readonly,
		findings_source: input.findings_source,
		expected_source: input.expected_source,
		calibrated: result.calibrated,
		recall: result.recall,
		threshold: result.threshold,
		cases: result.cases,
		hit_count: result.hit_count,
		hits: result.hits,
		misses: result.misses,
		rejected_confirmed: result.rejected_confirmed,
		rejected_hits: result.rejected_hits,
		by_workstream: result.by_workstream,
		narrow_briefs: result.narrow_briefs,
		detail: result.detail,
	};
	const recallFile = path.join(input.outDir, "recall.json");
	writeJsonAtomic(recallFile, recallJson);

	const lines: string[] = [];
	lines.push(`# Calibration — ${result.calibrated ? "CALIBRATED" : "UNCALIBRATED"}`, "");
	lines.push(`- Mode: ${input.mode}`);
	lines.push(`- Snapshot: ${input.snapshot} (read-only worktree: ${input.snapshot_readonly ? "yes" : "no"})`);
	lines.push(`- Findings source: ${input.findings_source}`);
	lines.push(`- Expected source: ${input.expected_source}`);
	lines.push(`- Generated: ${input.now ?? new Date().toISOString()}`);
	if (input.note) lines.push(`- Note: ${input.note}`);
	lines.push("");
	lines.push(`**Recall:** ${result.hit_count}/${result.cases} = ${result.recall.toFixed(3)} (threshold ${result.threshold})`);
	lines.push(`**Rejected Phase-A leads confirmed:** ${result.rejected_confirmed}`);
	lines.push(`**Verdict:** ${result.calibrated ? "the finders may grade the paper" : "the orchestrator refuses to grade; narrow the briefs and re-run S/M/R"}`);
	lines.push("");
	lines.push("## Per-workstream recall", "");
	lines.push("| Workstream | Expected | Hit | Recall |");
	lines.push("|---|---:|---:|---:|");
	for (const [workstream, counts] of Object.entries(result.by_workstream)) {
		lines.push(`| ${workstream} | ${counts.expected} | ${counts.hit} | ${counts.expected === 0 ? "n/a" : (counts.hit / counts.expected).toFixed(2)} |`);
	}
	lines.push("");
	lines.push("## Expected findings", "");
	lines.push("| Expected | Workstream | Result | Finding |");
	lines.push("|---|---|---|---|");
	const missesById = new Map(result.misses.map((miss) => [miss.expected, miss]));
	for (const testCase of CALIBRATION_EXPECTED.cases) {
		const hit = result.hits.find((entry) => entry.expected === testCase.id);
		const miss = missesById.get(testCase.id);
		lines.push(
			`| ${testCase.id} | ${testCase.workstream} | ${hit ? "HIT" : `MISS (${miss?.title ?? "not expected in this run"})`} | ${hit?.finding ?? "—"} |`,
		);
	}
	lines.push("");
	lines.push("## Rejected Phase-A leads (negative controls)", "");
	lines.push("| Lead | Result | Finding |");
	lines.push("|---|---|---|");
	for (const lead of CALIBRATION_EXPECTED.rejected_leads) {
		const confirmed = result.rejected_hits.filter((entry) => entry.lead === lead.id);
		lines.push(
			`| ${lead.id} | ${confirmed.length === 0 ? "not confirmed (correct)" : `CONFIRMED (false positive): ${confirmed.map((entry) => entry.finding).join(", ")}`} | ${confirmed[0]?.evidence ?? "—"} |`,
		);
	}
	lines.push("");
	if (!result.calibrated) {
		lines.push("## Why this run may not grade", "");
		if (result.recall < result.threshold) {
			lines.push(`- Recall ${result.recall.toFixed(3)} < ${result.threshold}: ${result.misses.map((miss) => miss.expected).join(", ")} were missed.`);
		}
		if (result.rejected_confirmed > 0) {
			lines.push(`- ${result.rejected_confirmed} rejected Phase-A lead(s) were confirmed: ${result.rejected_hits.map((entry) => entry.lead).join(", ")}.`);
		}
		lines.push("");
		lines.push("Narrowed briefs:");
		for (const [workstream, ids] of Object.entries(result.narrow_briefs)) {
			lines.push(`- WS-${workstream}: re-derive ${ids.join(", ")} with the command named in the historical register.`);
		}
		lines.push("- Every finding must be re-derived from the snapshot; the historical text is not a source.");
		lines.push("");
	}
	const report = path.join(input.outDir, "REPORT.md");
	fs.writeFileSync(report, `${lines.join("\n").trimEnd()}\n`);
	const marker = path.join(input.outDir, "UNCALIBRATED");
	if (!result.calibrated) fs.writeFileSync(marker, `${result.detail}\n`);
	return { report, recall: recallFile, uncalibrated: result.calibrated ? null : marker };
}
