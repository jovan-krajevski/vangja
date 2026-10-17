/**
 * Seeded-defect canary (plan/07_AUDIT_HARNESS.md §5 `canary`, §6 anti-gaming).
 *
 * This is not a finder: it is the sensitivity probe. A small synthetic repository
 * contains one instance of each defect class the audit must catch, plus negative
 * controls that must NOT be reported. Running the probes every cycle proves the
 * harness still detects known defects before it is trusted to grade a real
 * snapshot.
 *
 * Probes are deliberately simple and deterministic; they are matched to the
 * expectations in `expected.json` and every finding carries a command a human
 * can re-run in the fixture.
 *
 * No pi imports: usable from `node --test`.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import type { Finding, Severity, Workstream } from "./schema.ts";
import { severityRank } from "./schema.ts";

export interface SeededDefect {
	id: string;
	title: string;
	probe: string;
	severity: Severity;
	workstream: Workstream;
	expect: "detected" | "not_detected";
	location_contains?: string;
}

export interface SeededExpected {
	schema: string;
	fixture: string;
	defects: SeededDefect[];
}

export interface CanaryHit {
	expected: string;
	probe: string;
	finding: string;
	severity: Severity;
	title: string;
}

export interface CanaryMiss {
	expected: string;
	probe: string;
	severity: Severity;
	title: string;
	reason: string;
}

export interface CanaryFalsePositive {
	probe: string;
	finding: string;
	location: string;
	reason: string;
}

export interface CanaryResult {
	ok: boolean;
	fixture: string;
	threshold: number;
	findings: Finding[];
	hits: CanaryHit[];
	misses: CanaryMiss[];
	false_positives: CanaryFalsePositive[];
	negative_controls: Array<{ id: string; probe: string; title: string; reported: string | null }>;
	recall: number;
	expected_s0s1: number;
	detected_s0s1: number;
	severity_mismatches: Array<{ expected: string; finding: string; expected_severity: Severity; finding_severity: Severity }>;
	detail: string;
}

interface ProbeFinding {
	probe: string;
	workstream: Workstream;
	severity: Severity;
	title: string;
	location: string;
	command: string;
	expected: string;
	observed: string;
	consequence: string;
	owner: Finding["owner"];
}

function walk(root: string, predicate: (file: string) => boolean): string[] {
	const out: string[] = [];
	if (!fs.existsSync(root)) return out;
	const stack = [root];
	while (stack.length > 0) {
		const current = stack.pop() as string;
		for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
			if (entry.name === ".git" || entry.name === "node_modules" || entry.name === "__pycache__") continue;
			const full = path.join(current, entry.name);
			if (entry.isDirectory()) stack.push(full);
			else if (entry.isFile() && predicate(full)) out.push(full);
		}
	}
	return out.sort();
}

function rel(root: string, file: string): string {
	return path.relative(root, file).split(path.sep).join("/");
}

function readLines(file: string): string[] {
	try {
		return fs.readFileSync(file, "utf8").split("\n");
	} catch {
		return [];
	}
}

// ---------------------------------------------------------------------------
// Probes
// ---------------------------------------------------------------------------

export function probeWrongPotentialSign(root: string): ProbeFinding[] {
	const out: ProbeFinding[] = [];
	for (const file of walk(root, (candidate) => candidate.endsWith(".py") && candidate.includes(`${path.sep}src${path.sep}`))) {
		const lines = readLines(file);
		lines.forEach((line, index) => {
			if (!/pm\.Potential|potential\s*=/i.test(line)) return;
			if (!/\+\s*(self\.)?loss_factor/i.test(line)) return;
			if (!/(\*\*\s*2|abs\s*\()/i.test(line)) return;
			out.push({
				probe: "wrong_potential_sign",
				workstream: "S",
				severity: "S0",
				title: "Transfer potential rewards deviation from the transferred slope (wrong sign)",
				location: `${rel(root, file)}:${index + 1}`,
				command: `grep -n "loss_factor" ${rel(root, file)}`,
				expected: "the transfer potential is -loss·Σ(slope_g − slope_transferred)² (plan-level regulariser)",
				observed: `line ${index + 1} adds +loss_factor·(slope_g − slope_transferred)², an anti-regulariser`,
				consequence: "transfer moves estimates away from the context posterior, so every pooled transfer result is wrong",
				owner: "code",
			});
		});
	}
	return out;
}

export function probeTestSelection(root: string): ProbeFinding[] {
	const out: ProbeFinding[] = [];
	const files = walk(root, (file) => file.endsWith(".py") && /(runner|runner[s]?|scripts?|pipeline)/i.test(file));
	for (const file of files) {
		const lines = readLines(file);
		lines.forEach((line, index) => {
			const selects = /\b(min|max|sorted|sort_values|nlargest|nsmallest|idxmin|idxmax|argmin|argmax)\b/.test(line);
			const testMetric = /\btest[_-]?(mape|mae|rmse|error|score|metric)\b/i.test(line);
			const prev = lines.slice(Math.max(0, index - 2), index).join(" ");
			const selectionContext = /\b(best|select|choose|pick|winner)\b/i.test(prev);
			if ((selects && testMetric) || (testMetric && selectionContext)) {
				out.push({
					probe: "test_selection",
					workstream: "E",
					severity: "S1",
					title: "Runner selects the configuration on the test-horizon metric",
					location: `${rel(root, file)}:${index + 1}`,
					command: `grep -n "test_mape" ${rel(root, file)}`,
					expected: "configurations are chosen on a validation split, never on the test horizon",
					observed: `line ${index + 1} chooses a configuration using test_mape`,
					consequence: "reported accuracy is the maximum over the search grid, not an out-of-sample estimate",
					owner: "code",
				});
			}
		});
	}
	return out;
}

function countAssertions(file: string): number {
	return readLines(file).filter(
		(line) => !/^\s*#/.test(line) && /\b(assert\b|pytest\.raises|expect\()/.test(line),
	).length;
}

export function probeWeakenedTest(root: string): ProbeFinding[] {
	const out: ProbeFinding[] = [];
	const baselineDir = path.join(root, "tests_baseline");
	if (!fs.existsSync(baselineDir)) return out;
	const baselineFiles = walk(baselineDir, (file) => file.endsWith(".py"));
	for (const baseline of baselineFiles) {
		const relFromBaseline = path.relative(baselineDir, baseline);
		const current = path.join(root, "tests", relFromBaseline);
		if (!fs.existsSync(current)) {
			out.push({
				probe: "weakened_test",
				workstream: "I",
				severity: "S1",
				title: "Test removed while the behaviour it covered is still claimed",
				location: `tests/${relFromBaseline.split(path.sep).join("/")}`,
				command: `ls tests/${relFromBaseline.split(path.sep).join("/")}`,
				expected: "the test exists at the audited snapshot",
				observed: `tests_baseline/${relFromBaseline.split(path.sep).join("/")} has no counterpart under tests/`,
				consequence: "the behaviour is no longer protected by a regression test",
				owner: "code",
			});
			continue;
		}
		const before = countAssertions(baseline);
		const after = countAssertions(current);
		const commented = readLines(current).some((line) => /^\s*#\s*assert\b/.test(line));
		if (after < before || commented) {
			out.push({
				probe: "weakened_test",
				workstream: "I",
				severity: "S1",
				title: "Test assertion removed or commented out (weakened test)",
				location: `tests/${relFromBaseline.split(path.sep).join("/")}`,
				command: `grep -n "assert" tests/${relFromBaseline.split(path.sep).join("/")}`,
				expected: `at least ${before} assertion(s), as in tests_baseline`,
				observed: `${after} assertion(s) remain${commented ? "; an assertion is commented out" : ""}`,
				consequence: "a regression can pass the suite unnoticed",
				owner: "code",
			});
		}
	}
	return out;
}

interface ClaimsFile {
	claims?: Array<{ id?: string; artifact?: string; artifacts?: string[]; value?: number; numbers?: unknown }>;
}

function loadClaims(root: string): ClaimsFile | null {
	const file = path.join(root, "paper", "CLAIMS.json");
	if (!fs.existsSync(file)) return null;
	try {
		const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as ClaimsFile | Array<{ id?: string }>;
		return Array.isArray(parsed) ? { claims: parsed as ClaimsFile["claims"] } : parsed;
	} catch {
		return null;
	}
}

export function probeMissingArtifact(root: string): ProbeFinding[] {
	const out: ProbeFinding[] = [];
	const claims = loadClaims(root);
	if (!claims) return out;
	for (const [index, claim] of (claims.claims ?? []).entries()) {
		const id = claim.id ?? `claim[${index}]`;
		const artifacts = claim.artifacts ?? (claim.artifact ? [claim.artifact] : []);
		if (artifacts.length === 0) {
			out.push({
				probe: "missing_artifact",
				workstream: "C",
				severity: "S1",
				title: `Reported number ${id} has no artifact`,
				location: `paper/CLAIMS.json:${index + 1}`,
				command: `grep -n "${id}" paper/CLAIMS.json`,
				expected: "every claim names an artifact with a manifest and SHA",
				observed: `${id} names no artifact`,
				consequence: "the number cannot be recomputed and is not provenance-backed",
				owner: "paper",
			});
			continue;
		}
		for (const artifact of artifacts) {
			const manifest = path.join(root, artifact, "manifest.json");
			if (!fs.existsSync(manifest) && !artifact.endsWith("manifest.json")) {
				out.push({
					probe: "missing_artifact",
					workstream: "C",
					severity: "S1",
					title: `Claim ${id} points at a missing artifact`,
					location: `paper/CLAIMS.json:${index + 1}`,
					command: `ls ${artifact}`,
					expected: "the artifact and its manifest exist at the audited tag",
					observed: `${artifact} does not exist`,
					consequence: "the paper reports a number whose provenance is absent",
					owner: "paper",
				});
			} else if (!fs.existsSync(manifest)) {
				out.push({
					probe: "missing_artifact",
					workstream: "C",
					severity: "S1",
					title: `Claim ${id} artifact has no manifest`,
					location: `paper/CLAIMS.json:${index + 1}`,
					command: `ls ${artifact}/manifest.json`,
					expected: "the artifact directory carries manifest.json",
					observed: `${artifact}/manifest.json does not exist`,
					consequence: "the number cannot be hash-verified or traced to a commit",
					owner: "paper",
				});
			}
		}
	}
	return out;
}

export function probeRetrospectiveConfirmation(root: string): ProbeFinding[] {
	const out: ProbeFinding[] = [];
	const dataDir = path.join(root, "data");
	if (!fs.existsSync(dataDir)) return out;
	for (const file of walk(dataDir, (candidate) => candidate.endsWith(".csv") || candidate.endsWith(".json"))) {
		const lines = readLines(file).slice(0, 12);
		const hit = lines.findIndex((line) => /retrospective|retro-?active|chosen after|post[- ]?hoc/i.test(line));
		if (hit >= 0) {
			out.push({
				probe: "retrospective_confirmation",
				workstream: "D",
				severity: "S0",
				title: "Dataset labelled confirmation was selected retrospectively",
				location: `${rel(root, file)}:${hit + 1}`,
				command: `head -12 ${rel(root, file)}`,
				expected: "confirmation units are frozen before any result is viewed",
				observed: `line ${hit + 1} records retrospective/post-hoc selection`,
				consequence: "confirmation results are development results in disguise; the holdout claim is false",
				owner: "protocol",
			});
		}
	}
	const provenance = path.join(dataDir, "PROVENANCE.json");
	if (fs.existsSync(provenance)) {
		try {
			const parsed = JSON.parse(fs.readFileSync(provenance, "utf8")) as Record<string, unknown>;
			if (/retrospective|post[- ]?hoc|after results/i.test(JSON.stringify(parsed.selected ?? parsed.selection ?? ""))) {
				out.push({
					probe: "retrospective_confirmation",
					workstream: "D",
					severity: "S0",
					title: "Provenance records the confirmation set as selected after results",
					location: `${rel(root, provenance)}:1`,
					command: `cat ${rel(root, provenance)}`,
					expected: "the confirmation selection predates the results",
					observed: "the provenance record says the selection was retrospective",
					consequence: "the confirmation claim is unsupportable",
					owner: "protocol",
				});
			}
		} catch {
			// unreadable provenance is not this probe's finding
		}
	}
	return out;
}

export function probeMislabelledBaseline(root: string): ProbeFinding[] {
	const out: ProbeFinding[] = [];
	const files = walk(root, (file) => file.endsWith(".py"));
	for (const file of files) {
		const text = fs.readFileSync(file, "utf8");
		if (!/["']Prophet["']|\blabel\w*\s*=\s*["']Prophet["']/i.test(text)) continue;
		const importsProphet = /^\s*(from\s+prophet\s+import|import\s+prophet)\b/m.test(text);
		if (importsProphet) continue;
		const lineIndex = readLines(file).findIndex((line) => /["']Prophet["']/i.test(line));
		out.push({
			probe: "mislabelled_baseline",
			workstream: "M",
			severity: "S1",
			title: 'A baseline is labelled "Prophet" but never imports or runs Prophet',
			location: `${rel(root, file)}:${lineIndex + 1}`,
			command: `grep -n "Prophet" ${rel(root, file)}`,
			expected: "a baseline named after a package imports and runs that package",
			observed: `the file labels a run "Prophet" with no prophet import`,
			consequence: "the comparison is against a Vangja ablation, so every improvement claim over Prophet is mislabelled",
			owner: "code",
		});
	}
	return out;
}

export function probeFavourableSeed(root: string): ProbeFinding[] {
	const out: ProbeFinding[] = [];
	const files = walk(root, (file) => file.endsWith(".py") || file.endsWith(".json"));
	for (const file of files) {
		const lines = readLines(file);
		lines.forEach((line, index) => {
			if (!/(random_seed|BASE_SEED|\bseed\b)\s*=\s*\d+/i.test(line)) return;
			const window = lines.slice(Math.max(0, index - 2), index + 3).join(" ");
			if (!/best|favourab|chosen after|after seeing|tuned/i.test(window)) return;
			out.push({
				probe: "favourable_seed",
				workstream: "E",
				severity: "S1",
				title: "Sampler seed was chosen after seeing the result",
				location: `${rel(root, file)}:${index + 1}`,
				command: `grep -n -i "seed" ${rel(root, file)}`,
				expected: "seeds are fixed before the run and recorded in the manifest",
				observed: "the seed line is annotated as chosen after seeing results",
				consequence: "the reported result is a selected draw; reproducibility and uncertainty are invalid",
				owner: "code",
			});
		});
	}
	return out;
}

export function probeUncitedPaperNumber(root: string): ProbeFinding[] {
	const out: ProbeFinding[] = [];
	const claims = loadClaims(root);
	const paperDir = path.join(root, "paper");
	if (!fs.existsSync(paperDir)) return out;
	const claimed: number[] = [];
	for (const claim of claims?.claims ?? []) {
		if (typeof claim.value === "number") claimed.push(claim.value);
		for (const entry of (claim.numbers as Array<number | { value?: number }> | undefined) ?? []) {
			if (typeof entry === "number") claimed.push(entry);
			else if (entry && typeof entry.value === "number") claimed.push(entry.value);
		}
	}
	for (const file of walk(paperDir, (candidate) => candidate.endsWith(".tex"))) {
		readLines(file).forEach((line, index) => {
			if (/^\s*%/.test(line)) return;
			const stripped = line.replace(/\\[a-zA-Z]+\*?(\[[^\]]*\])?\{[^}]*\}/g, " ");
			for (const match of stripped.matchAll(/(?<![\w.])[-+]?\d+\.\d+(?![\w.])/g)) {
				const value = Number(match[0]);
				if (!Number.isFinite(value)) continue;
				if (claimed.some((known) => Math.abs(known - value) <= 1e-12)) continue;
				out.push({
					probe: "uncited_paper_number",
					workstream: "C",
					severity: "S1",
					title: "A number appears in the paper with no claims entry",
					location: `${rel(root, file)}:${index + 1}`,
					command: `grep -n "${match[0]}" ${rel(root, file)}`,
					expected: "every decimal number in the paper maps to paper/CLAIMS.json and an artifact",
					observed: `${match[0]} has no claim whose value matches`,
					consequence: "the number cannot be traced to an artifact and may be hand-typed",
					owner: "paper",
				});
			}
		});
	}
	return out;
}

export const PROBES: Record<string, (root: string) => ProbeFinding[]> = {
	wrong_potential_sign: probeWrongPotentialSign,
	test_selection: probeTestSelection,
	weakened_test: probeWeakenedTest,
	missing_artifact: probeMissingArtifact,
	retrospective_confirmation: probeRetrospectiveConfirmation,
	mislabelled_baseline: probeMislabelledBaseline,
	favourable_seed: probeFavourableSeed,
	uncited_paper_number: probeUncitedPaperNumber,
};

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

function toFinding(probe: ProbeFinding, index: number, fixture: string): Finding {
	const prefix = `${probe.workstream}-9${String(index + 1).padStart(2, "0")}`;
	return {
		id: prefix,
		title: probe.title,
		location: probe.location,
		snapshot: `canary/${fixture}`,
		command: probe.command,
		expected: probe.expected,
		observed: probe.observed,
		consequence: probe.consequence,
		severity: probe.severity,
		confidence: "high",
		status: "verified",
		reproducible: `yes — canary probe "${probe.probe}" reruns in the fixture`,
		evidence: [probe.location, `probe:${probe.probe}`],
		cross_refs: [],
		owner: probe.owner,
		found_in_cycle: 0,
		phase: "canary",
		author_agent: "auditkit.canary",
		probe: probe.probe,
	};
}

export function loadSeededExpected(fixtureRoot: string): SeededExpected {
	const file = path.join(fixtureRoot, "expected.json");
	if (!fs.existsSync(file)) throw new Error(`seeded fixture has no expected.json: ${file}`);
	return JSON.parse(fs.readFileSync(file, "utf8")) as SeededExpected;
}

/**
 * Run every probe on the seeded fixture and compare with `expected.json`.
 * Recall is over expected S0/S1 defects; negative controls count as false
 * positives when a probe reports them.
 */
export function runSeededAudit(fixtureRoot: string, options: { threshold?: number } = {}): CanaryResult {
	const expected = loadSeededExpected(fixtureRoot);
	const threshold = options.threshold ?? 0.8;
	const raw: ProbeFinding[] = [];
	for (const probe of Object.values(PROBES)) raw.push(...probe(fixtureRoot));
	const findings = raw.map((entry, index) => toFinding(entry, index, expected.fixture));

	const hits: CanaryHit[] = [];
	const misses: CanaryMiss[] = [];
	const severityMismatches: CanaryResult["severity_mismatches"] = [];
	const falsePositives: CanaryFalsePositive[] = [];
	const negativeControls: CanaryResult["negative_controls"] = [];
	const expectedS0S1 = expected.defects.filter(
		(defect) => defect.expect === "detected" && (defect.severity === "S0" || defect.severity === "S1"),
	);

	for (const defect of expected.defects) {
		if (defect.expect === "not_detected") {
			const offending = findings.filter(
				(finding) =>
					finding.probe === defect.probe &&
					(!defect.location_contains || finding.location.includes(defect.location_contains)),
			);
			for (const finding of offending) {
				falsePositives.push({
					probe: defect.probe,
					finding: finding.id,
					location: finding.location,
					reason: `negative control ${defect.id} was reported: ${defect.title}`,
				});
			}
			negativeControls.push({
				id: defect.id,
				probe: defect.probe,
				title: defect.title,
				reported: offending[0]?.id ?? null,
			});
			continue;
		}
		const matches = findings.filter(
			(finding) =>
				finding.probe === defect.probe &&
				(!defect.location_contains || finding.location.includes(defect.location_contains)),
		);
		const exact = matches.find((finding) => finding.severity === defect.severity);
		if (exact) {
			hits.push({
				expected: defect.id,
				probe: defect.probe,
				finding: exact.id,
				severity: exact.severity,
				title: exact.title,
			});
			continue;
		}
		if (matches.length > 0) {
			const first = matches[0];
			severityMismatches.push({
				expected: defect.id,
				finding: first.id,
				expected_severity: defect.severity,
				finding_severity: first.severity,
			});
		}
		misses.push({
			expected: defect.id,
			probe: defect.probe,
			severity: defect.severity,
			title: defect.title,
			reason: matches.length > 0 ? "detected at the wrong severity" : "no probe reported it",
		});
	}

	// Any finding that is not tied to an expected defect is an extra (not a false
	// positive unless it contradicts a negative control).
	const expectedProbes = new Set(expected.defects.map((defect) => defect.probe));
	const extras = findings.filter((finding) => !expectedProbes.has(String(finding.probe)));

	const detected = hits.filter((hit) => hit.severity === "S0" || hit.severity === "S1").length;
	const recall = expectedS0S1.length === 0 ? 1 : detected / expectedS0S1.length;
	const ok = recall >= threshold && falsePositives.length === 0;
	return {
		ok,
		fixture: expected.fixture,
		threshold,
		findings,
		hits,
		misses,
		false_positives: falsePositives,
		negative_controls: negativeControls,
		recall,
		expected_s0s1: expectedS0S1.length,
		detected_s0s1: detected,
		severity_mismatches: severityMismatches,
		detail: `${detected}/${expectedS0S1.length} seeded S0/S1 detected (recall ${recall.toFixed(3)}, threshold ${threshold}); ${falsePositives.length} false positive(s) on negative controls; ${extras.length} extra finding(s); ${severityMismatches.length} severity mismatch(es)`,
	};
}

/** Markdown matrix for `REPORT_BUILD.md` and the canary gate. */
export function renderCanaryMatrix(result: CanaryResult): string {
	const lines = [
		"| Defect | Probe | Expected | Result | Finding |",
		"|---|---|---|---|---|",
	];
	const all = [...result.hits.map((hit) => ({ ...hit, kind: "hit" })), ...result.misses.map((miss) => ({ ...miss, kind: "miss" }))];
	for (const entry of all) {
		lines.push(
			`| ${entry.expected} | ${entry.probe} | ${entry.severity} | ${entry.kind === "hit" ? "detected" : `MISS (${entry.reason})`} | ${"finding" in entry ? entry.finding : "—"} |`,
		);
	}
	for (const control of result.negative_controls) {
		lines.push(
			`| ${control.id} | ${control.probe} | not reported | ${
				control.reported ? `FALSE POSITIVE (${control.reported})` : "not reported (correct)"
			} | ${control.reported ?? "—"} |`,
		);
	}
	for (const falsePositive of result.false_positives) {
		if (result.negative_controls.some((control) => control.reported === falsePositive.finding)) continue;
		lines.push(`| negative control | ${falsePositive.probe} | not reported | FALSE POSITIVE | ${falsePositive.finding} |`);
	}
	return `${lines.join("\n")}\n`;
}

export function canarySeverityOf(finding: Finding): number {
	return severityRank(finding.severity);
}
