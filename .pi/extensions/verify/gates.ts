/**
 * The deterministic gates of plan/07_AUDIT_HARNESS.md §5.
 *
 * Every gate emits a machine result and is written to
 * `.pi/audit/cycle-<n>/gates.json` before any rubric scoring. Gates are
 * independent: a gate whose inputs are absent reports `skip` with the reason,
 * never a silent pass. A failed gate is an automatic S0/S1 finding assigned to a
 * workstream by the audit report assembler.
 *
 * No pi imports: usable from `node --test`.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { writeJsonAtomic } from "../../lib/io.ts";
import { runSeededAudit, renderCanaryMatrix } from "../auditkit/canary.ts";
import {
	DEFAULT_TIMEOUT_MS,
	defaultRunner,
	loadManifest,
	manifestDigest,
	resolveManifestPath,
	sha256File,
	verifyArtifact,
	verifyReproduce,
	verifyTests,
	verifyTraceability,
	walkFiles,
	type CommandRunner,
} from "./lib.ts";

export type GateStatus = "pass" | "fail" | "skip" | "error";

export interface GateCheck {
	id: string;
	ok: boolean;
	detail: string;
}

export interface GateEntry {
	id: string;
	label: string;
	status: GateStatus;
	detail: string;
	command?: string;
	output_tail?: string;
	checks?: GateCheck[];
	hash_ok?: boolean;
	waived?: boolean;
}

export interface GatesFile {
	schema: "vangja-gates/1";
	cycle: number;
	snapshot: string;
	generated_at: string;
	ok: boolean;
	hash_ok: boolean;
	gates: GateEntry[];
	failed: string[];
	skipped: string[];
	config_source: string;
	detail: Record<string, unknown>;
}

export interface AuditConfig {
	schema: string;
	testsCommand: string[];
	timeoutMs: number;
	protocolPaths: string[];
	protocolHashes: Record<string, string>;
	originsFile: string;
	selectionScan: string[];
	testDiffRange: string;
	testBaselineDir: string | null;
	resultsReport: string | null;
	failuresFile: string | null;
	contextSeries: string[];
	aggregateGlobs: string[];
	artifactPaths: string[];
	requiredArtifacts: string[];
	envPins: string[];
	traceability: { claimsPath: string; paperDir: string; ref: string; requireAncestor: boolean };
	canaryFixture: string | null;
	canaryRecallThreshold: number;
	reproduceTopN: number;
	skip: string[];
}

export const DEFAULT_AUDIT_CONFIG: AuditConfig = {
	schema: "pi-audit/1",
	testsCommand: ["uv", "run", "pytest", "-q"],
	timeoutMs: DEFAULT_TIMEOUT_MS,
	protocolPaths: ["fixed_case_studies/protocol"],
	protocolHashes: {},
	originsFile: "fixed_case_studies/protocol/CONFIRMATION_ORIGINS.csv",
	selectionScan: ["src", "fixed_case_studies"],
	testDiffRange: "HEAD~1..HEAD",
	testBaselineDir: null,
	resultsReport: null,
	failuresFile: null,
	contextSeries: ["^GSPC", "S&P 500"],
	aggregateGlobs: ["artifacts/**/aggregate*.csv", "results/aggregate*.csv"],
	artifactPaths: [".pi/artifacts"],
	requiredArtifacts: ["manifest.json"],
	envPins: ["pymc", "numpy", "numba"],
	traceability: { claimsPath: "paper/CLAIMS.json", paperDir: "paper", ref: "HEAD", requireAncestor: true },
	canaryFixture: ".pi/tests/audit/fixtures/seeded",
	canaryRecallThreshold: 0.8,
	reproduceTopN: 3,
	skip: [],
};

export function loadAuditConfig(root: string): AuditConfig {
	const file = path.join(root, ".pi", "config", "audit.json");
	if (!fs.existsSync(file)) return { ...DEFAULT_AUDIT_CONFIG };
	let parsed: Partial<AuditConfig>;
	try {
		parsed = JSON.parse(fs.readFileSync(file, "utf8")) as Partial<AuditConfig>;
	} catch (error) {
		throw new Error(`cannot parse ${file}: ${(error as Error).message}`);
	}
	return {
		...DEFAULT_AUDIT_CONFIG,
		...parsed,
		traceability: { ...DEFAULT_AUDIT_CONFIG.traceability, ...(parsed.traceability ?? {}) },
		skip: Array.isArray(parsed.skip) ? parsed.skip : [],
	};
}

function exists(file: string): boolean {
	return fs.existsSync(file);
}

function readLines(file: string): string[] {
	try {
		return fs.readFileSync(file, "utf8").split("\n");
	} catch {
		return [];
	}
}

function isGitRepo(root: string, run: CommandRunner): boolean {
	const result = run(["git", "rev-parse", "--is-inside-work-tree"], { cwd: root });
	return result.status === 0 && result.stdout.trim() === "true";
}

function globToRegex(pattern: string): RegExp {
	let out = "";
	for (let index = 0; index < pattern.length; index += 1) {
		const char = pattern[index];
		if (char === "*") {
			if (pattern[index + 1] === "*") {
				if (pattern[index + 2] === "/") {
					out += "(?:.*/)?";
					index += 2;
				} else {
					out += ".*";
					index += 1;
				}
			} else {
				out += "[^/]*";
			}
		} else if (char === "?") {
			out += "[^/]";
		} else {
			out += /[.+^${}()|[\]\\]/.test(char) ? `\\${char}` : char;
		}
	}
	return new RegExp(`^${out}$`);
}

export function expandGlobs(root: string, patterns: string[]): string[] {
	const files = walkFiles(root);
	const out = new Set<string>();
	for (const pattern of patterns) {
		const regex = globToRegex(pattern);
		for (const file of files) if (regex.test(file)) out.add(file);
	}
	return [...out].sort();
}

interface GateContext {
	root: string;
	config: AuditConfig;
	run: CommandRunner;
	cycle: number;
	snapshot: string;
}

function skipped(context: GateContext, id: string, label: string, reason: string): GateEntry {
	return { id, label, status: "skip", detail: reason };
}

// ---------------------------------------------------------------------------
// Gates
// ---------------------------------------------------------------------------

export function gateTests(context: GateContext): GateEntry {
	const id = "tests";
	if (context.config.skip.includes(id)) return skipped(context, id, "test suite", "skipped by .pi/config/audit.json");
	const result = verifyTests(context.root, {
		command: context.config.testsCommand,
		runner: context.run,
		timeoutMs: context.config.timeoutMs,
	});
	return {
		id,
		label: "test suite",
		status: result.ok ? "pass" : result.status === null ? "error" : "fail",
		detail: result.summary === "" ? `exited ${result.status}` : result.summary,
		command: result.command,
		output_tail: result.output_tail,
	};
}

export function gateProtocolHashes(context: GateContext): GateEntry {
	const id = "protocol-hashes";
	const checks: GateCheck[] = [];
	for (const [file, expected] of Object.entries(context.config.protocolHashes)) {
		const full = path.resolve(context.root, file);
		const actual = exists(full) ? sha256File(full) : null;
		checks.push({
			id: `hash:${file}`,
			ok: actual === expected,
			detail: actual === expected ? `${file} matches` : `${file}: expected ${expected}, actual ${actual ?? "missing"}`,
		});
	}
	for (const dir of context.config.protocolPaths) {
		for (const sidecar of expandGlobs(context.root, [`${dir}/**/*.sha256`])) {
			const full = path.resolve(context.root, sidecar);
			const content = readLines(full).map((line) => line.trim()).filter(Boolean)[0] ?? "";
			const match = /^([0-9a-fA-F]{64})\s+\*?(.+)$/.exec(content);
			const expected = match ? match[1].toLowerCase() : content.toLowerCase();
			const target = match ? path.resolve(path.dirname(full), match[2]) : full.replace(/\.sha256$/, "");
			const actual = exists(target) ? sha256File(target) : null;
			checks.push({
				id: `sidecar:${sidecar}`,
				ok: /^[0-9a-f]{64}$/.test(expected) && actual === expected,
				detail:
					actual === expected
						? `${sidecar} matches its sidecar`
						: `${sidecar}: expected ${expected}, actual ${actual ?? "missing"}`,
			});
		}
	}
	if (checks.length === 0) {
		return skipped(context, id, "protocol hashes", "no protocol sidecars or configured hashes");
	}
	const failed = checks.filter((check) => !check.ok);
	return {
		id,
		label: "protocol hashes",
		status: failed.length === 0 ? "pass" : "fail",
		detail: `${checks.length} frozen file(s); ${failed.length} mismatch(es)`,
		checks,
	};
}

export function gateConfirmationFrozen(context: GateContext): GateEntry {
	const id = "confirmation-frozen";
	const origins = path.resolve(context.root, context.config.originsFile);
	if (!exists(origins)) return skipped(context, id, "confirmation frozen", `no origins file at ${context.config.originsFile}`);
	const lines = readLines(origins).slice(0, 12);
	const markerLine = lines.findIndex((line) => /retrospective|retro-?active|chosen after|post[- ]?hoc/i.test(line));
	if (markerLine >= 0) {
		return {
			id,
			label: "confirmation frozen",
			status: "fail",
			detail: `${context.config.originsFile}:${markerLine + 1} records retrospective selection — the confirmation set was not frozen`,
			command: `head -12 ${context.config.originsFile}`,
			output_tail: lines.slice(0, 3).join("\n"),
		};
	}
	const ledger = path.join(context.root, ".pi", "state", "EVENTS.jsonl");
	if (!exists(ledger)) return skipped(context, id, "confirmation frozen", "no loop ledger to compare against");
	const frozen = readLines(ledger)
		.map((line) => line.trim())
		.filter(Boolean)
		.map((line) => {
			try {
				return JSON.parse(line) as Record<string, unknown>;
			} catch {
				return null;
			}
		})
		.filter((record): record is Record<string, unknown> => record !== null)
		.find((record) => /confirmation/i.test(JSON.stringify(record)) && /frozen|freeze/i.test(JSON.stringify(record)));
	if (!frozen || typeof frozen.ts !== "string") {
		return skipped(context, id, "confirmation frozen", "no confirmation-frozen event in the ledger");
	}
	const eventTime = Date.parse(frozen.ts);
	let fileTime: number;
	let source: string;
	if (isGitRepo(context.root, context.run)) {
		const log = context.run(["git", "log", "-1", "--format=%ct", "--", context.config.originsFile], {
			cwd: context.root,
		});
		fileTime = Number(log.stdout.trim()) * 1000;
		source = "git commit time";
	} else {
		fileTime = fs.statSync(origins).mtimeMs;
		source = "file mtime";
	}
	const ok = Number.isFinite(eventTime) && Number.isFinite(fileTime) && eventTime + 60_000 >= fileTime;
	return {
		id,
		label: "confirmation frozen",
		status: ok ? "pass" : "fail",
		detail: ok
			? `origins frozen at ${frozen.ts} (ledger) before the file time (${source})`
			: `origins file time (${source}) is after the frozen event: ${new Date(fileTime).toISOString()} > ${frozen.ts}`,
		command: `git log -1 --format=%ct -- ${context.config.originsFile}`,
	};
}

const SELECTION_LINE_RE = /\b(min|max|sorted|sort_values|nlargest|nsmallest|idxmin|idxmax|argmin|argmax)\b/;
const TEST_METRIC_RE = /\btest[_-]?(mape|mae|rmse|error|score|metric)\b/i;

export function gateNoTestSelection(context: GateContext): GateEntry {
	const id = "no-test-selection";
	const hits: string[] = [];
	for (const dir of context.config.selectionScan) {
		for (const file of expandGlobs(context.root, [`${dir}/**/*.py`])) {
			if (/(^|\/)(tests?|tests_baseline)\//.test(file)) continue;
			const lines = readLines(path.resolve(context.root, file));
			lines.forEach((line, index) => {
				if (/^\s*#/.test(line)) return;
				const selects = SELECTION_LINE_RE.test(line);
				const metric = TEST_METRIC_RE.test(line);
				const prev = lines.slice(Math.max(0, index - 2), index).join(" ");
				if ((selects && metric) || (metric && /\b(best|select|choose|pick|winner)\b/i.test(prev))) {
					hits.push(`${file}:${index + 1}`);
				}
			});
		}
	}
	if (hits.length === 0) {
		const scanned = context.config.selectionScan.filter((dir) => exists(path.resolve(context.root, dir)));
		if (scanned.length === 0) return skipped(context, id, "no test selection", "no source directories to scan");
		return {
			id,
			label: "no test selection",
			status: "pass",
			detail: `no test-metric selection in ${scanned.join(", ")}`,
			command: `grep -rn "test_mape" ${scanned.join(" ")}`,
		};
	}
	return {
		id,
		label: "no test selection",
		status: "fail",
		detail: `selection on the test horizon at ${hits.slice(0, 5).join(", ")}${hits.length > 5 ? ` (+${hits.length - 5} more)` : ""}`,
		command: `grep -rn "test_mape" ${context.config.selectionScan.join(" ")}`,
		output_tail: hits.slice(0, 10).join("\n"),
	};
}

export function gateTestDiff(context: GateContext): GateEntry {
	const id = "test-diff";
	const assertion = /\b(assert\b|(self\.)?assert[A-Z]\w*|pytest\.raises|expect\()/;
	if (!context.config.testBaselineDir && isGitRepo(context.root, context.run)) {
		const diff = context.run(["git", "diff", "-U0", context.config.testDiffRange, "--", "*test*"], {
			cwd: context.root,
		});
		let added = 0;
		let removed = 0;
		const problems: string[] = [];
		for (const line of diff.stdout.split("\n")) {
			if (line.startsWith("+") && !line.startsWith("+++") && assertion.test(line)) added += 1;
			if (line.startsWith("-") && !line.startsWith("---") && assertion.test(line)) removed += 1;
			if (
				line.startsWith("+") &&
				!line.startsWith("+++") &&
				/@pytest\.mark\.(skip|xfail)|test\.skip\(|it\.skip\(|@unittest\.skip/.test(line)
			) {
				problems.push(`added skip/xfail: ${line.trim().slice(0, 100)}`);
			}
		}
		if (removed > added) problems.push(`removed ${removed - added} more assertion(s) than added`);
		return {
			id,
			label: "test diff",
			status: problems.length === 0 ? "pass" : "fail",
			detail: problems.length === 0 ? `+${added}/-${removed} assertion lines; no weakening` : problems.join("; "),
			command: diff.command,
			output_tail: diff.stdout.split("\n").filter((line) => line.startsWith("+") || line.startsWith("-")).slice(0, 20).join("\n"),
		};
	}
	const baselineDir = context.config.testBaselineDir;
	if (!baselineDir) {
		return skipped(context, id, "test diff", "not a git repository and no testBaselineDir configured");
	}
	const baseline = path.resolve(context.root, baselineDir);
	if (!exists(baseline)) return skipped(context, id, "test diff", `baseline directory ${baselineDir} not found`);
	const problems: string[] = [];
	const baselineFiles = walkFiles(baseline).filter((file) => file.endsWith(".py"));
	for (const file of baselineFiles) {
		const current = path.resolve(context.root, "tests", file);
		const count = (target: string): number =>
			readLines(target).filter((line) => !/^\s*#/.test(line) && assertion.test(line)).length;
		const before = count(path.join(baseline, file));
		if (!exists(current)) {
			problems.push(`tests/${file} was removed (baseline had ${before} assertion(s))`);
			continue;
		}
		const after = count(current);
		if (after < before) problems.push(`tests/${file}: assertion count dropped from ${before} to ${after}`);
		const commented = readLines(current).some((line) => /^\s*#\s*(assert\b|(self\.)?assert[A-Z]\w*)/.test(line));
		if (commented) problems.push(`tests/${file}: an assertion is commented out`);
	}
	return {
		id,
		label: "test diff",
		status: problems.length === 0 ? "pass" : "fail",
		detail: problems.length === 0 ? `${baselineFiles.length} baseline test file(s); no weakening` : problems.join("; "),
		command: `diff -r ${baselineDir} tests`,
	};
}

export function gateNoFailuresHidden(context: GateContext): GateEntry {
	const id = "no-failures-hidden";
	const report = context.config.resultsReport;
	if (!report) return skipped(context, id, "no hidden failures", "no resultsReport configured");
	const file = path.resolve(context.root, report);
	if (!exists(file)) return skipped(context, id, "no hidden failures", `results report ${report} not found`);
	let data: Record<string, unknown>;
	try {
		data = JSON.parse(fs.readFileSync(file, "utf8")) as Record<string, unknown>;
	} catch (error) {
		return { id, label: "no hidden failures", status: "fail", detail: `${report} is unreadable: ${(error as Error).message}` };
	}
	const attempted = Number(data.attempted ?? data.n_attempted ?? Number.NaN);
	const succeeded = Number(data.succeeded ?? data.n_succeeded ?? Number.NaN);
	const failed = Number(data.failed ?? data.n_failed ?? Number.NaN);
	const reported = Number(data.reported ?? data.n_reported ?? data.n_units ?? attempted);
	const failuresFile = context.config.failuresFile ? path.resolve(context.root, context.config.failuresFile) : null;
	const problems: string[] = [];
	if (Number.isFinite(attempted) && Number.isFinite(succeeded) && Number.isFinite(failed)) {
		if (attempted !== succeeded + failed) {
			problems.push(`attempted ${attempted} ≠ succeeded ${succeeded} + failed ${failed}`);
		}
	}
	if (failed > 0 && failuresFile && !exists(failuresFile)) {
		problems.push(`${failed} failure(s) reported but ${context.config.failuresFile} is missing`);
	}
	if (Number.isFinite(succeeded) && Number.isFinite(reported) && reported !== succeeded) {
		problems.push(`reported ${reported} units but only ${succeeded} succeeded`);
	}
	return {
		id,
		label: "no hidden failures",
		status: problems.length === 0 ? "pass" : "fail",
		detail:
			problems.length === 0
				? `attempted=${attempted}, succeeded=${succeeded}, failed=${failed}, reported=${reported}; reconciles`
				: problems.join("; "),
		command: `cat ${report}`,
	};
}

export function gateContextExcluded(context: GateContext): GateEntry {
	const id = "context-excluded";
	const files = expandGlobs(context.root, context.config.aggregateGlobs);
	if (files.length === 0) return skipped(context, id, "context excluded", "no aggregate files matched");
	const hits: string[] = [];
	for (const file of files) {
		readLines(path.resolve(context.root, file)).forEach((line, index) => {
			if (/^\s*#/.test(line)) return;
			for (const series of context.config.contextSeries) {
				if (line.toLowerCase().includes(series.toLowerCase())) {
					hits.push(`${file}:${index + 1} (${series})`);
				}
			}
		});
	}
	return {
		id,
		label: "context excluded",
		status: hits.length === 0 ? "pass" : "fail",
		detail:
			hits.length === 0
				? `${files.length} aggregate file(s); no context series in target rows`
				: `context series scored as targets: ${hits.slice(0, 6).join(", ")}`,
		command: `grep -in "${context.config.contextSeries.join("\\|")}" ${files.join(" ")}`,
		output_tail: hits.slice(0, 10).join("\n"),
	};
}

export function gateArtifactsComplete(context: GateContext): GateEntry {
	const id = "artifacts-complete";
	const manifests: string[] = [];
	for (const dir of context.config.artifactPaths) {
		for (const file of expandGlobs(context.root, [`${dir}/**/manifest.json`])) manifests.push(file);
	}
	if (manifests.length === 0) return skipped(context, id, "artifacts complete", "no artifact manifests found");
	const checks: GateCheck[] = [];
	let hashOk = true;
	const requiredPresent = new Set<string>();
	for (const manifest of manifests) {
		const verification = verifyArtifact(context.root, path.resolve(context.root, manifest));
		const { manifest: view } = loadManifest(path.resolve(context.root, manifest));
		for (const file of view.files ?? []) requiredPresent.add(file.path);
		checks.push({ id: `artifact:${manifest}`, ok: verification.ok, detail: verification.detail });
		if (!verification.hash_ok) hashOk = false;
	}
	for (const required of context.config.requiredArtifacts) {
		const ok = requiredPresent.has(required) || manifests.some((manifest) => path.basename(manifest) === required);
		checks.push({ id: `required:${required}`, ok, detail: ok ? `${required} present` : `${required} missing from every artifact` });
	}
	const failed = checks.filter((check) => !check.ok);
	return {
		id,
		label: "artifacts complete",
		status: failed.length === 0 ? "pass" : "fail",
		detail: `${manifests.length} manifest(s); ${failed.length} problem(s)`,
		checks,
		hash_ok: hashOk && failed.length === 0,
	};
}

function claimEntries(root: string, claimsPath: string): Array<Record<string, unknown>> {
	const file = path.resolve(root, claimsPath);
	if (!exists(file)) return [];
	try {
		const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as unknown;
		if (Array.isArray(parsed)) return parsed as Array<Record<string, unknown>>;
		return ((parsed as { claims?: Array<Record<string, unknown>> }).claims ?? []) as Array<Record<string, unknown>>;
	} catch {
		return [];
	}
}

export function gateReproduceHeadline(context: GateContext): GateEntry {
	const id = "reproduce-headline";
	const claims = claimEntries(context.root, context.config.traceability.claimsPath);
	const candidates = claims
		.filter((claim) => typeof claim.artifact === "string")
		.sort((a, b) => Number(b.headline === true) - Number(a.headline === true));
	if (candidates.length === 0) return skipped(context, id, "reproduce headline", "no claims with artifacts");
	const checks: GateCheck[] = [];
	const drifts: string[] = [];
	for (const claim of candidates.slice(0, Math.max(1, context.config.reproduceTopN))) {
		const artifact = path.resolve(context.root, String(claim.artifact));
		const manifest = resolveManifestPath(artifact);
		if (!exists(manifest)) {
			checks.push({ id: `claim:${String(claim.id)}`, ok: false, detail: `artifact ${String(claim.artifact)} has no manifest` });
			continue;
		}
		const result = verifyReproduce(context.root, manifest, { runner: context.run, timeoutMs: context.config.timeoutMs });
		checks.push({
			id: `claim:${String(claim.id)}`,
			ok: result.ok,
			detail: result.ok ? "numbers match within tolerance" : (result.error ?? result.drift.replace(/\n/g, "; ")),
		});
		if (!result.ok) drifts.push(`${String(claim.id)}:\n${result.drift || result.error || "failed"}`);
	}
	const failed = checks.filter((check) => !check.ok);
	return {
		id,
		label: "reproduce headline",
		status: failed.length === 0 ? "pass" : "fail",
		detail: `${checks.length} headline claim(s); ${failed.length} drifted`,
		checks,
		command: "verify_reproduce on each claim artifact",
		output_tail: drifts.join("\n").slice(0, 4000),
	};
}

export function gateTraceability(context: GateContext): GateEntry {
	const id = "traceability";
	const result = verifyTraceability(context.root, {
		claimsPath: context.config.traceability.claimsPath,
		paperDir: context.config.traceability.paperDir,
		ref: context.config.traceability.ref,
		requireAncestor: context.config.traceability.requireAncestor,
		runner: context.run,
	});
	if (!result.exists) return skipped(context, id, "traceability", result.detail);
	return {
		id,
		label: "traceability",
		status: result.ok ? "pass" : "fail",
		detail: result.detail,
		command: "verify_traceability",
		output_tail: [
			...result.dangling.map((entry) => `dangling ${entry.claim}: ${entry.artifact} (${entry.reason})`),
			...result.uncited.map((entry) => `uncited ${entry.file}:${entry.line} ${entry.number}`),
		]
			.slice(0, 40)
			.join("\n"),
	};
}

export function gateEnvPinned(context: GateContext): GateEntry {
	const id = "env-pinned";
	const pyproject = path.resolve(context.root, "pyproject.toml");
	if (!exists(pyproject)) return skipped(context, id, "environment pinned", "no pyproject.toml");
	const text = fs.readFileSync(pyproject, "utf8");
	const missing = context.config.envPins.filter((name) => {
		const re = new RegExp(`["']?${name}(?:\\[[^\\]]*\\])?["']?\\s*(==|~=|>=)`);
		return !re.test(text);
	});
	const lock = path.resolve(context.root, "uv.lock");
	const lockOk = exists(lock) && fs.statSync(lock).size > 0;
	const problems: string[] = [];
	if (missing.length > 0) problems.push(`unpinned: ${missing.join(", ")}`);
	if (!lockOk) problems.push("uv.lock is missing or empty");
	return {
		id,
		label: "environment pinned",
		status: problems.length === 0 ? "pass" : "fail",
		detail: problems.length === 0 ? `pins present (${context.config.envPins.join(", ")}); uv.lock present` : problems.join("; "),
		command: "grep -E 'pymc|numpy|numba' pyproject.toml; ls uv.lock",
	};
}

export function gamingScannerPath(root: string): string | null {
	const local = path.join(root, ".pi", "skills", "scientific-integrity-and-anti-gaming", "scripts", "scan_gaming.py");
	if (exists(local)) return local;
	const harness = path.resolve(
		path.dirname(fileURLToPath(import.meta.url)),
		"..",
		"..",
		"skills",
		"scientific-integrity-and-anti-gaming",
		"scripts",
		"scan_gaming.py",
	);
	return exists(harness) ? harness : null;
}

export function gateGamingScan(context: GateContext): GateEntry {
	const id = "gaming-scan";
	const script = gamingScannerPath(context.root);
	if (!script) return skipped(context, id, "gaming scan", "scanner not installed");
	const result = context.run(["python3", script, "--root", context.root, "--fail-on", "none", "--json"], {
		cwd: context.root,
		timeoutMs: context.config.timeoutMs,
	});
	let report: { n_findings?: number; findings?: Array<Record<string, unknown>>; by_severity?: Record<string, number> } | null = null;
	try {
		report = JSON.parse(result.stdout) as typeof report;
	} catch {
		return {
			id,
			label: "gaming scan",
			status: "error",
			detail: `scanner did not return JSON (exit ${result.status})`,
			output_tail: result.stdout.slice(0, 1000) + result.stderr.slice(0, 1000),
		};
	}
	const count = report?.n_findings ?? report?.findings?.length ?? 0;
	return {
		id,
		label: "gaming scan",
		status: count === 0 ? "pass" : "fail",
		detail:
			count === 0
				? "no anti-gaming catalogue hits"
				: `${count} hit(s): ${(report?.findings ?? []).slice(0, 5).map((finding) => `${String(finding.severity)} ${String(finding.category)} ${String(finding.location)}`).join("; ")}`,
		command: result.command,
		output_tail: result.stdout.slice(0, 4000),
	};
}

export function gateCanary(context: GateContext): GateEntry {
	const id = "canary";
	if (!context.config.canaryFixture) return skipped(context, id, "canary", "no canaryFixture configured");
	const fixture = path.resolve(context.root, context.config.canaryFixture);
	if (!exists(fixture)) return skipped(context, id, "canary", `canary fixture ${context.config.canaryFixture} not found`);
	try {
		const result = runSeededAudit(fixture, { threshold: context.config.canaryRecallThreshold });
		return {
			id,
			label: "canary",
			status: result.ok ? "pass" : "fail",
			detail: result.detail,
			checks: [
				{ id: "recall", ok: result.recall >= result.threshold, detail: `recall ${result.recall.toFixed(3)}` },
				{ id: "negative-controls", ok: result.false_positives.length === 0, detail: `${result.false_positives.length} false positive(s)` },
			],
			output_tail: renderCanaryMatrix(result),
		};
	} catch (error) {
		return { id, label: "canary", status: "error", detail: (error as Error).message };
	}
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

export const GATE_ORDER = [
	"tests",
	"protocol-hashes",
	"confirmation-frozen",
	"no-test-selection",
	"test-diff",
	"no-failures-hidden",
	"context-excluded",
	"artifacts-complete",
	"reproduce-headline",
	"traceability",
	"env-pinned",
	"gaming-scan",
	"canary",
] as const;

export type GateId = (typeof GATE_ORDER)[number];

const GATE_FUNCTIONS: Record<GateId, (context: GateContext) => GateEntry> = {
	tests: gateTests,
	"protocol-hashes": gateProtocolHashes,
	"confirmation-frozen": gateConfirmationFrozen,
	"no-test-selection": gateNoTestSelection,
	"test-diff": gateTestDiff,
	"no-failures-hidden": gateNoFailuresHidden,
	"context-excluded": gateContextExcluded,
	"artifacts-complete": gateArtifactsComplete,
	"reproduce-headline": gateReproduceHeadline,
	traceability: gateTraceability,
	"env-pinned": gateEnvPinned,
	"gaming-scan": gateGamingScan,
	canary: gateCanary,
};

export interface RunGatesOptions {
	cycle: number;
	snapshot?: string;
	config?: Partial<AuditConfig>;
	runner?: CommandRunner;
	only?: readonly string[];
	skip?: string[];
	write?: boolean;
	now?: string;
}

export function cycleDir(root: string, cycle: number): string {
	return path.join(root, ".pi", "audit", `cycle-${cycle}`);
}

/**
 * Run the gates and (by default) write `.pi/audit/cycle-<n>/gates.json`.
 * `only` narrows the set; every gate still appears in the output as `skip`.
 */
export function runGates(rootInput: string, options: RunGatesOptions): GatesFile {
	const root = path.resolve(rootInput);
	const config: AuditConfig = { ...loadAuditConfig(root), ...(options.config ?? {}) };
	config.skip = [...config.skip, ...(options.skip ?? [])];
	const run = options.runner ?? defaultRunner;
	const context: GateContext = {
		root,
		config,
		run,
		cycle: options.cycle,
		snapshot: options.snapshot ?? "HEAD",
	};
	const only = options.only ? new Set(options.only) : null;
	const gates: GateEntry[] = [];
	for (const id of GATE_ORDER) {
		if (only && !only.has(id)) {
			const label = id;
			gates.push({ id, label, status: "skip", detail: "not selected for this run" });
			continue;
		}
		try {
			gates.push(GATE_FUNCTIONS[id](context));
		} catch (error) {
			gates.push({ id, label: id, status: "error", detail: (error as Error).message });
		}
	}
	const failed = gates.filter((gate) => gate.status === "fail" || gate.status === "error").map((gate) => gate.id);
	const skippedIds = gates.filter((gate) => gate.status === "skip").map((gate) => gate.id);
	const artifacts = gates.find((gate) => gate.id === "artifacts-complete");
	const file: GatesFile = {
		schema: "vangja-gates/1",
		cycle: options.cycle,
		snapshot: context.snapshot,
		generated_at: options.now ?? new Date().toISOString(),
		ok: failed.length === 0,
		hash_ok: artifacts?.hash_ok === true,
		gates,
		failed,
		skipped: skippedIds,
		config_source: exists(path.join(root, ".pi", "config", "audit.json"))
			? ".pi/config/audit.json"
			: "defaults",
		detail: {
			gate_order: [...GATE_ORDER],
			protocol_hashes: config.protocolHashes,
			canary_threshold: config.canaryRecallThreshold,
		},
	};
	if (options.write !== false) {
		writeJsonAtomic(path.join(cycleDir(root, options.cycle), "gates.json"), file);
	}
	return file;
}

/** Load a previously written gates.json (used by the rubric autofiller). */
export function loadGatesFile(root: string, cycle: number): GatesFile {
	const file = path.join(cycleDir(root, cycle), "gates.json");
	if (!exists(file)) throw new Error(`no gates.json for cycle ${cycle}: ${file}`);
	return JSON.parse(fs.readFileSync(file, "utf8")) as GatesFile;
}

export { manifestDigest };
