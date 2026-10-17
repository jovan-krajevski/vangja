/**
 * Deterministic verification primitives: tests, artifact hashes, reproduction
 * with numeric comparison, claims↔artifact traceability, and diff guards.
 *
 * Every function returns the exact command it ran plus a raw output tail, so a
 * failed check is reproducible evidence rather than an assertion. Process
 * execution is injectable (`CommandRunner`) so tests stay offline.
 *
 * No pi imports: usable from `node --test`.
 */

import { spawnSync } from "node:child_process";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { hashTree, missingOutputs, sha256File, sha256Text, walkFiles } from "../hpc/manifest.ts";

export interface CommandResult {
	status: number | null;
	stdout: string;
	stderr: string;
	command: string;
}

export type CommandRunner = (
	command: string[],
	options: { cwd: string; timeoutMs?: number; env?: Record<string, string> },
) => CommandResult;

export const DEFAULT_TIMEOUT_MS = 20 * 60_000;

export const defaultRunner: CommandRunner = (command, options) => {
	const result = spawnSync(command[0], command.slice(1), {
		cwd: options.cwd,
		env: { ...process.env, ...options.env },
		encoding: "utf8",
		timeout: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
		maxBuffer: 64 * 1024 * 1024,
	});
	return {
		status: result.status,
		stdout: result.stdout ?? "",
		stderr: result.stderr ?? "",
		command: command.join(" "),
	};
};

export function tail(text: string, lines = 40): string {
	const parts = text.trimEnd().split("\n");
	return parts.slice(Math.max(0, parts.length - lines)).join("\n");
}

function exists(file: string): boolean {
	return fs.existsSync(file);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

export interface TestsResult {
	ok: boolean;
	status: number | null;
	command: string;
	passed: number | null;
	failed: number | null;
	skipped: number | null;
	summary: string;
	failures: string[];
	output_tail: string;
}

export function verifyTests(
	root: string,
	options: { paths?: string[]; command?: string[]; runner?: CommandRunner; timeoutMs?: number } = {},
): TestsResult {
	const command = options.command ?? ["uv", "run", "pytest", "-q", ...(options.paths ?? [])];
	const run = options.runner ?? defaultRunner;
	const result = run(command, { cwd: root, timeoutMs: options.timeoutMs ?? DEFAULT_TIMEOUT_MS });
	const combined = `${result.stdout}\n${result.stderr}`;
	const count = (pattern: RegExp): number | null => {
		const matches = [...combined.matchAll(pattern)];
		if (matches.length === 0) return null;
		return matches.reduce((sum, match) => sum + Number(match[1]), 0);
	};
	const passed = count(/(\d+)\s+passed/g);
	const failed = count(/(\d+)\s+failed/g);
	const skipped = count(/(\d+)\s+skipped/g);
	const failures: string[] = [];
	for (const line of combined.split("\n")) {
		if (/^(FAILED|ERROR)\s+/.test(line.trim())) failures.push(line.trim());
		if (/^E\s+\w+Error:/.test(line.trim()) && failures.length < 10) failures.push(line.trim().slice(0, 200));
	}
	return {
		ok: result.status === 0,
		status: result.status,
		command: result.command,
		passed,
		failed,
		skipped,
		summary: [
			passed === null ? null : `${passed} passed`,
			failed === null ? null : `${failed} failed`,
			skipped === null ? null : `${skipped} skipped`,
		]
			.filter(Boolean)
			.join(", "),
		failures: [...new Set(failures)].slice(0, 10),
		output_tail: tail(combined),
	};
}

// ---------------------------------------------------------------------------
// Artifact hashes
// ---------------------------------------------------------------------------

export interface ArtifactFile {
	path: string;
	sha256: string;
	size?: number;
	mtime?: string;
}

export interface ArtifactManifestView {
	schema?: string;
	experiment_id?: string;
	commit_sha?: string;
	repro_command?: string | string[];
	numbers?: Record<string, number>;
	expected_numbers?: Record<string, number>;
	tolerance?: number;
	tolerances?: Record<string, number>;
	files?: ArtifactFile[];
	missing?: string[];
	manifest_sha256?: string;
	[key: string]: unknown;
}

export interface ArtifactVerification {
	ok: boolean;
	hash_ok: boolean;
	manifest_path: string;
	schema_ok: boolean;
	manifest_sha256_ok: boolean | null;
	files: number;
	missing_files: string[];
	declared_missing: string[];
	mismatched: Array<{ path: string; expected: string; actual: string | null }>;
	detail: string;
}

export function resolveManifestPath(target: string): string {
	if (fs.existsSync(target) && fs.statSync(target).isDirectory()) return path.join(target, "manifest.json");
	return target;
}

export function loadManifest(target: string): { manifest: ArtifactManifestView; file: string; dir: string } {
	const file = resolveManifestPath(target);
	if (!exists(file)) throw new Error(`manifest not found: ${file}`);
	let manifest: ArtifactManifestView;
	try {
		manifest = JSON.parse(fs.readFileSync(file, "utf8")) as ArtifactManifestView;
	} catch (error) {
		throw new Error(`cannot parse manifest ${file}: ${(error as Error).message}`);
	}
	return { manifest, file, dir: path.dirname(file) };
}

/** Manifest SHA-256 is computed over the JSON body without `manifest_sha256`. */
export function manifestDigest(manifest: ArtifactManifestView): string {
	const body = { ...manifest };
	delete body.manifest_sha256;
	return sha256Text(`${JSON.stringify(body, null, 2)}\n`);
}

/**
 * Verify that every declared file exists and matches its hash; that declared
 * missing outputs are absent; and that the manifest's own digest matches.
 */
export function verifyArtifact(root: string, target: string): ArtifactVerification {
	const { manifest, file, dir } = loadManifest(target);
	const schemaOk =
		typeof manifest.schema === "string" && /^(pi-artifact\/1|vangja-artifact\/1)$/.test(manifest.schema);
	const files = Array.isArray(manifest.files) ? manifest.files : [];
	const missingFiles: string[] = [];
	const mismatched: ArtifactVerification["mismatched"] = [];
	for (const entry of files) {
		const full = path.join(dir, entry.path);
		if (!exists(full)) {
			missingFiles.push(entry.path);
			mismatched.push({ path: entry.path, expected: entry.sha256, actual: null });
			continue;
		}
		const actual = sha256File(full);
		if (actual !== entry.sha256) mismatched.push({ path: entry.path, expected: entry.sha256, actual });
	}
	const declaredMissing = Array.isArray(manifest.missing) ? manifest.missing.map(String) : [];
	const manifestShaOk =
		typeof manifest.manifest_sha256 === "string" ? manifestDigest(manifest) === manifest.manifest_sha256 : null;
	const hashOk = schemaOk && files.length > 0 && missingFiles.length === 0 && mismatched.length === 0 && manifestShaOk !== false;
	const detail = [
		`${files.length} declared file(s)`,
		missingFiles.length > 0 ? `missing: ${missingFiles.join(", ")}` : null,
		mismatched.length > 0
			? `hash mismatches: ${mismatched
					.map((entry) => `${entry.path} (expected ${entry.expected.slice(0, 12)}…, actual ${entry.actual?.slice(0, 12) ?? "absent"}…)`)
					.join("; ")}`
			: null,
		manifestShaOk === false ? "manifest_sha256 does not match its body" : null,
		declaredMissing.length > 0 ? `declared missing outputs: ${declaredMissing.join(", ")}` : null,
		!schemaOk ? `schema is ${String(manifest.schema) || "absent"}` : null,
	]
		.filter(Boolean)
		.join("; ");
	return {
		ok: hashOk && declaredMissing.length === 0,
		hash_ok: hashOk,
		manifest_path: file,
		schema_ok: schemaOk,
		manifest_sha256_ok: manifestShaOk,
		files: files.length,
		missing_files: missingFiles,
		declared_missing: declaredMissing,
		mismatched,
		detail: detail || `${path.relative(root, file)} verifies`,
	};
}

/** `verify_artifact` also checks that declared outputs expected by the record exist. */
export function verifyArtifactAgainstExpected(
	root: string,
	target: string,
	expectedOutputs: string[],
): ArtifactVerification & { missing_expected: string[] } {
	const base = verifyArtifact(root, target);
	const file = resolveManifestPath(target);
	const dir = path.dirname(file);
	const present = new Set(fs.existsSync(dir) ? walkFiles(dir) : []);
	const missing = expectedOutputs.filter((pattern) => {
		if (!pattern.includes("*")) return !present.has(pattern);
		const regex = new RegExp(`^${pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&").replace(/\*/g, "[^/]*")}$`);
		return ![...present].some((item) => regex.test(item));
	});
	return { ...base, ok: base.ok && missing.length === 0, missing_expected: missing };
}

// ---------------------------------------------------------------------------
// Reproduction with numeric comparison
// ---------------------------------------------------------------------------

export interface NumberDiff {
	name: string;
	expected: number;
	actual: number | null;
	delta: number | null;
	tolerance: number;
	within: boolean;
}

export interface ReproduceResult {
	ok: boolean;
	command: string;
	artifact: string;
	tolerance: number;
	numbers: Record<string, number>;
	diffs: NumberDiff[];
	drift: string;
	output_tail: string;
	status: number | null;
	error?: string;
}

function flattenNumbers(value: unknown, prefix = ""): Record<string, number> {
	const out: Record<string, number> = {};
	if (value === null || value === undefined) return out;
	if (typeof value === "number" && Number.isFinite(value)) {
		out[prefix || "value"] = value;
		return out;
	}
	if (Array.isArray(value)) {
		value.forEach((item, index) => {
			const key = item && typeof item === "object" && "name" in item ? String((item as { name: unknown }).name) : `${prefix}[${index}]`;
			Object.assign(out, flattenNumbers(item && typeof item === "object" && "value" in item ? (item as { value: unknown }).value : item, key));
		});
		return out;
	}
	if (typeof value === "object") {
		for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
			const next = prefix ? `${prefix}.${key}` : key;
			if (typeof item === "object" && item !== null) Object.assign(out, flattenNumbers(item, next));
			else if (typeof item === "number" && Number.isFinite(item)) out[next] = item;
		}
	}
	return out;
}

function parseNumbers(stdout: string): { numbers: Record<string, number>; error?: string } {
	const trimmed = stdout.trim();
	if (trimmed === "") return { numbers: {}, error: "repro command printed nothing" };
	const attempts = [trimmed];
	const first = trimmed.indexOf("{");
	const last = trimmed.lastIndexOf("}");
	if (first >= 0 && last > first) attempts.push(trimmed.slice(first, last + 1));
	for (const attempt of attempts) {
		try {
			const parsed = JSON.parse(attempt) as unknown;
			const container =
				parsed && typeof parsed === "object" && "numbers" in parsed
					? (parsed as { numbers: unknown }).numbers
					: parsed && typeof parsed === "object" && "metrics" in parsed
						? (parsed as { metrics: unknown }).metrics
						: parsed;
			const numbers = flattenNumbers(container);
			if (Object.keys(numbers).length > 0) return { numbers };
		} catch {
			// try the next candidate
		}
	}
	return { numbers: {}, error: "repro command did not print a JSON object of numbers" };
}

/**
 * Execute the artifact's recorded `repro_command` in a fresh temp dir and compare
 * the printed numbers with the manifest's recorded values within a stated
 * tolerance. Reports a per-number diff (expected, actual, delta, tolerance).
 */
export function verifyReproduce(
	root: string,
	target: string,
	options: { runner?: CommandRunner; tolerance?: number; timeoutMs?: number; keepTemp?: boolean } = {},
): ReproduceResult {
	const { manifest, dir } = loadManifest(target);
	const command = manifest.repro_command;
	const expected = manifest.numbers ?? manifest.expected_numbers;
	const base: ReproduceResult = {
		ok: false,
		command: Array.isArray(command) ? command.join(" ") : String(command ?? ""),
		artifact: resolveManifestPath(target),
		tolerance: options.tolerance ?? 1e-9,
		numbers: {},
		diffs: [],
		drift: "",
		output_tail: "",
		status: null,
	};
	if (!command || (Array.isArray(command) && command.length === 0)) {
		return { ...base, error: "manifest has no repro_command" };
	}
	if (!expected || Object.keys(expected).length === 0) {
		return { ...base, error: "manifest records no numbers to compare" };
	}
	const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vangja-reproduce-"));
	const defaultTolerance = options.tolerance ?? (typeof manifest.tolerance === "number" ? manifest.tolerance : 1e-9);
	try {
		const argv = Array.isArray(command) ? command : ["bash", "-lc", command];
		const run = options.runner ?? defaultRunner;
		const result = run(argv, {
			cwd: tempDir,
			timeoutMs: options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
			env: { PI_REPO_ROOT: root, PI_ARTIFACT_DIR: dir },
		});
		const parsed = parseNumbers(result.stdout);
		const diffs: NumberDiff[] = [];
		for (const [name, expectedValue] of Object.entries(expected)) {
			const actual = parsed.numbers[name] ?? null;
			const tolerance =
				options.tolerance ??
				(typeof manifest.tolerances?.[name] === "number" ? manifest.tolerances[name] : defaultTolerance);
			const delta = actual === null ? null : actual - expectedValue;
			const within = delta !== null && Math.abs(delta) <= tolerance + Number.EPSILON;
			diffs.push({ name, expected: expectedValue, actual, delta, tolerance, within });
		}
		const allWithin = diffs.length > 0 && diffs.every((diff) => diff.within);
		const drift = diffs
			.map((diff) => {
				const actual = diff.actual === null ? "missing" : diff.actual.toPrecision(12);
				const delta = diff.delta === null ? "n/a" : diff.delta.toExponential(3);
				return `${diff.name}: expected=${diff.expected} actual=${actual} delta=${delta} tolerance=${diff.tolerance}${
					diff.within ? "" : " ❌"
				}`;
			})
			.join("\n");
		return {
			ok: result.status === 0 && allWithin && !parsed.error,
			command: result.command,
			artifact: base.artifact,
			tolerance: defaultTolerance,
			numbers: expected,
			diffs,
			drift,
			output_tail: tail(`${result.stdout}\n${result.stderr}`),
			status: result.status,
			error: parsed.error && !allWithin ? parsed.error : parsed.error,
		};
	} finally {
		if (!options.keepTemp) fs.rmSync(tempDir, { recursive: true, force: true });
	}
}

// ---------------------------------------------------------------------------
// Traceability
// ---------------------------------------------------------------------------

export interface ClaimEntry {
	id?: string;
	statement?: string;
	value?: number;
	number?: string | number;
	tex?: string;
	artifact?: string;
	artifacts?: string[];
	numbers?: Array<number | { name?: string; value: number; tex?: string }>;
	headline?: boolean;
	sha?: string;
	[key: string]: unknown;
}

export interface TraceabilityResult {
	ok: boolean;
	exists: boolean;
	claims: number;
	dangling: Array<{ claim: string; artifact: string; reason: string }>;
	uncited: Array<{ file: string; line: number; number: string }>;
	scanned_files: string[];
	checked_ancestry: number;
	detail: string;
}

const TEX_SKIP_RE = /\\(cite\w*|ref|label|includegraphics|input|include|begin|end|usepackage|documentclass|hspace|vspace|setlength|url|href)\*?(\[[^\]]*\])?\{[^}]*\}/g;

function texFiles(paperDir: string): string[] {
	if (!fs.existsSync(paperDir)) return [];
	return walkFiles(paperDir)
		.filter((file) => file.endsWith(".tex"))
		.map((file) => path.join(paperDir, file));
}

/** Numeric claims covered by a claim entry (value, number, numbers[]). */
function claimNumbers(claim: ClaimEntry): number[] {
	const values: number[] = [];
	if (typeof claim.value === "number") values.push(claim.value);
	if (typeof claim.number === "number") values.push(claim.number);
	if (typeof claim.number === "string" && claim.number.trim() !== "" && Number.isFinite(Number(claim.number))) {
		values.push(Number(claim.number));
	}
	for (const entry of claim.numbers ?? []) {
		if (typeof entry === "number") values.push(entry);
		else if (entry && typeof entry.value === "number") values.push(entry.value);
	}
	return values;
}

function claimArtifacts(claim: ClaimEntry): string[] {
	if (Array.isArray(claim.artifacts)) return claim.artifacts.map(String);
	if (typeof claim.artifact === "string") return [claim.artifact];
	return [];
}

/**
 * Check `paper/CLAIMS.json`: every claim has an artifact whose manifest SHA is an
 * ancestor of the audited ref; every decimal number in `paper/**` has a claim.
 */
export function verifyTraceability(
	root: string,
	options: {
		claimsPath?: string;
		paperDir?: string;
		ref?: string;
		runner?: CommandRunner;
		requireAncestor?: boolean;
	} = {},
): TraceabilityResult {
	const claimsPath = path.resolve(root, options.claimsPath ?? path.join("paper", "CLAIMS.json"));
	const paperDir = path.resolve(root, options.paperDir ?? "paper");
	const ref = options.ref ?? "HEAD";
	const requireAncestor = options.requireAncestor ?? true;
	const run = options.runner ?? defaultRunner;
	const result: TraceabilityResult = {
		ok: false,
		exists: false,
		claims: 0,
		dangling: [],
		uncited: [],
		scanned_files: [],
		checked_ancestry: 0,
		detail: "",
	};
	if (!exists(claimsPath)) {
		result.detail = `no claims registry at ${path.relative(root, claimsPath)}`;
		return result;
	}
	result.exists = true;
	let parsed: unknown;
	try {
		parsed = JSON.parse(fs.readFileSync(claimsPath, "utf8"));
	} catch (error) {
		result.detail = `cannot parse ${path.relative(root, claimsPath)}: ${(error as Error).message}`;
		return result;
	}
	const claims: ClaimEntry[] = Array.isArray(parsed)
		? (parsed as ClaimEntry[])
		: ((parsed as { claims?: ClaimEntry[] }).claims ?? []);
	result.claims = claims.length;

	for (const [index, claim] of claims.entries()) {
		const id = claim.id ?? `claim[${index}]`;
		const artifacts = claimArtifacts(claim);
		if (artifacts.length === 0) {
			result.dangling.push({ claim: id, artifact: "(none)", reason: "claim has no artifact" });
			continue;
		}
		for (const artifact of artifacts) {
			const artifactPath = path.resolve(root, artifact);
			const manifestPath = resolveManifestPath(artifactPath);
			if (!exists(manifestPath)) {
				result.dangling.push({ claim: id, artifact, reason: "artifact/manifest does not exist" });
				continue;
			}
			let manifest: ArtifactManifestView;
			try {
				manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8")) as ArtifactManifestView;
			} catch (error) {
				result.dangling.push({ claim: id, artifact, reason: `manifest unreadable: ${(error as Error).message}` });
				continue;
			}
			const sha = (typeof claim.sha === "string" ? claim.sha : undefined) ?? manifest.commit_sha;
			if (requireAncestor) {
				if (!sha) {
					result.dangling.push({ claim: id, artifact, reason: "manifest records no commit SHA" });
					continue;
				}
				result.checked_ancestry += 1;
				const check = run(["git", "merge-base", "--is-ancestor", sha, ref], { cwd: root });
				if (check.status !== 0) {
					result.dangling.push({
						claim: id,
						artifact,
						reason: `SHA ${sha.slice(0, 12)} is not an ancestor of ${ref}`,
					});
				}
			}
		}
	}

	const claimedValues = claims.flatMap(claimNumbers);
	const numberRe = /(?<![\w.])[-+]?\d+\.\d+(?![\w.])/g;
	for (const file of texFiles(paperDir)) {
		const rel = path.relative(root, file).split(path.sep).join("/");
		result.scanned_files.push(rel);
		const text = fs.readFileSync(file, "utf8");
		const lines = text.split("\n");
		lines.forEach((line, index) => {
			if (/^\s*%/.test(line)) return;
			const stripped = line.replace(TEX_SKIP_RE, " ");
			for (const match of stripped.matchAll(numberRe)) {
				const token = match[0];
				const value = Number(token);
				if (!Number.isFinite(value)) continue;
				const covered = claimedValues.some((claimed) => Math.abs(claimed - value) <= 1e-12);
				if (!covered) result.uncited.push({ file: rel, line: index + 1, number: token });
			}
		});
	}

	result.ok = result.dangling.length === 0 && result.uncited.length === 0;
	result.detail = `${claims.length} claim(s), ${result.checked_ancestry} ancestry check(s), ${result.scanned_files.length} tex file(s); ${result.dangling.length} dangling, ${result.uncited.length} uncited number(s)`;
	return result;
}

// ---------------------------------------------------------------------------
// Guards (protected paths, test diff, gaming scan)
// ---------------------------------------------------------------------------

const PROTECTED_PREFIXES: Array<{ prefix: string; label: string }> = [
	{ prefix: "Vangja-project", label: "historical audit archive" },
	{ prefix: "research_agent/plan", label: "frozen build plan" },
	{ prefix: "fixed_case_studies/protocol", label: "frozen protocol" },
	{ prefix: "AGENTS.md", label: "shared project instructions" },
	{ prefix: ".git", label: "git store" },
	{ prefix: ".pi/state", label: "loop state (ledger tools only)" },
];

export interface GuardsResult {
	ok: boolean;
	range: string;
	changed_files: string[];
	protected_violations: string[];
	test_diff: { ok: boolean; added: number; removed: number; problems: string[] };
	gaming: { ok: boolean; findings: unknown[]; by_severity?: Record<string, number>; error?: string };
	command: string;
	output_tail: string;
}

export function verifyGuards(
	root: string,
	options: { range?: string; runner?: CommandRunner; allowPaper?: boolean; allowProtected?: boolean } = {},
): GuardsResult {
	const range = options.range ?? "HEAD~1..HEAD";
	const run = options.runner ?? defaultRunner;
	const diff = run(["git", "diff", "--name-only", range], { cwd: root });
	const changed = diff.stdout.split("\n").map((line) => line.trim()).filter(Boolean);
	const violations: string[] = [];
	if (!options.allowProtected) {
		for (const file of changed) {
			const rel = file.split(path.sep).join("/");
			for (const rule of PROTECTED_PREFIXES) {
				if (rel === rule.prefix || rel.startsWith(`${rule.prefix}/`)) violations.push(`${rel} (${rule.label})`);
			}
			if (!options.allowPaper && (rel === "paper" || rel.startsWith("paper/"))) violations.push(`${rel} (paper outside PAPER state)`);
		}
	}

	const assertion = /\b(assert\b|pytest\.raises|expect\()/;
	const testDiff = run(["git", "diff", "-U0", range, "--", "*test*"], { cwd: root });
	let added = 0;
	let removed = 0;
	const problems: string[] = [];
	for (const line of testDiff.stdout.split("\n")) {
		if (line.startsWith("+") && !line.startsWith("+++") && assertion.test(line)) added += 1;
		if (line.startsWith("-") && !line.startsWith("---") && assertion.test(line)) removed += 1;
		if (line.startsWith("+") && !line.startsWith("+++") && /@pytest\.mark\.(skip|xfail)|test\.skip\(|it\.skip\(|@unittest\.skip/.test(line)) {
			problems.push(`added a skip/xfail marker: ${line.trim().slice(0, 100)}`);
		}
	}
	if (removed > added) problems.push(`removed ${removed - added} more assertion(s) than added`);
	const testDiffOk = problems.length === 0;

	let gaming: GuardsResult["gaming"] = { ok: true, findings: [] };
	const script = path.join(root, ".pi", "skills", "scientific-integrity-and-anti-gaming", "scripts", "scan_gaming.py");
	if (exists(script)) {
		const scan = run(
			["python3", script, "--root", root, "--diff", range, "--fail-on", "none", "--json"],
			{ cwd: root, timeoutMs: DEFAULT_TIMEOUT_MS },
		);
		try {
			const report = JSON.parse(scan.stdout) as { findings?: unknown[]; by_severity?: Record<string, number> };
			gaming = { ok: (report.findings ?? []).length === 0, findings: report.findings ?? [], by_severity: report.by_severity };
		} catch {
			gaming = { ok: scan.status === 0, findings: [], error: `scanner did not return JSON (status ${scan.status})` };
		}
	} else {
		gaming = { ok: true, findings: [], error: "scanner not installed" };
	}

	return {
		ok: violations.length === 0 && testDiffOk && gaming.ok,
		range,
		changed_files: changed,
		protected_violations: violations,
		test_diff: { ok: testDiffOk, added, removed, problems },
		gaming,
		command: `${diff.command}; ${testDiff.command}`,
		output_tail: tail(`${diff.stderr}\n${testDiff.stderr}`),
	};
}

// ---------------------------------------------------------------------------
// Helpers shared with gates
// ---------------------------------------------------------------------------

export { hashTree, missingOutputs, sha256File, sha256Text, walkFiles };
export const _private = { parseNumbers, flattenNumbers, claimNumbers, claimArtifacts, texFiles, crypto };
