/**
 * The pre-merge gate runner, the HPC gate hook, and human gates
 * (plan/02_ARCHITECTURE.md §5, plan/05_EXTENSIONS_AND_TOOLS.md §4).
 *
 * Pre-merge checks, in the plan's order: tests, protected paths, test diff,
 * integrity scan, provenance, hygiene, claims. Every check records the exact
 * command and a raw output tail, and the whole run is written to
 * `state/gate-runs/<pr>-<ts>.json` and summarised in the completion event, so a
 * failed gate is evidence a human can re-run, not an assertion.
 *
 * A failing check blocks the merge unless a waiver record names it
 * (`gate: "pre-merge:<check-id>"`, optionally with the PR id). Waivers are the
 * only way past a red check, and they are supervisor-authored and ledger-visible.
 *
 * Human gates live in `GATES.jsonl` (kinds: merge, confirmation, audit,
 * submission, enqueue). The HPC gate is a hook: S5 implements it; here it
 * reports `not-configured` and never pretends a cluster exists.
 */

import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { appendRecord } from "../ledger/store.ts";
import { loadHpcConfig } from "../../lib/config.ts";
import { readJsonFile, writeFileAtomic } from "../../lib/io.ts";
import { statePath } from "../../lib/paths.ts";
import { readLedger } from "../ledger/store.ts";
import { scanForSecrets, secretPathDecision } from "../../lib/secrets.ts";
import { latestGates, type GateView, type HumanGateKind } from "./state-machine.ts";
import {
	assertCleanCheckout,
	changedFiles,
	diffText,
	git,
	isAncestor,
} from "./git.ts";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type CheckStatus = "pass" | "fail" | "skip" | "waived" | "error";

export interface GateCheck {
	id: string;
	label: string;
	status: CheckStatus;
	detail: string;
	command?: string;
	output_tail?: string;
}

export interface GateRun {
	schema: "pi-gate-run/1";
	pr_id: string;
	task_id: string;
	base_sha: string;
	head_sha: string;
	state: string;
	run_id: string | null;
	ok: boolean;
	checks: GateCheck[];
	waived: string[];
	started_at: string;
	ended_at: string;
	path: string;
}

export interface GateConfig {
	/** Command run for the `tests` check. */
	testsCommand: string[];
	/** Checks skipped deliberately (recorded as `skip`, not silently ignored). */
	skip: string[];
	timeoutMs: number;
}

export const DEFAULT_TESTS_COMMAND = ["uv", "run", "pytest", "-q"];
export const DEFAULT_GATE_TIMEOUT_MS = 20 * 60_000;

/** `.pi/config/gates.json`; defaults are the plan's commands, never a silent skip. */
export function loadGateConfig(root: string): GateConfig {
	const raw = readJsonFile<Record<string, unknown>>(path.join(root, ".pi", "config", "gates.json"));
	if (!raw) return { testsCommand: [...DEFAULT_TESTS_COMMAND], skip: [], timeoutMs: DEFAULT_GATE_TIMEOUT_MS };
	const testsCommand =
		Array.isArray(raw.testsCommand) && raw.testsCommand.length > 0 && raw.testsCommand.every((part) => typeof part === "string")
			? (raw.testsCommand as string[])
			: [...DEFAULT_TESTS_COMMAND];
	const skip = Array.isArray(raw.skip) ? (raw.skip as unknown[]).filter((id): id is string => typeof id === "string") : [];
	const timeoutMs = typeof raw.timeoutMs === "number" && raw.timeoutMs > 0 ? raw.timeoutMs : DEFAULT_GATE_TIMEOUT_MS;
	return { testsCommand, skip, timeoutMs };
}

// ---------------------------------------------------------------------------
// Pre-merge checks
// ---------------------------------------------------------------------------

export interface PreMergeOptions {
	root: string;
	prId: string;
	taskId: string;
	baseSha: string;
	headSha: string;
	branch: string;
	kind: string;
	state: string;
	artifacts: string[];
	/** Paths the PR declares as intentionally protected (supervisor-reviewed). */
	protectedPaths: string[];
	reviewActor: string;
	runId: string | null;
	amendmentCovers?: (rel: string) => boolean;
	config?: GateConfig;
	actor?: string;
}

const PROTECTED_RULES: Array<{
	prefix: string;
	label: string;
	allowed: (options: PreMergeOptions) => boolean;
	alternative: string;
}> = [
	{
		prefix: "Vangja-project",
		label: "read-only historical audit archive",
		allowed: () => false,
		alternative: "put new analysis under .pi/artifacts/ or a path named by the task brief",
	},
	{
		prefix: "research_agent/plan",
		label: "frozen build plan",
		allowed: () => false,
		alternative: "record a decision/amendment instead of editing plan/ (only S7 may edit it)",
	},
	{
		prefix: "fixed_case_studies/protocol",
		label: "frozen study protocol",
		allowed: (options) => options.protectedPaths.some((rel) => rel.startsWith("fixed_case_studies/protocol")),
		alternative: "record an amendment with amendment_record and name the path in the PR footer",
	},
	{
		prefix: "paper",
		label: "paper tree (PAPER state only)",
		allowed: (options) => options.state === "PAPER" || options.protectedPaths.some((rel) => rel.startsWith("paper")),
		alternative: "wait for the PAPER state, or declare the path in the PR footer as intentionally protected",
	},
	{
		prefix: "AGENTS.md",
		label: "shared project instructions",
		allowed: (options) =>
			options.protectedPaths.includes("AGENTS.md") &&
			options.reviewActor === "supervisor",
		alternative: "name AGENTS.md in the PR footer and have the supervisor review the change",
	},
	{
		prefix: ".pi/config",
		label: "loop configuration",
		allowed: (options) =>
			options.protectedPaths.some((rel) => rel.startsWith(".pi/config")) &&
			options.reviewActor === "supervisor",
		alternative: "name the .pi/config path in the PR footer and have the supervisor review the change",
	},
];

function tail(text: string, lines = 12, maxChars = 4000): string {
	const split = text.trimEnd().split("\n");
	const kept = split.slice(-lines).join("\n");
	return kept.length > maxChars ? `${kept.slice(-maxChars)}` : kept;
}

export function runCommand(
	command: string[],
	cwd: string,
	timeoutMs: number,
): { status: number | null; stdout: string; stderr: string } {
	const [bin, ...args] = command;
	const result = spawnSync(bin, args, { cwd, encoding: "utf8", timeout: timeoutMs, maxBuffer: 32 * 1024 * 1024 });
	return {
		status: result.status,
		stdout: String(result.stdout ?? ""),
		stderr: String(result.stderr ?? result.error?.message ?? ""),
	};
}

function configSkip(options: PreMergeOptions, id: string): boolean {
	return (options.config ?? loadGateConfig(options.root)).skip.includes(id);
}

export function checkTests(options: PreMergeOptions): GateCheck {
	const id = "tests";
	if (configSkip(options, id)) return { id, label: "uv run pytest", status: "skip", detail: "skipped by .pi/config/gates.json" };
	const config = options.config ?? loadGateConfig(options.root);
	if (!fs.existsSync(path.join(options.root, "pyproject.toml"))) {
		return { id, label: "uv run pytest", status: "skip", detail: `no pyproject.toml in ${options.root}; no python test suite to run` };
	}
	const result = runCommand(config.testsCommand, options.root, config.timeoutMs);
	const command = config.testsCommand.join(" ");
	if (result.status === 0) {
		return { id, label: command, status: "pass", detail: "test suite green", command, output_tail: tail(result.stdout) };
	}
	return {
		id,
		label: command,
		status: result.status === null ? "error" : "fail",
		detail: `test suite exited ${result.status ?? "timeout"}`,
		command,
		output_tail: tail(`${result.stdout}\n${result.stderr}`),
	};
}

export function checkProtectedPaths(options: PreMergeOptions): GateCheck {
	const id = "protected-paths";
	const files = changedFiles(options.root, options.baseSha, options.headSha);
	const violations: string[] = [];
	for (const file of files) {
		const rel = file.split(path.sep).join("/");
		for (const rule of PROTECTED_RULES) {
			const matched = rel === rule.prefix || rel.startsWith(`${rule.prefix}/`);
			if (!matched) continue;
			if (options.amendmentCovers?.(rel)) continue;
			if (rule.allowed(options)) continue;
			violations.push(`${rel} (${rule.label}; ${rule.alternative})`);
		}
	}
	if (violations.length === 0) {
		return { id, label: "protected paths", status: "pass", detail: `${files.length} changed files, none violating the protected-path rules` };
	}
	return { id, label: "protected paths", status: "fail", detail: violations.join("; ") };
}

export function checkTestDiff(options: PreMergeOptions): GateCheck {
	const id = "test-diff";
	if (configSkip(options, id)) return { id, label: "test diff", status: "skip", detail: "skipped by .pi/config/gates.json" };
	const files = changedFiles(options.root, options.baseSha, options.headSha).filter(
		(file) => /(^|\/)(tests?|test_)/.test(file) || /_test\.(py|ts)$/.test(file) || /\.test\.ts$/.test(file),
	);
	if (files.length === 0) return { id, label: "test diff", status: "pass", detail: "no test files changed" };
	const problems: string[] = [];

	// (a) Every commit in the range: a commit that removes more assertions than it
	// adds, or that adds a skip/xfail marker, is flagged. A base..head endpoint
	// diff would hide weakening that a later commit re-adds.
	const log = git(options.root, [
		"log",
		"-p",
		"--no-merges",
		"--format=commit %H",
		`${options.baseSha}..${options.headSha}`,
		"--",
		...files,
	]).stdout;
	let commit = "?";
	let removed = 0;
	let added = 0;
	const flush = (): void => {
		if (removed > added) {
			problems.push(`${commit.slice(0, 12)}: removed ${removed - added} more assertion(s) than it added`);
		}
		removed = 0;
		added = 0;
	};
	const assertion = /\b(assert|pytest\.raises|expect\()/;
	for (const line of log.split("\n")) {
		if (line.startsWith("commit ")) {
			flush();
			commit = line.slice("commit ".length).trim();
			continue;
		}
		if (line.startsWith("-") && !line.startsWith("---") && assertion.test(line)) removed += 1;
		if (line.startsWith("+") && !line.startsWith("+++") && assertion.test(line)) added += 1;
		if (line.startsWith("+") && !line.startsWith("+++") && /@pytest\.mark\.(skip|xfail)|test\.skip\(|it\.skip\(|@unittest\.skip/.test(line)) {
			problems.push(`${commit.slice(0, 12)}: added a skip/xfail marker (${line.trim().slice(0, 80)})`);
		}
	}
	flush();

	// (b) Net check per file: the head must not assert less than the base.
	for (const file of files) {
		const count = (ref: string): number | null => {
			const result = git(options.root, ["show", `${ref}:${file}`], { allowFailure: true });
			if (result.code !== 0) return null;
			return result.stdout.split("\n").filter((line) => assertion.test(line)).length;
		};
		const before = count(options.baseSha);
		const after = count(options.headSha);
		if (before !== null && after !== null && after < before) {
			problems.push(`${file}: assertion count dropped from ${before} to ${after}`);
		}
	}
	if (problems.length === 0) {
		return { id, label: "test diff", status: "pass", detail: `${files.length} test file(s) changed; no weakened assertions detected` };
	}
	return {
		id,
		label: "test diff",
		status: "fail",
		detail: `${problems.join("; ")} — a removal needs a waiver naming pre-merge:test-diff`,
	};
}

export function checkIntegrity(options: PreMergeOptions): GateCheck {
	const id = "integrity";
	if (configSkip(options, id)) return { id, label: "integrity scan", status: "skip", detail: "skipped by .pi/config/gates.json" };
	const script = path.join(options.root, ".pi", "skills", "scientific-integrity-and-anti-gaming", "scripts", "scan_gaming.py");
	if (!fs.existsSync(script)) {
		return { id, label: "integrity scan", status: "skip", detail: `scanner not found at ${script}` };
	}
	const command = ["python3", script, "--root", options.root, "--diff", options.baseSha, "--fail-on", "none", "--json"];
	const result = runCommand(command, options.root, options.config?.timeoutMs ?? DEFAULT_GATE_TIMEOUT_MS);
	if (result.stdout.trim() === "" && result.status !== 0) {
		return { id, label: "integrity scan", status: "error", detail: result.stderr.trim() || "scanner produced no output", command: command.join(" ") };
	}
	let findings: Array<Record<string, unknown>> = [];
	try {
		const parsed = JSON.parse(result.stdout) as { findings?: Array<Record<string, unknown>> } | Array<Record<string, unknown>>;
		findings = Array.isArray(parsed) ? parsed : (parsed.findings ?? []);
	} catch {
		return {
			id,
			label: "integrity scan",
			status: "error",
			detail: "scanner output was not JSON; inspect it before merging",
			command: command.join(" "),
			output_tail: tail(result.stdout),
		};
	}
	const severe = findings.filter((finding) => ["S0", "S1"].includes(String(finding.severity)));
	if (severe.length === 0) {
		return {
			id,
			label: "integrity scan",
			status: "pass",
			detail: `${findings.length} finding(s), none S0/S1`,
			command: command.join(" "),
		};
	}
	return {
		id,
		label: "integrity scan",
		status: "fail",
		detail: `${severe.length} S0/S1 integrity finding(s): ${severe
			.map((finding) => `${String(finding.id ?? "?")} ${String(finding.title ?? finding.claim ?? "").slice(0, 80)}`)
			.join("; ")}`,
		command: command.join(" "),
		output_tail: tail(result.stdout),
	};
}

export function checkProvenance(options: PreMergeOptions): GateCheck {
	const id = "provenance";
	if (configSkip(options, id)) return { id, label: "provenance", status: "skip", detail: "skipped by .pi/config/gates.json" };
	if (options.artifacts.length === 0) {
		return { id, label: "provenance", status: "pass", detail: "the PR declares no new artifacts" };
	}
	const problems: string[] = [];
	for (const artifact of options.artifacts) {
		const abs = path.isAbsolute(artifact) ? artifact : path.join(options.root, artifact);
		if (!fs.existsSync(abs)) {
			problems.push(`${artifact}: does not exist`);
			continue;
		}
		if (/^data\/|^results\/|\.(csv|h5|nc|parquet)$/.test(artifact)) {
			problems.push(`${artifact}: results/data files belong under .pi/artifacts/ or results/ with a manifest`);
			continue;
		}
		const manifestCandidates = [`${abs}/manifest.json`, `${abs}.manifest.json`];
		const manifestFile = manifestCandidates.find((candidate) => fs.existsSync(candidate));
		if (manifestFile) {
			const manifest = readJsonFile<Record<string, unknown>>(manifestFile);
			if (!manifest || typeof manifest.commit_sha !== "string" || manifest.commit_sha.trim() === "") {
				problems.push(`${artifact}: manifest ${path.relative(options.root, manifestFile)} has no commit_sha`);
			}
			if (manifest && manifest.seeds === undefined) {
				problems.push(`${artifact}: manifest ${path.relative(options.root, manifestFile)} has no seeds`);
			}
		} else if (!options.artifacts.some((other) => other !== artifact && abs.startsWith(path.join(options.root, other)))) {
			problems.push(`${artifact}: no manifest.json next to the artifact`);
		}
	}
	if (problems.length === 0) {
		return { id, label: "provenance", status: "pass", detail: `${options.artifacts.length} artifact(s) have manifests` };
	}
	return { id, label: "provenance", status: "fail", detail: problems.join("; ") };
}

export function checkHygiene(options: PreMergeOptions): GateCheck {
	const id = "hygiene";
	const problems: string[] = [];
	if (!isAncestor(options.root, options.baseSha, options.headSha)) {
		problems.push(`${options.headSha.slice(0, 12)} is not a descendant of ${options.baseSha.slice(0, 12)} (history was rewritten)`);
	}
	const files = changedFiles(options.root, options.baseSha, options.headSha);
	for (const file of files) {
		const rel = file.split(path.sep).join("/");
		const decision = secretPathDecision(rel);
		if (decision.blocked) problems.push(`${rel}: ${decision.reason ?? "secret-bearing path"}`);
	}
	const diff = diffText(options.root, options.baseSha, options.headSha);
	const added = diff
		.split("\n")
		.filter((line) => line.startsWith("+") && !line.startsWith("+++"))
		.map((line) => line.slice(1))
		.join("\n");
	const secrets = scanForSecrets(added);
	if (secrets.findings.length > 0) {
		problems.push(`added lines contain secret-like material (${secrets.findings.map((finding) => finding.kind).join(", ")})`);
	}
	for (const file of files) {
		const abs = path.join(options.root, file);
		try {
			const stat = fs.statSync(abs);
			if (stat.isFile() && stat.size > 5 * 1024 * 1024) problems.push(`${file}: ${(stat.size / 1e6).toFixed(1)} MB data dump`);
		} catch {
			// Deleted file: nothing to size.
		}
	}
	if (problems.length === 0) {
		return { id, label: "hygiene", status: "pass", detail: `${files.length} changed files; no secrets, dumps, or rewritten history` };
	}
	return { id, label: "hygiene", status: "fail", detail: problems.join("; ") };
}

export function checkClaims(options: PreMergeOptions): GateCheck {
	const id = "claims";
	const files = changedFiles(options.root, options.baseSha, options.headSha).map((file) => file.split(path.sep).join("/"));
	const touchesPaper = files.some((file) => file.startsWith("paper/"));
	const script = path.join(options.root, ".pi", "skills", "scientific-writing-latex", "scripts", "check_claims.py");
	if (!touchesPaper) {
		return { id, label: "claims registry", status: "pass", detail: "no paper/** changes" };
	}
	if (!fs.existsSync(script)) {
		return { id, label: "claims registry", status: "skip", detail: `checker not found at ${script}` };
	}
	const command = [
		"python3",
		script,
		"--repo",
		options.root,
		"--paper",
		path.join(options.root, "paper", "paper.tex"),
		"--claims",
		path.join(options.root, "paper", "CLAIMS.json"),
		"--gate",
	];
	const result = runCommand(command, options.root, options.config?.timeoutMs ?? DEFAULT_GATE_TIMEOUT_MS);
	if (result.status === 0) {
		return { id, label: "claims registry", status: "pass", detail: "paper numbers are backed by claims entries", command: command.join(" ") };
	}
	return {
		id,
		label: "claims registry",
		status: "fail",
		detail: "check_claims.py --gate failed; every number in the paper needs a CLAIMS.json entry pointing at an artifact",
		command: command.join(" "),
		output_tail: tail(`${result.stdout}\n${result.stderr}`),
	};
}

export const PRE_MERGE_CHECKS: Array<{ id: string; run: (options: PreMergeOptions) => GateCheck }> = [
	{ id: "tests", run: checkTests },
	{ id: "protected-paths", run: checkProtectedPaths },
	{ id: "test-diff", run: checkTestDiff },
	{ id: "integrity", run: checkIntegrity },
	{ id: "provenance", run: checkProvenance },
	{ id: "hygiene", run: checkHygiene },
	{ id: "claims", run: checkClaims },
];

export interface WaiverLookup {
	gate: string;
	reason: string;
	pr_id?: string;
	task_id?: string;
}

export function loadWaivers(root: string): WaiverLookup[] {
	return readLedger(root, "waiver").map((waiver) => ({
		gate: String(waiver.gate ?? ""),
		reason: String(waiver.reason ?? ""),
		pr_id: typeof waiver.pr_id === "string" ? waiver.pr_id : undefined,
		task_id: typeof waiver.task_id === "string" ? waiver.task_id : undefined,
	}));
}

export function waiverFor(root: string, checkId: string, options: PreMergeOptions): WaiverLookup | null {
	const waivers = loadWaivers(root);
	return (
		waivers.find(
			(waiver) =>
				(waiver.gate === `pre-merge:${checkId}` || waiver.gate === "pre-merge") &&
				(waiver.pr_id === undefined || waiver.pr_id === options.prId),
		) ?? null
	);
}

/** Run every pre-merge check, apply waivers, and persist the run as evidence. */
export function runPreMergeGates(options: PreMergeOptions): GateRun {
	assertCleanCheckout(options.root, `run the pre-merge gate for ${options.prId}`);
	const started = new Date().toISOString();
	const checks: GateCheck[] = [];
	const waived: string[] = [];
	for (const check of PRE_MERGE_CHECKS) {
		let result: GateCheck;
		try {
			result = check.run(options);
		} catch (error) {
			result = { id: check.id, label: check.id, status: "error", detail: (error as Error).message };
		}
		if (result.status === "fail" || result.status === "error") {
			const waiver = waiverFor(options.root, check.id, options);
			if (waiver) {
				waived.push(check.id);
				result = {
					...result,
					status: "waived",
					detail: `${result.detail} — WAIVED (${waiver.reason})`,
				};
			}
		}
		checks.push(result);
	}
	const ok = checks.every((check) => check.status !== "fail" && check.status !== "error");
	const ended = new Date().toISOString();
	const dir = statePath(options.root, "gate-runs");
	fs.mkdirSync(dir, { recursive: true });
	const file = path.join(dir, `${options.prId}-${started.replace(/[:.]/g, "-")}.json`);
	const run: GateRun = {
		schema: "pi-gate-run/1",
		pr_id: options.prId,
		task_id: options.taskId,
		base_sha: options.baseSha,
		head_sha: options.headSha,
		state: options.state,
		run_id: options.runId,
		ok,
		checks,
		waived,
		started_at: started,
		ended_at: ended,
		path: file,
	};
	writeFileAtomic(file, `${JSON.stringify(run, null, 2)}\n`);
	appendRecord(
		options.root,
		"event",
		{
			summary: `pre-merge gate for ${options.prId}: ${ok ? "green" : "RED"} (${checks.filter((check) => check.status === "fail" || check.status === "error").map((check) => check.id).join(", ") || "no failures"}${waived.length > 0 ? `; waivers: ${waived.join(", ")}` : ""})`,
			pr_id: options.prId,
			task_id: options.taskId,
			gate_run: path.relative(options.root, file),
			gate_ok: ok,
			checks: checks.map((check) => ({ id: check.id, status: check.status })),
		},
		{ root: options.root, role: "conductor", actor: options.actor ?? "conductor" },
	);
	return run;
}

// ---------------------------------------------------------------------------
// HPC gate hook (S5 implements the real one)
// ---------------------------------------------------------------------------

export interface HpcGateResult {
	status: "not-configured" | "ok" | "blocked";
	reason: string;
}

/** S4 stub: reports the configured/disabled state; submission is S5's job. */
export function hpcGate(root: string): HpcGateResult {
	try {
		const loaded = loadHpcConfig(root);
		return loaded.enabled
			? { status: "ok", reason: loaded.reason }
			: { status: "not-configured", reason: `${loaded.reason}; submission is disabled until S5 implements the collector` };
	} catch (error) {
		return { status: "blocked", reason: (error as Error).message };
	}
}

// ---------------------------------------------------------------------------
// Human gates
// ---------------------------------------------------------------------------

export interface OpenGateOptions {
	gate: HumanGateKind;
	note: string;
	pr_id?: string;
	task_id?: string;
	run_id?: string | null;
}

export function openGate(root: string, options: OpenGateOptions, actor = "conductor"): Record<string, unknown> {
	return appendRecord(
		root,
		"gate",
		{
			gate: options.gate,
			status: "pending",
			note: options.note,
			pr_id: options.pr_id,
			task_id: options.task_id,
			run_id: options.run_id ?? undefined,
		},
		{ root, role: "conductor", actor },
	);
}

export function findGate(root: string, gateId: string): GateView | null {
	const gate = latestGates(readLedger(root, "gate")).find((record) => record.id === gateId);
	return gate ?? null;
}

export function listGates(root: string): GateView[] {
	return latestGates(readLedger(root, "gate"));
}

function decidableGate(root: string, gateId: string): GateView {
	const gate = findGate(root, gateId);
	if (!gate) throw new Error(`unknown gate ${gateId}; list them with \`loop gate list\``);
	if (gate.status !== "pending") throw new Error(`gate ${gateId} is already ${gate.status}; it cannot be decided twice`);
	return gate;
}

export function approveGate(root: string, gateId: string, note: string, actor = "human"): Record<string, unknown> {
	const gate = decidableGate(root, gateId);
	return appendRecord(
		root,
		"gate",
		{ id: gate.id, gate: gate.gate, status: "approved", note, pr_id: gate.pr_id, task_id: gate.task_id },
		{ root, role: "conductor", actor },
	);
}

export function rejectGate(root: string, gateId: string, note: string, actor = "human"): Record<string, unknown> {
	const gate = decidableGate(root, gateId);
	return appendRecord(
		root,
		"gate",
		{ id: gate.id, gate: gate.gate, status: "rejected", note, pr_id: gate.pr_id, task_id: gate.task_id },
		{ root, role: "conductor", actor },
	);
}

export function gateSummary(gate: GateView): string {
	return `${gate.id} ${gate.gate} ${gate.status}: ${gate.note}`;
}
