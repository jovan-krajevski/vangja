/**
 * Environment check for `make -C .pi env-check` / `verify-env`.
 *
 * Prints one PASS/FAIL line per check, exits non-zero if any check fails.
 * Read-only: it never writes state and never calls a model.
 *
 * Usage:
 *   node .pi/tests/check_env.ts            # full environment check
 *   node .pi/tests/check_env.ts --status   # read-only loop status (status.sh fallback)
 */

import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { loadAgentsManifest, loadBudgetConfig, loadHpcConfig, loadHpcExample, loadModelPin, loadProjectConfig, DEFAULT_BUDGETS } from "../lib/config.ts";
import { scanForSecrets } from "../lib/secrets.ts";
import { readJsonFile } from "../lib/io.ts";
import { statePath } from "../lib/paths.ts";
import { readStatus } from "../extensions/status/status.ts";
import { composeStatusLines } from "../extensions/status/status.ts";
import { PI_DIR, REPO_ROOT, piVersion } from "./helpers.ts";

interface Check {
	name: string;
	ok: boolean;
	detail: string;
}

const checks: Check[] = [];

function check(name: string, fn: () => string): void {
	try {
		checks.push({ name, ok: true, detail: fn() });
	} catch (error) {
		checks.push({ name, ok: false, detail: (error as Error).message });
	}
}

function assert(condition: unknown, message: string): void {
	if (!condition) throw new Error(message);
}

if (process.argv.includes("--status")) {
	for (const line of composeStatusLines(readStatus(REPO_ROOT))) process.stdout.write(`${line}\n`);
	process.exit(0);
}

check("pi binary", () => {
	const version = execFileSync("pi", ["--version"], { encoding: "utf8" }).trim();
	assert(/^\d+\.\d+\.\d+/.test(version), `unexpected version string "${version}"`);
	return `pi ${version} on PATH`;
});

check("settings pin", () => {
	const settings = readJsonFile<Record<string, unknown>>(path.join(PI_DIR, "settings.json"));
	assert(settings, ".pi/settings.json is missing");
	assert(settings.$schema === "https://pi.dev/schemas/settings.schema.json", "missing $schema");
	assert(settings.defaultProvider === "opencode-go", "defaultProvider drift");
	assert(settings.defaultModel === "deepseek-v4.1-flash", "defaultModel drift");
	assert(settings.defaultThinkingLevel === "max", "defaultThinkingLevel drift");
	assert(
		Array.isArray(settings.enabledModels) && settings.enabledModels.length === 1 && settings.enabledModels[0] === "opencode-go/deepseek-v4.1-flash",
		"enabledModels drift",
	);
	const compaction = settings.compaction as { reserveTokens?: number; keepRecentTokens?: number } | undefined;
	assert(compaction?.reserveTokens === 32768, "compaction.reserveTokens drift");
	assert(compaction?.keepRecentTokens === 120000, "compaction.keepRecentTokens drift");
	return "model pin, enabledModels, +codemode, and compaction values match plan/02 §7";
});

check("model.json", () => {
	const pin = loadModelPin(REPO_ROOT);
	return `${pin.provider}/${pin.model} thinking=${pin.thinking} window=${pin.contextWindow} enforce=${pin.enforce}`;
});

check("budgets.json", () => {
	const budgets = loadBudgetConfig(REPO_ROOT);
	for (const [key, value] of Object.entries(DEFAULT_BUDGETS)) {
		assert(budgets[key as keyof typeof DEFAULT_BUDGETS] === value, `${key} differs from the documented default`);
	}
	return `all ${Object.keys(DEFAULT_BUDGETS).length} caps equal plan/02_ARCHITECTURE.md §6`;
});

check("project.json", () => {
	const project = loadProjectConfig(REPO_ROOT);
	assert(project.baseBranch === "update_pymc", "baseBranch drift");
	assert(project.stateDir === ".pi/state", "stateDir drift");
	return `${project.venue}; base=${project.baseBranch}; requireHumanMerge=${project.requireHumanMerge}; dryRun=${project.dryRun}`;
});

check("agents.json", () => {
	const manifest = loadAgentsManifest(REPO_ROOT);
	const files = manifest.agents.map((agent) => agent.file);
	const present = files.filter((file) => fs.existsSync(path.join(REPO_ROOT, file)));
	return `${manifest.agents.length} agents, all on ${manifest.pinnedModel} thinking=${manifest.pinnedThinking}; ` +
		`${present.length}/${files.length} definition files present (the rest arrive in S3)`;
});

check("hpc.example.json", () => {
	const example = loadHpcExample(REPO_ROOT);
	assert(example.maxParallelJobs === 1, "maxParallelJobs must be 1");
	assert(example.wallClockHours === 24, "wallClockHours must be 24");
	return `host="${example.host}" wall=${example.wallClockHours}h singleSlot=${example.maxParallelJobs}`;
});

check("hpc.json (runtime)", () => {
	const loaded = loadHpcConfig(REPO_ROOT);
	return loaded.enabled ? `enabled (${loaded.reason})` : `disabled: ${loaded.reason}`;
});

check("loop shims", () => {
	for (const shim of ["step.sh", "status.sh", "collect.sh"]) {
		const file = path.join(PI_DIR, "loop", shim);
		assert(fs.existsSync(file), `${file} is missing`);
		assert((fs.statSync(file).mode & 0o111) !== 0, `${file} is not executable`);
	}
	return "step.sh, status.sh, collect.sh present and executable (S4/S5 stubs)";
});

check("gitignore coverage", () => {
	const text = fs.readFileSync(path.join(REPO_ROOT, ".gitignore"), "utf8");
	const required = [".pi/state/", ".pi/artifacts/*/", ".pi/config/hpc.json", ".pi/**/node_modules/", ".pi/loop/worktrees/", ".pi/sessions/"];
	for (const entry of required) assert(text.includes(entry), `.gitignore is missing "${entry}"`);
	return `${required.length} runtime paths ignored`;
});

check("no key material under .pi", () => {
	const findings: string[] = [];
	const walk = (dir: string): void => {
		for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
			if (entry.name === "node_modules" || entry.name === "state" || entry.name === "artifacts") continue;
			const full = path.join(dir, entry.name);
			if (entry.isDirectory()) {
				walk(full);
			} else if (entry.isFile() && /\.(ts|json|md|sh|txt|example)$/.test(entry.name)) {
				const text = fs.readFileSync(full, "utf8");
				if (scanForSecrets(text).findings.length > 0) findings.push(path.relative(REPO_ROOT, full));
			}
		}
	};
	walk(PI_DIR);
	assert(findings.length === 0, `secret-like content in: ${findings.join(", ")}`);
	return "no PEM blocks or provider tokens in versioned .pi files";
});

check("state root", () => {
	const stateExists = fs.existsSync(statePath(REPO_ROOT, "LOOP_STATE.json"));
	return stateExists
		? `${statePath(REPO_ROOT, "LOOP_STATE.json")} exists (run status.sh for details)`
		: "no LOOP_STATE.json yet — the orchestrator (S4) creates it on BOOT";
});

let failures = 0;
for (const result of checks) {
	const status = result.ok ? "PASS" : "FAIL";
	if (!result.ok) failures += 1;
	process.stdout.write(`${status}  ${result.name}: ${result.detail}\n`);
}
process.stdout.write(`\n${checks.length - failures}/${checks.length} checks passed\n`);
if (failures > 0) process.exit(1);
