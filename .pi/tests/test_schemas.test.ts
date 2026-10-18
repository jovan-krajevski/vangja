/**
 * Config schemas: every `.pi/config/*.json` validates, the budget defaults match
 * plan/02_ARCHITECTURE.md §6 exactly, and the loaders fail loudly on drift.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import {
	DEFAULT_BUDGETS,
	loadAgentsManifest,
	loadBudgetConfig,
	loadHpcConfig,
	loadHpcExample,
	loadModelPin,
	loadProjectConfig,
	parseBudgetConfig,
	parseModelPin,
	resolveBaseBranch,
} from "../lib/config.ts";
import { PI_DIR, readRepoConfig, REPO_ROOT, scratchRoot, writeJson } from "./helpers.ts";

const EXPECTED_AGENTS = [
	"phd_candidate",
	"collector",
	"supervisor",
	"audit_finder",
	"audit_verifier",
	"audit_synthesizer",
	"audit_adjudicator",
	"scout",
	"planner",
	"methodologist",
	"experimenter",
	"repro_engineer",
	"literature_scout",
	"paper_writer",
	"critic",
	"meta_review",
];

test("model.json loads and matches the pin", () => {
	const pin = loadModelPin(REPO_ROOT);
	assert.equal(pin.provider, "opencode-go");
	assert.equal(pin.model, "deepseek-v4.1-flash");
	assert.equal(pin.thinking, "max");
	assert.equal(pin.contextWindow, 1_000_000);
	assert.equal(pin.enforce, true);
});

test("model.json rejects a smaller context window while enforce=true", () => {
	const file = "/tmp/model.json";
	assert.throws(
		() => parseModelPin({ provider: "p", model: "m", thinking: "max", contextWindow: 200_000, enforce: true }, file),
		/contextWindow.*1000000/,
	);
});

test("model.json rejects a missing field naming the file", () => {
	assert.throws(
		() => parseModelPin({ provider: "p", model: "m", thinking: "max", contextWindow: 1_000_000 }, "/tmp/model.json"),
		/invalid config \/tmp\/model\.json.*enforce/,
	);
});

test("budgets.json equals the documented defaults exactly", () => {
	const budgets = loadBudgetConfig(REPO_ROOT);
	assert.deepEqual(budgets, DEFAULT_BUDGETS);
	const onDisk = readRepoConfig("budgets.json");
	assert.deepEqual(onDisk, DEFAULT_BUDGETS);
});

test("budgets.json rejects an incomplete file", () => {
	assert.throws(() => parseBudgetConfig({ maxSessionsPerDay: 10 }, "/tmp/budgets.json"), /maxParallelWorkers/);
});

test("project.json carries the base branch, venue, and gates", () => {
	const project = loadProjectConfig(REPO_ROOT);
	assert.equal(project.schema, "pi-project/1");
	assert.equal(project.repoRoot, ".");
	assert.equal(project.baseBranch, "@current");
	assert.equal(project.venue, "Journal of Forecasting");
	assert.equal(project.requireHumanMerge, true);
	assert.equal(project.dryRun, false);
	assert.equal(project.stateDir, ".pi/state");
	assert.equal(project.sessionDir, ".pi/state/sessions");
	assert.equal(project.artifactDir, ".pi/artifacts");
	assert.equal(project.worktreeDir, ".pi/state/worktrees");
	assert.equal(project.sandboxPath, ".pi/state/sandbox");
});

test("resolveBaseBranch follows @current and refuses a detached HEAD", () => {
	assert.equal(resolveBaseBranch("@current", "vangja_loop_20261018"), "vangja_loop_20261018");
	assert.equal(resolveBaseBranch("update_pymc", "somewhere-else"), "update_pymc");
	assert.throws(() => resolveBaseBranch("@current", null), /detached HEAD/);
	assert.throws(() => resolveBaseBranch("@current", "HEAD"), /detached HEAD/);
});

test("agents.json has the full roster from plan/03_AGENTS.md", () => {
	const manifest = loadAgentsManifest(REPO_ROOT);
	assert.equal(manifest.schema, "pi-agents/1");
	assert.deepEqual(
		manifest.agents.map((agent) => agent.name).sort(),
		[...EXPECTED_AGENTS].sort(),
	);
	for (const agent of manifest.agents) {
		assert.ok(agent.file.startsWith(".pi/agents/"), `${agent.name}.file must be .pi/agents/<name>.md`);
		assert.ok(agent.file.endsWith(".md"), `${agent.name}.file must be a Markdown file`);
		assert.ok(agent.tools.length > 0, `${agent.name} must declare tools`);
		assert.ok(agent.skills.length > 0, `${agent.name} must declare skills`);
		assert.ok(agent.maxParallel >= 1, `${agent.name}.maxParallel`);
	}
	const files = manifest.agents.map((agent) => agent.file);
	assert.equal(new Set(files).size, files.length, "agent files must be unique");
	const roles = manifest.agents.map((agent) => agent.role);
	assert.equal(new Set(roles).size, roles.length, "agent roles must be unique");
});

test("hpc.example.json has the disabled-by-default schema", () => {
	const example = loadHpcExample(REPO_ROOT);
	assert.equal(example.host, "");
	assert.equal(example.port, 22);
	assert.equal(example.forwardGitAgent, true);
	assert.equal(example.wallClockHours, 24);
	assert.equal(example.maxParallelJobs, 1);
	assert.equal(example.maxAttemptsPerExperiment, 6);
	assert.equal(example.maxPolls, 40);
});

test("loadHpcConfig reports disabled when hpc.json is absent", () => {
	const root = scratchRoot();
	const loaded = loadHpcConfig(root);
	assert.equal(loaded.enabled, false);
	assert.match(loaded.reason, /HPC path is disabled/);
});

test("loadHpcConfig rejects maxParallelJobs > 1 without a recorded amendment", () => {
	const example = readRepoConfig<Record<string, unknown>>("hpc.example.json");
	const root = scratchRoot();
	writeJson(path.join(root, ".pi", "config", "hpc.json"), { ...example, host: "cluster.example", maxParallelJobs: 2 });
	assert.throws(() => loadHpcConfig(root), /maxParallelJobs=2/);
	// With a recorded amendment covering the change the config loads.
	fs.appendFileSync(
		path.join(root, ".pi", "state", "AMENDMENTS.jsonl"),
		`${JSON.stringify({
			schema: "pi-ledger/1",
			kind: "amendment",
			id: "A-001",
			seq: 1,
			ts: new Date().toISOString(),
			actor: "supervisor",
			path: ".pi/config/hpc.json",
			change: "raise maxParallelJobs to 2 for a partition with fast preemption",
			justification: "measured queue latency",
		})}\n`,
	);
	assert.equal(loadHpcConfig(root).enabled, true);
});

test("no committed config file contains a real credential", () => {
	const configDir = path.join(PI_DIR, "config");
	const files = fs.readdirSync(configDir).filter((name) => name.endsWith(".json"));
	assert.ok(files.includes("hpc.example.json"), "hpc.example.json must exist");
	for (const file of files) {
		const text = fs.readFileSync(path.join(configDir, file), "utf8");
		assert.ok(!/BEGIN [A-Z ]*PRIVATE KEY/.test(text), `${file} contains a PEM key`);
		assert.ok(!/("|')(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,})/.test(text), `${file} contains a token`);
	}
});
