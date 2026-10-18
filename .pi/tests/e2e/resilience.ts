#!/usr/bin/env node
/**
 * Resilience matrix (`make -C .pi smoke` is the happy path; this is the kill/restart path).
 *
 * For each kill point the conductor can die at, the scenario:
 *   1. builds a fresh scratch clone with the synthetic task + fixture,
 *   2. drives the loop to the boundary,
 *   3. kills (a real SIGKILL of the child, or the documented effect-without-event
 *      state produced by a SIGKILL at that exact boundary),
 *   4. restarts with the operator commands (`pause` + `resume`) and continues,
 *   5. asserts exactly-once semantics for branches, PRs, merges, jobs, and artifacts.
 *
 * Every scenario prints `KILLPOINT <name>: PASS|FAIL — detail`; exit non-zero if
 * any fails. The matrix is documented in `.pi/tests/e2e/resilience.md`.
 */

import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.dirname(path.dirname(path.dirname(HERE)));
const CLI = path.join(REPO_ROOT, ".pi", "extensions", "orchestrator", "cli.ts");
const CHILD = path.join(REPO_ROOT, ".pi", "tests", "e2e", "e2e_child.mjs");
const FIXTURE = path.join(HERE, "fixture");
const RUN_ROOT = path.join(REPO_ROOT, ".pi", "state", "e2e-tests", `resilience-${new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14)}`);
const keep = process.env.E2E_KEEP === "1";

interface ScenarioResult {
	name: string;
	ok: boolean;
	detail: string;
}
const results: ScenarioResult[] = [];

function record(name: string, ok: boolean, detail: string): void {
	results.push({ name, ok, detail });
	process.stdout.write(`KILLPOINT ${name}: ${ok ? "PASS" : "FAIL"} — ${detail}\n`);
}

function sh(cmd: string, args: string[], cwd = REPO_ROOT, env: NodeJS.ProcessEnv = {}): { status: number; stdout: string; stderr: string } {
	const result = spawnSync(cmd, args, { cwd, encoding: "utf8", timeout: 600_000, env: { ...process.env, ...env } });
	return { status: result.status ?? -1, stdout: String(result.stdout ?? ""), stderr: String(result.stderr ?? "") };
}

function git(root: string, args: string[]): string {
	const result = sh("git", ["-C", root, ...args]);
	if (result.status !== 0) throw new Error(`git ${args.join(" ")} failed: ${result.stderr}`);
	return result.stdout.trim();
}

function overlay(source: string, target: string): void {
	const stat = fs.statSync(source);
	if (stat.isDirectory()) {
		fs.mkdirSync(target, { recursive: true });
		for (const entry of fs.readdirSync(source)) overlay(path.join(source, entry), path.join(target, entry));
		return;
	}
	fs.mkdirSync(path.dirname(target), { recursive: true });
	fs.copyFileSync(source, target);
}

function readJsonl(file: string): Array<Record<string, unknown>> {
	if (!fs.existsSync(file)) return [];
	return fs
		.readFileSync(file, "utf8")
		.split("\n")
		.filter((line) => line.trim() !== "")
		.map((line) => JSON.parse(line) as Record<string, unknown>);
}

function latest(records: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
	const byId = new Map<string, Record<string, unknown>>();
	for (const record of records) if (typeof record.id === "string") byId.set(record.id, record);
	return [...byId.values()];
}

async function freshSandbox(label: string): Promise<string> {
	const root = path.join(RUN_ROOT, label);
	fs.mkdirSync(RUN_ROOT, { recursive: true });
	const clone = sh("git", ["clone", "--quiet", "--local", REPO_ROOT, root]);
	if (clone.status !== 0) throw new Error(`clone failed: ${clone.stderr}`);
	git(root, ["checkout", "--quiet", "update_pymc"]);
	for (const dir of ["src", "tests", "paper", "docs", "fixed_case_studies", "case_studies", "images"]) {
		fs.rmSync(path.join(root, dir), { recursive: true, force: true });
	}
	overlay(path.join(FIXTURE, "synthetic_fixture"), path.join(root, "synthetic_fixture"));
	overlay(path.join(FIXTURE, "experiments"), path.join(root, "experiments"));
	fs.mkdirSync(path.join(root, ".pi", "config"), { recursive: true });
	fs.writeFileSync(
		path.join(root, ".pi", "config", "gates.json"),
		`${JSON.stringify({ testsCommand: ["python3", "experiments/run_fixture_tests.py"] }, null, 2)}\n`,
	);
	fs.mkdirSync(path.join(root, ".pi", "audit", "calibration", "latest"), { recursive: true });
	fs.writeFileSync(
		path.join(root, ".pi", "audit", "calibration", "latest", "recall.json"),
		`${JSON.stringify({ calibrated: true, recall: 1, rejected_confirmed: 0 }, null, 2)}\n`,
	);
	const store = await import(pathToFileURL(path.join(root, ".pi", "extensions", "ledger", "store.ts")).href);
	(store as { appendRecord: (...args: unknown[]) => Record<string, unknown> }).appendRecord(
		root,
		"task",
		{
			id: "T-001",
			title: "add a docstring and a passing test to a fixture module",
			origin: "PLAN:S7-smoke",
			priority: "P2",
			objective: "Add a docstring to synthetic_fixture.calculator.add and a passing unittest for it.",
			acceptance_criteria: ["the docstring exists", "the unittest passes"],
			deliverables: ["synthetic_fixture/calculator.py", "synthetic_fixture/tests/test_calculator.py"],
			compute: "local",
			status: "open",
			effort: "small",
		},
		{ root, role: "conductor", actor: "e2e-resilience" },
	);
	fs.mkdirSync(path.join(root, ".pi", "state", "tasks"), { recursive: true });
	fs.writeFileSync(path.join(root, ".pi", "state", "tasks", "T-001.md"), "# T-001 — fixture docstring + test\n");
	git(root, ["add", "-A"]);
	git(root, ["-c", "user.name=e2e", "-c", "user.email=e2e@example.invalid", "commit", "-qm", "e2e setup"]);
	return root;
}

function cli(root: string, args: string[], env: NodeJS.ProcessEnv = {}): { status: number; stdout: string; stderr: string } {
	return sh("node", [CLI, ...args, "--sandbox", root, "--headless", "--command", CHILD], REPO_ROOT, env);
}

function step(root: string, env: NodeJS.ProcessEnv = {}): { status: number; stdout: string; stderr: string } {
	return cli(root, ["step"], env);
}

/** The documented operator restart: pause, then resume to the recorded state. */
function restart(root: string): string {
	const paused = cli(root, ["pause", "resilience matrix restart"]);
	const resumed = cli(root, ["resume"]);
	if (resumed.status !== 0) throw new Error(`resume failed: ${resumed.stderr || resumed.stdout}`);
	return `${paused.stdout.trim()} / ${resumed.stdout.trim()}`;
}

function boot(root: string): void {
	const first = step(root);
	if (first.status !== 0) throw new Error(`BOOT failed: ${first.stderr || first.stdout}`);
	const second = step(root);
	if (second.status !== 0) throw new Error(`SELECT_TASK failed: ${second.stderr || second.stdout}`);
}

function driveToCollected(root: string): void {
	boot(root);
	step(root); // work:T-001:1 queues the experiment
	step(root); // hpc-disabled / local-fallback bookkeep
	const local = step(root, { PI_E2E_MODE: "local" });
	if (local.status !== 0) throw new Error(`local run failed: ${local.stderr || local.stdout}`);
}

function startCount(root: string, prefix: string): number {
	return readJsonl(path.join(root, ".pi", "state", "EVENTS.jsonl")).filter(
		(event) => event.action_phase === "start" && String(event.action_key ?? "").startsWith(prefix),
	).length;
}

async function scenarioBeforeChildSpawn(): Promise<void> {
	const root = await freshSandbox("before-child-spawn");
	boot(root);
	// A crash after the lock was taken but before the child process existed.
	fs.mkdirSync(path.join(root, ".pi", "state", "locks"), { recursive: true });
	fs.writeFileSync(
		path.join(root, ".pi", "state", "locks", "child.json"),
		`${JSON.stringify({ pid: 4194303, action_key: "work:T-001:1", agent: "phd_candidate", argv: [], started_at: new Date(0).toISOString() }, null, 2)}\n`,
	);
	const killed = step(root);
	const restarted = restart(root);
	const resumed = step(root);
	const events = readJsonl(path.join(root, ".pi", "state", "EVENTS.jsonl"));
	const stale = events.filter((event) => event.action_phase === "stale");
	const experiments = latest(readJsonl(path.join(root, ".pi", "state", "EXPERIMENTS.jsonl")));
	const ok =
		stale.length === 1 &&
		startCount(root, "work:T-001:1") === 1 &&
		startCount(root, "work:T-001:") === 1 &&
		experiments.length === 1 &&
		/not PAUSED|resumed/.test(restarted);
	record(
		"before-child-spawn",
		ok,
		`stale=${stale.length} starts(work:T-001:1)=${startCount(root, "work:T-001:1")} experiments=${experiments.length}; step=${killed.stdout.trim()} resumed=${resumed.stdout.trim()}`,
	);
}

async function scenarioDuringChildBeforeEffect(): Promise<void> {
	const root = await freshSandbox("during-child-before-effect");
	boot(root);
	// Real SIGKILL: the child dies before touching the ledger.
	const killed = step(root, { PI_E2E_MODE: "work-kill-early" });
	const restarted = restart(root);
	const recovered = step(root);
	const events = readJsonl(path.join(root, ".pi", "state", "EVENTS.jsonl"));
	const failed = events.filter((event) => event.action_phase === "complete" && event.exit_state === "failed");
	const experiments = latest(readJsonl(path.join(root, ".pi", "state", "EXPERIMENTS.jsonl")));
	const ok =
		startCount(root, "work:T-001:1") === 1 &&
		startCount(root, "work:T-001:2") === 1 &&
		failed.length === 1 &&
		experiments.length === 1;
	record(
		"during-child-before-effect",
		ok,
		`failed=${failed.length} starts(work:1)=${startCount(root, "work:T-001:1")} starts(work:2)=${startCount(root, "work:T-001:2")} experiments=${experiments.length}; resumed=${restarted}`,
	);
}

async function scenarioAfterChild(): Promise<void> {
	const root = await freshSandbox("after-child");
	driveToCollected(root);
	const opened = step(root); // work:T-001:3 opens the PR
	const replay = step(root); // the completed spawn key must be a no-op
	const prs = latest(readJsonl(path.join(root, ".pi", "state", "PRS.jsonl")));
	const state = JSON.parse(fs.readFileSync(path.join(root, ".pi", "state", "LOOP_STATE.json"), "utf8")) as Record<string, unknown>;
	const ok = startCount(root, "work:T-001:3") === 1 && prs.length === 1 && state.state === "REVIEW";
	record(
		"after-child",
		ok,
		`starts(work:T-001:3)=${startCount(root, "work:T-001:3")} PRs=${prs.length}; replay=${replay.stdout.trim()}`,
	);
}

async function scenarioBeforeLedgerAppend(): Promise<void> {
	const root = await freshSandbox("before-ledger-append");
	driveToCollected(root);
	// Real SIGKILL after the branch + commit exist but before the PR record.
	const killed = step(root, { PI_E2E_KILL: "before-record" });
	const restarted = restart(root);
	const recovered = step(root);
	const branches = git(root, ["branch", "--list", "work/*"]).split("\n").filter(Boolean);
	const prs = latest(readJsonl(path.join(root, ".pi", "state", "PRS.jsonl")));
	const ok =
		branches.length === 1 &&
		prs.length === 1 &&
		startCount(root, "work:T-001:3") === 1 &&
		startCount(root, "work:T-001:4") === 1 &&
		/killed|failed/.test(killed.stderr + killed.stdout);
	record(
		"before-ledger-append",
		ok,
		`branches=${branches.length} PRs=${prs.length} starts(work:3)=${startCount(root, "work:T-001:3")} starts(work:4)=${startCount(root, "work:T-001:4")}; resumed=${restarted}`,
	);
}

async function scenarioAfterLedgerAppend(): Promise<void> {
	const root = await freshSandbox("after-ledger-append");
	driveToCollected(root);
	// Real SIGKILL after the PR record exists but before the completion event.
	const killed = step(root, { PI_E2E_MODE: "work-then-kill" });
	const restarted = restart(root);
	const recovered = step(root);
	const prs = latest(readJsonl(path.join(root, ".pi", "state", "PRS.jsonl")));
	const state = JSON.parse(fs.readFileSync(path.join(root, ".pi", "state", "LOOP_STATE.json"), "utf8")) as Record<string, unknown>;
	const ok =
		prs.length === 1 &&
		startCount(root, "work:T-001:3") === 1 &&
		state.state === "REVIEW" &&
		/REVIEW/.test(recovered.stdout) &&
		/killed|failed/.test(killed.stderr + killed.stdout);
	record(
		"after-ledger-append",
		ok,
		`PRs=${prs.length} starts(work:3)=${startCount(root, "work:T-001:3")} state=${String(state.state)}; resumed=${restarted}`,
	);
}

async function scenarioDuringMerge(): Promise<void> {
	const root = await freshSandbox("during-merge");
	driveToCollected(root);
	step(root); // open the PR (work:T-001:3)
	step(root, { PI_E2E_MODE: "review" }); // REVIEW: supervisor accepts
	step(root); // REVIEW → MERGE
	step(root); // open the merge human gate
	step(root); // wait on the gate (no-op while pending)
	const gates = latest(readJsonl(path.join(root, ".pi", "state", "GATES.jsonl"))).filter((gate) => gate.status === "pending");
	if (gates.length !== 1) throw new Error(`expected one pending merge gate, found ${gates.length}`);
	const approved = cli(root, ["gate", "approve", String(gates[0].id), "resilience matrix approval"]);
	if (approved.status !== 0) throw new Error(`gate approve failed: ${approved.stderr || approved.stdout}`);
	// Simulate the crash between `git merge --no-ff` and the completion event: land
	// the merge by hand with the conductor's message, then let it recover.
	const pr = latest(readJsonl(path.join(root, ".pi", "state", "PRS.jsonl")))[0];
	const review = latest(readJsonl(path.join(root, ".pi", "state", "REVIEWS.jsonl")))[0];
	const baseBefore = git(root, ["rev-parse", "update_pymc"]);
	git(root, ["merge", "--no-ff", String(pr.branch), "-m", `merge ${String(pr.id)} (review ${String(review.id)}): ${String(pr.summary)}`]);
	const restarted = restart(root);
	const recovered = step(root);
	const log = git(root, ["log", "--format=%s", `${baseBefore}..update_pymc`]).split("\n");
	const events = readJsonl(path.join(root, ".pi", "state", "EVENTS.jsonl"));
	const mergeCompletions = events.filter((event) => event.action_kind === "merge" && event.action_phase === "complete");
	const ok =
		log.filter((line) => line.startsWith(`merge ${String(pr.id)}`)).length === 1 &&
		mergeCompletions.length === 1 &&
		mergeCompletions[0].already_merged === true &&
		/already up to date|merged/.test(recovered.stdout) &&
		/not PAUSED|resumed/.test(restarted);
	record(
		"during-merge",
		ok,
		`merge commits=${log.filter((line) => line.startsWith(`merge ${String(pr.id)}`)).length} completions=${mergeCompletions.length} already_merged=${String(mergeCompletions[0]?.already_merged)}; resumed=${restarted}`,
	);
}

async function scenarioDuringCollection(): Promise<void> {
	const root = await freshSandbox("during-collection");
	boot(root);
	step(root); // work:T-001:1 queues the experiment
	step(root); // hpc-disabled / local-fallback
	const marker = path.join(RUN_ROOT, "collection-kill.marker");
	// Real SIGKILL inside the experiment after the first checkpoint unit.
	const killed = step(root, { PI_E2E_MODE: "local", TINY_RUN_KILL_AFTER: "1", TINY_RUN_KILL_MARKER: marker });
	const restarted = restart(root);
	const recovered = step(root, { PI_E2E_MODE: "local" });
	const experiments = latest(readJsonl(path.join(root, ".pi", "state", "EXPERIMENTS.jsonl")));
	const progress = experiments[0]?.progressHistory as Array<{ attempt: number; completed: number }> | undefined;
	const checkpoint = path.join(root, ".pi", "artifacts", "E-0001", "checkpoints", "progress.json");
	const checkpointState = fs.existsSync(checkpoint) ? (JSON.parse(fs.readFileSync(checkpoint, "utf8")) as Record<string, unknown>) : null;
	const ok =
		experiments.length === 1 &&
		experiments[0].status === "collected" &&
		progress?.length === 1 &&
		progress[0].completed === 3 &&
		fs.existsSync(marker) &&
		checkpointState?.completed !== undefined &&
		(checkpointState.completed as unknown[]).length === 3 &&
		/killed|failed/.test(killed.stderr + killed.stdout);
	record(
		"during-collection",
		ok,
		`experiments=${experiments.length} status=${String(experiments[0]?.status)} progress=${JSON.stringify(progress)} checkpoint=${JSON.stringify(checkpointState?.completed)}; resumed=${restarted}`,
	);
}

async function main(): Promise<number> {
	const startedAt = Date.now();
	await scenarioBeforeChildSpawn();
	await scenarioDuringChildBeforeEffect();
	await scenarioAfterChild();
	await scenarioBeforeLedgerAppend();
	await scenarioAfterLedgerAppend();
	await scenarioDuringMerge();
	await scenarioDuringCollection();
	const failed = results.filter((entry) => !entry.ok);
	process.stdout.write(
		`\nRESILIENCE ${failed.length === 0 ? "PASS" : "FAIL"}: ${results.length - failed.length}/${results.length} kill point(s) recovered with exactly-once semantics in ${((Date.now() - startedAt) / 1000).toFixed(1)}s\n`,
	);
	if (!keep) fs.rmSync(RUN_ROOT, { recursive: true, force: true });
	else process.stdout.write(`sandboxes kept: ${RUN_ROOT}\n`);
	return failed.length === 0 ? 0 : 1;
}

main()
	.then((code) => {
		process.exitCode = code;
	})
	.catch((error) => {
		process.stderr.write(`${(error as Error).stack ?? String(error)}\n`);
		process.exitCode = 1;
	});
