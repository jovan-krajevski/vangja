/**
 * CLI surface: `status`, `peek`, `verify-env`, and `.pi/loop/*.sh` parity.
 *
 * The shims must contain no logic: running `bash .pi/loop/step.sh --sandbox X`
 * and `node .pi/extensions/orchestrator/cli.ts step --sandbox X` on identical
 * starting states must produce identical events (modulo timestamps and paths).
 */

import { strict as assert } from "node:assert";
import { test } from "node:test";
import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { CLI, STUB_CHILD, createSandbox, fileHashes, readEvents, readState, runCli, seedTask, writeGatesConfig } from "./orchestrator_helpers.ts";
import { REPO_ROOT } from "../helpers.ts";

const LOOP_DIR = path.join(REPO_ROOT, ".pi", "loop");

function stable(event: Record<string, unknown>, root: string): Record<string, unknown> {
	// Events are identical when the action semantics match; timestamps, pids,
	// durations, absolute paths, and run ids legitimately differ between runs.
	return {
		action_key: event.action_key ?? null,
		action_phase: event.action_phase ?? null,
		action_kind: event.action_kind ?? null,
		action_state: event.action_state ?? null,
		from: event.from ?? null,
		to: event.to ?? null,
		agent: event.agent ?? null,
		attempt: event.attempt ?? null,
		exit_state: event.exit_state ?? null,
		gate: event.gate ?? null,
		gate_ok: event.gate_ok ?? null,
		effects: event.effects ?? null,
		summary: String(event.summary ?? "")
			.split(root)
			.join("<root>")
			.replace(/\b[0-9a-f]{12}\b/g, "<sha>")
			.replace(/run-\d{14}-[0-9a-f]{8}/g, "<run>"),
	};
}

test("status reports every field the runbook asks for, and changes nothing", () => {
	const sandbox = createSandbox("status");
	try {
		seedTask(sandbox.root);
		writeGatesConfig(sandbox.root);
		runCli(sandbox.root, ["step"]);
		runCli(sandbox.root, ["step"]);
		const stateBefore = fileHashes(path.join(sandbox.root, ".pi", "state"));
		const status = runCli(sandbox.root, ["status"]);
		assert.equal(status.status, 0, status.stderr);
		for (const needle of ["state:", "run:", "task:", "pr:", "experiment:", "budget:", "open human gates:", "last 5 events:", "nextAction:"]) {
			assert.ok(status.stdout.includes(needle), `status output is missing "${needle}":\n${status.stdout}`);
		}
		assert.match(status.stdout, /nextAction: spawn/);

		const json = runCli(sandbox.root, ["status", "--json"]);
		const report = JSON.parse(json.stdout) as Record<string, unknown>;
		assert.equal(report.state, readState(sandbox.root).state);
		for (const key of [
			"state",
			"task_id",
			"pr_id",
			"experiment_id",
			"attempt",
			"budget",
			"caps",
			"open_gates",
			"last_events",
			"next_action",
			"child",
		]) {
			assert.ok(key in report, `status --json is missing "${key}"`);
		}
		assert.ok((report.next_action as Record<string, unknown>).agent);
		// A read-only command must not rewrite state.
		assert.deepEqual(fileHashes(path.join(sandbox.root, ".pi", "state")), stateBefore);
	} finally {
		sandbox.cleanup();
	}
});

test("peek is read-only and names the action a step would run", () => {
	const sandbox = createSandbox("peek");
	try {
		const before = fileHashes(path.join(sandbox.root, ".pi", "state"));
		const peek = runCli(sandbox.root, ["peek"]);
		assert.equal(peek.status, 0, peek.stderr);
		assert.match(peek.stdout, /^boot state=UNINITIALIZED key=boot/);
		assert.deepEqual(fileHashes(path.join(sandbox.root, ".pi", "state")), before);
	} finally {
		sandbox.cleanup();
	}
});

test("step.sh and the CLI produce identical events for the same starting state", () => {
	const a = createSandbox("parity-cli");
	const b = createSandbox("parity-shim");
	try {
		for (const sandbox of [a, b]) {
			seedTask(sandbox.root);
			writeGatesConfig(sandbox.root);
		}
		const runBoth = (args: string[], env: NodeJS.ProcessEnv = {}) => {
			const cli = spawnSync("node", [CLI, ...args, "--sandbox", a.root, "--headless", "--command", STUB_CHILD], {
				cwd: REPO_ROOT,
				encoding: "utf8",
				env: { ...process.env, ...env },
			});
			const shim = spawnSync("bash", [path.join(LOOP_DIR, "step.sh"), ...args, "--sandbox", b.root, "--command", STUB_CHILD], {
				cwd: REPO_ROOT,
				encoding: "utf8",
				env: { ...process.env, ...env },
			});
			return { cli, shim };
		};
		for (const [args, env] of [
			[["step"], {}],
			[["step"], {}],
			[["step"], { PI_STUB_MODE: "work" }],
		] as Array<[string[], NodeJS.ProcessEnv]>) {
			const { cli, shim } = runBoth(args, env);
			assert.equal(cli.status, 0, cli.stderr);
			assert.equal(shim.status, 0, shim.stderr);
			const viaCli = readEvents(a.root).map((event) => stable(event, a.root));
			const viaShim = readEvents(b.root).map((event) => stable(event, b.root));
			assert.equal(viaCli.length, viaShim.length);
			assert.deepEqual(viaCli, viaShim);
		}
		assert.equal(readState(a.root).task_id, readState(b.root).task_id);
		assert.match(readState(a.root).state, /WORK|REVIEW/);
	} finally {
		a.cleanup();
		b.cleanup();
	}
});

test("status.sh is a thin wrapper that prints the same summary as the CLI", () => {
	const sandbox = createSandbox("status-shim");
	try {
		const viaCli = runCli(sandbox.root, ["status"]);
		const viaShim = spawnSync("bash", [path.join(LOOP_DIR, "status.sh"), "--sandbox", sandbox.root], {
			cwd: REPO_ROOT,
			encoding: "utf8",
		});
		assert.equal(viaShim.status, 0, viaShim.stderr);
		const strip = (text: string) => text.split("\n").filter((line) => !line.startsWith("  - ")).join("\n");
		assert.equal(strip(viaShim.stdout).trim(), strip(viaCli.stdout).trim());
	} finally {
		sandbox.cleanup();
	}
});

test("the loop scripts refuse to run against the real repository via --sandbox", () => {
	const result = spawnSync("bash", [path.join(LOOP_DIR, "step.sh"), "--sandbox", REPO_ROOT], { cwd: REPO_ROOT, encoding: "utf8" });
	assert.equal(result.status, 1);
	assert.match(result.stderr, /resolves to the real repository root/);
});

test("an unknown CLI command exits 2 with the usage text", () => {
	const result = spawnSync("node", [CLI, "frobnicate"], { cwd: REPO_ROOT, encoding: "utf8" });
	assert.equal(result.status, 2);
	assert.match(result.stderr, /unknown command "frobnicate"/);
});

test("a missing state directory is not silently created by status", () => {
	const sandbox = createSandbox("status-missing");
	try {
		fs.rmSync(path.join(sandbox.root, ".pi", "state"), { recursive: true, force: true });
		const result = runCli(sandbox.root, ["status", "--json"]);
		assert.equal(result.status, 0, result.stderr);
		const report = JSON.parse(result.stdout) as Record<string, unknown>;
		assert.equal(report.state, "UNINITIALIZED");
	} finally {
		sandbox.cleanup();
	}
});
