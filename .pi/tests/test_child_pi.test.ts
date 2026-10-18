/**
 * Child-pi guard test (offline): a real `pi` process launched with a drifted
 * model must be blocked before any model turn. No network and no tokens are
 * used because the guard drops the input and aborts the run.
 */

import assert from "node:assert/strict";
import * as path from "node:path";
import { test } from "node:test";
import { PI_DIR, REPO_ROOT, spawnSyncNode } from "./helpers.ts";

const GUARD_EXTENSION = path.join(PI_DIR, "extensions", "guards", "index.ts");

function runPi(args: string[], env: NodeJS.ProcessEnv = {}) {
	return spawnSyncNode("pi", args, {
		cwd: REPO_ROOT,
		env: { PI_LOOP_ROOT: REPO_ROOT, ...env },
		timeoutMs: 120_000,
	});
}

test("a child pi with the wrong model is blocked and names model.json", () => {
	const result = runPi([
		"--print",
		"--no-session",
		"-na",
		"-e",
		GUARD_EXTENSION,
		"--provider",
		"opencode-go",
		"--model",
		"opencode-go/gpt-5.6-luna",
		"--thinking",
		"max",
		"Reply with exactly: SHOULD-NOT-RUN",
	]);
	assert.notEqual(result.status, 0, `expected a non-zero exit, got ${result.status}\nstderr:\n${result.stderr}`);
	assert.equal(result.stdout.trim(), "", "no model output may be produced");
	assert.match(result.stderr, /\.pi\/config\/model\.json/, `stderr must name the pin file:\n${result.stderr}`);
	assert.match(result.stderr, /gpt-5\.6-luna/, `stderr must name the drifting model:\n${result.stderr}`);
	assert.ok(!result.stdout.includes("SHOULD-NOT-RUN"));
});

test("a child pi with the wrong thinking level is blocked", () => {
	const result = runPi([
		"--print",
		"--no-session",
		"-na",
		"-e",
		GUARD_EXTENSION,
		"--provider",
		"opencode-go",
		"--model",
		"opencode-go/deepseek-v4.1-flash",
		"--thinking",
		"low",
		"Reply with exactly: SHOULD-NOT-RUN",
	]);
	assert.notEqual(result.status, 0);
	assert.equal(result.stdout.trim(), "");
	assert.match(result.stderr, /thinking level is "low"/);
	assert.match(result.stderr, /\.pi\/config\/model\.json/);
});
