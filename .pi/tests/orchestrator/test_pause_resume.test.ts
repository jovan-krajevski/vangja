/**
 * Pause/resume and idempotency.
 *
 * 1. The replay matrix: for every transition boundary in a cycle, execute the
 *    action, then replay it; the replay must be a no-op (no event, no ledger
 *    effect, no second child/PR/merge).
 * 2. A real kill: the conductor is SIGKILLed while a child runs; the stale pid
 *    lock is reported, and the resumed run produces exactly one PR and one
 *    completion.
 * 3. Pause before the merge and resume: exactly one merge commit.
 */

import { strict as assert } from "node:assert";
import { test } from "node:test";
import { spawn, spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { executeAction, type ExecutionResult } from "../../extensions/orchestrator/execute.ts";
import { commandPause, commandResume, loadLoopOptions, peekAction } from "../../extensions/orchestrator/commands.ts";
import type { RunContext } from "../../extensions/orchestrator/sandbox.ts";
import { REPO_ROOT } from "../helpers.ts";
import type { Action } from "../../extensions/orchestrator/state-machine.ts";
import {
	CLI,
	STUB_CHILD,
	createSandbox,
	git,
	latestById,
	readEvents,
	readLedger,
	readState,
	seedTask,
	writeGatesConfig,
} from "./orchestrator_helpers.ts";

function contextFor(root: string): RunContext {
	return { root, realRoot: REPO_ROOT, sandbox: root, dryRun: false, headless: true, command: STUB_CHILD };
}

async function execute(context: RunContext, env: NodeJS.ProcessEnv, action: Action): Promise<ExecutionResult> {
	const previous = { ...process.env };
	Object.assign(process.env, env);
	try {
		return await executeAction({ context, action, options: loadLoopOptions(context) });
	} finally {
		for (const key of Object.keys(env)) {
			if (previous[key] === undefined) delete process.env[key];
			else process.env[key] = previous[key];
		}
	}
}

async function replay(context: RunContext, action: Action): Promise<ExecutionResult> {
	return executeAction({ context, action, options: loadLoopOptions(context) });
}

test("replaying a completed action is a no-op at every transition boundary", async () => {
	const sandbox = createSandbox("replay");
	try {
		seedTask(sandbox.root);
		writeGatesConfig(sandbox.root);
		// No human gate: the replay matrix should exercise the merge directly.
		const project = JSON.parse(fs.readFileSync(path.join(sandbox.root, ".pi", "config", "project.json"), "utf8"));
		project.requireHumanMerge = false;
		fs.writeFileSync(path.join(sandbox.root, ".pi", "config", "project.json"), `${JSON.stringify(project, null, 2)}\n`);
		// BOOT requires a clean checkout, so the fixture's config change is committed
		// like any other change would be.
		git(sandbox.root, ["add", ".pi/config/project.json", ".pi/config/gates.json"]);
		git(sandbox.root, ["-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-m", "test: sandbox config"]);
		const context = contextFor(sandbox.root);
		const boundaries: Array<{ label: string; env?: NodeJS.ProcessEnv }> = [
			{ label: "BOOT" },
			{ label: "SELECT_TASK → WORK" },
			{ label: "WORK: open the PR", env: { PI_STUB_MODE: "work" } },
			{ label: "WORK → REVIEW" },
			{ label: "REVIEW: accept", env: { PI_STUB_MODE: "review" } },
			{ label: "REVIEW → MERGE" },
			{ label: "MERGE: gate + merge" },
			{ label: "POST_MERGE → SELECT_TASK" },
		];
		const seen: string[] = [];
		for (const boundary of boundaries) {
			const action = peekAction(context);
			const executed = await execute(context, boundary.env ?? {}, action);
			assert.equal(executed.performed, true, `${boundary.label}: ${executed.message}`);
			seen.push(`${boundary.label}: ${action.kind} ${action.key}`);
			const eventsBefore = readEvents(sandbox.root).length;
			const prsBefore = readLedger(sandbox.root, "PRS.jsonl").length;
			const replayResult = await replay(context, action);
			assert.equal(replayResult.skipped, "completed", `${boundary.label}: replay should be a no-op`);
			assert.equal(readEvents(sandbox.root).length, eventsBefore, `${boundary.label}: replay appended an event`);
			assert.equal(readLedger(sandbox.root, "PRS.jsonl").length, prsBefore, `${boundary.label}: replay wrote a PR`);
			// A new action is now available because the state moved.
			const next = peekAction(context);
			assert.notEqual(next.key, action.key, `${boundary.label}: state did not advance`);
		}
		// Exactly one PR, one review, one merge commit, one tag.
		assert.equal(latestById(readLedger(sandbox.root, "PRS.jsonl")).length, 1);
		assert.equal(latestById(readLedger(sandbox.root, "REVIEWS.jsonl")).length, 1);
		assert.equal(git(sandbox.root, ["tag", "--list", "run/*"]).split("\n").filter(Boolean).length, 1);
		assert.equal(git(sandbox.root, ["log", "--format=%s", "-1", "update_pymc"]).startsWith("merge PR-001"), true);
		const merges = readEvents(sandbox.root).filter(
			(event) => event.action_phase === "complete" && String(event.action_key ?? "").startsWith("merge:"),
		);
		assert.equal(merges.length, 1);
	} finally {
		sandbox.cleanup();
	}
});

test("killing the conductor mid-child leaves a stale lock, and the resumed run has one PR and one completion", async () => {
	const sandbox = createSandbox("kill");
	try {
		seedTask(sandbox.root);
		writeGatesConfig(sandbox.root);
		// Walk to WORK (the state whose action is the child spawn).
		for (const env of [{}, {}]) {
			const step = spawnSync("node", [CLI, "step", "--sandbox", sandbox.root, "--headless", "--command", STUB_CHILD], {
				cwd: REPO_ROOT,
				encoding: "utf8",
				env: { ...process.env, ...env },
			});
			assert.equal(step.status, 0, step.stderr);
		}
		assert.equal(readState(sandbox.root).state, "WORK");

		// Start a slow work unit, then SIGKILL the conductor and the child.
		const conductor = spawn("node", [CLI, "step", "--sandbox", sandbox.root, "--headless", "--command", STUB_CHILD], {
			cwd: REPO_ROOT,
			env: { ...process.env, PI_STUB_MODE: "slow", PI_STUB_SLEEP_MS: "15000" },
			stdio: ["ignore", "pipe", "pipe"],
		});
		const lockFile = path.join(sandbox.root, ".pi", "state", "locks", "child.json");
		const deadline = Date.now() + 15_000;
		const conductorPid = conductor.pid;
		let lock: { pid: number; action_key: string } | null = null;
		while (Date.now() < deadline) {
			if (fs.existsSync(lockFile)) {
				const candidate = JSON.parse(fs.readFileSync(lockFile, "utf8")) as { pid: number; action_key: string };
				// The provisional lock carries the conductor pid; wait for onSpawn to
				// replace it with the child pid so the kill below also kills the child.
				if (candidate.pid !== conductorPid) {
					lock = candidate;
					break;
				}
			}
			await new Promise((resolve) => setTimeout(resolve, 25));
		}
		assert.ok(lock, "child lock with the child pid was never written");
		// The start event is appended before the lock: a kill can no longer leave a
		// lock with no start event (the V2 race).
		const startsBeforeKill = readEvents(sandbox.root).filter(
			(event) => event.action_phase === "start" && String(event.action_key ?? "").startsWith("work:"),
		);
		assert.equal(startsBeforeKill.length, 1, "the start event must precede the child lock");
		conductor.kill("SIGKILL");
		try {
			process.kill(lock.pid, "SIGKILL");
		} catch {
			// The child already exited; the stale-pid path is what we want to test.
		}
		await new Promise((resolve) => conductor.on("close", resolve));
		const startEvents = readEvents(sandbox.root).filter(
			(event) => event.action_phase === "start" && String(event.action_key ?? "").startsWith("work:"),
		);
		const completeBefore = readEvents(sandbox.root).filter(
			(event) => event.action_phase === "complete" && String(event.action_key ?? "").startsWith("work:"),
		);
		assert.equal(startEvents.length, 1);
		assert.equal(completeBefore.length, 0, "the killed child must not have a completion event");

		// Resume with the normal stub: the dead lock is reported, one PR appears.
		const resume = spawnSync("node", [CLI, "step", "--sandbox", sandbox.root, "--headless", "--command", STUB_CHILD], {
			cwd: REPO_ROOT,
			encoding: "utf8",
			env: { ...process.env, PI_STUB_MODE: "work" },
		});
		assert.equal(resume.status, 0, resume.stderr);
		const stale = readEvents(sandbox.root).filter((event) => event.action_phase === "stale");
		assert.equal(stale.length, 1, "the stale child lock must be reported");
		assert.match(String(stale[0].summary), /stale child lock/);
		const completions = readEvents(sandbox.root).filter(
			(event) => event.action_phase === "complete" && String(event.action_key ?? "").startsWith("work:"),
		);
		assert.equal(completions.length, 1);
		assert.equal(latestById(readLedger(sandbox.root, "PRS.jsonl")).length, 1);
		assert.equal(fs.existsSync(lockFile), false, "the lock must be released after the child finishes");
	} finally {
		sandbox.cleanup();
	}
});

test("pausing before the merge and resuming produces exactly one merge commit", () => {
	const sandbox = createSandbox("pause-merge");
	try {
		seedTask(sandbox.root);
		writeGatesConfig(sandbox.root);
		const run = (mode?: string) =>
			spawnSync("node", [CLI, "step", "--sandbox", sandbox.root, "--headless", "--command", STUB_CHILD], {
				cwd: REPO_ROOT,
				encoding: "utf8",
				env: { ...process.env, ...(mode ? { PI_STUB_MODE: mode } : {}) },
			});
		run();
		run();
		run("work");
		run();
		run("review");
		run();
		assert.equal(readState(sandbox.root).state, "MERGE");
		const pause = spawnSync("node", [CLI, "pause", "waiting for the human to read the review", "--sandbox", sandbox.root], {
			cwd: REPO_ROOT,
			encoding: "utf8",
		});
		assert.equal(pause.status, 0, pause.stderr);
		assert.equal(readState(sandbox.root).state, "PAUSED");
		const resume = spawnSync("node", [CLI, "resume", "--sandbox", sandbox.root], { cwd: REPO_ROOT, encoding: "utf8" });
		assert.equal(resume.status, 0, resume.stderr);
		assert.equal(readState(sandbox.root).state, "MERGE");
		// The MERGE action was never completed before the pause, so it runs now:
		// with requireHumanMerge the first step opens the gate.
		run();
		const gate = latestById(readLedger<{ id: string; status: string }>(sandbox.root, "GATES.jsonl"))[0];
		assert.equal(gate.status, "pending");
		const approve = spawnSync("node", [CLI, "gate", "approve", gate.id, "reviewed after the pause", "--sandbox", sandbox.root], {
			cwd: REPO_ROOT,
			encoding: "utf8",
		});
		assert.equal(approve.status, 0, approve.stderr);
		run();
		assert.equal(git(sandbox.root, ["log", "--format=%s", "-1", "update_pymc"]).startsWith("merge PR-001"), true);
		// A step again is a no-op or the next bookkeeping — never a second merge.
		run();
		const merges = readEvents(sandbox.root).filter(
			(event) => event.action_phase === "complete" && String(event.action_key ?? "").startsWith("merge:"),
		);
		assert.equal(merges.length, 1);
		assert.equal(git(sandbox.root, ["tag", "--list", "run/*"]).split("\n").filter(Boolean).length, 1);
	} finally {
		sandbox.cleanup();
	}
});
