/**
 * Sandbox end-to-end cycle: SELECT_TASK → WORK → PR → REVIEW(accept) → MERGE
 * (human gate + pre-merge gate) → POST_MERGE.
 *
 * The whole cycle runs in a scratch clone driven through the real CLI with the
 * deterministic child stub. Assertions cover the invariants the prompt calls
 * out: the merge happens only after the gate run, the human gate blocks the
 * merge until approved, the sandbox base branch advances, the real repository
 * does not, and there is exactly one merge commit.
 */

import { strict as assert } from "node:assert";
import { test } from "node:test";
import * as fs from "node:fs";
import * as path from "node:path";
import {
	createSandbox,
	git,
	latestById,
	readEvents,
	readLedger,
	readState,
	runCli,
	seedTask,
	writeGatesConfig,
} from "./orchestrator_helpers.ts";
import { REPO_ROOT } from "../helpers.ts";

interface CycleResult {
	transcript: string[];
	sandbox: ReturnType<typeof createSandbox>;
	mergeEvent: Record<string, unknown>;
}

/** Run the happy path; fail loudly on the first unexpected step. */
function runCycle(sandbox: ReturnType<typeof createSandbox>, env: NodeJS.ProcessEnv = {}): CycleResult {
	const transcript: string[] = [];
	const step = (label: string, extraEnv: NodeJS.ProcessEnv = {}): string => {
		const result = runCli(sandbox.root, ["step"], { ...env, ...extraEnv });
		assert.equal(result.status, 0, `${label} failed: ${result.stdout}\n${result.stderr}`);
		transcript.push(`$ loop step   # ${label}\n${result.stdout.trim()}`);
		return result.stdout;
	};

	step("BOOT");
	step("SELECT_TASK → WORK");
	step("WORK: phd_candidate opens the PR", { PI_STUB_MODE: "work" });
	step("WORK → REVIEW");
	step("REVIEW: supervisor accepts", { PI_STUB_MODE: "review" });
	step("REVIEW → MERGE");
	const gate = step("MERGE: open the human gate");
	assert.match(gate, /human gate G-\d+ \(merge\) is pending/);
	const waiting = step("MERGE: waiting on the human gate");
	assert.match(waiting, /skipped\(no-action\) waiting for human gate/);
	return { transcript, sandbox, mergeEvent: {} };
}

test("the sandbox cycle merges only after the gate run, the human gate, and one accepted review", () => {
	const sandbox = createSandbox("cycle");
	const baseBefore = git(sandbox.root, ["rev-parse", "update_pymc"]);
	const realBefore = git(REPO_ROOT, ["rev-parse", "update_pymc"]);
	const realRefsBefore = git(REPO_ROOT, ["for-each-ref", "--format=%(refname) %(objectname)"]);
	try {
		seedTask(sandbox.root);
		writeGatesConfig(sandbox.root);
		const { transcript } = runCycle(sandbox);

		// The merge must not have happened yet: the human gate is pending.
		assert.equal(git(sandbox.root, ["rev-parse", "update_pymc"]), baseBefore);
		const pendingGates = latestById(readLedger<{ id: string; status: string }>(sandbox.root, "GATES.jsonl"));
		assert.equal(pendingGates.length, 1);
		assert.equal(pendingGates[0].status, "pending");

		// Approve, then merge.
		const approve = runCli(sandbox.root, ["gate", "approve", pendingGates[0].id, "reviewed the stub evidence"]);
		assert.equal(approve.status, 0, approve.stderr);
		transcript.push(`$ loop gate approve ${pendingGates[0].id} "reviewed the stub evidence"\n${approve.stdout.trim()}`);
		const merge = runCli(sandbox.root, ["step"]);
		assert.equal(merge.status, 0, merge.stderr);
		transcript.push(`$ loop step   # MERGE\n${merge.stdout.trim()}`);
		assert.match(merge.stdout, /merged PR-001 → POST_MERGE/);

		// The base branch advanced in the sandbox and not in the real repository.
		const baseAfter = git(sandbox.root, ["rev-parse", "update_pymc"]);
		assert.notEqual(baseAfter, baseBefore);
		assert.equal(git(REPO_ROOT, ["rev-parse", "update_pymc"]), realBefore);
		// No branch, tag, or commit ref in the real repository changed: the loop
		// only ever operated in the sandbox.
		assert.equal(git(REPO_ROOT, ["for-each-ref", "--format=%(refname) %(objectname)"]), realRefsBefore);

		// Exactly one merge commit, with the review id in the message.
		const log = git(sandbox.root, ["log", "--format=%s", `${baseBefore}..update_pymc`]);
		const mergeCommits = log.split("\n").filter((line) => line.startsWith("merge PR-001"));
		assert.equal(mergeCommits.length, 1, log);
		assert.match(mergeCommits[0], /\(review RV-001\)/);

		// The tag names the run and the sequence.
		const tags = git(sandbox.root, ["tag", "--list", "run/*/merge/*"]).split("\n").filter(Boolean);
		assert.equal(tags.length, 1);
		assert.match(tags[0], /^run\/run-\d{14}-[0-9a-f]{8}\/merge\/1$/);

		// The merge event points at the gate run and the review.
		const mergeEvent = readEvents(sandbox.root).find((event) => event.action_key === "merge:PR-001:RV-001");
		assert.ok(mergeEvent, "merge completion event missing");
		assert.equal(mergeEvent.action_phase, "complete");
		assert.equal(mergeEvent.gate_ok, true);
		assert.ok(
			typeof mergeEvent.gate_run === "string" && fs.existsSync(path.resolve(sandbox.root, String(mergeEvent.gate_run))),
			`gate run ${String(mergeEvent.gate_run)} is missing`,
		);
		const gateRun = JSON.parse(fs.readFileSync(path.resolve(sandbox.root, String(mergeEvent.gate_run)), "utf8")) as {
			checks: Array<{ id: string; status: string }>;
			ok: boolean;
		};
		assert.equal(gateRun.ok, true);
		assert.deepEqual(
			gateRun.checks.map((check) => check.id),
			["tests", "protected-paths", "test-diff", "integrity", "provenance", "hygiene", "claims"],
		);

		// POST_MERGE closes the task and returns to SELECT_TASK.
		const post = runCli(sandbox.root, ["step"]);
		assert.equal(post.status, 0, post.stderr);
		assert.match(post.stdout, /POST_MERGE → SELECT_TASK/);
		const tasks = latestById(readLedger<{ id: string; status: string }>(sandbox.root, "TASKS.jsonl"));
		assert.equal(tasks[0].status, "merged");
		assert.equal(readState(sandbox.root).state, "SELECT_TASK");

		// Replaying the merge does not create a second commit or tag.
		const mergeActionEvents = readEvents(sandbox.root).filter((event) => event.action_key === "merge:PR-001:RV-001");
		assert.equal(mergeActionEvents.filter((event) => event.action_phase === "complete").length, 1);
		assert.equal(git(sandbox.root, ["rev-list", "--count", `${baseBefore}..update_pymc`]).trim(), "2");

		process.stdout.write(`\n--- sandbox cycle transcript ---\n${transcript.join("\n")}\n--- end transcript ---\n`);
	} finally {
		sandbox.cleanup();
	}
});

test("merging is refused while the human gate is pending, and rejected gates send the PR back", () => {
	const sandbox = createSandbox("cycle-reject");
	try {
		seedTask(sandbox.root);
		writeGatesConfig(sandbox.root);
		runCycle(sandbox);
		const gate = latestById(readLedger<{ id: string; status: string }>(sandbox.root, "GATES.jsonl"))[0];
		const reject = runCli(sandbox.root, ["gate", "reject", gate.id, "the evidence does not match the criterion"]);
		assert.equal(reject.status, 0, reject.stderr);
		const step = runCli(sandbox.root, ["step"]);
		assert.equal(step.status, 0, step.stderr);
		assert.match(step.stdout, /human gate G-\d+ rejected the merge/);
		assert.equal(readState(sandbox.root).state, "WORK");
		// No merge commit exists.
		const log = git(sandbox.root, ["log", "--format=%s", "update_pymc"]);
		assert.equal(log.includes("merge PR-001"), false);
	} finally {
		sandbox.cleanup();
	}
});
