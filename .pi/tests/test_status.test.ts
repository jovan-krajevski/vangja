/**
 * Status composition: the same lines feed the TUI widget and `status.sh`.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { composeStatusLine, composeStatusLines, readStatus } from "../extensions/status/status.ts";
import { handleExperimentQueue, handleStateSet, handleTaskEnqueue } from "../extensions/ledger/handlers.ts";
import { scratchRoot } from "./helpers.ts";

const ctx = (root: string) => ({ root, role: "conductor" as const, actor: "test" });

test("an uninitialized loop reports UNINITIALIZED without inventing state", () => {
	const root = scratchRoot();
	const snapshot = readStatus(root);
	assert.equal(snapshot.initialized, false);
	assert.equal(snapshot.state, "UNINITIALIZED");
	const lines = composeStatusLines(snapshot);
	assert.match(lines[0], /UNINITIALIZED/);
	assert.match(lines[1], /tasks: 0/);
	assert.match(composeStatusLine(snapshot), /UNINITIALIZED/);
});

test("a working loop summarises state, task, experiment, budget, and gates", () => {
	const root = scratchRoot();
	handleStateSet({ state: "BOOT", run_id: "run-7" }, ctx(root));
	handleStateSet({ state: "SELECT_TASK" }, ctx(root));
	handleTaskEnqueue(
		{
			brief: {
				title: "Add calibration table",
				origin: "PLAN:P1-2",
				priority: "P1",
				objective: "x",
				acceptance_criteria: ["table builds"],
				deliverables: ["paper/generated/calibration.tex"],
			},
		},
		ctx(root),
	);
	handleStateSet({ state: "WORK", task_id: "T-001" }, ctx(root));
	handleExperimentQueue(
		{
			record: {
				task_id: "T-001",
				branch: "work/T-001",
				commit_sha: "d".repeat(40),
				data_hash: "sha256:d",
				command: "uv run python run.py",
				resources: { kind: "local" },
				checkpoints: { dir: "d", progressFile: "p", resumeCommand: "r" },
				expectedOutputs: ["out.csv"],
				seeds: [42],
				idempotencyKey: "k",
				resumeSmokeTest: "ok",
			},
		},
		ctx(root),
	);
	// The conductor records the active job in LOOP_STATE when it submits.
	handleStateSet({ state: "WORK", task_id: "T-001", experiment_id: "E-0001", attempt: 1 }, ctx(root));

	const snapshot = readStatus(root);
	assert.equal(snapshot.state, "WORK");
	assert.equal(snapshot.taskId, "T-001");
	assert.equal(snapshot.openTasks, 1);
	assert.equal(snapshot.totalTasks, 1);
	assert.equal(snapshot.activeExperiments.length, 1);
	assert.equal(snapshot.activeExperiments[0].id, "E-0001");
	assert.equal(snapshot.activeExperiments[0].attempt, 1);
	assert.equal(snapshot.caps?.maxSessionsPerDay, 120);

	const line = composeStatusLine(snapshot);
	assert.match(line, /LOOP WORK/);
	assert.match(line, /task=T-001/);
	assert.match(line, /exp=E-0001#1/);

	const lines = composeStatusLines(snapshot);
	assert.match(lines[0], /LOOP WORK/);
	assert.match(lines[1], /budget: no BUDGET.json/);
	assert.match(lines[2], /tasks 1\/1 open/);
	assert.match(lines[2], /E-0001\(queued#1\)/);
});

test("a paused loop reports where it will resume from", () => {
	const root = scratchRoot();
	handleStateSet({ state: "BOOT" }, ctx(root));
	handleStateSet({ state: "SELECT_TASK" }, ctx(root));
	handleStateSet({ state: "PAUSED", reason: "awaiting confirmation gate" }, ctx(root));
	const snapshot = readStatus(root);
	assert.equal(snapshot.state, "PAUSED");
	assert.equal(snapshot.pausedFrom, "SELECT_TASK");
	assert.match(composeStatusLine(snapshot), /LOOP PAUSED\(from SELECT_TASK\)/);
	assert.match(composeStatusLines(snapshot).join("\n"), /awaiting confirmation gate/);
});
