/**
 * The conductor's retry table (plan/06_HPC_AND_COLLECTOR.md §6) and the
 * sacct classifier parity with the Python reference implementation
 * (`.pi/skills/hpc-slurm-and-checkpointing/scripts/parse_sacct.py`).
 *
 * No cluster: pure functions and the recorded fixture.
 */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { PI_DIR, REPO_ROOT } from "../helpers.ts";
import { decideRetry, progressStats } from "../../extensions/hpc/retry.ts";
import { classifySacctJob } from "../../extensions/hpc/slurm.ts";
import { heartbeatCount, hpcDisabledRecorded, findUnfinishedExperiment } from "../../extensions/orchestrator/state-machine.ts";

const SACCT_FIXTURE = path.join(PI_DIR, "tests", "skills", "fixtures", "hpc", "sacct_states.json");
const PARSE_SACCT = path.join(PI_DIR, "skills", "hpc-slurm-and-checkpointing", "scripts", "parse_sacct.py");

function input(overrides: Record<string, unknown> = {}) {
	return {
		status: "requeue_requested",
		attempt: 2,
		maxAttempts: 6,
		progressHistory: [
			{ attempt: 1, completed: 1 },
			{ attempt: 2, completed: 4 },
		],
		memoryDoubled: false,
		memoryGb: 8,
		partitionMemGbLimit: 64,
		sacctState: "timed_out",
		exitCode: 99,
		stderrTail: null,
		heartbeatCount: 0,
		maxPolls: 40,
		missingOutputs: [],
		taskId: "T-001",
		jobId: "900001",
		...overrides,
	};
}

test("progressStats detects advancement and consecutive no-progress attempts", () => {
	assert.deepEqual(progressStats([{ attempt: 1, completed: 0 }, { attempt: 2, completed: 0 }], 2), {
		advanced: false,
		noProgressAttempts: 2,
		current: 0,
		previous: 0,
	});
	assert.deepEqual(progressStats([{ attempt: 1, completed: 2 }, { attempt: 2, completed: 2 }, { attempt: 3, completed: 2 }], 3), {
		advanced: false,
		noProgressAttempts: 2,
		current: 2,
		previous: 2,
	});
	assert.equal(progressStats([{ attempt: 1, completed: 1 }, { attempt: 2, completed: 3 }], 2).advanced, true);
});

test("the retry table: requeue with progress, failed-final without, OOM doubled once", () => {
	const requeue = decideRetry(input());
	assert.equal(requeue.action, "requeue");
	assert.match(requeue.reason, /checkpoints advanced/);

	const noProgress = decideRetry(
		input({ progressHistory: [{ attempt: 1, completed: 2 }, { attempt: 2, completed: 2 }] }),
	);
	assert.equal(noProgress.action, "failed-final");
	assert.match(noProgress.question ?? "", /smoke test/i);

	const oomFirst = decideRetry(input({ sacctState: "oom", exitCode: 137 }));
	assert.equal(oomFirst.action, "requeue");
	assert.equal(oomFirst.memoryGb, 16);

	const oomAgain = decideRetry(input({ sacctState: "oom", exitCode: 137, memoryDoubled: true, memoryGb: 16 }));
	assert.equal(oomAgain.action, "failed-final");
	assert.match(oomAgain.reason, /after the memory doubling/);

	const oomUnknownLimit = decideRetry(input({ sacctState: "oom", exitCode: 137, partitionMemGbLimit: null }));
	assert.equal(oomUnknownLimit.action, "failed-final");
	assert.match(oomUnknownLimit.reason, /partitionMemGbLimit is unknown/);

	const maxAttempts = decideRetry(input({ attempt: 6 }));
	assert.equal(maxAttempts.action, "failed-final");
	assert.match(maxAttempts.reason, /maxAttemptsPerExperiment 6/);
});

test("the retry table: collector statuses, stuck polls, code bugs, collection failures", () => {
	assert.equal(decideRetry(input({ status: "running" })).action, "collect");
	assert.equal(decideRetry(input({ status: "queued" })).action, "collect");
	const stuck = decideRetry(input({ status: "running", heartbeatCount: 40 }));
	assert.equal(stuck.action, "blocked");
	assert.match(stuck.reason, /stuck in running/);

	const bug = decideRetry(input({ sacctState: "failed", exitCode: 1, progressHistory: [] }));
	assert.equal(bug.action, "open-task");
	assert.match(bug.question ?? "", /exit 1/);

	const missing = decideRetry(input({ status: "collection-failed", missingOutputs: ["results/a.csv"] }));
	assert.equal(missing.action, "open-task");
	assert.match(missing.reason, /results\/a\.csv/);

	assert.equal(decideRetry(input({ status: "collected" })).action, "wait");
	assert.equal(decideRetry(input({ status: "cancelled" })).action, "blocked");
});

test("state-machine helpers see unfinished experiments, heartbeats, and the disabled event", () => {
	const experiments = [
		{ id: "E-0001", status: "collected", attempt: 1, task_id: "T-001" },
		{ id: "E-0002", status: "timed_out", attempt: 2, task_id: "T-001" },
	];
	assert.equal(findUnfinishedExperiment(experiments)?.id, "E-0002");
	assert.equal(findUnfinishedExperiment([experiments[0]]), null);
	const events = [
		{ seq: 1, ts: "", actor: "", summary: "h", experiment_id: "E-0002", heartbeat: true },
		{ seq: 2, ts: "", actor: "", summary: "h", experiment_id: "E-0002", heartbeat: true },
		{ seq: 3, ts: "", actor: "", summary: "d", experiment_id: "E-0002", hpc_disabled: true },
	];
	assert.equal(heartbeatCount(events, "E-0002"), 2);
	assert.equal(hpcDisabledRecorded(events, "E-0002"), true);
	assert.equal(hpcDisabledRecorded(events, "E-0001"), false);
});

test("classifySacctJob matches parse_sacct.py on the recorded fixture", () => {
	const fixture = JSON.parse(fs.readFileSync(SACCT_FIXTURE, "utf8")) as { jobs: Array<Record<string, unknown>> };
	const optionSets: Array<{ args: string[]; options: Parameters<typeof classifySacctJob>[1] }> = [
		{ args: [], options: {} },
		{ args: ["--progress-advanced"], options: { progressAdvanced: true } },
		{ args: ["--progress-advanced", "--attempt", "6"], options: { progressAdvanced: true, attempt: 6 } },
		{ args: ["--memory-can-double", "--progress-advanced"], options: { memoryCanDouble: true, progressAdvanced: true } },
	];
	for (const set of optionSets) {
		const python = execFileSync("python3", [PARSE_SACCT, SACCT_FIXTURE, "--json", ...set.args], { encoding: "utf8" });
		const report = JSON.parse(python) as { jobs: Array<Record<string, unknown>> };
		for (const job of fixture.jobs) {
			const expected = report.jobs.find((entry) => entry.job_id === String(job.job_id));
			assert.ok(expected, `python did not classify job ${job.job_id}`);
			const actual = classifySacctJob(job, set.options);
			assert.equal(actual.state, expected.state, `state for ${String(job.job_id)} with ${set.args.join(" ")}`);
			assert.equal(actual.action, expected.action, `action for ${String(job.job_id)} with ${set.args.join(" ")}`);
			assert.equal(actual.exit_code, expected.exit_code, `exit for ${String(job.job_id)}`);
			assert.equal(actual.signal, expected.signal, `signal for ${String(job.job_id)}`);
			assert.equal(actual.elapsed_seconds, expected.elapsed_seconds, `elapsed for ${String(job.job_id)}`);
			assert.equal(actual.node, expected.node, `node for ${String(job.job_id)}`);
		}
	}
	assert.ok(REPO_ROOT.length > 0);
});
