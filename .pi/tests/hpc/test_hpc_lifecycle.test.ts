/**
 * Scenarios 2–6 — the experiment lifecycle on the mock cluster.
 *
 * 2. Happy path: queued → running → completed → outputs + checkpoints fetched
 *    → collected, with a manifest carrying sacct accounting and a SHA-256 per file.
 * 3. Timeout at 24h (exit 99): checkpoints fetched first, progress recorded,
 *    requeue_requested; a second submission before the first ended is refused;
 *    after the requeue the same idempotency key is reused at attempt 2.
 * 4. OOM: one requeue with doubled memory; a second OOM is failed-final + BLOCKED.
 * 5. No progress across two attempts → failed-final + BLOCKED.
 * 6. Missing declared output after completion → collection-failed, a task is
 *    opened, no fabricated result.
 */

import assert from "node:assert/strict";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { REPO_ROOT } from "../helpers.ts";
import { loadLoopOptions } from "../../extensions/orchestrator/commands.ts";
import { loadLedgerView } from "../../extensions/orchestrator/ledgers.ts";
import { nextAction } from "../../extensions/orchestrator/state-machine.ts";
import { executeAction } from "../../extensions/orchestrator/execute.ts";
import type { RunContext } from "../../extensions/orchestrator/sandbox.ts";
import { appendRecord, readLedger } from "../../extensions/ledger/store.ts";
import { loadLoopState } from "../../lib/state.ts";
import { handleStateSet } from "../../extensions/ledger/handlers.ts";
import { handleHpcCheckpoints, handleHpcSubmit, handleHpcRequeue } from "../../extensions/hpc/handlers.ts";
import {
	collectorPass,
	createHpcSandbox,
	latestExperimentRecord,
	seedExperiment,
	setUpCluster,
	type CollectorPass,
} from "./mock_cluster.ts";

const OUTPUTS = {
	"results/units_a.csv": "unit,value\na,1\nb,2\n",
	"results/manifest_a.json": '{"units":2}\n',
};

function runContext(root: string): RunContext {
	return { root, realRoot: REPO_ROOT, sandbox: root, dryRun: false, headless: true, command: "pi" };
}

function driveCollector(root: string, id: string, terminal = "requeue_requested", maxPasses = 10): CollectorPass {
	let pass: CollectorPass | null = null;
	const allTransitions: string[] = [];
	for (let i = 0; i < maxPasses; i += 1) {
		pass = collectorPass(root, id);
		allTransitions.push(...pass.transitions);
		if (pass.statusAfter === terminal || pass.statusAfter === "collection-failed" || pass.statusAfter === "collected") break;
	}
	assert.equal(pass?.statusAfter, terminal, `collector did not reach ${terminal}: ${pass?.message}`);
	return { ...(pass as CollectorPass), transitions: allTransitions };
}

async function conductorStep(root: string) {
	const context = runContext(root);
	const options = loadLoopOptions(context);
	const action = nextAction(loadLoopState(root).state, loadLedgerView(root), options);
	const result = await executeAction({ context, action, options });
	return { action, result };
}

/** Boot (if needed) and put the loop in WAITING_HPC for the experiment. */
async function ensureWaitingHpc(root: string, id: string, attempt: number): Promise<void> {
	if (loadLoopState(root).state.state === "UNINITIALIZED") {
		await conductorStep(root); // boot
	}
	const state = loadLoopState(root).state.state;
	if (state !== "WAITING_HPC") {
		const target = state === "SELECT_TASK" ? "WORK" : state;
		if (target !== state) handleStateSet({ state: "WORK", task_id: "T-001" }, { root, role: "conductor", actor: "test" });
		handleStateSet(
			{ state: "WAITING_HPC", task_id: "T-001", experiment_id: id, attempt },
			{ root, role: "conductor", actor: "test" },
		);
	}
}

function sha256(text: string): string {
	return crypto.createHash("sha256").update(text).digest("hex");
}

test("scenario 2: happy path — submit, poll, complete, collect with a hashed manifest", async (t) => {
	const sandbox = createHpcSandbox("hpc-lifecycle-happy");
	const cluster = setUpCluster(sandbox.root, "cluster");
	t.after(() => {
		cluster.cleanup();
		sandbox.cleanup();
	});
	const root = sandbox.root;
	const { id } = seedExperiment(root, cluster);
	cluster.writeScenario({
		states: ["PENDING", "RUNNING", "RUNNING", "COMPLETED"],
		exitCode: 0,
		elapsed: 754,
		maxRss: "1024M",
		progress: { version: 1, completed: ["2023-01-01:main", "2023-01-02:main"] },
		checkpoints: { "fit_a.pkl": "checkpoint-a", "fit_b.pkl": "checkpoint-b" },
		outputs: OUTPUTS,
		logLines: { out: ["started", "two units done"], err: [] },
	});

	const submitted = handleHpcSubmit({ experiment_id: id }, { root, role: "phd_candidate", actor: "test:phd_candidate" });
	assert.equal(submitted.isError, undefined, submitted.content[0].text);
	const jobId = String(submitted.details.job_id);
	assert.match(submitted.content[0].text, new RegExp(`submitted as job ${jobId}`));

	const pass = driveCollector(root, id, "collected");
	assert.deepEqual(pass.transitions, ["running", "collecting", "collected"]);
	const record = latestExperimentRecord(root, id);
	assert.equal(record?.status, "collected");
	assert.equal(record?.job_id, jobId);

	// Manifest: sacct accounting + SHA-256 for every fetched file.
	const manifestPath = path.join(root, ".pi", "artifacts", id, "manifest.json");
	assert.ok(fs.existsSync(manifestPath), manifestPath);
	const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8")) as {
		status: string;
		files: Array<{ path: string; sha256: string; size: number }>;
		checkpoints: Array<{ path: string; sha256: string }>;
		missing: string[];
		sacct: Record<string, unknown>;
		manifest_sha256: string;
	};
	assert.equal(manifest.status, "collected");
	assert.deepEqual(manifest.missing, []);
	const fileNames = manifest.files.map((file) => file.path).sort();
	assert.deepEqual(fileNames, ["results/manifest_a.json", "results/units_a.csv"]);
	for (const file of manifest.files) {
		const actual = sha256(fs.readFileSync(path.join(root, ".pi", "artifacts", id, file.path), "utf8"));
		assert.equal(file.sha256, actual, `sha256 for ${file.path}`);
	}
	assert.ok(manifest.checkpoints.some((file) => file.path === "checkpoints/progress.json"));
	assert.ok(manifest.checkpoints.length >= 3, "progress.json plus two pkl checkpoints");
	for (const field of ["State", "ExitCode", "Elapsed", "MaxRSS", "NodeList", "AllocCPUS", "Start", "End", "JobID"]) {
		assert.ok(field in manifest.sacct, `manifest.sacct.${field} is missing`);
	}
	assert.equal(manifest.sacct.State, "completed");
	assert.equal(manifest.sacct.JobID, jobId);
	assert.equal(manifest.sacct.MaxRSS, "1024M");
	assert.match(manifest.manifest_sha256, /^[0-9a-f]{64}$/);
	// The artifact dir is what the manifest describes: no extra/missing files.
	const listed = [...manifest.files, ...manifest.checkpoints].map((file) => file.path).sort();
	const onDisk = (function walk(dir: string, base = ""): string[] {
		return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
			const rel = base === "" ? entry.name : `${base}/${entry.name}`;
			return entry.isDirectory() ? walk(path.join(dir, entry.name), rel) : [rel];
		});
	})(path.join(root, ".pi", "artifacts", id)).filter((rel) => rel !== "manifest.json");
	assert.deepEqual([...listed].sort(), onDisk.sort());
});

test("scenario 3: timeout (exit 99) fetches checkpoints first, refuses a second job, and requeues at attempt 2", async (t) => {
	const sandbox = createHpcSandbox("hpc-lifecycle-timeout");
	const cluster = setUpCluster(sandbox.root, "cluster");
	t.after(() => {
		cluster.cleanup();
		sandbox.cleanup();
	});
	const root = sandbox.root;
	const { id } = seedExperiment(root, cluster);
	cluster.writeScenario({
		states: ["PENDING", "RUNNING", "TIMEOUT"],
		exitCode: 99,
		elapsed: 86400,
		progress: { version: 1, completed: ["2023-01-01:main"] },
		checkpoints: { "fit_a.pkl": "checkpoint-a" },
		outputs: OUTPUTS,
		partialOutputs: { "results/units_a.csv": "unit,value\na,1\n" },
	});

	const submitted = handleHpcSubmit({ experiment_id: id }, { root, role: "phd_candidate", actor: "test" });
	assert.equal(submitted.isError, undefined, submitted.content[0].text);
	const firstJob = String(submitted.details.job_id);
	const pass = driveCollector(root, id, "requeue_requested");
	assert.deepEqual(pass.transitions, ["running", "timed_out", "requeue_requested"]);
	const timedOut = latestExperimentRecord(root, id);
	assert.equal(timedOut?.lastState, "timed_out");
	assert.equal(timedOut?.exitCode, 99);
	assert.deepEqual(timedOut?.progressHistory, [{ attempt: 1, completed: 1 }]);
	assert.equal(timedOut?.checkpointsFetched, true);
	// The derived view shows id/attempt/status/progress/job.
	const experimentsView = fs.readFileSync(path.join(root, ".pi", "state", "EXPERIMENTS.md"), "utf8");
	assert.match(experimentsView, /\| E-0001 \| T-001 \| requeue_requested \| 1 \| 1 units \| 900001 \|/);
	// Checkpoints were fetched before the transition: their hashes are on disk.
	assert.ok(fs.existsSync(path.join(root, ".pi", "artifacts", id, "checkpoints", "progress.json")));
	assert.ok(fs.existsSync(path.join(root, ".pi", "artifacts", id, "checkpoints", "fit_a.pkl")));

	// A second job while E-0001 is active is refused and names it.
	appendRecord(
		root,
		"experiment",
		{
			id: "E-0002",
			task_id: "T-002",
			branch: "work/T-002",
			commit_sha: cluster.commitSha,
			data_hash: "sha256:mock-data",
			uv_lock_sha: cluster.uvLockSha,
			command: "uv run python mock_runner.py",
			resources: { kind: "hpc", cpus: 2, memGb: 8, gpus: 0, wallClockHours: 24 },
			checkpoints: {
				dir: path.join(cluster.remoteScratch, "E-0002", "checkpoints"),
				progressFile: "progress.json",
				resumeCommand: "uv run python mock_runner.py --resume $CKPT",
				flushOnSignal: true,
			},
			expectedOutputs: ["results/units_a.csv"],
			seeds: [42],
			idempotencyKey: "sha256:second",
			resumeSmokeTest: "ok",
			status: "queued",
			attempt: 1,
		},
		{ root, role: "conductor", actor: "test" },
	);
	const second = handleHpcSubmit({ experiment_id: "E-0002" }, { root, role: "phd_candidate", actor: "test" });
	assert.equal(second.isError, true);
	assert.match(second.content[0].text, /E-0001/);
	assert.match(second.content[0].text, /single-slot|active/i);
	// Retire E-0002 again (it was only here to prove the refusal) so E-0001 can requeue.
	appendRecord(
		root,
		"experiment",
		{ ...(latestExperimentRecord(root, "E-0002") as Record<string, unknown>), id: "E-0002", status: "cancelled", evidence: ["test: retired after the single-slot refusal"] },
		{ root, role: "conductor", actor: "test" },
	);

	// Requeue: attempt+1 with the same idempotency key, only after the job ended.
	const requeue = handleHpcRequeue(
		{ experiment_id: id, reason: "TIMEOUT with 1 checkpointed unit; conductor decision: requeue" },
		{ root, role: "collector", actor: "test:collector" },
	);
	assert.equal(requeue.isError, undefined, requeue.content[0].text);
	const requeued = latestExperimentRecord(root, id);
	assert.equal(requeued?.attempt, 2);
	assert.equal(requeued?.status, "queued");
	assert.equal(requeued?.idempotencyKey, "sha256:mock-idempotency-key");
	assert.equal(requeued?.previous_job_id, firstJob);
	assert.equal(requeued?.job_id, null);

	// A second submit before the first ended was refused; attempt 2 submits anew.
	const resubmitted = handleHpcSubmit({ experiment_id: id }, { root, role: "phd_candidate", actor: "test" });
	assert.equal(resubmitted.isError, undefined, resubmitted.content[0].text);
	assert.equal(resubmitted.details.attempt, 2);
	assert.notEqual(String(resubmitted.details.job_id), firstJob);
	assert.equal(latestExperimentRecord(root, id)?.idempotencyKey, "sha256:mock-idempotency-key");
});

test("scenario 4: OOM requeues once with doubled memory, then failed-final + BLOCKED", async (t) => {
	const sandbox = createHpcSandbox("hpc-lifecycle-oom");
	const cluster = setUpCluster(sandbox.root, "cluster");
	t.after(() => {
		cluster.cleanup();
		sandbox.cleanup();
	});
	const root = sandbox.root;
	const { id } = seedExperiment(root, cluster);
	cluster.writeScenario({
		states: ["PENDING", "RUNNING", "OUT_OF_MEMORY"],
		exitCode: 137,
		elapsed: 1800,
		maxRss: "8.1G",
		progress: { version: 1, completed: ["2023-01-01:main"] },
		checkpoints: { "fit_a.pkl": "checkpoint-a" },
	});

	assert.equal(handleHpcSubmit({ experiment_id: id }, { root, role: "phd_candidate", actor: "test" }).isError, undefined);
	driveCollector(root, id, "requeue_requested");
	await ensureWaitingHpc(root, id, 1);

	// Conductor decision 1: requeue with 16G (the record declares a 64G limit).
	const first = await conductorStep(root);
	assert.equal(first.action.kind, "hpc");
	if (first.action.kind === "hpc") {
		assert.equal(first.action.op, "requeue");
		assert.equal(first.action.memoryGb, 16);
	}
	const afterFirst = latestExperimentRecord(root, id);
	assert.equal(afterFirst?.attempt, 2);
	assert.equal(afterFirst?.memoryDoubled, true);
	assert.equal((afterFirst?.resources as { memGb: number }).memGb, 16);
	assert.equal(afterFirst?.idempotencyKey, "sha256:mock-idempotency-key");
	assert.equal(loadLoopState(root).state.state, "WAITING_HPC");

	// The conductor submits attempt 2 on the next step.
	const second = await conductorStep(root);
	assert.equal(second.action.kind, "hpc");
	if (second.action.kind === "hpc") assert.equal(second.action.op, "submit");

	// Second OOM: no further doubling.
	cluster.writeScenario({
		states: ["PENDING", "RUNNING", "OUT_OF_MEMORY"],
		exitCode: 137,
		elapsed: 1800,
		progress: { version: 1, completed: ["2023-01-01:main"] },
		checkpoints: { "fit_a.pkl": "checkpoint-a" },
	});
	driveCollector(root, id, "requeue_requested");
	const third = await conductorStep(root);
	assert.equal(third.action.kind, "block");
	assert.equal(loadLoopState(root).state.state, "BLOCKED");
	const final = latestExperimentRecord(root, id);
	assert.equal(final?.status, "failed-final");
	assert.match(String(loadLoopState(root).state.reason), /OOM again|after the memory doubling/);
});

test("scenario 5: no progress across two attempts → failed-final + BLOCKED", async (t) => {
	const sandbox = createHpcSandbox("hpc-lifecycle-noprogress");
	const cluster = setUpCluster(sandbox.root, "cluster");
	t.after(() => {
		cluster.cleanup();
		sandbox.cleanup();
	});
	const root = sandbox.root;
	const { id } = seedExperiment(root, cluster);
	const noProgress = {
		states: ["PENDING", "RUNNING", "TIMEOUT"],
		exitCode: 99,
		elapsed: 86400,
		progress: { version: 1, completed: [] },
		checkpoints: {},
	};
	cluster.writeScenario(noProgress);
	assert.equal(handleHpcSubmit({ experiment_id: id }, { root, role: "phd_candidate", actor: "test" }).isError, undefined);
	driveCollector(root, id, "requeue_requested");
	await ensureWaitingHpc(root, id, 1);
	const first = await conductorStep(root);
	// Attempt 1 without progress requeues once (with a warning), not silently.
	assert.equal(first.action.kind, "hpc");
	assert.equal(latestExperimentRecord(root, id)?.attempt, 2);
	await conductorStep(root); // submit attempt 2

	cluster.writeScenario(noProgress);
	driveCollector(root, id, "requeue_requested");
	const final = await conductorStep(root);
	assert.equal(final.action.kind, "block");
	assert.equal(loadLoopState(root).state.state, "BLOCKED");
	assert.equal(latestExperimentRecord(root, id)?.status, "failed-final");
	assert.match(String(loadLoopState(root).state.reason), /no checkpoint progress across 2 consecutive attempts/);
});

test("scenario 6: a missing declared output is collection-failed, a task is opened, nothing is fabricated", async (t) => {
	const sandbox = createHpcSandbox("hpc-lifecycle-missing");
	const cluster = setUpCluster(sandbox.root, "cluster");
	t.after(() => {
		cluster.cleanup();
		sandbox.cleanup();
	});
	const root = sandbox.root;
	const { id } = seedExperiment(root, cluster);
	cluster.writeScenario({
		states: ["PENDING", "RUNNING", "COMPLETED"],
		exitCode: 0,
		elapsed: 120,
		progress: { version: 1, completed: ["2023-01-01:main"] },
		checkpoints: { "fit_a.pkl": "checkpoint-a" },
		// Only one of the two declared outputs exists.
		outputs: { "results/units_a.csv": "unit,value\na,1\n" },
	});
	assert.equal(handleHpcSubmit({ experiment_id: id }, { root, role: "phd_candidate", actor: "test" }).isError, undefined);
	const pass = driveCollector(root, id, "collection-failed");
	assert.deepEqual(pass.transitions, ["running", "collecting", "collection-failed"]);
	const record = latestExperimentRecord(root, id);
	assert.equal(record?.status, "collection-failed");
	assert.deepEqual(record?.missingOutputs, ["results/manifest_a.json"]);
	const manifest = JSON.parse(
		fs.readFileSync(path.join(root, ".pi", "artifacts", id, "manifest.json"), "utf8"),
	) as { status: string; missing: string[] };
	assert.equal(manifest.status, "collection-failed");
	assert.deepEqual(manifest.missing, ["results/manifest_a.json"]);

	await ensureWaitingHpc(root, id, 1);
	const blocked = await conductorStep(root);
	assert.equal(blocked.action.kind, "block");
	assert.equal(loadLoopState(root).state.state, "BLOCKED");
	assert.match(String(loadLoopState(root).state.reason), /collection failed/);
	const tasks = readLedger(root, "task");
	const opened = tasks.find((task) => task.origin === `REVIEW:${id}`);
	assert.ok(opened, `a task must be opened for ${id}`);
	assert.equal(opened?.status, "open");
	assert.match(String(opened?.title), /missing outputs/);
});

test("hpc_checkpoints lists hashes and progress while the job runs", async (t) => {
	const sandbox = createHpcSandbox("hpc-checkpoints");
	const cluster = setUpCluster(sandbox.root, "cluster");
	t.after(() => {
		cluster.cleanup();
		sandbox.cleanup();
	});
	const root = sandbox.root;
	const { id } = seedExperiment(root, cluster);
	cluster.writeScenario({
		states: ["RUNNING", "RUNNING"],
		progress: { version: 1, completed: ["2023-01-01:main"] },
		checkpoints: { "fit_a.pkl": "checkpoint-a" },
	});
	assert.equal(handleHpcSubmit({ experiment_id: id }, { root, role: "phd_candidate", actor: "test" }).isError, undefined);
	collectorPass(root, id); // one RUNNING poll materializes checkpoints
	const listing = handleHpcCheckpoints({ experiment_id: id }, { root, role: "collector", actor: "test" });
	assert.equal(listing.isError, undefined, listing.content[0].text);
	const files = listing.details.files as Array<{ path: string; size: number }>;
	assert.ok(files.some((file) => file.path === "progress.json"));
	assert.ok(files.some((file) => file.path === "fit_a.pkl"));
	const hashes = listing.details.hashes as Record<string, string>;
	assert.equal(hashes["fit_a.pkl"], sha256("checkpoint-a"));
	assert.deepEqual(listing.details.progress, { completed: ["2023-01-01:main"], version: 1 });
	assert.match(listing.content[0].text, /checkpoint file/);
});
