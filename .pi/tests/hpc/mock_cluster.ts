/**
 * Shared helpers for the S5 mock-cluster tests.
 *
 * Every test runs against a scratch clone of the repository (never the real
 * checkout) plus a directory-backed fake cluster under
 * `<sandbox>/.pi/state/hpc-tests/<label>/`. The mock shims are put first on
 * PATH, so the HPC extension invokes the real `ssh`/`scp`/`rsync`/`sbatch`
 * command lines and the shims answer them from recorded scenarios.
 */

import { execFileSync, spawnSync } from "node:child_process";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { REPO_ROOT } from "../helpers.ts";
import { appendRecord, readLedger } from "../../extensions/ledger/store.ts";
import { handleExperimentQueue, handleExperimentUpdate } from "../../extensions/ledger/handlers.ts";
import { handleHpcFetch, handleHpcStatus, recordHeartbeat } from "../../extensions/hpc/handlers.ts";
import { projectLatest } from "../../lib/records.ts";

export const MOCKBIN = path.join(REPO_ROOT, ".pi", "tests", "hpc", "fixtures", "mockbin");
export const TESTS_ROOT = path.join(REPO_ROOT, ".pi", "state", "hpc-tests");

export interface MockScenario {
	host?: string;
	user?: string;
	startJobId?: number;
	states?: string[];
	exitCode?: number;
	elapsed?: number;
	maxRss?: string;
	node?: string;
	progress?: { version: number; completed: string[] } | null;
	checkpoints?: Record<string, string>;
	outputs?: Record<string, string>;
	partialOutputs?: Record<string, string>;
	logLines?: { out?: string[]; err?: string[] };
	failSsh?: boolean;
	sbatchExit?: number;
	cpus?: number;
}

export interface Cluster {
	sandbox: string;
	dir: string;
	remoteRepo: string;
	remoteScratch: string;
	commitSha: string;
	uvLockSha: string;
	scenarioPath: string;
	callsPath: string;
	writeScenario: (scenario: MockScenario) => void;
	readCalls: () => Array<Record<string, unknown>>;
	cleanup: () => void;
}

function git(args: string[], cwd?: string): string {
	return execFileSync("git", args, { cwd, encoding: "utf8" }).trim();
}

export function createHpcSandbox(label: string): { root: string; cleanup: () => void } {
	fs.mkdirSync(TESTS_ROOT, { recursive: true });
	const dir = fs.mkdtempSync(path.join(TESTS_ROOT, `${label}-`));
	const result = spawnSync("git", ["clone", "--quiet", "--local", REPO_ROOT, dir], { encoding: "utf8" });
	if (result.status !== 0) throw new Error(`git clone into ${dir} failed: ${result.stderr || result.stdout}`);
	git(["-C", dir, "checkout", "--quiet", "update_pymc"]);
	fs.mkdirSync(path.join(dir, ".pi", "state"), { recursive: true });
	fs.rmSync(path.join(dir, ".pi", "config", "hpc.json"), { force: true });
	return {
		root: dir,
		cleanup: () => {
			try {
				fs.rmSync(dir, { recursive: true, force: true });
			} catch {
				// Best-effort cleanup of gitignored test residue.
			}
		},
	};
}

export interface ClusterOptions {
	hpcEnabled?: boolean;
	containerSha?: string;
	maxParallelJobs?: number;
	gitKeyMissing?: boolean;
	forwardGitAgent?: boolean;
}

/**
 * Build the fake remote + `hpc.json` inside `root`. The remote repo is a real
 * git repository (so `git fetch`/`worktree add`/`status` in the preflight run
 * for real), while SLURM itself is the scenario-driven shim.
 */
export function setUpCluster(root: string, label: string, options: ClusterOptions = {}): Cluster {
	const dir = fs.mkdtempSync(path.join(root, ".pi", "state", `${label}-`));
	const remoteRoot = path.join(dir, "remote");
	const remoteRepo = path.join(remoteRoot, "repo");
	const remoteScratch = path.join(remoteRoot, "scratch");
	const keys = path.join(dir, "keys");
	fs.mkdirSync(remoteRepo, { recursive: true });
	fs.mkdirSync(remoteScratch, { recursive: true });
	fs.mkdirSync(path.join(remoteRoot, "home"), { recursive: true });
	fs.mkdirSync(keys, { recursive: true });

	// A real repo with a root-looking finisher: gate checks run real git.
	fs.copyFileSync(path.join(root, "uv.lock"), path.join(remoteRepo, "uv.lock"));
	fs.writeFileSync(path.join(remoteRepo, "README.mock.md"), "# mock remote\n");
	git(["init", "--quiet", "-b", "main"], remoteRepo);
	git(["-C", remoteRepo, "config", "user.email", "mock@example.invalid"]);
	git(["-C", remoteRepo, "config", "user.name", "Mock Cluster"]);
	git(["-C", remoteRepo, "add", "-A"]);
	git(["-C", remoteRepo, "commit", "--quiet", "-m", "mock remote checkout"]);
	const commitSha = git(["-C", remoteRepo, "rev-parse", "HEAD"]);
	const origin = path.join(remoteRoot, "origin.git");
	git(["init", "--quiet", "--bare", origin]);
	git(["-C", remoteRepo, "remote", "add", "origin", origin]);
	git(["-C", remoteRepo, "push", "--quiet", "origin", "main"]);
	const uvLockSha = crypto.createHash("sha256").update(fs.readFileSync(path.join(remoteRepo, "uv.lock"))).digest("hex");

	const hpcKey = path.join(keys, "id_mock");
	const gitKey = path.join(keys, "git_mock");
	fs.writeFileSync(hpcKey, "MOCK-HPC-KEY-PLACEHOLDER-DO-NOT-USE\n");
	fs.writeFileSync(gitKey, "MOCK-GIT-KEY-PLACEHOLDER-DO-NOT-USE\n");

	const container = options.containerSha ? path.join(remoteRoot, "container.sif") : "";
	if (options.containerSha) fs.writeFileSync(container, options.containerSha);

	const scenarioPath = path.join(dir, "scenario.json");
	const callsPath = path.join(dir, "calls.jsonl");
	const writeScenario = (scenario: MockScenario): void => {
		fs.writeFileSync(
			scenarioPath,
			`${JSON.stringify({ host: "mock-cluster", user: "mockuser", ...scenario }, null, 2)}\n`,
		);
	};
	writeScenario({ states: ["PENDING", "RUNNING", "RUNNING", "COMPLETED"], exitCode: 0, elapsed: 754 });

	const hpcJson = {
		host: options.hpcEnabled === false ? "" : "mock-cluster",
		user: options.hpcEnabled === false ? "" : "mockuser",
		port: 22,
		remoteRepoDir: options.hpcEnabled === false ? "" : remoteRepo,
		remoteScratchDir: options.hpcEnabled === false ? "" : remoteScratch,
		hpcKey: options.hpcEnabled === false ? "" : hpcKey,
		gitKey: options.gitKeyMissing ? "" : gitKey,
		forwardGitAgent: options.forwardGitAgent !== false,
		container: options.hpcEnabled === false ? "" : container,
		partition: "",
		cpus: 2,
		memGb: 8,
		gpus: 0,
		wallClockHours: 24,
		pollSeconds: 1,
		maxPolls: 40,
		maxAttemptsPerExperiment: 6,
		maxParallelJobs: options.maxParallelJobs ?? 1,
	};
	fs.mkdirSync(path.join(root, ".pi", "config"), { recursive: true });
	fs.writeFileSync(path.join(root, ".pi", "config", "hpc.json"), `${JSON.stringify(hpcJson, null, 2)}\n`);

	process.env.MOCK_CLUSTER_DIR = dir;
	process.env.PATH = `${MOCKBIN}:${process.env.PATH ?? ""}`;
	delete process.env.MOCK_FAIL_IF_SSH;

	return {
		sandbox: root,
		dir,
		remoteRepo,
		remoteScratch,
		commitSha,
		uvLockSha,
		scenarioPath,
		callsPath,
		writeScenario,
		readCalls: () => {
			if (!fs.existsSync(callsPath)) return [];
			return fs
				.readFileSync(callsPath, "utf8")
				.split("\n")
				.filter((line) => line.trim() !== "")
				.map((line) => JSON.parse(line) as Record<string, unknown>);
		},
		cleanup: () => {
			delete process.env.MOCK_CLUSTER_DIR;
			try {
				fs.rmSync(dir, { recursive: true, force: true });
			} catch {
				// Best effort.
			}
		},
	};
}

export interface SeededExperiment {
	id: string;
	record: Record<string, unknown>;
}

/**
 * Seed one queued experiment through the real `experiment_queue` handler (so the
 * single-slot and checkpoint gates are exercised). The first record in a fresh
 * sandbox is always `E-0001`.
 */
export function seedExperiment(
	root: string,
	cluster: Cluster,
	overrides: Record<string, unknown> = {},
): SeededExperiment {
	const id = String(overrides.id ?? "E-0001");
	const checkpoints = {
		dir: path.join(cluster.remoteScratch, id, "checkpoints"),
		progressFile: "progress.json",
		resumeCommand: `uv run python mock_runner.py --out results --resume $CKPT`,
		flushOnSignal: true,
		...(overrides.checkpoints as Record<string, unknown> | undefined),
	};
	const record: Record<string, unknown> = {
		task_id: "T-001",
		branch: "work/T-001",
		commit_sha: cluster.commitSha,
		data_hash: "sha256:mock-data",
		uv_lock_sha: cluster.uvLockSha,
		command: "uv run python mock_runner.py --out results",
		resources: { kind: "hpc", cpus: 2, memGb: 8, gpus: 0, wallClockHours: 24, partitionMemGbLimit: 64 },
		checkpoints,
		expectedOutputs: ["results/units_a.csv", "results/manifest_a.json"],
		seeds: [42, 20240816],
		idempotencyKey: "sha256:mock-idempotency-key",
		resumeSmokeTest: "ran 1 unit, killed, --resume skipped it; observed in .pi/state/hpc-tests/smoke.txt",
		...overrides,
		checkpoints,
	};
	const result = handleExperimentQueue({ record }, { root, role: "phd_candidate", actor: "test:phd_candidate" });
	if (result.isError) throw new Error(`seedExperiment refused: ${result.content[0].text}`);
	return { id: String((result.details.record as { id: string }).id), record: result.details.record as Record<string, unknown> };
}

export function latestExperimentRecord(root: string, id: string): Record<string, unknown> | null {
	const records = readLedger(root, "experiment");
	const latest = projectLatest(records as Array<{ id?: string; seq: number }>).find((record) => record.id === id);
	return (latest as Record<string, unknown> | undefined) ?? null;
}

export function readExperimentHistory(root: string, id: string): Array<Record<string, unknown>> {
	return readLedger(root, "experiment").filter((record) => record.id === id);
}

export function collectorContext(root: string) {
	return { root, role: "collector" as const, actor: "test:collector" };
}

export interface CollectorPass {
	experimentId: string;
	statusBefore: string;
	statusAfter: string;
	transitions: string[];
	message: string;
	error?: string;
}

function evidenceOf(statusResult: ReturnType<typeof handleHpcStatus>, experimentId: string): string {
	const jobs = (statusResult.details.jobs ?? []) as Array<Record<string, unknown>>;
	const job = jobs.find((entry) => entry.experiment_id === experimentId) ?? jobs[0];
	if (!job) return statusResult.content[0].text;
	return JSON.stringify({
		job_id: job.jobId,
		squeue: job.squeue,
		sacct: job.sacct,
		raw_squeue: statusResult.details.squeue_raw,
		raw_sacct: statusResult.details.sacct_raw,
	});
}

/**
 * One deterministic collector pass, mirroring `.pi/agents/collector.md`:
 * poll → evidence → transition/heartbeat → fetch checkpoints first on
 * interruption → write requeue_requested (never a resubmission).
 */
export function collectorPass(root: string, experimentId: string): CollectorPass {
	const ctx = collectorContext(root);
	const before = latestExperimentRecord(root, experimentId);
	const transitions: string[] = [];
	if (!before) return { experimentId, statusBefore: "?", statusAfter: "?", transitions, message: "unknown experiment" };
	const statusBefore = String(before.status);
	if (statusBefore === "queued" && !before.job_id) {
		return { experimentId, statusBefore, statusAfter: statusBefore, transitions, message: "queued but not submitted; the conductor submits" };
	}
	const statusResult = handleHpcStatus({}, ctx);
	if (statusResult.isError) {
		return { experimentId, statusBefore, statusAfter: statusBefore, transitions, message: "hpc_status failed", error: statusResult.content[0].text };
	}
	const jobs = (statusResult.details.jobs ?? []) as Array<Record<string, unknown>>;
	const job = jobs.find((entry) => entry.experiment_id === experimentId) ?? jobs[0];
	const squeue = job?.squeue as { state?: string } | null | undefined;
	const sacct = job?.sacct as { state?: string; exit_code?: number | null; reason?: string } | null | undefined;
	const observed = String(squeue?.state ?? sacct?.raw_state ?? sacct?.state ?? "UNKNOWN").toUpperCase();
	const classified = String(sacct?.state ?? "unknown");
	const exitCode = typeof sacct?.exit_code === "number" ? sacct.exit_code : null;
	const evidence = evidenceOf(statusResult, experimentId);
	const transition = (status: string, extraEvidence: string, fields: Record<string, unknown> = {}): void => {
		const result = handleExperimentUpdate(
			{ experiment_id: experimentId, status, evidence: extraEvidence, fields },
			ctx,
		);
		if (result.isError) throw new Error(`collector transition ${status} refused: ${result.content[0].text}`);
		transitions.push(status);
	};

	const activeStates = ["PENDING", "CONFIGURING", "REQUEUED", "RUNNING", "COMPLETING", "SUSPENDED"];
	const terminal = !activeStates.includes(observed);

	if (!terminal) {
		const latest = latestExperimentRecord(root, experimentId) ?? before;
		if (String(latest.status) === "queued" && observed === "RUNNING") {
			transition("running", evidence, { lastState: "running" });
		} else if (latest.lastState !== observed.toLowerCase()) {
			handleExperimentUpdate(
				{ experiment_id: experimentId, evidence, fields: { lastState: observed.toLowerCase() } },
				ctx,
			);
		}
		const heartbeats = readLedger(root, "event").filter(
			(event) => event.experiment_id === experimentId && event.heartbeat === true,
		).length;
		recordHeartbeat(root, experimentId, heartbeats + 1, evidence, ctx.actor);
		return {
			experimentId,
			statusBefore,
			statusAfter: String((latestExperimentRecord(root, experimentId) ?? before).status),
			transitions,
			message: `job still ${observed}; one heartbeat recorded`,
		};
	}
	// A completed job is collected; an interrupted/failed one fetches checkpoints
	// first, records the progress delta, then writes requeue_requested.
	if (classified === "completed" && exitCode !== 99) {
		if (String((latestExperimentRecord(root, experimentId) ?? before).status) === "running") {
			transition("collecting", evidence, { lastState: "completed" });
		}
		const collected = handleHpcFetch({ experiment_id: experimentId }, ctx);
		if (collected.isError) {
			transition("collection-failed", `${evidence}\n${collected.content[0].text}`, {
				missingOutputs: collected.details.missing ?? [],
				lastState: "completed",
			});
			return {
				experimentId,
				statusBefore,
				statusAfter: "collection-failed",
			transitions,
				message: collected.content[0].text,
			};
		}
		transition(
			"collected",
			`${evidence}\nmanifest ${String(collected.details.manifest)} sha256 ${String(collected.details.manifest_sha256)}`,
			{ lastState: "completed" },
		);
		return {
			experimentId,
			statusBefore,
			statusAfter: "collected",
			transitions,
			message: `collected ${String(collected.details.manifest)}`,
		};
	}

	const fetch = handleHpcFetch({ experiment_id: experimentId, checkpointsOnly: true }, ctx);
	if (fetch.isError) {
		transition("failed", `checkpoint fetch failed: ${fetch.content[0].text}`, { lastState: "failed" });
		return { experimentId, statusBefore, statusAfter: "failed", transitions, message: "checkpoint fetch failed" };
	}
	const progress = (fetch.details.progress ?? null) as { completed?: string[] } | null;
	const history = Array.isArray(before.progressHistory) ? (before.progressHistory as Array<Record<string, unknown>>) : [];
	const attempt = Number(before.attempt ?? 1);
	const completedCount = progress?.completed?.length ?? 0;
	const nextHistory = [...history.filter((entry) => Number(entry.attempt) !== attempt), { attempt, completed: completedCount }];
	const fields = {
		progressHistory: nextHistory,
		progress: progress ? { version: progress.version ?? 1, completed: progress.completed ?? [] } : { completed: [] },
		checkpointsFetched: true,
		lastState: classified,
		exitCode,
		stderrTail: String(statusResult.details.sacct_raw ?? "").slice(-400),
	};
	const interruptedStatus =
		classified === "interrupted_clean"
			? "interrupted_clean"
			: classified === "timed_out" || classified === "preempted"
				? "timed_out"
				: "failed";
	transition(interruptedStatus, evidence, fields);
	transition("requeue_requested", `${evidence}\nprogress: ${completedCount} completed unit(s)`);
	return {
		experimentId,
		statusBefore,
		statusAfter: "requeue_requested",
		transitions,
		message: `checkpoints fetched (${completedCount} units); requeue_requested`,
	};
}

/** One collector pass for a normal completion: collecting → fetch → collected. */
export function collectCompleted(root: string, experimentId: string): CollectorPass {
	const ctx = collectorContext(root);
	const before = latestExperimentRecord(root, experimentId);
	if (!before) return { experimentId, statusBefore: "?", statusAfter: "?", transitions: [], message: "unknown experiment" };
	const statusBefore = String(before.status);
	const transitions: string[] = [];
	const statusResult = handleHpcStatus({}, ctx);
	const jobs = (statusResult.details.jobs ?? []) as Array<Record<string, unknown>>;
	const job = jobs.find((entry) => entry.experiment_id === experimentId) ?? jobs[0];
	const evidence = evidenceOf(statusResult, experimentId);
	if (statusBefore === "queued") {
		const result = handleExperimentUpdate(
			{ experiment_id: experimentId, status: "running", evidence, fields: { lastState: "running" } },
			ctx,
		);
		if (result.isError) throw new Error(result.content[0].text);
		transitions.push("running");
	}
	if (String((latestExperimentRecord(root, experimentId) ?? before).status) === "running") {
		const result = handleExperimentUpdate(
			{ experiment_id: experimentId, status: "collecting", evidence, fields: { lastState: "completed" } },
			ctx,
		);
		if (result.isError) throw new Error(result.content[0].text);
		transitions.push("collecting");
	}
	const fetch = handleHpcFetch({ experiment_id: experimentId }, ctx);
	if (fetch.isError) {
		const result = handleExperimentUpdate(
			{ experiment_id: experimentId, status: "collection-failed", evidence: `${evidence}\n${fetch.content[0].text}`, fields: { missingOutputs: fetch.details.missing ?? [] } },
			ctx,
		);
		if (result.isError) throw new Error(result.content[0].text);
		transitions.push("collection-failed");
		return { experimentId, statusBefore, statusAfter: "collection-failed", transitions, message: fetch.content[0].text };
	}
	const result = handleExperimentUpdate(
		{
			experiment_id: experimentId,
			status: "collected",
			evidence: `${evidence}\nmanifest ${String(fetch.details.manifest)} sha256 ${String(fetch.details.manifest_sha256)}`,
			fields: { lastState: "completed", job_id: String(job?.jobId ?? "") },
		},
		ctx,
	);
	if (result.isError) throw new Error(result.content[0].text);
	transitions.push("collected");
	return { experimentId, statusBefore, statusAfter: "collected", transitions, message: `collected ${String(fetch.details.manifest)}` };
}


export { appendRecord };
