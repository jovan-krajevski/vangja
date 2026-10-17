/**
 * Seed the `run_mock_cluster.sh` sandbox for the literal `/loop:step` check:
 * one open task, one queued experiment, loop state WORK. `force: true` is used
 * because this is a purpose-built acceptance sandbox (the loop never jumps
 * UNINITIALIZED → WORK in production).
 *
 * Usage: SANDBOX=<dir> node .pi/tests/hpc/seed_disabled_loop.mjs
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";

const root = process.env.SANDBOX;
if (!root || !fs.existsSync(path.join(root, ".pi"))) {
	process.stderr.write("SANDBOX must point at a loop checkout\n");
	process.exit(1);
}

const store = await import(pathToFileURL(path.join(root, ".pi", "extensions", "ledger", "store.ts")).href);
const handlers = await import(pathToFileURL(path.join(root, ".pi", "extensions", "ledger", "handlers.ts")).href);
const ctx = { root, role: "conductor", actor: "acceptance" };
store.appendRecord(
	root,
	"task",
	{
		id: "T-001",
		title: "hpc acceptance task",
		origin: "PLAN:s5",
		priority: "P2",
		objective: "Exercise disabled-HPC routing in a work unit.",
		acceptance_criteria: ["an hpc-disabled event exists", "the loop proceeds to a local run"],
		deliverables: ["README.md"],
		status: "in_progress",
		effort: "small",
	},
	ctx,
);
store.appendRecord(
	root,
	"experiment",
	{
		id: "E-0001",
		task_id: "T-001",
		branch: "work/T-001",
		commit_sha: "a".repeat(40),
		data_hash: "sha256:acceptance",
		uv_lock_sha: "b".repeat(64),
		command: "uv run python mock_runner.py",
		resources: { kind: "hpc", cpus: 2, memGb: 8, gpus: 0, wallClockHours: 24 },
		checkpoints: {
			dir: "/tmp/mock/E-0001/checkpoints",
			progressFile: "progress.json",
			resumeCommand: "uv run python mock_runner.py --resume $CKPT",
			flushOnSignal: true,
		},
		expectedOutputs: ["results/units_a.csv"],
		seeds: [42],
		idempotencyKey: "sha256:acceptance",
		resumeSmokeTest: "ran 1 unit, killed, --resume skipped it",
		status: "queued",
		attempt: 1,
	},
	ctx,
);
handlers.handleStateSet({ state: "WORK", task_id: "T-001", force: true }, ctx);
process.stdout.write("seeded T-001 + E-0001 and set state WORK\n");
