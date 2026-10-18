/**
 * Raw acceptance snippets for `run_mock_cluster.sh` (S5 prompt):
 * - scenario 1: every hpc_* tool returns {enabled:false} with no ssh;
 * - the single-slot refusal message names the active experiment id.
 *
 * Prints machine-greppable evidence on stdout and exits non-zero on failure.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { appendRecord, readLedger } from "../../extensions/ledger/store.ts";
import { projectLatest } from "../../lib/records.ts";
import {
	handleHpcCancel,
	handleHpcCheckpoints,
	handleHpcConfig,
	handleHpcFetch,
	handleHpcRequeue,
	handleHpcStatus,
	handleHpcSubmit,
	handleHpcTail,
} from "../../extensions/hpc/handlers.ts";
import { createHpcSandbox, seedExperiment, setUpCluster } from "./mock_cluster.ts";

function latest(root, id) {
	return projectLatest(readLedger(root, "experiment")).find((record) => record.id === id);
}

function fail(message) {
	process.stderr.write(`FAIL ${message}\n`);
	process.exit(1);
}

// 1. Disabled: no ssh, {enabled:false} everywhere.
{
	const sandbox = createHpcSandbox("hpc-accept-disabled");
	const cluster = setUpCluster(sandbox.root, "cluster", { hpcEnabled: false });
	try {
		process.env.MOCK_FAIL_IF_SSH = "1";
		seedExperiment(sandbox.root, cluster);
		const ctx = { root: sandbox.root, role: "phd_candidate", actor: "acceptance" };
		const results = [
			handleHpcConfig({}, ctx),
			handleHpcStatus({}, ctx),
			handleHpcSubmit({ experiment_id: "E-0001" }, ctx),
			handleHpcCheckpoints({ experiment_id: "E-0001" }, ctx),
			handleHpcFetch({ experiment_id: "E-0001" }, ctx),
			handleHpcTail({ experiment_id: "E-0001" }, ctx),
			handleHpcCancel({ experiment_id: "E-0001", reason: "acceptance" }, { ...ctx, role: "supervisor" }),
			handleHpcRequeue({ experiment_id: "E-0001", reason: "acceptance" }, { ...ctx, role: "collector" }),
		];
		if (results.some((result) => result.details.enabled !== false)) fail("a tool did not return enabled:false");
		if (cluster.readCalls().length !== 0) fail("a shim was invoked while HPC was disabled");
		delete process.env.MOCK_FAIL_IF_SSH;
		process.stdout.write(`disabled: 8/8 tools returned {enabled:false}; 0 ssh invocations\n`);
	} finally {
		delete process.env.MOCK_FAIL_IF_SSH;
		cluster.cleanup();
		sandbox.cleanup();
	}
}

// 2. Second submit while a job is active is refused and names the active id.
{
	const sandbox = createHpcSandbox("hpc-accept-slot");
	const cluster = setUpCluster(sandbox.root, "cluster");
	try {
		const { id } = seedExperiment(sandbox.root, cluster);
		const ctx = { root: sandbox.root, role: "phd_candidate", actor: "acceptance" };
		const first = handleHpcSubmit({ experiment_id: id }, ctx);
		if (first.isError) fail(first.content[0].text);
		const record = latest(sandbox.root, id);
		appendRecord(
			sandbox.root,
			"experiment",
			{
				...record,
				id: "E-0002",
				task_id: "T-002",
				idempotencyKey: "sha256:acceptance-second",
				job_id: null,
				status: "queued",
				attempt: 1,
			},
			{ root: sandbox.root, role: "conductor", actor: "acceptance" },
		);
		const second = handleHpcSubmit({ experiment_id: "E-0002" }, ctx);
		if (!second.isError) fail("the second submit was not refused");
		const message = second.content[0].text;
		if (!message.includes(id)) fail(`the refusal does not name the active experiment ${id}: ${message}`);
		process.stdout.write(`second-submit refusal names ${id}: ${message}\n`);
		if (!fs.existsSync(path.join(sandbox.root, ".pi", "state", "EXPERIMENTS.jsonl"))) fail("no experiment ledger");
	} finally {
		cluster.cleanup();
		sandbox.cleanup();
	}
}

process.stdout.write("acceptance snippets: OK\n");
