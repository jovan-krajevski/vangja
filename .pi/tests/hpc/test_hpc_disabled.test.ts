/**
 * Scenario 1 — HPC disabled (`host: ""`).
 *
 * Every `hpc_*` tool must return `{enabled:false}` and perform no I/O, the
 * collector must exit with exactly one `hpc-disabled` event, the conductor must
 * skip `WAITING_HPC`, and no ssh may be attempted. `MOCK_FAIL_IF_SSH=1` makes the
 * ssh shim fail the test if it is ever called.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { REPO_ROOT } from "../helpers.ts";
import { loadLoopOptions, commandCollect } from "../../extensions/orchestrator/commands.ts";
import { loadLedgerView } from "../../extensions/orchestrator/ledgers.ts";
import { nextAction } from "../../extensions/orchestrator/state-machine.ts";
import { executeAction } from "../../extensions/orchestrator/execute.ts";
import type { RunContext } from "../../extensions/orchestrator/sandbox.ts";
import { handleStateSet } from "../../extensions/ledger/handlers.ts";
import { appendRecord } from "../../extensions/ledger/store.ts";
import { loadLoopState } from "../../lib/state.ts";
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

function runContext(root: string): RunContext {
	return { root, realRoot: REPO_ROOT, sandbox: root, dryRun: false, headless: true, command: "pi" };
}

test("every hpc_* tool returns {enabled:false} and never calls ssh", async (t) => {
	const sandbox = createHpcSandbox("hpc-disabled");
	const cluster = setUpCluster(sandbox.root, "cluster", { hpcEnabled: false });
	t.after(() => {
		cluster.cleanup();
		sandbox.cleanup();
	});
	const root = sandbox.root;
	seedExperiment(root, cluster);
	const ctx = { root, role: "phd_candidate" as const, actor: "test" };
	// Any ssh call would be recorded as a violation and exit 97.
	process.env.MOCK_FAIL_IF_SSH = "1";

	const calls = [
		handleHpcConfig({}, ctx),
		handleHpcStatus({}, ctx),
		handleHpcSubmit({ experiment_id: "E-0001" }, ctx),
		handleHpcCheckpoints({ experiment_id: "E-0001" }, ctx),
		handleHpcFetch({ experiment_id: "E-0001" }, ctx),
		handleHpcTail({ experiment_id: "E-0001" }, ctx),
		handleHpcCancel({ experiment_id: "E-0001", reason: "test" }, { ...ctx, role: "supervisor" }),
		handleHpcRequeue({ experiment_id: "E-0001", reason: "test" }, { ...ctx, role: "collector" }),
	];
	for (const result of calls) {
		assert.equal(result.isError, undefined, result.content[0].text);
		assert.equal(result.details.enabled, false, `enabled must be false: ${result.content[0].text}`);
		assert.match(result.content[0].text, /disabled/);
	}
	delete process.env.MOCK_FAIL_IF_SSH;
	const invocations = cluster.readCalls();
	assert.deepEqual(invocations, [], `no shim may run while HPC is disabled: ${JSON.stringify(invocations)}`);
});

test("the collector records one hpc-disabled event and exits cleanly", async (t) => {
	const sandbox = createHpcSandbox("hpc-disabled-collect");
	const cluster = setUpCluster(sandbox.root, "cluster", { hpcEnabled: false });
	t.after(() => {
		cluster.cleanup();
		sandbox.cleanup();
	});
	seedExperiment(sandbox.root, cluster);
	const first = await commandCollect({ context: runContext(sandbox.root) });
	assert.equal(first.disabled, true);
	assert.match(first.message, /hpc-disabled/);
	const events = loadLedgerView(sandbox.root).events.filter((event) => event.hpc_disabled === true);
	assert.equal(events.length, 1, "exactly one hpc-disabled event");
	const second = await commandCollect({ context: runContext(sandbox.root) });
	assert.equal(second.disabled, true);
	assert.match(second.message, /already recorded/);
	assert.equal(loadLedgerView(sandbox.root).events.filter((event) => event.hpc_disabled === true).length, 1);
	assert.deepEqual(cluster.readCalls(), []);
});

test("the loop skips WAITING_HPC and runs the experiment locally (plan/06 §7)", async (t) => {
	const sandbox = createHpcSandbox("hpc-disabled-loop");
	const cluster = setUpCluster(sandbox.root, "cluster", { hpcEnabled: false });
	t.after(() => {
		cluster.cleanup();
		sandbox.cleanup();
	});
	const root = sandbox.root;
	seedExperiment(root, cluster);
	appendRecord(
		root,
		"task",
		{
			id: "T-001",
			title: "local fallback task",
			origin: "PLAN:s5",
			priority: "P2",
			objective: "exercise the disabled-HPC routing",
			acceptance_criteria: ["an event is recorded"],
			deliverables: ["README.md"],
			status: "in_progress",
			effort: "small",
		},
		{ root, role: "conductor", actor: "test" },
	);
	handleStateSet(
		{ state: "WORK", task_id: "T-001" },
		{ root, role: "conductor", actor: "test" },
	);
	const context = runContext(root);
	const options = loadLoopOptions(context);
	// BOOT first (the loop never jumps UNINITIALIZED → WORK).
	const boot = nextAction(loadLoopState(root).state, loadLedgerView(root), options);
	assert.equal(boot.kind, "boot");
	await executeAction({ context, action: boot, options });
	handleStateSet({ state: "WORK", task_id: "T-001" }, { root, role: "conductor", actor: "test" });

	// First step: bookkeeping — record hpc-disabled + mark the experiment local.
	const first = nextAction(loadLoopState(root).state, loadLedgerView(root), options);
	assert.equal(first.kind, "bookkeep");
	assert.notEqual(first.kind === "bookkeep" ? first.to : "", "WAITING_HPC");
	const firstResult = await executeAction({ context, action: first, options });
	assert.equal(firstResult.performed, true, firstResult.message);
	assert.equal(loadLoopState(root).state.state, "WORK");
	const event = loadLedgerView(root).events.find((entry) => entry.hpc_disabled === true);
	assert.ok(event, "hpc-disabled event recorded");
	const record = loadLedgerView(root).experiments.find((entry) => entry.id === "E-0001");
	assert.equal(record?.status, "running");
	assert.equal(record?.resourcesKind, "local");

	// Second step: the local run is handed to phd_candidate, still not WAITING_HPC.
	const second = nextAction(loadLoopState(root).state, loadLedgerView(root), options);
	assert.equal(second.kind, "spawn");
	if (second.kind === "spawn") {
		assert.equal(second.agent, "phd_candidate");
		assert.match(second.key, /^work-local:/);
		assert.match(second.brief, /local fallback/i);
	}
	// Nothing was sent to the cluster.
	assert.deepEqual(cluster.readCalls(), []);
	assert.ok(fs.existsSync(path.join(root, ".pi", "state", "EXPERIMENTS.jsonl")));
});
