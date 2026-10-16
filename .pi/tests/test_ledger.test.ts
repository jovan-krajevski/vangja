/**
 * Ledger round-trip, validation, projections, derived views, role gates, and
 * the state file. All offline, all in scratch roots.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { readJsonlFile } from "../lib/io.ts";
import { projectLatest } from "../lib/records.ts";
import { renderContextMd } from "../lib/views.ts";
import { loadLoopState } from "../lib/state.ts";
import {
	handleAmendmentRecord,
	handleDecisionRecord,
	handleExperimentQueue,
	handleExperimentUpdate,
	handleLedgerAppend,
	handleLedgerRead,
	handlePrOpen,
	handleReviewWrite,
	handleStateGet,
	handleStateSet,
	handleTaskClose,
	handleTaskEnqueue,
	handleWaiverRecord,
	readLedger,
	type LedgerContext,
} from "../extensions/ledger/handlers.ts";
import { scratchRoot } from "./helpers.ts";

const CONDUCTOR: LedgerContext = { root: "", role: "conductor", actor: "test-conductor" };

function ctx(root: string, role: LedgerContext["role"] = "conductor"): LedgerContext {
	return { ...CONDUCTOR, root, role, actor: role === "conductor" ? "test-conductor" : `test-${role}` };
}

const BRIEF = {
	title: "Fix the transfer-trend potential sign",
	origin: "PLAN:P0-3",
	priority: "P0",
	effort: "medium",
	objective: "The trend potential must match plan/05 and the existing test.",
	acceptance_criteria: ["tests/test_regularization.py::test_trend_potential passes", "diff shows no sign change in the seasonal cap"],
	deliverables: ["src/vangja/components/linear.py", "tests/test_regularization.py"],
	compute: "none",
	status: "open",
};

test("round-trip: append then read, with assigned id and seq", () => {
	const root = scratchRoot();
	const appended = handleTaskEnqueue({ brief: BRIEF }, ctx(root));
	assert.equal(appended.isError, undefined, appended.content[0].text);
	const id = (appended.details.record as { id: string }).id;
	assert.equal(id, "T-001");
	const read = handleLedgerRead({ kind: "task" }, ctx(root));
	assert.equal(read.details.returned, 1);
	assert.equal((read.details.records as Array<{ id: string }>)[0].id, "T-001");
	assert.match(read.content[0].text, /Fix the transfer-trend potential sign/);
});

test("append-only updates repeat the id and projectLatest keeps the newest", () => {
	const root = scratchRoot();
	handleTaskEnqueue({ brief: BRIEF }, ctx(root));
	const closed = handleTaskClose({ task_id: "T-001", status: "merged", evidence: "pytest tests/test_regularization.py green" }, ctx(root));
	assert.equal(closed.isError, undefined, closed.content[0].text);
	const records = readLedger(root, "task");
	assert.equal(records.length, 2, "an update appends a second record");
	assert.equal(new Set(records.map((record) => record.id)).size, 1);
	const latest = projectLatest(records as Array<{ id?: string; seq: number }>);
	assert.equal(latest.length, 1);
	assert.equal(latest[0].status, "merged");
});

test("task_enqueue rejects briefs without acceptance criteria or a known origin", () => {
	const root = scratchRoot();
	const noCriteria = handleTaskEnqueue({ brief: { ...BRIEF, acceptance_criteria: [] } }, ctx(root));
	assert.equal(noCriteria.isError, true);
	assert.match(noCriteria.content[0].text, /acceptance_criteria/);
	const badOrigin = handleTaskEnqueue({ brief: { ...BRIEF, origin: "make-it-up" } }, ctx(root));
	assert.equal(badOrigin.isError, true);
	assert.match(badOrigin.content[0].text, /origin "make-it-up"/);
	const findingOrigin = handleTaskEnqueue({ brief: { ...BRIEF, origin: "FINDING:WS-3-07" } }, ctx(root));
	assert.equal(findingOrigin.isError, undefined);
});

test("schema validation rejects a malformed record and names the field", () => {
	const root = scratchRoot();
	const result = handleLedgerAppend({ kind: "review", record: { pr_id: "PR-001", decision: "maybe" } }, ctx(root));
	assert.equal(result.isError, true);
	assert.match(result.content[0].text, /missing required field "id"|decision/);
});

test("pr_open requires the machine-readable footer fields", () => {
	const root = scratchRoot();
	const missing = handlePrOpen({ footer: { task_id: "T-001" } }, ctx(root));
	assert.equal(missing.isError, true);
	assert.match(missing.content[0].text, /head_sha/);
	const opened = handlePrOpen(
		{
			footer: {
				task_id: "T-001",
				head_sha: "a".repeat(40),
				base_sha: "b".repeat(40),
				branch: "work/T-001-sign",
				summary: "fix the sign",
				artifacts: [],
				tests: ["tests/test_regularization.py"],
				claims: [],
				hpc_jobs: [],
			},
		},
		ctx(root),
	);
	assert.equal(opened.isError, undefined, opened.content[0].text);
	assert.equal((opened.details.record as { id: string }).id, "PR-001");
});

test("review_write is supervisor/conductor only and validates the decision", () => {
	const root = scratchRoot();
	const phd = handleReviewWrite(
		{ record: { pr_id: "PR-001", decision: "accept", evidence: ["x"], required_changes: [], residual_risk: "low" } },
		ctx(root, "phd_candidate"),
	);
	assert.equal(phd.isError, true);
	assert.match(phd.content[0].text, /phd_candidate.*may not use review_write/);
	const supervisor = handleReviewWrite(
		{ record: { pr_id: "PR-001", decision: "accept", evidence: ["pytest green"], required_changes: [], residual_risk: "none" } },
		ctx(root, "supervisor"),
	);
	assert.equal(supervisor.isError, undefined, supervisor.content[0].text);
});

test("experiment_queue enforces the single-slot rule and checkpoint metadata", () => {
	const root = scratchRoot();
	const base = {
		task_id: "T-001",
		branch: "work/T-001",
		commit_sha: "c".repeat(40),
		data_hash: "sha256:data",
		command: "uv run python fixed_case_studies/stocks/02_run.py",
		resources: { kind: "local", cpus: 8, memGb: 32, gpus: 0, wallClockHours: 24 },
		checkpoints: { dir: "/scratch/E-0001/checkpoints", progressFile: "progress.json", resumeCommand: "… --resume /scratch/E-0001/checkpoints" },
		expectedOutputs: ["results/units_*.csv"],
		seeds: [42, 20240816],
		idempotencyKey: "sha256(branch+commit+command+seeds+data_hash)",
	};
	const noSmoke = handleExperimentQueue({ record: base }, ctx(root, "phd_candidate"));
	assert.equal(noSmoke.isError, true);
	assert.match(noSmoke.content[0].text, /resumeSmokeTest/);

	const noCheckpoints = handleExperimentQueue(
		{ record: { ...base, checkpoints: { dir: "d" }, resumeSmokeTest: "ran 1 unit, --resume skipped it" } },
		ctx(root, "phd_candidate"),
	);
	assert.equal(noCheckpoints.isError, true);
	assert.match(noCheckpoints.content[0].text, /progressFile/);

	const queued = handleExperimentQueue(
		{ record: { ...base, resumeSmokeTest: "ran 1 unit, --resume skipped it" } },
		ctx(root, "phd_candidate"),
	);
	assert.equal(queued.isError, undefined, queued.content[0].text);
	assert.equal((queued.details.record as { id: string }).id, "E-0001");

	const second = handleExperimentQueue(
		{ record: { ...base, idempotencyKey: "other", resumeSmokeTest: "ok" } },
		ctx(root, "phd_candidate"),
	);
	assert.equal(second.isError, true);
	assert.match(second.content[0].text, /single-slot|active/);
});

test("experiment_update only allows legal lifecycle transitions", () => {
	const root = scratchRoot();
	const record = {
		task_id: "T-001",
		branch: "work/T-001",
		commit_sha: "c".repeat(40),
		data_hash: "sha256:data",
		command: "cmd",
		resources: {},
		checkpoints: { dir: "d", progressFile: "p", resumeCommand: "r" },
		expectedOutputs: ["o"],
		seeds: [42],
		idempotencyKey: "k",
		resumeSmokeTest: "ok",
	};
	handleExperimentQueue({ record }, ctx(root, "phd_candidate"));
	assert.equal(handleExperimentUpdate({ experiment_id: "E-0001", status: "running", evidence: "squeue R" }, ctx(root, "collector")).isError, undefined);
	const illegal = handleExperimentUpdate({ experiment_id: "E-0001", status: "collected", evidence: "wishful" }, ctx(root, "collector"));
	assert.equal(illegal.isError, true);
	assert.match(illegal.content[0].text, /illegal experiment transition running → collected/);
	assert.equal(handleExperimentUpdate({ experiment_id: "E-0001", status: "collecting", evidence: "sacct complete" }, ctx(root, "collector")).isError, undefined);
	assert.equal(handleExperimentUpdate({ experiment_id: "E-0001", status: "collected", evidence: "hashes recorded" }, ctx(root, "collector")).isError, undefined);
	const terminal = handleExperimentUpdate({ experiment_id: "E-0001", status: "running", evidence: "zombie" }, ctx(root, "collector"));
	assert.equal(terminal.isError, true);
	assert.match(terminal.content[0].text, /terminal/);
});

test("decisions, waivers, and amendments record ids and open a human gate", () => {
	const root = scratchRoot();
	const decision = handleDecisionRecord(
		{
			adr: {
				title: "Keep Relative MAE as the primary metric",
				context: "The audit found mixed metrics across runners.",
				decision: "Relative MAE vs persistence, median over target×origin units.",
				alternatives: ["RMSE", "MASE"],
				consequences: ["Comparable with the historical audit", "Needs the ε-exclusion rule"],
			},
		},
		ctx(root),
	);
	assert.equal(decision.isError, undefined, decision.content[0].text);
	assert.equal((decision.details.record as { id: string }).id, "D-001");

	const waiver = handleWaiverRecord({ waiver: { gate: "pytest", reason: "one flaky network test unrelated to the diff" } }, ctx(root, "supervisor"));
	assert.equal(waiver.isError, undefined, waiver.content[0].text);

	const amendment = handleAmendmentRecord(
		{ amendment: { path: "fixed_case_studies/protocol", change: "add origin 2024-06-28", justification: "extend confirmation window" } },
		ctx(root, "supervisor"),
	);
	assert.equal(amendment.isError, undefined, amendment.content[0].text);
	assert.equal((amendment.details.record as { id: string }).id, "A-001");
	const gates = readLedger(root, "gate");
	assert.equal(gates.length, 1);
	assert.match(String(gates[0].note), /A-001/);
});

test("state_get/state_set validate the state machine and write atomically", () => {
	const root = scratchRoot();
	const before = handleStateGet({}, ctx(root));
	assert.equal(before.details.initialized, false);
	assert.equal((before.details.state as { state: string }).state, "UNINITIALIZED");

	const boot = handleStateSet({ state: "BOOT", run_id: "run-0001" }, ctx(root));
	assert.equal(boot.isError, undefined, boot.content[0].text);
	const illegal = handleStateSet({ state: "PAPER" }, ctx(root));
	assert.equal(illegal.isError, true);
	assert.match(illegal.content[0].text, /illegal transition BOOT → PAPER/);

	const loaded = loadLoopState(root);
	assert.equal(loaded.state.state, "BOOT");
	assert.equal(loaded.state.run_id, "run-0001");
	assert.equal(fs.existsSync(path.join(root, ".pi", "state", "LOOP_STATE.json")), true);
	assert.equal(fs.existsSync(path.join(root, ".pi", "state", "LOOP_STATE.json.lock")), false, "locks are released");
});

test("state_set is conductor-only", () => {
	const root = scratchRoot();
	const denied = handleStateSet({ state: "BOOT" }, ctx(root, "phd_candidate"));
	assert.equal(denied.isError, true);
	assert.match(denied.content[0].text, /state_set.*conductor/);
});

test("derived views are written deterministically", () => {
	const root = scratchRoot();
	handleTaskEnqueue({ brief: BRIEF }, ctx(root));
	handleExperimentQueue(
		{
			record: {
				task_id: "T-001",
				branch: "work/T-001",
				commit_sha: "c".repeat(40),
				data_hash: "sha256:data",
				command: "cmd",
				resources: {},
				checkpoints: { dir: "d", progressFile: "p", resumeCommand: "r" },
				expectedOutputs: ["o"],
				seeds: [42],
				idempotencyKey: "k",
				resumeSmokeTest: "ok",
			},
		},
		ctx(root, "phd_candidate"),
	);
	const stateDir = path.join(root, ".pi", "state");
	for (const view of ["CONTEXT.md", "TASKS.md", "EXPERIMENTS.md"]) {
		assert.ok(fs.existsSync(path.join(stateDir, view)), `${view} must exist`);
	}
	const context = fs.readFileSync(path.join(stateDir, "CONTEXT.md"), "utf8");
	assert.match(context, /# Loop context/);
	assert.match(context, /## State/);
	assert.match(context, /T-001/);
	assert.match(context, /E-0001/);
	assert.ok(context.split("\n").length <= 200, "CONTEXT.md must stay ≤200 lines");

	// Determinism: rendering the same ledgers twice yields byte-identical output.
	const tasks = readLedger(root, "task");
	const first = renderContextMd({
		state: loadLoopState(root).state as unknown as Record<string, unknown>,
		budget: {},
		tasks,
		experiments: [],
		reviews: [],
		prs: [],
		gates: [],
		events: [],
	});
	const second = renderContextMd({
		state: loadLoopState(root).state as unknown as Record<string, unknown>,
		budget: {},
		tasks,
		experiments: [],
		reviews: [],
		prs: [],
		gates: [],
		events: [],
	});
	assert.equal(first, second);
});

test("ledger_read is bounded and reports where the full ledger lives", () => {
	const root = scratchRoot();
	for (let i = 0; i < 12; i += 1) {
		handleLedgerAppend({ kind: "event", record: { summary: `event ${i}` } }, ctx(root));
	}
	const bounded = handleLedgerRead({ kind: "event", limit: 5 }, ctx(root));
	assert.equal(bounded.details.returned, 5);
	assert.equal(bounded.details.truncated, true);
	assert.match(bounded.content[0].text, /full ledger: .*EVENTS\.jsonl/);
	assert.equal(readJsonlFile(path.join(root, ".pi", "state", "EVENTS.jsonl")).length, 12);
	const tail = handleLedgerRead({ kind: "event", limit: 3, tail: true }, ctx(root));
	assert.equal((tail.details.records as Array<{ summary: string }>)[0].summary, "event 9");
});

test("unknown ledger kinds are refused with the valid list", () => {
	const root = scratchRoot();
	const result = handleLedgerRead({ kind: "nonsense" }, ctx(root));
	assert.equal(result.isError, true);
	assert.match(result.content[0].text, /valid kinds: event, task, pr/);
});
