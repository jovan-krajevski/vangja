/**
 * State-machine table tests: every state × condition → the expected action.
 *
 * `nextAction` is pure, so these build snapshots by hand — no filesystem, no
 * spawning, no clock. The coverage list mirrors plan/02_ARCHITECTURE.md §2.
 */

import { strict as assert } from "node:assert";
import { test } from "node:test";
import { nextAction, describeAction, readinessCriteria, completions } from "../../extensions/orchestrator/state-machine.ts";
import {
	budget,
	event,
	experiment,
	gate,
	loopOptions,
	loopState,
	pr,
	review,
	task,
	view,
} from "./orchestrator_helpers.ts";

test("UNINITIALIZED bootstraps and the action carries a state-changing target", () => {
	const action = nextAction(loopState({ state: "UNINITIALIZED" }), view(), loopOptions());
	assert.equal(action.kind, "boot");
	assert.equal(action.to, "SELECT_TASK");
});

test("BOOT re-runs only when BOOT did not complete", () => {
	const action = nextAction(loopState({ state: "BOOT" }), view(), loopOptions());
	assert.equal(action.kind, "boot");
	assert.equal(action.to, "SELECT_TASK");
});

test("STOP flag pauses before any other rule, with the resume command", () => {
	const action = nextAction(loopState({ state: "WORK" }), view({ stopFlag: true, tasks: [task()] }), loopOptions());
	assert.equal(action.kind, "pause");
	assert.match(action.reason ?? "", /STOP flag/);
	assert.match(action.reason ?? "", /cli\.ts resume/);
});

test("DONE is terminal and wins over the STOP flag", () => {
	const action = nextAction(loopState({ state: "DONE" }), view({ stopFlag: true }), loopOptions());
	assert.equal(action.kind, "done");
});

test("PAUSED and BLOCKED return no-action with the recorded reason", () => {
	const paused = nextAction(
		loopState({ state: "PAUSED", paused_from: "WORK", reason: "budget cap" }),
		view(),
		loopOptions(),
	);
	assert.equal(paused.kind, "none");
	assert.match(paused.summary, /PAUSED/);
	const blocked = nextAction(loopState({ state: "BLOCKED", reason: "missing data" }), view(), loopOptions());
	assert.equal(blocked.kind, "none");
	assert.match(blocked.summary, /missing data/);
});

test("budget caps pause with the cap name, the values, and the resume command", () => {
	const caps = { maxSessionsPerDay: 2 };
	const action = nextAction(
		loopState({ state: "WORK" }),
		view({ tasks: [task()], budget: budget({ sessionsToday: 2 }) }),
		loopOptions({ budgets: { ...loopOptions().budgets, ...caps } }),
	);
	assert.equal(action.kind, "pause");
	assert.match(action.reason ?? "", /maxSessionsPerDay cap reached: today's child sessions \(2 ≥ 2\)/);
	assert.match(action.reason ?? "", /resume/);
});

test("SELECT_TASK with no work and mechanical readiness green enters READINESS", () => {
	const action = nextAction(loopState({ state: "SELECT_TASK" }), view(), loopOptions());
	assert.equal(action.kind, "bookkeep");
	assert.equal(action.to, "READINESS");
});

test("SELECT_TASK does not enter READINESS while a P0/P1 task or S0/S1 finding is open", () => {
	const p0 = nextAction(loopState({ state: "SELECT_TASK" }), view({ tasks: [task({ priority: "P0" })] }), loopOptions());
	assert.equal(p0.kind, "bookkeep");
	assert.equal(p0.to, "WORK");
	const s0 = nextAction(
		loopState({ state: "SELECT_TASK" }),
		view({ findings: [{ id: "F-001", severity: "S0", status: "verified" }] }),
		loopOptions(),
	);
	assert.equal(s0.kind, "spawn");
	assert.equal(s0.agent, "phd_candidate");
});

test("SELECT_TASK assigns the highest-priority open task and resets the work unit", () => {
	const action = nextAction(
		loopState({ state: "SELECT_TASK", attempt: 3 }),
		view({
			tasks: [
				task({ id: "T-002", priority: "P3" }),
				task({ id: "T-001", priority: "P1" }),
			],
		}),
		loopOptions(),
	);
	assert.equal(action.kind, "bookkeep");
	assert.equal(action.to, "WORK");
	assert.equal(action.task_id, "T-001");
	assert.deepEqual(action.effects, [
		{ type: "set_task", task_id: "T-001", attempt: 0 },
		{ type: "reset_failures" },
	]);
});

test("WORK without a task returns to SELECT_TASK", () => {
	const action = nextAction(loopState({ state: "WORK", task_id: null }), view(), loopOptions());
	assert.equal(action.kind, "bookkeep");
	assert.equal(action.to, "SELECT_TASK");
});

test("WORK with a blocked task is BLOCKED with the smallest decision", () => {
	const action = nextAction(
		loopState({ state: "WORK", task_id: "T-001" }),
		view({ tasks: [task({ status: "blocked" })] }),
		loopOptions(),
	);
	assert.equal(action.kind, "block");
	assert.match(action.question ?? "", /resolve the blocker/);
});

test("WORK with an open PR or review status moves to REVIEW", () => {
	const withPr = nextAction(
		loopState({ state: "WORK", task_id: "T-001" }),
		view({ tasks: [task({ status: "in_progress" })], prs: [pr()] }),
		loopOptions(),
	);
	assert.equal(withPr.kind, "bookkeep");
	assert.equal(withPr.to, "REVIEW");
	assert.equal(withPr.pr_id, "PR-001");
	const marked = nextAction(
		loopState({ state: "WORK", task_id: "T-001" }),
		view({ tasks: [task({ status: "review" })] }),
		loopOptions(),
	);
	assert.equal(marked.to, "REVIEW");
});

test("WORK with an active experiment waits for the collector", () => {
	const action = nextAction(
		loopState({ state: "WORK", task_id: "T-001" }),
		view({ tasks: [task({ status: "in_progress" })], experiments: [experiment()] }),
		loopOptions(),
	);
	assert.equal(action.kind, "bookkeep");
	assert.equal(action.to, "WAITING_HPC");
	assert.deepEqual(action.effects, [{ type: "set_experiment", experiment_id: "E-0001", attempt: 1 }]);
});

test("WORK spawns phd_candidate with a keyed, attempt-numbered action", () => {
	const before = nextAction(
		loopState({ state: "WORK", task_id: "T-001", attempt: 0 }),
		view({ tasks: [task({ status: "in_progress" })] }),
		loopOptions(),
	);
	assert.equal(before.kind, "spawn");
	assert.equal(before.agent, "phd_candidate");
	assert.equal(before.key, "work:T-001:1");
	assert.equal(before.attempt, 1);
	assert.match(before.brief, /T-001/);
	// After one completed work unit the key moves on: no replay of a finished action.
	const after = nextAction(
		loopState({ state: "WORK", task_id: "T-001", attempt: 2 }),
		view({
			tasks: [task({ status: "in_progress" })],
			events: [event({ action_key: "work:T-001:1", action_phase: "complete" })],
		}),
		loopOptions(),
	);
	assert.equal(after.key, "work:T-001:3");
	assert.equal(completions(after === null ? [] : [event({ action_key: "work:T-001:1", action_phase: "complete" })], "work:"), 1);
});

test("WAITING_HPC returns to WORK when no experiment is unfinished, else gives the collector a pass", () => {
	const back = nextAction(loopState({ state: "WAITING_HPC" }), view(), loopOptions());
	assert.equal(back.kind, "bookkeep");
	assert.equal(back.to, "WORK");
	const collect = nextAction(
		loopState({ state: "WAITING_HPC", task_id: "T-001" }),
		view({ experiments: [experiment({ status: "running", attempt: 2, job_id: "900001" })] }),
		loopOptions(),
	);
	assert.equal(collect.kind, "spawn");
	assert.equal(collect.agent, "collector");
	assert.equal(collect.key, "collect:E-0001:a2:running");
});

test("REVIEW decisions: accept → MERGE, changes → WORK, reject → SELECT_TASK, none → supervisor", () => {
	const base = { tasks: [task({ status: "review" })], prs: [pr({ status: "review" })] };
	const accept = nextAction(loopState({ state: "REVIEW", task_id: "T-001", pr_id: "PR-001" }), view({ ...base, reviews: [review()] }), loopOptions());
	assert.equal(accept.kind, "bookkeep");
	assert.equal(accept.to, "MERGE");
	assert.equal(accept.pr_id, "PR-001");

	const changes = nextAction(
		loopState({ state: "REVIEW", task_id: "T-001", pr_id: "PR-001" }),
		view({ ...base, reviews: [review({ decision: "changes_requested" })] }),
		loopOptions(),
	);
	assert.equal(changes.kind, "bookkeep");
	assert.equal(changes.to, "WORK");
	assert.ok(changes.effects.some((effect) => effect.type === "bump_attempt"));

	const reject = nextAction(
		loopState({ state: "REVIEW", task_id: "T-001", pr_id: "PR-001" }),
		view({ ...base, reviews: [review({ decision: "reject" })] }),
		loopOptions(),
	);
	assert.equal(reject.to, "SELECT_TASK");
	assert.ok(reject.effects.some((effect) => effect.type === "task_status" && effect.status === "rejected"));

	const noReview = nextAction(
		loopState({ state: "REVIEW", task_id: "T-001", pr_id: "PR-001" }),
		view(base),
		loopOptions(),
	);
	assert.equal(noReview.kind, "spawn");
	assert.equal(noReview.agent, "supervisor");
	assert.equal(noReview.key, "review:PR-001:1");

	const noPr = nextAction(loopState({ state: "REVIEW", task_id: "T-001" }), view({ tasks: [task({ status: "review" })] }), loopOptions());
	assert.equal(noPr.kind, "bookkeep");
	assert.equal(noPr.to, "WORK");
});

test("MERGE opens the human gate, waits on it, resumes after approval, and unblocks on rejection", () => {
	const state = loopState({ state: "MERGE", task_id: "T-001", pr_id: "PR-001" });
	const base = { tasks: [task({ status: "review" })], prs: [pr({ status: "review" })], reviews: [review()] };

	const open = nextAction(state, view(base), loopOptions({ requireHumanMerge: true }));
	assert.equal(open.kind, "gate");
	assert.equal(open.gate, "merge");
	assert.equal(open.pr_id, "PR-001");

	const pending = nextAction(state, view({ ...base, gates: [gate()] }), loopOptions({ requireHumanMerge: true }));
	assert.equal(pending.kind, "none");
	assert.match(pending.summary, /waiting for human gate G-001/);
	assert.equal(pending.waiting_on_gate, "G-001");

	const approved = nextAction(
		state,
		view({ ...base, gates: [gate({ status: "approved" })] }),
		loopOptions({ requireHumanMerge: true }),
	);
	assert.equal(approved.kind, "merge");
	assert.equal(approved.pr_id, "PR-001");
	assert.equal(approved.review_id, "RV-001");
	assert.equal(approved.to, "POST_MERGE");

	const rejected = nextAction(
		state,
		view({ ...base, gates: [gate({ status: "rejected", note: "tests were not re-run" })] }),
		loopOptions({ requireHumanMerge: true }),
	);
	assert.equal(rejected.kind, "bookkeep");
	assert.equal(rejected.to, "WORK");
	assert.match(rejected.reason, /tests were not re-run/);

	const noHumanGate = nextAction(state, view(base), loopOptions({ requireHumanMerge: false }));
	assert.equal(noHumanGate.kind, "merge");
	assert.equal(noHumanGate.to, "POST_MERGE");

	const paper = nextAction(
		state,
		view({ ...base, prs: [pr({ status: "review", kind: "paper" })] }),
		loopOptions({ requireHumanMerge: false }),
	);
	assert.equal(paper.kind, "merge");
	assert.equal(paper.to, "AUDIT");
});

test("MERGE without an accepted review returns to REVIEW", () => {
	const action = nextAction(
		loopState({ state: "MERGE", task_id: "T-001", pr_id: "PR-001" }),
		view({ tasks: [task({ status: "review" })], prs: [pr({ status: "review" })] }),
		loopOptions(),
	);
	assert.equal(action.kind, "bookkeep");
	assert.equal(action.to, "REVIEW");
});

test("POST_MERGE closes the task and returns to SELECT_TASK", () => {
	const action = nextAction(
		loopState({ state: "POST_MERGE", task_id: "T-001", pr_id: "PR-001" }),
		view({ tasks: [task({ status: "review" })], prs: [pr({ status: "merged" })] }),
		loopOptions(),
	);
	assert.equal(action.kind, "bookkeep");
	assert.equal(action.to, "SELECT_TASK");
	assert.deepEqual(action.effects, [
		{ type: "task_status", task_id: "T-001", status: "merged", evidence: "PR-001 merged into update_pymc at " + "a".repeat(40) },
		{ type: "clear_pr" },
	]);
});

test("READINESS passes to PAPER, fails to SELECT_TASK, otherwise spawns the supervisor", () => {
	const pass = nextAction(
		loopState({ state: "READINESS" }),
		view({ readiness: { pass: true, reason: "all criteria met" } }),
		loopOptions(),
	);
	assert.equal(pass.kind, "bookkeep");
	assert.equal(pass.to, "PAPER");
	const fail = nextAction(
		loopState({ state: "READINESS" }),
		view({ readiness: { pass: false, reason: "confirmation runs missing" } }),
		loopOptions(),
	);
	assert.equal(fail.kind, "bookkeep");
	assert.equal(fail.to, "SELECT_TASK");
	const spawn = nextAction(loopState({ state: "READINESS" }), view(), loopOptions());
	assert.equal(spawn.kind, "spawn");
	assert.equal(spawn.agent, "supervisor");
	assert.equal(spawn.key, "readiness:1");
});

test("PAPER spawns the writer, then supervises its PR, then merges and audits", () => {
	const write = nextAction(loopState({ state: "PAPER", attempt: 0 }), view(), loopOptions());
	assert.equal(write.kind, "spawn");
	assert.equal(write.agent, "phd_candidate");
	assert.equal(write.key, "paper:1");

	const paperPr = pr({ kind: "paper", task_id: "PAPER" });
	const supervise = nextAction(loopState({ state: "PAPER" }), view({ prs: [paperPr] }), loopOptions());
	assert.equal(supervise.kind, "bookkeep");
	assert.equal(supervise.to, "PAPER_REVIEW");
	assert.equal(supervise.pr_id, "PR-001");

	const accepted = nextAction(
		loopState({ state: "PAPER_REVIEW", pr_id: "PR-001" }),
		view({ prs: [paperPr], reviews: [review()] }),
		loopOptions(),
	);
	assert.equal(accepted.kind, "bookkeep");
	assert.equal(accepted.to, "MERGE");

	const changes = nextAction(
		loopState({ state: "PAPER_REVIEW", pr_id: "PR-001" }),
		view({ prs: [paperPr], reviews: [review({ decision: "changes_requested" })] }),
		loopOptions(),
	);
	assert.equal(changes.to, "PAPER");

	const spawn = nextAction(loopState({ state: "PAPER_REVIEW", pr_id: "PR-001" }), view({ prs: [paperPr] }), loopOptions());
	assert.equal(spawn.kind, "spawn");
	assert.equal(spawn.agent, "supervisor");
});

test("AUDIT stops at the cycle cap and spawns the agent that owns the current phase", () => {
	const capped = nextAction(
		loopState({ state: "AUDIT" }),
		view({ audit: { cycle: 6, phase: "find", complete: false } }),
		loopOptions({ budgets: { ...loopOptions().budgets, maxAuditCycles: 5 } }),
	);
	assert.equal(capped.kind, "block");
	assert.match(capped.reason, /maxAuditCycles=5/);
	// S6 contract: gates → find → verify → synthesise → adjudicate, each phase
	// spawned to the agent that owns it (the S4 contract spawned the finder for
	// every phase; that was the placeholder this stage replaces).
	const phases: Array<[string, string]> = [
		["find", "audit_finder"],
		["verify", "audit_verifier"],
		["synthesise", "audit_synthesizer"],
		["adjudicate", "audit_adjudicator"],
		["gates", "supervisor"],
	];
	for (const [phase, agent] of phases) {
		const running = nextAction(
			loopState({ state: "AUDIT" }),
			view({ audit: { cycle: 2, phase, complete: false } }),
			loopOptions(),
		);
		assert.equal(running.kind, "spawn", phase);
		assert.equal(running.agent, agent, phase);
		assert.equal(running.key, `audit:2:${phase}:1`);
	}
	const complete = nextAction(
		loopState({ state: "AUDIT" }),
		view({ audit: { cycle: 2, phase: "adjudicate", complete: true } }),
		loopOptions(),
	);
	assert.equal(complete.kind, "bookkeep");
	assert.equal(complete.to, "DECIDE");
});

test("DECIDE applies the decision rule: more work, DONE sign-off, or a waiting gate", () => {
	const more = nextAction(
		loopState({ state: "DECIDE" }),
		view({ decision: { done: false, grade: "weak reject", reason: "S1 open" } }),
		loopOptions(),
	);
	assert.equal(more.kind, "bookkeep");
	assert.equal(more.to, "SELECT_TASK");
	const spawn = nextAction(loopState({ state: "DECIDE" }), view(), loopOptions());
	assert.equal(spawn.kind, "spawn");
	assert.equal(spawn.agent, "supervisor");
	assert.equal(spawn.key, "decide:1");
	const openGate = nextAction(
		loopState({ state: "DECIDE" }),
		view({ decision: { done: true, grade: "accept" } }),
		loopOptions(),
	);
	assert.equal(openGate.kind, "gate");
	assert.equal(openGate.gate, "submission");
	const waiting = nextAction(
		loopState({ state: "DECIDE" }),
		view({ decision: { done: true, grade: "accept" }, gates: [gate({ gate: "submission", pr_id: undefined })] }),
		loopOptions(),
	);
	assert.equal(waiting.kind, "none");
	assert.match(waiting.summary, /submission/);
	const done = nextAction(
		loopState({ state: "DECIDE" }),
		view({
			decision: { done: true, grade: "accept" },
			gates: [gate({ gate: "submission", status: "approved", pr_id: undefined })],
		}),
		loopOptions(),
	);
	assert.equal(done.kind, "done");
	const rejected = nextAction(
		loopState({ state: "DECIDE" }),
		view({
			decision: { done: true, grade: "accept" },
			gates: [gate({ gate: "submission", status: "rejected", note: "not ready", pr_id: undefined })],
		}),
		loopOptions(),
	);
	assert.equal(rejected.kind, "bookkeep");
	assert.equal(rejected.to, "SELECT_TASK");
});

test("readinessCriteria reports the mechanical blockers it can check", () => {
	const clean = readinessCriteria(view());
	assert.equal(clean.green, true);
	const withP1 = readinessCriteria(view({ tasks: [task({ priority: "P1", status: "open" })] }));
	assert.equal(withP1.green, false);
	assert.match(withP1.blockers.join(" "), /T-001/);
	const waived = readinessCriteria(
		view({
			findings: [{ id: "F-001", severity: "S0", status: "verified" }],
			waivers: [{ gate: "finding:F-001", reason: "superseded by a later fix" }],
		}),
	);
	assert.equal(waived.green, true);
	const refuted = readinessCriteria(view({ findings: [{ id: "F-002", severity: "S1", status: "refuted" }] }));
	assert.equal(refuted.green, true);
});

test("nextAction is pure: same snapshot in, same action out, snapshot not mutated", () => {
	const snapshot = view({
		tasks: [task({ status: "in_progress" })],
		budget: budget({ sessionsToday: 1 }),
		events: [event({ action_key: "work:T-001:1", action_phase: "complete" })],
	});
	const before = JSON.stringify(snapshot);
	const first = nextAction(loopState({ state: "WORK", task_id: "T-001", attempt: 1 }), snapshot, loopOptions());
	const second = nextAction(loopState({ state: "WORK", task_id: "T-001", attempt: 1 }), snapshot, loopOptions());
	assert.deepEqual(first, second);
	assert.equal(JSON.stringify(snapshot), before);
	assert.equal(describeAction(first), `spawn state=WORK key=work:T-001:2 agent=phd_candidate attempt=2`);
});

test("every rule has a distinct id and the table ends with the fallback", async () => {
	const { RULES } = await import("../../extensions/orchestrator/state-machine.ts");
	const ids = RULES.map((rule) => rule.id);
	assert.equal(new Set(ids).size, ids.length);
	assert.equal(ids[ids.length - 1], "fallback");
	// One rule per state that can act, so a new state is a visible table change.
	for (const state of ["SELECT_TASK", "WORK", "WAITING_HPC", "REVIEW", "MERGE", "POST_MERGE", "READINESS", "PAPER", "PAPER_REVIEW", "AUDIT", "DECIDE"]) {
		assert.ok(ids.some((id) => id.startsWith(state.toLowerCase().replace(/_/g, "-")) || id.includes(state.toLowerCase())), `no rule for ${state}`);
	}
});
