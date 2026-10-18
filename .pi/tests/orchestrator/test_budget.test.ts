/**
 * Budget accounting and caps (plan/02_ARCHITECTURE.md §6).
 *
 * Pure tests for the counters and the five violation messages, plus one sandbox
 * check that a cap actually writes PAUSED with the documented resume command.
 */

import { strict as assert } from "node:assert";
import { test } from "node:test";
import * as fs from "node:fs";
import * as path from "node:path";
import {
	budgetViolations,
	emptyBudget,
	ledgerDigest,
	loadBudget,
	recordOutcome,
	recordSession,
	resumeCommand,
	saveBudget,
	utcDayKey,
} from "../../extensions/orchestrator/budget.ts";
import { nextAction } from "../../extensions/orchestrator/state-machine.ts";
import { DEFAULT_BUDGETS } from "../../lib/config.ts";
import { createSandbox, loopOptions, loopState, readState, runCli, view } from "./orchestrator_helpers.ts";

test("an empty budget exceeds nothing", () => {
	assert.deepEqual(budgetViolations(emptyBudget(), DEFAULT_BUDGETS), []);
});

test("each cap produces exactly one violation with its cap name and values", () => {
	const cases: Array<[Partial<ReturnType<typeof emptyBudget>>, keyof typeof DEFAULT_BUDGETS, string]> = [
		[{ sessionsToday: 120 }, "maxSessionsPerDay", "today's child sessions"],
		[{ wallClockMinutes: 360 }, "maxTaskWallClockMinutes", "wall-clock minutes"],
		[{ costUsd: 250 }, "maxCostUsdPerRun", "cost in USD"],
		[{ consecutiveFailures: 3 }, "maxConsecutiveFailures", "consecutive child failures"],
		[{ noProgressCycles: 2 }, "maxNoProgressCycles", "changed no ledger"],
		[{ auditCycles: 6 }, "maxAuditCycles", "audit cycles already run"],
	];
	for (const [patch, cap, detail] of cases) {
		const violations = budgetViolations({ ...emptyBudget(), ...patch }, DEFAULT_BUDGETS);
		assert.equal(violations.length, 1, `${cap}: expected one violation, got ${JSON.stringify(violations)}`);
		assert.equal(violations[0].cap, cap);
		assert.equal(violations[0].limit, DEFAULT_BUDGETS[cap]);
		assert.match(violations[0].reason, new RegExp(`${cap} cap reached`));
		assert.match(violations[0].reason, new RegExp(detail.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
	}
});

test("the pause reason carries the exact resume command", () => {
	const action = nextAction(
		loopState({ state: "WORK" }),
		view({ budget: { ...emptyBudget(), costUsd: 300 } }),
		loopOptions({ budgets: DEFAULT_BUDGETS, sandbox: "/tmp/scratch" }),
	);
	assert.equal(action.kind, "pause");
	assert.ok(action.reason?.includes(resumeCommand("/tmp/scratch")), action.reason);
	assert.equal(resumeCommand("/tmp/scratch"), "node .pi/extensions/orchestrator/cli.ts resume --sandbox /tmp/scratch");
});

test("recordSession accumulates sessions, wall clock, cost, and the progress digest", () => {
	const sandbox = createSandbox("budget-accounting");
	try {
		saveBudget(sandbox.root, { sessionsTotal: 0 }, "test");
		const digestA = ledgerDigest(sandbox.root);
		const after = recordSession(
			sandbox.root,
			{ usage: { cost: 0.25 }, durationMs: 30_000, ledgerDigest: digestA, reason: "first child" },
			"test",
		);
		assert.equal(after.sessionsToday, 1);
		assert.equal(after.sessionsTotal, 1);
		assert.equal(after.wallClockMinutes, 0.5);
		assert.equal(after.costUsd, 0.25);
		// Same digest twice → one no-progress cycle.
		const again = recordSession(
			sandbox.root,
			{ usage: { cost: 0.25 }, durationMs: 30_000, ledgerDigest: digestA, reason: "second child" },
			"test",
		);
		assert.equal(again.noProgressCycles, 1);
		// A changed digest resets the counter.
		fs.writeFileSync(path.join(sandbox.root, ".pi", "state", "EVENTS.jsonl"), '{"seq":1}\n');
		const progress = recordSession(
			sandbox.root,
			{ usage: { cost: 0 }, durationMs: 1000, ledgerDigest: ledgerDigest(sandbox.root), reason: "third child" },
			"test",
		);
		assert.equal(progress.noProgressCycles, 0);
	} finally {
		sandbox.cleanup();
	}
});

test("recordOutcome extends and resets the consecutive-failure streak", () => {
	const sandbox = createSandbox("budget-streak");
	try {
		recordOutcome(sandbox.root, false, "first failure", "test");
		recordOutcome(sandbox.root, false, "second failure", "test");
		assert.equal(loadBudget(sandbox.root).consecutiveFailures, 2);
		recordOutcome(sandbox.root, true, "recovered", "test");
		assert.equal(loadBudget(sandbox.root).consecutiveFailures, 0);
	} finally {
		sandbox.cleanup();
	}
});

test("the daily session counter rolls over at the UTC day boundary", () => {
	const sandbox = createSandbox("budget-day");
	try {
		saveBudget(sandbox.root, { sessionsToday: 40 }, "test");
		const file = path.join(sandbox.root, ".pi", "state", "BUDGET.json");
		const raw = JSON.parse(fs.readFileSync(file, "utf8")) as Record<string, unknown>;
		raw.dayKey = "2020-01-01";
		fs.writeFileSync(file, `${JSON.stringify(raw, null, 2)}\n`);
		const loaded = loadBudget(sandbox.root);
		assert.equal(loaded.sessionsToday, 0);
		assert.equal(loaded.dayKey, utcDayKey());
	} finally {
		sandbox.cleanup();
	}
});

test("a cap pauses the loop in a sandbox, and resume returns to the recorded state", () => {
	const sandbox = createSandbox("budget-pause");
	try {
		saveBudget(sandbox.root, { costUsd: DEFAULT_BUDGETS.maxCostUsdPerRun }, "test");
		const step = runCli(sandbox.root, ["step"]);
		assert.equal(step.status, 0, step.stderr);
		assert.match(step.stdout, /paused: maxCostUsdPerRun cap reached/);
		assert.match(step.stdout, /cli\.ts resume/);
		const state = readState(sandbox.root);
		assert.equal(state.state, "PAUSED");
		assert.equal(state.paused_from, "UNINITIALIZED");
		// Resume puts the loop back where it was; the cap is unchanged, so the next
		// step pauses again (the human must raise the cap deliberately).
		const resume = runCli(sandbox.root, ["resume"]);
		assert.equal(resume.status, 0, resume.stderr);
		assert.match(resume.stdout, /resumed to UNINITIALIZED/);
		assert.equal(readState(sandbox.root).state, "UNINITIALIZED");
		const pauseAgain = runCli(sandbox.root, ["step"]);
		assert.match(pauseAgain.stdout, /paused: maxCostUsdPerRun cap reached/);
		assert.equal(readState(sandbox.root).state, "PAUSED");
	} finally {
		sandbox.cleanup();
	}
});
