/**
 * Budget accounting for the loop (plan/02_ARCHITECTURE.md §6).
 *
 * Two files, two jobs:
 * - `state/BUDGET.json` — the current snapshot (fast to read, rewritten atomically);
 * - `state/BUDGET.jsonl` — the append-only history, one record per child/outcome.
 *
 * Every counter has a cap in `.pi/config/budgets.json`. Exceeding a cap is not a
 * warning: the state machine turns it into `PAUSED` with the exact value, the cap,
 * and the command that resumes the loop after the human raises the cap.
 *
 * Where a counter is defined:
 * - sessions: one per completed child process, reset at the UTC day boundary;
 * - wallClockMinutes: summed child duration (per task; reset by `assign`/`WORK` start);
 * - costUsd: summed provider cost reported by the children;
 * - consecutiveFailures: children that exited non-ok, reset by any ok child;
 * - noProgressCycles: completed children after which the ledgers were unchanged;
 * - auditCycles: completed audit cycles (S6 drives the increments; capped here).
 */

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import { appendRecord } from "../ledger/store.ts";
import type { BudgetConfig } from "../../lib/config.ts";
import { readJsonFile, writeJsonAtomic, withLock } from "../../lib/io.ts";
import { ledgerPath, statePath } from "../../lib/paths.ts";
import { LEDGER_FILES, type LedgerKind } from "../../lib/records.ts";

export interface BudgetSnapshot {
	schema: "pi-budget/1";
	/** UTC date of the last write; `sessionsToday` resets when it changes. */
	dayKey: string;
	sessionsToday: number;
	sessionsTotal: number;
	wallClockMinutes: number;
	costUsd: number;
	consecutiveFailures: number;
	noProgressCycles: number;
	auditCycles: number;
	startedAt: string;
	updatedAt: string;
	/** SHA-256 over the ledger JSONL files after the last completed action. */
	lastLedgerDigest: string | null;
}

export function utcDayKey(now: Date = new Date()): string {
	return now.toISOString().slice(0, 10);
}

export function emptyBudget(): BudgetSnapshot {
	const now = new Date().toISOString();
	return {
		schema: "pi-budget/1",
		dayKey: utcDayKey(),
		sessionsToday: 0,
		sessionsTotal: 0,
		wallClockMinutes: 0,
		costUsd: 0,
		consecutiveFailures: 0,
		noProgressCycles: 0,
		auditCycles: 0,
		startedAt: now,
		updatedAt: now,
		lastLedgerDigest: null,
	};
}

export function budgetPath(root: string): string {
	return statePath(root, "BUDGET.json");
}

export function loadBudget(root: string): BudgetSnapshot {
	const raw = readJsonFile<Partial<BudgetSnapshot>>(budgetPath(root));
	const budget = { ...emptyBudget(), ...(raw ?? {}) };
	if (budget.dayKey !== utcDayKey()) {
		budget.sessionsToday = 0;
		budget.dayKey = utcDayKey();
	}
	return budget;
}

export interface BudgetPatch {
	sessionsToday?: number;
	sessionsTotal?: number;
	wallClockMinutes?: number;
	costUsd?: number;
	consecutiveFailures?: number;
	noProgressCycles?: number;
	auditCycles?: number;
	lastLedgerDigest?: string | null;
	reason?: string;
}

/** Merge `patch` into the snapshot under the state lock and rewrite it atomically. */
export function saveBudget(root: string, patch: BudgetPatch, actor = "conductor"): BudgetSnapshot {
	const file = budgetPath(root);
	return withLock(`${file}.lock`, () => {
		const current = loadBudget(root);
		const next: BudgetSnapshot = {
			...current,
			...patch,
			schema: "pi-budget/1",
			dayKey: utcDayKey(),
			updatedAt: new Date().toISOString(),
		};
		writeJsonAtomic(file, next);
		return next;
	});
}

export interface UsageLike {
	cost: number;
}

export interface SessionAccounting {
	usage?: UsageLike;
	durationMs: number;
	ledgerDigest?: string | null;
	reason?: string;
}

/** One completed child: sessions/day, wall clock, cost, and the progress digest. */
export function recordSession(root: string, accounting: SessionAccounting, actor = "conductor"): BudgetSnapshot {
	const current = loadBudget(root);
	const minutes = accounting.durationMs / 60_000;
	const progress =
		accounting.ledgerDigest !== undefined &&
		accounting.ledgerDigest !== null &&
		(current.lastLedgerDigest === null || accounting.ledgerDigest !== current.lastLedgerDigest);
	const patch: BudgetPatch = {
		sessionsToday: current.sessionsToday + 1,
		sessionsTotal: current.sessionsTotal + 1,
		wallClockMinutes: Number((current.wallClockMinutes + minutes).toFixed(3)),
		costUsd: Number((current.costUsd + (accounting.usage?.cost ?? 0)).toFixed(6)),
		noProgressCycles: progress ? 0 : current.noProgressCycles + 1,
		lastLedgerDigest: accounting.ledgerDigest ?? current.lastLedgerDigest,
		reason: accounting.reason ?? "session completed",
	};
	const next = saveBudget(root, patch, actor);
	appendRecord(
		root,
		"budget",
		{
			sessions: next.sessionsTotal,
			wallClockMinutes: next.wallClockMinutes,
			costUsd: next.costUsd,
			reason: accounting.reason ?? "session completed",
			sessionsToday: next.sessionsToday,
			consecutiveFailures: next.consecutiveFailures,
			noProgressCycles: next.noProgressCycles,
		},
		{ root, role: "conductor", actor },
	);
	return next;
}

/** A child that exited ok resets the failure streak; anything else extends it. */
export function recordOutcome(root: string, ok: boolean, reason: string, actor = "conductor"): BudgetSnapshot {
	const current = loadBudget(root);
	return saveBudget(
		root,
		{
			consecutiveFailures: ok ? 0 : current.consecutiveFailures + 1,
			reason,
		},
		actor,
	);
}

/** Compute the digest the no-progress counter compares. Never throws for absent files. */
export function ledgerDigest(root: string): string {
	const hash = crypto.createHash("sha256");
	for (const kind of Object.keys(LEDGER_FILES).sort() as LedgerKind[]) {
		const file = ledgerPath(root, LEDGER_FILES[kind]);
		hash.update(`${kind}:`);
		if (fs.existsSync(file)) hash.update(fs.readFileSync(file));
		hash.update("\n");
	}
	const stateFile = statePath(root, "LOOP_STATE.json");
	if (fs.existsSync(stateFile)) hash.update(fs.readFileSync(stateFile));
	return hash.digest("hex");
}

export interface BudgetViolation {
	cap: keyof BudgetConfig;
	limit: number;
	value: number;
	reason: string;
}

/** The exact resume command quoted in every PAUSED reason. */
export function resumeCommand(sandbox?: string): string {
	const sandboxArg = sandbox ? ` --sandbox ${sandbox}` : "";
	return `node .pi/extensions/orchestrator/cli.ts resume${sandboxArg}`;
}

function violation(
	cap: keyof BudgetConfig,
	limit: number,
	value: number,
	detail: string,
): BudgetViolation {
	return {
		cap,
		limit,
		value,
		reason: `${cap} cap reached: ${detail} (${value} ≥ ${limit}); the loop is paused, no further step will run`,
	};
}

/** Every cap this snapshot exceeds, in a fixed order (deterministic messages). */
export function budgetViolations(
	budget: BudgetSnapshot,
	caps: BudgetConfig,
	sandbox?: string,
): BudgetViolation[] {
	const violations: BudgetViolation[] = [];
	if (budget.sessionsToday >= caps.maxSessionsPerDay) {
		violations.push(
			violation("maxSessionsPerDay", caps.maxSessionsPerDay, budget.sessionsToday, "today's child sessions"),
		);
	}
	if (budget.wallClockMinutes >= caps.maxTaskWallClockMinutes) {
		violations.push(
			violation(
				"maxTaskWallClockMinutes",
				caps.maxTaskWallClockMinutes,
				Math.round(budget.wallClockMinutes),
				"wall-clock minutes spent on the current task",
			),
		);
	}
	if (budget.costUsd >= caps.maxCostUsdPerRun) {
		violations.push(violation("maxCostUsdPerRun", caps.maxCostUsdPerRun, budget.costUsd, "cost in USD for this run"));
	}
	if (budget.consecutiveFailures >= caps.maxConsecutiveFailures) {
		violations.push(
			violation(
				"maxConsecutiveFailures",
				caps.maxConsecutiveFailures,
				budget.consecutiveFailures,
				"consecutive child failures",
			),
		);
	}
	if (budget.noProgressCycles >= caps.maxNoProgressCycles) {
		violations.push(
			violation(
				"maxNoProgressCycles",
				caps.maxNoProgressCycles,
				budget.noProgressCycles,
				"completed actions that changed no ledger",
			),
		);
	}
	if (budget.auditCycles > caps.maxAuditCycles) {
		violations.push(
			violation("maxAuditCycles", caps.maxAuditCycles, budget.auditCycles, "audit cycles already run"),
		);
	}
	return violations;
}

export function budgetViolationReason(violation: BudgetViolation, sandbox?: string): string {
	return `${violation.reason}. Resume with: ${resumeCommand(sandbox)}`;
}
