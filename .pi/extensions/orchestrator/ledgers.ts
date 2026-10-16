/**
 * Read-only adapter from `.pi/state/**` on disk to the `LedgerView` the pure
 * state machine consumes. Contains no policy and writes nothing: every fact the
 * machine reacts to is loaded here, and anything missing is `null`/empty rather
 * than a guessed default.
 */

import * as fs from "node:fs";
import { readJsonFile } from "../../lib/io.ts";
import { statePath } from "../../lib/paths.ts";
import { readLedger } from "../ledger/store.ts";
import { loadBudget, type BudgetSnapshot } from "./budget.ts";
import {
	latestExperiments,
	latestGates,
	latestPrs,
	latestReviews,
	latestTasks,
	type AuditView,
	type DecisionView,
	type EventView,
	type FindingView,
	type LedgerView,
	type ReadinessView,
	type WaiverView,
} from "./state-machine.ts";

function latestFinding(finding: Record<string, unknown>): FindingView {
	return {
		id: String(finding.id ?? ""),
		severity: String(finding.severity ?? "S3"),
		status: String(finding.status ?? "to-verify"),
	};
}

function latestWaiver(waiver: Record<string, unknown>): WaiverView {
	return {
		id: typeof waiver.id === "string" ? waiver.id : undefined,
		gate: String(waiver.gate ?? ""),
		reason: String(waiver.reason ?? ""),
	};
}

function eventsFrom(records: Array<Record<string, unknown>>): EventView[] {
	return records.map((event) => ({
		seq: Number(event.seq ?? 0),
		ts: String(event.ts ?? ""),
		actor: String(event.actor ?? ""),
		summary: String(event.summary ?? ""),
		action_key: typeof event.action_key === "string" ? event.action_key : undefined,
		action_phase: typeof event.action_phase === "string" ? event.action_phase : undefined,
		exit_state: typeof event.exit_state === "string" ? event.exit_state : undefined,
		from: typeof event.from === "string" ? event.from : undefined,
		to: typeof event.to === "string" ? event.to : undefined,
	}));
}

export function loadAuditView(root: string): AuditView | null {
	const raw = readJsonFile<Record<string, unknown>>(statePath(root, "AUDIT.json"));
	if (!raw) return null;
	return {
		cycle: Number(raw.cycle ?? 1),
		complete: raw.complete === true,
		phase: String(raw.phase ?? "find"),
	};
}

export function loadReadinessView(root: string): ReadinessView | null {
	const raw = readJsonFile<Record<string, unknown>>(statePath(root, "READINESS.json"));
	if (!raw || typeof raw.pass !== "boolean") return null;
	return { pass: raw.pass, reason: String(raw.reason ?? "") };
}

export function loadDecisionView(root: string): DecisionView | null {
	const raw = readJsonFile<Record<string, unknown>>(statePath(root, "DECISION.json"));
	if (!raw || typeof raw.done !== "boolean") return null;
	return {
		done: raw.done,
		grade: typeof raw.grade === "string" ? raw.grade : undefined,
		reason: typeof raw.reason === "string" ? raw.reason : undefined,
	};
}

/** Detect a stale child lock: the pid is dead or the lock is unreadable. */
export interface ChildLock {
	pid: number;
	action_key: string;
	argv: string[];
	started_at: string;
	alive: boolean;
}

export function childLockPath(root: string): string {
	return statePath(root, "locks/child.json");
}

export function readChildLock(root: string): ChildLock | null {
	const raw = readJsonFile<Record<string, unknown>>(childLockPath(root));
	if (!raw) return null;
	const pid = Number(raw.pid ?? 0);
	let alive = false;
	if (pid > 0) {
		try {
			process.kill(pid, 0);
			alive = true;
		} catch {
			alive = false;
		}
	}
	return {
		pid,
		action_key: String(raw.action_key ?? ""),
		argv: Array.isArray(raw.argv) ? (raw.argv as string[]) : [],
		started_at: String(raw.started_at ?? ""),
		alive,
	};
}

export interface LoadLedgerOptions {
	/** Injected for deterministic tests; production reads `.pi/state/`. */
	budget?: BudgetSnapshot | null;
	stopFlag?: boolean;
	baseSha?: string | null;
	repoClean?: boolean;
}

export function loadLedgerView(root: string, options: LoadLedgerOptions = {}): LedgerView {
	return {
		tasks: latestTasks(readLedger(root, "task")),
		prs: latestPrs(readLedger(root, "pr")),
		reviews: latestReviews(readLedger(root, "review")),
		experiments: latestExperiments(readLedger(root, "experiment")),
		gates: latestGates(readLedger(root, "gate")),
		events: eventsFrom(readLedger(root, "event")),
		findings: readLedger(root, "finding").map(latestFinding),
		waivers: readLedger(root, "waiver").map(latestWaiver),
		budget: options.budget ?? loadBudget(root),
		readiness: loadReadinessView(root),
		decision: loadDecisionView(root),
		audit: loadAuditView(root),
		stopFlag: options.stopFlag ?? fs.existsSync(statePath(root, "STOP")),
		baseSha: options.baseSha ?? null,
		repoClean: options.repoClean ?? true,
	};
}

/** True when the last recorded start for `key` has no completion (a killed child). */
export function hasIncompleteStart(events: EventView[], key: string): boolean {
	let started = false;
	let completed = false;
	for (const event of events) {
		if (event.action_key !== key) continue;
		if (event.action_phase === "start") started = true;
		if (event.action_phase === "complete") completed = true;
	}
	return started && !completed;
}

export function isCompleted(events: EventView[], key: string): boolean {
	return events.some((event) => event.action_key === key && event.action_phase === "complete");
}
