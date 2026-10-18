/**
 * LOOP_STATE.json — the one mutable state file.
 *
 * Written atomically (temp + fsync + rename) under the same lock as the ledgers.
 * Transitions are validated against `plan/02_ARCHITECTURE.md` §2 so a state can
 * only move where the architecture says it can.
 */

import * as fs from "node:fs";
import { VangjaLoopError } from "./errors.ts";
import { readJsonFile, writeJsonAtomic, withLock } from "./io.ts";
import { statePath } from "./paths.ts";

export const LOOP_STATES = [
	"UNINITIALIZED",
	"BOOT",
	"SELECT_TASK",
	"WORK",
	"WAITING_HPC",
	"REVIEW",
	"MERGE",
	"POST_MERGE",
	"READINESS",
	"PAPER",
	"PAPER_REVIEW",
	"AUDIT",
	"DECIDE",
	"PAUSED",
	"BLOCKED",
	"DONE",
] as const;

export type LoopStateName = (typeof LOOP_STATES)[number];

const TRANSITIONS: Record<LoopStateName, LoopStateName[]> = {
	UNINITIALIZED: ["BOOT", "SELECT_TASK", "BLOCKED", "PAUSED", "DONE"],
	BOOT: ["SELECT_TASK", "BLOCKED", "PAUSED", "DONE"],
	SELECT_TASK: ["WORK", "READINESS", "BLOCKED", "PAUSED", "DONE"],
	WORK: ["REVIEW", "WAITING_HPC", "BLOCKED", "PAUSED", "DONE"],
	WAITING_HPC: ["WORK", "BLOCKED", "PAUSED", "DONE"],
	REVIEW: ["WORK", "MERGE", "SELECT_TASK", "BLOCKED", "PAUSED", "DONE"],
	// MERGE → WORK is the human-gate rejection path: plan/09_OPERATIONS_RUNBOOK.md §3
	// ("rejecting a gate sends the loop back with your note attached to the task; it
	// does not stop the run") needs a legal edge back to WORK, otherwise a rejected
	// merge gate deadlocks the loop.
	MERGE: ["POST_MERGE", "AUDIT", "WORK", "BLOCKED", "PAUSED", "DONE"],
	POST_MERGE: ["SELECT_TASK", "READINESS", "BLOCKED", "PAUSED", "DONE"],
	READINESS: ["PAPER", "SELECT_TASK", "BLOCKED", "PAUSED", "DONE"],
	PAPER: ["PAPER_REVIEW", "BLOCKED", "PAUSED", "DONE"],
	PAPER_REVIEW: ["MERGE", "PAPER", "BLOCKED", "PAUSED", "DONE"],
	AUDIT: ["DECIDE", "BLOCKED", "PAUSED", "DONE"],
	DECIDE: ["SELECT_TASK", "DONE", "BLOCKED", "PAUSED"],
	PAUSED: [...LOOP_STATES],
	BLOCKED: ["SELECT_TASK", "WORK", "PAUSED", "DONE"],
	DONE: [],
};

export interface LoopState {
	schema: "pi-loop-state/1";
	run_id: string | null;
	/** Branch this run merges into; pinned at BOOT (see resolveBaseBranch). */
	base_branch: string | null;
	state: LoopStateName;
	paused_from: LoopStateName | null;
	task_id: string | null;
	pr_id: string | null;
	experiment_id: string | null;
	attempt: number;
	next_gate: string | null;
	reason: string | null;
	updated_at: string;
	updated_by: string;
}

export function defaultLoopState(): LoopState {
	return {
		schema: "pi-loop-state/1",
		run_id: null,
		base_branch: null,
		state: "UNINITIALIZED",
		paused_from: null,
		task_id: null,
		pr_id: null,
		experiment_id: null,
		attempt: 0,
		next_gate: null,
		reason: null,
		updated_at: "",
		updated_by: "",
	};
}

export function loopStatePath(root: string): string {
	return statePath(root, "LOOP_STATE.json");
}

export function isLoopStateName(value: string): value is LoopStateName {
	return (LOOP_STATES as readonly string[]).includes(value);
}

export function assertTransition(
	from: LoopStateName,
	to: LoopStateName,
	pausedFrom: LoopStateName | null,
): void {
	if (from === to) return; // idempotent re-write of the same state is allowed
	if (from === "PAUSED") {
		if (pausedFrom === null) {
			throw new VangjaLoopError(
				"STATE_INVALID",
				`LOOP_STATE.json is PAUSED without "paused_from"; refusing to resume to ${to}. ` +
					`Fix .pi/state/LOOP_STATE.json (or rerun the transition that paused it) — do not guess the resume state.`,
			);
		}
		if (to !== pausedFrom) {
			throw new VangjaLoopError(
				"STATE_INVALID",
				`resume must return to the state recorded in paused_from (${pausedFrom}), not ${to}`,
			);
		}
		return;
	}
	if (!TRANSITIONS[from].includes(to)) {
		throw new VangjaLoopError(
			"STATE_INVALID",
			`illegal transition ${from} → ${to} (plan/02_ARCHITECTURE.md §2). ` +
				`Allowed from ${from}: ${TRANSITIONS[from].join(", ") || "none"}`,
		);
	}
}

export interface LoadedState {
	state: LoopState;
	initialized: boolean;
	path: string;
}

export function loadLoopState(root: string): LoadedState {
	const file = loopStatePath(root);
	const raw = readJsonFile<Record<string, unknown>>(file);
	if (!raw) return { state: defaultLoopState(), initialized: false, path: file };
	if (raw.schema !== "pi-loop-state/1") {
		throw new VangjaLoopError("STATE_INVALID", `${file}: expected schema "pi-loop-state/1", got ${String(raw.schema)}`);
	}
	if (typeof raw.state !== "string" || !isLoopStateName(raw.state)) {
		throw new VangjaLoopError("STATE_INVALID", `${file}: unknown state ${String(raw.state)}`);
	}
	const state = { ...defaultLoopState(), ...(raw as unknown as LoopState) };
	return { state, initialized: true, path: file };
}

export interface SaveStateOptions {
	actor: string;
	/** When true, transition validation is skipped (used to repair a corrupt file). */
	force?: boolean;
}

/**
 * Merge `patch` into the current state and write it atomically. Validates the
 * transition unless `force` is set.
 */
export function saveLoopState(root: string, patch: Partial<LoopState>, options: SaveStateOptions): LoopState {
	const file = loopStatePath(root);
	const lockPath = `${file}.lock`;
	return withLock(lockPath, () => {
		const current = loadLoopState(root).state;
		const next: LoopState = {
			...current,
			...patch,
			schema: "pi-loop-state/1",
			updated_at: new Date().toISOString(),
			updated_by: options.actor,
		};
		if (next.state === "PAUSED" && next.paused_from === null && current.state !== "PAUSED") {
			next.paused_from = current.state;
		}
		if (next.state !== "PAUSED" && next.paused_from !== null && current.state === "PAUSED") {
			next.paused_from = null;
		}
		if (!options.force) {
			assertTransition(current.state, next.state, current.state === "PAUSED" ? current.paused_from : null);
		}
		writeJsonAtomic(file, next);
		return next;
	});
}

export function isStopFlagSet(root: string): boolean {
	return fs.existsSync(statePath(root, "STOP"));
}

export function stopFlagPath(root: string): string {
	return statePath(root, "STOP");
}
