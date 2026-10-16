/**
 * Status composition for the TUI widget and the `renderResult` of ledger tools.
 * Pure of pi imports so `node` can test it.
 */

import { readJsonFile } from "../../lib/io.ts";
import { projectLatest, ACTIVE_EXPERIMENT_STATUSES } from "../../lib/records.ts";
import { loadBudgetConfig, type BudgetConfig } from "../../lib/config.ts";
import { loadLoopState, isStopFlagSet } from "../../lib/state.ts";
import { statePath } from "../../lib/paths.ts";
import { readLedger } from "../ledger/handlers.ts";

export interface StatusSnapshot {
	initialized: boolean;
	state: string;
	pausedFrom: string | null;
	runId: string | null;
	taskId: string | null;
	prId: string | null;
	experimentId: string | null;
	attempt: number;
	nextGate: string | null;
	reason: string | null;
	updatedAt: string;
	openTasks: number;
	totalTasks: number;
	activeExperiments: Array<{ id: string; status: string; attempt: number }>;
	pendingGates: Array<{ id: string; gate: string; note: string }>;
	stopFlag: boolean;
	budget: { sessions: number; wallClockMinutes: number; costUsd: number } | null;
	caps: BudgetConfig | null;
}

export function readStatus(root: string): StatusSnapshot {
	const loaded = loadLoopState(root);
	const tasks = projectLatest(readLedger(root, "task") as Array<{ id?: string; seq: number }>);
	const openTasks = tasks.filter((task) => !["merged", "closed", "rejected"].includes(String(task.status)));
	const experiments = readLedger(root, "experiment");
	const active = projectLatest(experiments as Array<{ id?: string; seq: number }>)
		.filter((experiment) => (ACTIVE_EXPERIMENT_STATUSES as readonly string[]).includes(String(experiment.status)))
		.map((experiment) => ({
			id: String(experiment.id),
			status: String(experiment.status),
			attempt: Number(experiment.attempt ?? 0),
		}));
	const gates = projectLatest(readLedger(root, "gate") as Array<{ id?: string; seq: number }>)
		.filter((gate) => gate.status === "pending")
		.map((gate) => ({ id: String(gate.id), gate: String(gate.gate), note: String(gate.note ?? "") }));
	const budget = readJsonFile<Record<string, number>>(statePath(root, "BUDGET.json"));
	let caps: BudgetConfig | null = null;
	try {
		caps = loadBudgetConfig(root);
	} catch {
		caps = null;
	}
	return {
		initialized: loaded.initialized,
		state: loaded.state.state,
		pausedFrom: loaded.state.paused_from,
		runId: loaded.state.run_id,
		taskId: loaded.state.task_id,
		prId: loaded.state.pr_id,
		experimentId: loaded.state.experiment_id,
		attempt: loaded.state.attempt,
		nextGate: loaded.state.next_gate,
		reason: loaded.state.reason,
		updatedAt: loaded.state.updated_at,
		openTasks: openTasks.length,
		totalTasks: tasks.length,
		activeExperiments: active,
		pendingGates: gates,
		stopFlag: isStopFlagSet(root),
		budget: budget
			? {
					sessions: Number(budget.sessions ?? 0),
					wallClockMinutes: Number(budget.wallClockMinutes ?? 0),
					costUsd: Number(budget.costUsd ?? 0),
				}
			: null,
		caps,
	};
}

/** One line for `ctx.ui.setStatus`. */
export function composeStatusLine(snapshot: StatusSnapshot): string {
	if (!snapshot.initialized) return "LOOP UNINITIALIZED — orchestrator arrives in S4";
	const experiment = snapshot.experimentId ? ` exp=${snapshot.experimentId}#${snapshot.attempt}` : "";
	const gate = snapshot.nextGate ? ` gate=${snapshot.nextGate}` : "";
	return `LOOP ${snapshot.state}${snapshot.pausedFrom ? `(from ${snapshot.pausedFrom})` : ""} task=${snapshot.taskId ?? "none"}${experiment}${gate}${snapshot.stopFlag ? " STOP" : ""}`;
}

/** Widget lines (2–4 lines; compact by design). */
export function composeStatusLines(snapshot: StatusSnapshot): string[] {
	if (!snapshot.initialized) {
		return [
			"LOOP: UNINITIALIZED — no .pi/state/LOOP_STATE.json yet (the orchestrator arrives in S4)",
			`tasks: ${snapshot.totalTasks} · experiments: ${snapshot.activeExperiments.length} · gates: ${snapshot.pendingGates.length}`,
		];
	}
	const lines: string[] = [];
	lines.push(composeStatusLine(snapshot));
	if (snapshot.budget && snapshot.caps) {
		lines.push(
			`budget: sessions ${snapshot.budget.sessions}/${snapshot.caps.maxSessionsPerDay} · ` +
				`wall ${snapshot.budget.wallClockMinutes}/${snapshot.caps.maxTaskWallClockMinutes}min · ` +
				`$${snapshot.budget.costUsd.toFixed(2)}/${snapshot.caps.maxCostUsdPerRun}`,
		);
	} else {
		lines.push("budget: no BUDGET.json yet (accounting arrives with S4)");
	}
	const active = snapshot.activeExperiments.map((job) => `${job.id}(${job.status}#${job.attempt})`).join(", ");
	lines.push(
		`tasks ${snapshot.openTasks}/${snapshot.totalTasks} open · experiments active: ${active || "none"} · ` +
			`gates pending: ${snapshot.pendingGates.map((gate) => gate.id).join(", ") || "none"} · STOP: ${snapshot.stopFlag ? "yes" : "no"}`,
	);
	if (snapshot.reason) lines.push(`note: ${snapshot.reason}`);
	return lines;
}

/** Short summary used by the ledger `renderResult`. */
export function firstLine(text: string): string {
	const line = text.split("\n")[0] ?? "";
	return line.length > 160 ? `${line.slice(0, 157)}...` : line;
}
