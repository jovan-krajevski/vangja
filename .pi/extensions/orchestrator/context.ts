/**
 * Deterministic `state/CONTEXT.md`, the derived Markdown views, and the text
 * `status` report.
 *
 * The views are a pure function of the ledgers (same renderers as the ledger
 * store), so `CONTEXT.md` after any number of runs is byte-identical to what a
 * fresh process would render. `status` adds exactly the facts the operations
 * runbook asks for: state, current task/PR/experiment + attempt, budget, open
 * human gates, the last 5 events, and the action a `step` would perform.
 */

import * as fs from "node:fs";
import { writeFileAtomic } from "../../lib/io.ts";
import { statePath } from "../../lib/paths.ts";
import { loadBudget } from "./budget.ts";
import { readLedger } from "../ledger/store.ts";
import {
	renderContextMd,
	renderExperimentsMd,
	renderPrsMd,
	renderReviewsMd,
	renderTasksMd,
} from "../../lib/views.ts";
import { childLockPath, loadLedgerView, readChildLock, type ChildLock } from "./ledgers.ts";
import {
	describeAction,
	nextAction,
	type Action,
	type LedgerView,
	type LoopOptions,
} from "./state-machine.ts";
import { loadLoopState, type LoopState } from "../../lib/state.ts";

/** Rewrite every derived view (called after state changes, never inside a lock). */
export function refreshViews(root: string): void {
	const write = (name: string, content: string): void => writeFileAtomic(statePath(root, name), content);
	write("TASKS.md", renderTasksMd(readLedger(root, "task")));
	write("EXPERIMENTS.md", renderExperimentsMd(readLedger(root, "experiment")));
	write("REVIEWS.md", renderReviewsMd(readLedger(root, "review")));
	write("PRS.md", renderPrsMd(readLedger(root, "pr")));
	const loaded = loadLoopState(root);
	write(
		"CONTEXT.md",
		renderContextMd({
			state: loaded.state as unknown as Record<string, unknown>,
			budget: loadBudget(root) as unknown as Record<string, unknown>,
			tasks: readLedger(root, "task"),
			experiments: readLedger(root, "experiment"),
			reviews: readLedger(root, "review"),
			prs: readLedger(root, "pr"),
			gates: readLedger(root, "gate"),
			events: readLedger(root, "event"),
		}),
	);
}

export interface StatusReport {
	state: string;
	paused_from: string | null;
	run_id: string | null;
	task_id: string | null;
	pr_id: string | null;
	experiment_id: string | null;
	attempt: number;
	next_gate: string | null;
	reason: string | null;
	updated_at: string;
	stop_flag: boolean;
	budget: Record<string, number>;
	caps: Record<string, number>;
	open_gates: Array<{ id: string; gate: string; note: string }>;
	last_events: Array<{ ts: string; actor: string; from?: string; to?: string; summary: string }>;
	next_action: { kind: string; key: string; summary: string; agent?: string };
	child: ChildLock | null;
}

export function childLockFor(root: string): ChildLock | null {
	return fs.existsSync(childLockPath(root)) ? readChildLock(root) : null;
}

export function composeStatus(root: string, options: LoopOptions): { report: StatusReport; view: LedgerView; action: Action; state: LoopState } {
	const loaded = loadLoopState(root);
	const state = loaded.state;
	const view = loadLedgerView(root, { budget: loadBudget(root) });
	const action = nextAction(state, view, options);
	const budget = view.budget;
	const report: StatusReport = {
		state: state.state,
		paused_from: state.paused_from,
		run_id: state.run_id,
		task_id: state.task_id,
		pr_id: state.pr_id,
		experiment_id: state.experiment_id,
		attempt: state.attempt,
		next_gate: state.next_gate,
		reason: state.reason,
		updated_at: state.updated_at,
		stop_flag: view.stopFlag,
		budget: {
			sessionsToday: budget.sessionsToday,
			sessionsTotal: budget.sessionsTotal,
			wallClockMinutes: budget.wallClockMinutes,
			costUsd: budget.costUsd,
			consecutiveFailures: budget.consecutiveFailures,
			noProgressCycles: budget.noProgressCycles,
			auditCycles: budget.auditCycles,
		},
		caps: { ...options.budgets },
		open_gates: view.gates
			.filter((gate) => gate.status === "pending")
			.map((gate) => ({ id: gate.id, gate: gate.gate, note: gate.note })),
		last_events: [...view.events]
			.sort((a, b) => a.seq - b.seq)
			.slice(-5)
			.map((event) => ({ ts: event.ts, actor: event.actor, from: event.from, to: event.to, summary: event.summary })),
		next_action: action.kind === "spawn"
			? { kind: action.kind, key: action.key, summary: action.summary, agent: action.agent }
			: { kind: action.kind, key: action.key, summary: action.summary },
		child: childLockFor(root),
	};
	return { report, view, action, state };
}

/** Exactly the runbook's order; `--json` prints `report` instead. */
export function renderStatus(report: StatusReport): string[] {
	const lines: string[] = [];
	lines.push(`state: ${report.state}${report.paused_from ? ` (paused from ${report.paused_from})` : ""}`);
	lines.push(`run: ${report.run_id ?? "none"}`);
	lines.push(
		`task: ${report.task_id ?? "none"} · pr: ${report.pr_id ?? "none"} · experiment: ${report.experiment_id ?? "none"} (attempt ${report.attempt})`,
	);
	lines.push(`next gate: ${report.next_gate ?? "none"}${report.reason ? ` · reason: ${report.reason}` : ""}`);
	lines.push(
		`budget: sessions ${report.budget.sessionsToday}/${report.caps.maxSessionsPerDay} today, ` +
			`${report.budget.wallClockMinutes}/${report.caps.maxTaskWallClockMinutes} min, ` +
			`$${report.budget.costUsd.toFixed(2)}/${report.caps.maxCostUsdPerRun}, ` +
			`failures ${report.budget.consecutiveFailures}/${report.caps.maxConsecutiveFailures}, ` +
			`no-progress ${report.budget.noProgressCycles}/${report.caps.maxNoProgressCycles}, ` +
			`audit cycles ${report.budget.auditCycles}/${report.caps.maxAuditCycles}`,
	);
	lines.push(`stop flag: ${report.stop_flag ? "SET (.pi/state/STOP)" : "clear"}`);
	if (report.child) {
		lines.push(
			`child: pid ${report.child.pid} ${report.child.alive ? "alive" : "STALE (dead pid; the next step will re-run)"} ` +
				`key=${report.child.action_key}`,
		);
	}
	lines.push(`open human gates: ${report.open_gates.length}`);
	for (const gate of report.open_gates) lines.push(`  - ${gate.id} ${gate.gate}: ${gate.note}`);
	lines.push("last 5 events:");
	for (const event of report.last_events) {
		lines.push(`  - ${event.ts} ${event.actor} ${event.from ?? ""}${event.from || event.to ? "→" : ""}${event.to ?? ""}: ${event.summary}`);
	}
	lines.push(`nextAction: ${report.next_action.kind} — ${report.next_action.summary}`);
	return lines;
}

/** One-line form for the TUI status widget and the `start` loop. */
export function renderStatusLine(report: StatusReport): string {
	return `LOOP ${report.state} task=${report.task_id ?? "-"} pr=${report.pr_id ?? "-"} attempt=${report.attempt} gate=${report.next_gate ?? "-"} next=${report.next_action.kind}`;
}

export { describeAction };
