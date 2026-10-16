/**
 * `merge_pr` — the S3 stub of the conductor's merge gate.
 *
 * The real orchestrator (S4) owns worktrees, the pre-merge gate, and
 * `git merge --no-ff`; this stub accepts a supervisor/conductor call, records
 * it in the event ledger, and returns `merged: false`. It must not touch git.
 */

import { appendRecord, checkCapability, fail, ok, requireString, type LedgerContext, type ToolTextResult } from "../ledger/store.ts";

export interface MergePrArgs {
	pr_id: string;
	task_id?: string;
	reason?: string;
}

export function handleMergePrStub(args: Record<string, unknown>, ctx: LedgerContext): ToolTextResult {
	try {
		checkCapability(ctx, "merge_pr");
		const prId = requireString(args, "pr_id", "the PR record id (PR-###) that the conductor should merge");
		const taskId = typeof args.task_id === "string" && args.task_id !== "" ? args.task_id : undefined;
		const reason = typeof args.reason === "string" && args.reason !== "" ? args.reason : "supervisor merge recommendation";
		const event = appendRecord(
			ctx.root,
			"event",
			{
				summary: `merge_pr STUB accepted for ${prId} (no merge performed; orchestrator arrives in S4)`,
				pr_id: prId,
				task_id: taskId,
				merge_stub: true,
				merged: false,
				reason,
			},
			ctx,
		);
		return ok(
			`STUB: merge_pr accepted for ${prId}. The orchestrator (S4) performs the real merge; this call recorded ` +
				`event ${String(event.id ?? "-")} and did not modify any branch.`,
			{ stub: true, merged: false, pr_id: prId, event_id: event.id, event_seq: event.seq },
		);
	} catch (error) {
		return fail((error as Error).message, { stub: true, merged: false });
	}
}
