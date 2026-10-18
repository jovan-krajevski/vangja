/**
 * `merge_pr` — the supervisor's request to the conductor (replaces the S3 stub).
 *
 * Semantics:
 * - only conductor/supervisor may call it (role matrix, enforced by the guard and
 *   re-checked here);
 * - the PR must have an accepted review record; anything else is refused with the
 *   reason and the ledger pointer;
 * - when `requireHumanMerge` is true and the merge gate has not been decided, the
 *   call opens the pending human gate and returns without merging;
 * - otherwise the conductor runs the pre-merge gate and performs
 *   `git merge --no-ff` through the same code path the MERGE state uses, so the
 *   tool and the loop cannot diverge.
 */

import { loadProjectConfig } from "../../lib/config.ts";
import { readJsonlFile } from "../../lib/io.ts";
import { statePath } from "../../lib/paths.ts";
import { loadLoopState, saveLoopState } from "../../lib/state.ts";
import { appendRecord, checkCapability, fail, ok, requireString, type LedgerContext, type ToolTextResult } from "../ledger/store.ts";
import { executeAction } from "./execute.ts";
import { openGate } from "./gates.ts";
import { loadLoopOptions } from "./commands.ts";
import { refreshViews } from "./context.ts";
import type { RunContext } from "./sandbox.ts";
import * as path from "node:path";

export interface MergePrArgs {
	pr_id?: unknown;
	task_id?: unknown;
	reason?: unknown;
}

/** Build a RunContext from the session environment (children run under PI_LOOP_ROOT). */
export function contextFromEnvironment(root: string, realRoot?: string, command?: string): RunContext {
	return {
		root,
		realRoot: realRoot ?? root,
		sandbox: realRoot && path.resolve(realRoot) !== path.resolve(root) ? root : null,
		dryRun: false,
		headless: true,
		command: command ?? process.env.PI_LOOP_COMMAND ?? "pi",
	};
}

export async function handleMergePr(args: MergePrArgs, ctx: LedgerContext): Promise<ToolTextResult> {
	try {
		checkCapability(ctx, "merge_pr");
		const prId = requireString(args as Record<string, unknown>, "pr_id", "the PR record id (PR-###) to merge");
		const prs = readJsonlFile<Record<string, unknown>>(statePath(ctx.root, "PRS.jsonl")).filter((record) => record.id === prId);
		const pr = prs.pop();
		if (!pr) {
			return fail(`unknown PR ${prId}; known PRs: ${readJsonlFile<Record<string, unknown>>(statePath(ctx.root, "PRS.jsonl")).map((record) => record.id).join(", ") || "none"}`);
		}
		const reviews = readJsonlFile<Record<string, unknown>>(statePath(ctx.root, "REVIEWS.jsonl")).filter(
			(record) => record.pr_id === prId,
		);
		const review = reviews.pop();
		if (pr.status === "merged" || typeof pr.merge_sha === "string") {
			return ok(`PR ${prId} is already merged (${String(pr.merge_sha ?? "sha unrecorded")}); no duplicate merge performed.`, {
				stub: false,
				merged: true,
				pr_id: prId,
				status: "already-merged",
			});
		}
		if (!review || review.decision !== "accept") {
			return fail(
				`merge refused for ${prId}: ${review ? `review ${String(review.id)} decided "${String(review.decision)}"` : "no review record exists"}. ` +
					`The conductor merges only an accepted review; write the review with review_write first.`,
				{ pr_id: prId, review_id: review?.id ?? null },
			);
		}
		const project = loadProjectConfig(ctx.root);
		const currentState = loadLoopState(ctx.root).state.state;
		if (!["REVIEW", "PAPER_REVIEW", "MERGE"].includes(currentState)) {
			return fail(
				`merge refused for ${prId}: the loop is in ${currentState}, not at the merge gate. ` +
					`merge_pr is valid from REVIEW/PAPER_REVIEW (the review accepted the PR) or MERGE (the gate is decided).`,
				{ pr_id: prId, state: currentState },
			);
		}
		const action = {
			kind: "merge" as const,
			state: currentState,
			key: `merge:${prId}:${String(review.id)}`,
			summary: `merge ${prId} (review ${String(review.id)})`,
			pr_id: prId,
			review_id: String(review.id),
			to: String(pr.pr_kind ?? (pr.kind === "paper" ? "paper" : "work")) === "paper" ? ("AUDIT" as const) : ("POST_MERGE" as const),
		};
		if (project.requireHumanMerge) {
			const gates = readJsonlFile<Record<string, unknown>>(statePath(ctx.root, "GATES.jsonl"));
			const latest = new Map<string, Record<string, unknown>>();
			for (const record of gates) if (typeof record.id === "string") latest.set(record.id, record);
			const mergeGates = [...latest.values()].filter((record) => record.gate === "merge" && record.pr_id === prId);
			const decided = mergeGates.find((record) => record.status !== "pending");
			if (!decided) {
				const pending = mergeGates.find((record) => record.status === "pending");
				if (!pending) {
					const gate = openGate(
						ctx.root,
						{
							gate: "merge",
							note: `merge ${prId}: approve to let the conductor land it on ${project.baseBranch}`,
							pr_id: prId,
							task_id: typeof pr.task_id === "string" ? pr.task_id : undefined,
							run_id: loadLoopState(ctx.root).state.run_id,
						},
						ctx.actor,
					);
					appendRecord(
						ctx.root,
						"event",
						{ summary: `merge request for ${prId} recorded; human gate ${String(gate.id)} opened`, pr_id: prId, actor: ctx.actor },
						ctx,
					);
					refreshViews(ctx.root);
					return ok(
						`merge request for ${prId} recorded. requireHumanMerge=true, so the conductor opened gate ${String(gate.id)}; ` +
							`approve with \`gate approve ${String(gate.id)} "<what you verified>"\` and the next step will merge.`,
						{ stub: false, merged: false, pr_id: prId, gate_id: gate.id, status: "pending-human-gate" },
					);
				}
				return ok(`merge for ${prId} already waits on human gate ${String(pending.id)}; no new action taken.`, {
					stub: false,
					merged: false,
					pr_id: prId,
					gate_id: pending.id,
					status: "pending-human-gate",
				});
			}
			if (decided.status === "rejected") {
				return fail(`merge refused for ${prId}: human gate ${String(decided.id)} was rejected (${String(decided.note ?? "")})`, {
					pr_id: prId,
					gate_id: decided.id,
					status: "human-gate-rejected",
				});
			}
		}
		const context = contextFromEnvironment(ctx.root, process.env.PI_LOOP_REAL_ROOT, process.env.PI_LOOP_COMMAND);
		// The supervisor's session runs in REVIEW/PAPER_REVIEW; the conductor's merge
		// always happens at the MERGE state, so the tool walks the loop there first
		// (REVIEW → MERGE and PAPER_REVIEW → MERGE are the plan's transitions).
		if (currentState !== "MERGE") {
			saveLoopState(
				ctx.root,
				{ state: "MERGE", pr_id: prId, reason: `review ${String(review.id)} accepted ${prId}; the conductor will run the gate and merge` },
				{ actor: ctx.actor },
			);
			refreshViews(ctx.root);
		}
		const result = await executeAction({
			context,
			action,
			options: loadLoopOptions(context),
			actor: ctx.actor,
		});
		if (!result.performed && result.skipped === "completed") {
			return ok(`merge for ${prId} was already recorded as complete; no duplicate merge performed.`, {
				stub: false,
				merged: true,
				pr_id: prId,
				status: "already-merged",
			});
		}
		if (result.transition?.to === "WORK") {
			return fail(result.message, { pr_id: prId, gate_run: result.gate_run?.path, status: "pre-merge-gate-failed" });
		}
		return ok(result.message, {
			stub: false,
			merged: true,
			pr_id: prId,
			review_id: String(review.id),
			gate_run: result.gate_run?.path,
			status: "merged",
		});
	} catch (error) {
		return fail((error as Error).message, { stub: false, merged: false });
	}
}
