/**
 * orchestrator (stub) — S3 placeholder for the conductor from
 * plan/05_EXTENSIONS_AND_TOOLS.md §2.
 *
 * Only `merge_pr` is registered: the guard's role matrix allows supervisor and
 * conductor, the handler checks the capability again, and the stub logs the
 * request without merging anything. S4 replaces this file with the state
 * machine; the tool contract (reject without a green gate or a waiver) arrives
 * with it.
 */

import { defineTool, type ExtensionAPI, type ExtensionToolContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { resolveRepoRoot } from "../../lib/paths.ts";
import { parseRole } from "../../lib/roles.ts";
import { handleMergePrStub } from "./stub.ts";

const MergePrOutputSchema = Type.Object({
	stub: Type.Boolean(),
	merged: Type.Boolean(),
	pr_id: Type.String(),
});

export default function (pi: ExtensionAPI) {
	pi.registerTool(
		defineTool({
			name: "merge_pr",
			label: "Merge PR (stub)",
			description:
				"STUB for the conductor's merge gate (S4): accepts pr_id from supervisor/conductor, records the request " +
				"and does not merge. Never call this expecting a branch to change.",
			parameters: Type.Object({
				pr_id: Type.String({ description: "PR record id (PR-###)" }),
				task_id: Type.Optional(Type.String({ description: "Task id for bookkeeping" })),
				reason: Type.Optional(Type.String({ description: "Why the merge is recommended" })),
			}),
			outputSchema: MergePrOutputSchema,
			executionMode: "sequential",
			async execute(_toolCallId, params, _signal, _update, ctx: ExtensionToolContext) {
				const root = resolveRepoRoot(process.env.PI_LOOP_ROOT ?? ctx.cwd);
				const role = parseRole(process.env.PI_LOOP_ROLE);
				const actor = process.env.PI_AGENT_NAME ?? `agent:${role}`;
				return handleMergePrStub(params as Record<string, unknown>, { root, role, actor });
			},
		}),
	);
}
