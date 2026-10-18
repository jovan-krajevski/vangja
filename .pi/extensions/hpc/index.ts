/**
 * hpc — SSH/SLURM tools with single-slot and checkpoint enforcement.
 *
 * Registers the tools from plan/05_EXTENSIONS_AND_TOOLS.md §5. All behaviour
 * lives in `./handlers.ts` (pi-free, tested with `node`); the mock-cluster
 * harness in `.pi/tests/hpc/` drives the same handlers through the real `ssh`,
 * `scp`, `rsync`, `sbatch`, `squeue`, and `sacct` invocations on `PATH`.
 *
 * With `host: ""` every tool returns `{enabled:false}` and performs no I/O.
 */

import { Type } from "typebox";
import { defineTool, type ExtensionAPI, type ExtensionToolContext } from "@earendil-works/pi-coding-agent";
import { resolveRepoRoot } from "../../lib/paths.ts";
import { parseRole, type LoopRole } from "../../lib/roles.ts";
import {
	failText,
	handleHpcCancel,
	handleHpcCheckpoints,
	handleHpcConfig,
	handleHpcFetch,
	handleHpcRequeue,
	handleHpcStatus,
	handleHpcSubmit,
	handleHpcTail,
	type HpcContext,
	type ToolTextResult,
} from "./handlers.ts";

function contextFor(ctx: ExtensionToolContext): HpcContext {
	const root = resolveRepoRoot(process.env.PI_LOOP_ROOT ?? ctx.cwd);
	const role: LoopRole = parseRole(process.env.PI_LOOP_ROLE);
	return { root, role, actor: process.env.PI_AGENT_NAME ?? `agent:${role}` };
}

function guard<T>(fn: (params: T, ctx: HpcContext) => ToolTextResult) {
	return (params: T, ctx: HpcContext): ToolTextResult => {
		try {
			return fn(params, ctx);
		} catch (error) {
			return failText((error as Error).message);
		}
	};
}

export default function (pi: ExtensionAPI) {
	pi.registerTool(
		defineTool({
			name: "hpc_config",
			label: "HPC config",
			description:
				"Report the .pi/config/hpc.json state (host redacted, resources, git-forwarding). " +
				"Returns {enabled:false} and does nothing when host is empty.",
			parameters: Type.Object({}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleHpcConfig)(params as Record<string, unknown>, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "hpc_status",
			label: "HPC status",
			description:
				"Run squeue and sacct for the active experiment's job(s) and record the raw output as evidence. " +
				"Accepts an explicit list of job ids. Returns {enabled:false} when host is empty.",
			parameters: Type.Object({
				jobs: Type.Optional(Type.Array(Type.String({ description: "SLURM job ids (default: active experiments)" }))),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleHpcStatus)(params as { jobs?: string[] }, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "hpc_submit",
			label: "HPC submit",
			description:
				"Gate-checked sbatch submission for one experiment record (single-slot, checkpoint metadata, " +
				"local resume smoke test, clean detached remote checkout, uv.lock/container hash, ≤24h wall clock). " +
				"Refused while any other experiment is active.",
			parameters: Type.Object({
				experiment_id: Type.String({ description: "Experiment record id, e.g. E-0007" }),
				stage: Type.Optional(Type.Number({ description: "Stage index for a record split into stages>1" })),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleHpcSubmit)(params as { experiment_id: string; stage?: number }, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "hpc_checkpoints",
			label: "HPC checkpoints",
			description:
				"List checkpoint files with sizes/mtimes/SHA-256 and read the progress marker. Works while the job runs.",
			parameters: Type.Object({ experiment_id: Type.String({ description: "Experiment record id" }) }),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleHpcCheckpoints)(params as { experiment_id: string }, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "hpc_fetch",
			label: "HPC fetch",
			description:
				"Copy declared outputs and checkpoints, SHA-256 every file, and write " +
				".pi/artifacts/<id>/manifest.json with sacct accounting. Fails loudly when a declared output is missing.",
			parameters: Type.Object({
				experiment_id: Type.String({ description: "Experiment record id" }),
				checkpointsOnly: Type.Optional(
					Type.Boolean({ description: "Fetch only the checkpoint dir (timeout/preempt path)" }),
				),
				paths: Type.Optional(Type.Array(Type.String({ description: "Override the declared outputs (run-dir relative)" }))),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleHpcFetch)(params as { experiment_id: string; checkpointsOnly?: boolean; paths?: string[] }, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "hpc_tail",
			label: "HPC tail",
			description: "Tail the SLURM stdout/stderr of the experiment's current job.",
			parameters: Type.Object({
				experiment_id: Type.String({ description: "Experiment record id" }),
				stream: Type.Optional(Type.Union([Type.Literal("out"), Type.Literal("err")])),
				lines: Type.Optional(Type.Number({ description: "Lines to tail (1..2000, default 50)" })),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleHpcTail)(params as { experiment_id: string; stream?: "out" | "err"; lines?: number }, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "hpc_cancel",
			label: "HPC cancel",
			description:
				"scancel a job — destructive; refused unless an instruction record (decision or event) names the " +
				"experiment and the cancellation. Never call this to break a stuck poll.",
			parameters: Type.Object({
				experiment_id: Type.String({ description: "Experiment record id" }),
				reason: Type.String({ description: "The instruction being executed" }),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleHpcCancel)(params as { experiment_id: string; reason: string }, contextFor(ctx));
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "hpc_requeue",
			label: "HPC requeue",
			description:
				"Create attempt+1 for a requeue_requested experiment with the same idempotency key. Allowed only " +
				"after the previous job is confirmed ended; refuses while another experiment is active.",
			parameters: Type.Object({
				experiment_id: Type.String({ description: "Experiment record id" }),
				reason: Type.String({ description: "The collector/conductor decision being executed" }),
				memoryGb: Type.Optional(Type.Number({ description: "Doubled memory for an OOM retry" })),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(handleHpcRequeue)(
					params as { experiment_id: string; reason: string; memoryGb?: number },
					contextFor(ctx),
				);
			},
		}),
	);
}
