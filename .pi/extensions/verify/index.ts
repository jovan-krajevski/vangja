/**
 * verify — deterministic verification tools (plan/05_EXTENSIONS_AND_TOOLS.md §6).
 *
 * Registers verify_tests, verify_artifact, verify_reproduce, verify_traceability,
 * verify_guards, and run_gates. All behaviour lives in `./lib.ts` and
 * `./gates.ts` (pi-free, tested with `node`); this file declares the
 * model-facing schemas and forwards calls.
 */

import { Type } from "typebox";
import { defineTool, type ExtensionAPI, type ExtensionToolContext } from "@earendil-works/pi-coding-agent";
import { resolveRepoRoot } from "../../lib/paths.ts";
import { parseRole, roleGate, type LoopRole } from "../../lib/roles.ts";
import {
	defaultRunner,
	verifyArtifact,
	verifyArtifactAgainstExpected,
	verifyGuards,
	verifyReproduce,
	verifyTests,
	verifyTraceability,
} from "./lib.ts";
import { runGates } from "./gates.ts";

interface ToolResult {
	content: Array<{ type: "text"; text: string }>;
	details: Record<string, unknown>;
	isError?: boolean;
}

function ok(text: string, details: Record<string, unknown> = {}): ToolResult {
	return { content: [{ type: "text", text }], details };
}

function fail(message: string, details: Record<string, unknown> = {}): ToolResult {
	return { content: [{ type: "text", text: message }], details: { ...details, error: message }, isError: true };
}

function rootFor(ctx: ExtensionToolContext): string {
	return resolveRepoRoot(process.env.PI_LOOP_ROOT ?? ctx.cwd);
}

function roleFor(): LoopRole {
	return parseRole(process.env.PI_LOOP_ROLE);
}

function render(value: unknown): string {
	return `${JSON.stringify(value, null, 2)}`;
}

function guard<T>(fn: (params: T) => ToolResult): (params: T) => ToolResult {
	return (params) => {
		try {
			return fn(params);
		} catch (error) {
			return fail((error as Error).message);
		}
	};
}

function allowWriteTool(): ToolResult | null {
	const role = roleFor();
	if (role === "human") return null;
	const decision = roleGate("auditkit_write", role);
	if (!decision.allowed) return fail(decision.reason ?? "role may not write audit results");
	return null;
}

export default function (pi: ExtensionAPI) {
	pi.registerTool(
		defineTool({
			name: "verify_tests",
			label: "Verify tests",
			description:
				"Run the test suite (default `uv run pytest -q`) and return pass/fail counts, first failures, duration, and the exact command.",
			parameters: Type.Object({
				paths: Type.Optional(Type.Array(Type.String({ description: "Test paths to run (default: the whole suite)" }))),
				command: Type.Optional(Type.Array(Type.String({ description: "Override the test command argv" }))),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard((args: { paths?: string[]; command?: string[] }) => {
					const result = verifyTests(rootFor(ctx), { paths: args.paths, command: args.command });
					const text = [
						`${result.ok ? "PASS" : "FAIL"} — ${result.summary || `exited ${result.status}`}`,
						`command: ${result.command}`,
						result.failures.length > 0 ? `failures:\n${result.failures.join("\n")}` : "",
					]
						.filter(Boolean)
						.join("\n");
					return { ...(result.ok ? ok(text, { ...result }) : fail(text, { ...result })) };
				})(params as { paths?: string[]; command?: string[] });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "verify_artifact",
			label: "Verify artifact",
			description:
				"Verify an artifact manifest: schema valid, declared files exist, SHA-256 matches, and the manifest digest matches its body. Returns the missing/mismatched list and hash_ok.",
			parameters: Type.Object({
				path: Type.String({ description: "Artifact directory or manifest.json path" }),
				expected_outputs: Type.Optional(
					Type.Array(Type.String({ description: "Glob patterns that must exist in the artifact directory" })),
				),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard((args: { path: string; expected_outputs?: string[] }) => {
					const root = rootFor(ctx);
					const result = args.expected_outputs
						? verifyArtifactAgainstExpected(root, args.path, args.expected_outputs)
						: verifyArtifact(root, args.path);
					const text = `${result.ok ? "PASS" : "FAIL"} — hash_ok=${result.hash_ok}; ${result.detail}`;
					return result.ok ? ok(text, { ...result }) : fail(text, { ...result });
				})(params as { path: string; expected_outputs?: string[] });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "verify_reproduce",
			label: "Verify reproduction",
			description:
				"Run the artifact's recorded repro_command in a fresh temp dir and compare the printed numbers with the manifest values within a stated tolerance. Reports per-number drift.",
			parameters: Type.Object({
				artifact_id: Type.Optional(Type.String({ description: "Experiment/artifact id under .pi/artifacts, or a manifest path" })),
				path: Type.Optional(Type.String({ description: "Manifest path (alternative to artifact_id)" })),
				tolerance: Type.Optional(Type.Number({ description: "Absolute tolerance override" })),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard((args: { artifact_id?: string; path?: string; tolerance?: number }) => {
					const root = rootFor(ctx);
					const target = args.path ?? (args.artifact_id ? `${root}/.pi/artifacts/${args.artifact_id}` : "");
					if (!target) return fail("verify_reproduce needs artifact_id or path");
					const result = verifyReproduce(root, target, { tolerance: args.tolerance });
					const text = [
						`${result.ok ? "PASS" : "FAIL"} — ${result.diffs.length} number(s), tolerance ${result.tolerance}`,
						result.error ? `error: ${result.error}` : "",
						result.drift,
					]
						.filter(Boolean)
						.join("\n");
					return result.ok ? ok(text, { ...result }) : fail(text, { ...result });
				})(params as { artifact_id?: string; path?: string; tolerance?: number });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "verify_traceability",
			label: "Verify traceability",
			description:
				"Check paper/CLAIMS.json: every claim has an artifact whose manifest SHA is an ancestor of the audited ref; every decimal number in paper/** has a claims entry. Returns dangling/uncited lists.",
			parameters: Type.Object({
				claims_path: Type.Optional(Type.String()),
				paper_dir: Type.Optional(Type.String()),
				ref: Type.Optional(Type.String({ description: "Audited git ref (default HEAD)" })),
				require_ancestor: Type.Optional(Type.Boolean()),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard((args: { claims_path?: string; paper_dir?: string; ref?: string; require_ancestor?: boolean }) => {
					const result = verifyTraceability(rootFor(ctx), {
						claimsPath: args.claims_path,
						paperDir: args.paper_dir,
						ref: args.ref,
						requireAncestor: args.require_ancestor,
					});
					const text = `${result.ok ? "PASS" : "FAIL"} — ${result.detail}${result.dangling.length ? `\ndangling:\n${result.dangling.map((entry) => `- ${entry.claim}: ${entry.artifact} (${entry.reason})`).join("\n")}` : ""}${result.uncited.length ? `\nuncited:\n${result.uncited.slice(0, 40).map((entry) => `- ${entry.file}:${entry.line} ${entry.number}`).join("\n")}` : ""}`;
					return result.ok ? ok(text, { ...result }) : fail(text, { ...result });
				})(params as { claims_path?: string; paper_dir?: string; ref?: string; require_ancestor?: boolean });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "verify_guards",
			label: "Verify guards",
			description:
				"Run the protected-path scan, the test-diff scan, and the anti-gaming scan on a diff range. Returns violations and scanner findings with raw output.",
			parameters: Type.Object({
				range: Type.Optional(Type.String({ description: "Git diff range, e.g. 8028c51..HEAD" })),
				allow_paper: Type.Optional(Type.Boolean({ description: "Allow paper/** changes (PAPER state)" })),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard((args: { range?: string; allow_paper?: boolean }) => {
					const result = verifyGuards(rootFor(ctx), { range: args.range, allowPaper: args.allow_paper });
					const text = [
						`${result.ok ? "PASS" : "FAIL"} — ${result.changed_files.length} changed file(s) in ${result.range}`,
						result.protected_violations.length > 0 ? `protected: ${result.protected_violations.join("; ")}` : "",
						result.test_diff.problems.length > 0 ? `test diff: ${result.test_diff.problems.join("; ")}` : "",
						result.gaming.findings.length > 0 ? `gaming scan: ${result.gaming.findings.length} hit(s)` : "",
					]
						.filter(Boolean)
						.join("\n");
					return result.ok ? ok(text, { ...result }) : fail(text, { ...result });
				})(params as { range?: string; allow_paper?: boolean });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "run_gates",
			label: "Run audit gates",
			description:
				"Run the 13 deterministic gates of plan/07_AUDIT_HARNESS.md §5 and write .pi/audit/cycle-<n>/gates.json. Every gate reports pass/fail/skip/error; a missing input is a skip with a reason.",
			parameters: Type.Object({
				cycle: Type.Number({ description: "Audit cycle number" }),
				snapshot: Type.Optional(Type.String({ description: "Snapshot tag@sha" })),
				only: Type.Optional(Type.Array(Type.String({ description: "Run only these gate ids" }))),
				skip: Type.Optional(Type.Array(Type.String({ description: "Skip these gate ids" }))),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard((args: { cycle: number; snapshot?: string; only?: string[]; skip?: string[] }) => {
					const blocked = allowWriteTool();
					if (blocked) return blocked;
					if (!Number.isInteger(args.cycle) || args.cycle < 1) return fail("cycle must be a positive integer");
					const result = runGates(rootFor(ctx), {
						cycle: args.cycle,
						snapshot: args.snapshot,
						only: args.only,
						skip: args.skip,
						runner: defaultRunner,
					});
					const text = [
						`${result.ok ? "ALL GREEN" : "FAILED"} — ${result.gates.length} gate(s); failed: ${result.failed.join(", ") || "none"}; skipped: ${result.skipped.join(", ") || "none"}`,
						`hash_ok=${result.hash_ok}`,
						...result.gates.map((gate) => `- ${gate.id}: ${gate.status} — ${gate.detail}`),
					].join("\n");
					return result.ok ? ok(text, { ...result }) : fail(text, { ...result });
				})(params as { cycle: number; snapshot?: string; only?: string[]; skip?: string[] });
			},
		}),
	);
}
