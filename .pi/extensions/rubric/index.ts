/**
 * rubric — rubric trees: load / autofill deterministic leaves / record scores
 * with evidence pointers / validate / render (plan/05_EXTENSIONS_AND_TOOLS.md §6).
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { Type } from "typebox";
import { defineTool, type ExtensionAPI, type ExtensionToolContext } from "@earendil-works/pi-coding-agent";
import { writeJsonAtomic } from "../../lib/io.ts";
import { resolveRepoRoot } from "../../lib/paths.ts";
import { parseRole, roleGate, type LoopRole } from "../../lib/roles.ts";
import {
	autofillRubric,
	loadRubric,
	renderRubric,
	rubricPath,
	scoreRubric,
	validateRubric,
	type AutofillInput,
	type DecisionRubric,
	type RubricTree,
	type ScoreEntry,
} from "./lib.ts";

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

function asTree(rubric: ReturnType<typeof loadRubric>): RubricTree {
	if (rubric.kind !== "rubric") throw new Error(`rubric "${rubric.id}" is a ${rubric.kind} mapping, not a score tree`);
	return rubric as RubricTree;
}

function loadGates(root: string, args: { cycle?: number; gates_path?: string }): AutofillInput["gates"] | undefined {
	const file = args.gates_path
		? path.resolve(root, args.gates_path)
		: args.cycle !== undefined
			? path.join(root, ".pi", "audit", `cycle-${args.cycle}`, "gates.json")
			: null;
	if (!file || !fs.existsSync(file)) return undefined;
	return JSON.parse(fs.readFileSync(file, "utf8")) as AutofillInput["gates"];
}

function loadCalibration(root: string, args: { calibration_path?: string }): AutofillInput["calibration"] | undefined {
	const file = args.calibration_path
		? path.resolve(root, args.calibration_path)
		: path.join(root, ".pi", "audit", "calibration", "latest", "recall.json");
	if (!fs.existsSync(file)) return undefined;
	if (fs.existsSync(path.join(path.dirname(file), "UNCALIBRATED"))) {
		return { calibrated: false, recall: 0, rejected_confirmed: 1 };
	}
	const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as {
		calibrated?: boolean;
		recall?: number;
		rejected_confirmed?: number;
	};
	if (typeof parsed.recall !== "number") return undefined;
	return {
		calibrated: parsed.calibrated === true,
		recall: parsed.recall,
		rejected_confirmed: parsed.rejected_confirmed ?? 0,
	};
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

function role(): LoopRole {
	return parseRole(process.env.PI_LOOP_ROLE);
}

function requireScoreRole(): ToolResult | null {
	if (role() === "human") return null;
	const decision = roleGate("rubric_score", role());
	return decision.allowed ? null : fail(decision.reason ?? "role may not score rubrics");
}

export default function (pi: ExtensionAPI) {
	pi.registerTool(
		defineTool({
			name: "rubric_load",
			label: "Load rubric",
			description: "Load and validate a rubric tree from .pi/rubrics/<name>.json (or an explicit path).",
			parameters: Type.Object({ name: Type.String({ description: "Rubric name or path" }) }),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard((args: { name: string }) => {
					const rubric = loadRubric(rootFor(ctx), args.name);
					const text = `${rubric.id} (${rubric.kind}) — ${rubric.title}`;
					return ok(text, { ...rubric });
				})(params as { name: string });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "rubric_validate",
			label: "Validate rubric",
			description:
				"Structurally validate a rubric: unique ids, leaf types, auto names, one-line criteria, weights summing to 1 per branch and overall.",
			parameters: Type.Object({ name: Type.String({ description: "Rubric name or path" }) }),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard((args: { name: string }) => {
					const root = rootFor(ctx);
					const file = rubricPath(root, args.name);
					if (!fs.existsSync(file)) return fail(`rubric not found: ${file}`);
					const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as unknown;
					const validation = validateRubric(parsed);
					const text = validation.ok
						? `VALID — ${validation.kind}; ${validation.branches} branch(es), ${validation.leaves} leaf(leaves)`
						: `INVALID — ${validation.errors.length} error(s):\n- ${validation.errors.join("\n- ")}`;
					return validation.ok ? ok(text, { ...validation, path: file }) : fail(text, { ...validation, path: file });
				})(params as { name: string });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "rubric_autofill",
			label: "Autofill rubric",
			description:
				"Fill every `auto` leaf from the cycle's gates.json (and the latest calibration recall). A skipped gate stays unresolved — it proves nothing.",
			parameters: Type.Object({
				name: Type.String(),
				cycle: Type.Optional(Type.Number({ description: "Audit cycle whose gates.json to read" })),
				gates_path: Type.Optional(Type.String()),
				calibration_path: Type.Optional(Type.String()),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard((args: { name: string; cycle?: number; gates_path?: string; calibration_path?: string }) => {
					const root = rootFor(ctx);
					const tree = asTree(loadRubric(root, args.name));
					const result = autofillRubric(tree, {
						gates: loadGates(root, args),
						calibration: loadCalibration(root, args),
					});
					const text = [
						`${result.filled.length} leaf(leaves) autofilled; ${result.unresolved.length} unresolved`,
						...result.unresolved.map((entry) => `- ${entry.id}: ${entry.reason}`),
					].join("\n");
					return ok(text, { ...result });
				})(params as { name: string; cycle?: number; gates_path?: string; calibration_path?: string });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "rubric_score",
			label: "Score rubric",
			description:
				"Record LLM scores for non-auto leaves (each requires an evidence pointer), merge with autofilled leaves, and compute branch and total scores. Writes rubric.json when a cycle is given.",
			parameters: Type.Object({
				name: Type.String(),
				scores: Type.Array(
					Type.Object({
						leaf_id: Type.String(),
						score: Type.Number(),
						evidence: Type.String({ description: "Finding id, gate id, or artifact path proving the score" }),
						note: Type.Optional(Type.String()),
					}),
				),
				cycle: Type.Optional(Type.Number()),
				gates_path: Type.Optional(Type.String()),
				calibration_path: Type.Optional(Type.String()),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(
					(args: {
						name: string;
						scores: ScoreEntry[];
						cycle?: number;
						gates_path?: string;
						calibration_path?: string;
					}) => {
						const blocked = requireScoreRole();
						if (blocked) return blocked;
						const root = rootFor(ctx);
						const loaded = loadRubric(root, args.name);
						if (loaded.kind === "decision") {
							const decision = loaded as DecisionRubric;
							return ok(`decision mapping loaded: ${Object.keys(decision.mapping).length} grades`, { ...decision });
						}
						const tree = asTree(loaded);
						const autofilled = autofillRubric(tree, {
							gates: loadGates(root, args),
							calibration: loadCalibration(root, args),
						});
						const result = scoreRubric(autofilled.tree, args.scores);
						const text = [
							`${result.ok ? "SCORED" : "ERRORS"} — total ${result.total === null ? "incomplete" : result.total.toFixed(3)}`,
							result.unresolved.length > 0 ? `unresolved: ${result.unresolved.join(", ")}` : "",
							...result.errors.map((error) => `- ${error}`),
						]
							.filter(Boolean)
							.join("\n");
						if (args.cycle !== undefined && result.ok) {
							writeJsonAtomic(path.join(root, ".pi", "audit", `cycle-${args.cycle}`, "rubric.json"), {
								schema: "vangja-rubric-score/1",
								rubric: args.name,
								cycle: args.cycle,
								complete: result.complete,
								total: result.total,
								unresolved: result.unresolved,
								branches: result.branches,
								tree: autofilled.tree,
							});
						}
						return result.ok ? ok(text, { ...result, autofilled: autofilled.filled }) : fail(text, { ...result });
					},
				)(params as { name: string; scores: ScoreEntry[]; cycle?: number; gates_path?: string; calibration_path?: string });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "rubric_render",
			label: "Render rubric",
			description: "Render a rubric tree (optionally with autofilled scores) as the markdown table GRADE.md cites.",
			parameters: Type.Object({
				name: Type.String(),
				cycle: Type.Optional(Type.Number()),
				gates_path: Type.Optional(Type.String()),
				calibration_path: Type.Optional(Type.String()),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard((args: { name: string; cycle?: number; gates_path?: string; calibration_path?: string }) => {
					const root = rootFor(ctx);
					const loaded = loadRubric(root, args.name);
					if (loaded.kind === "decision") {
						const decision = loaded as DecisionRubric;
						const lines = [`### Decision mapping: ${decision.title}`, "", "| Grade | Action | Re-audit | Done |", "|---|---|---|---|"];
						for (const [grade, entry] of Object.entries(decision.mapping)) {
							lines.push(`| ${grade} | ${entry.action} | ${entry.reaudit ? "yes" : "no"} | ${entry.done ? "yes" : "no"} |`);
						}
						return ok(`${lines.join("\n")}\n`, { ...decision });
					}
					const autofilled = autofillRubric(asTree(loaded), {
						gates: loadGates(root, args),
						calibration: loadCalibration(root, args),
					});
					return ok(renderRubric(autofilled.tree), { rubric: autofilled.tree, filled: autofilled.filled });
				})(params as { name: string; cycle?: number; gates_path?: string; calibration_path?: string });
			},
		}),
	);
}
