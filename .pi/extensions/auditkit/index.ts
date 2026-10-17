/**
 * auditkit — finding schema validation, dedupe/register, report assembly, and
 * grade computation (plan/05_EXTENSIONS_AND_TOOLS.md §6,
 * plan/07_AUDIT_HARNESS.md §4–§7).
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { Type } from "typebox";
import { defineTool, type ExtensionAPI, type ExtensionToolContext } from "@earendil-works/pi-coding-agent";
import { resolveRepoRoot } from "../../lib/paths.ts";
import { parseRole, roleGate, type LoopRole } from "../../lib/roles.ts";
import { auditGrade, applyDecision, type CalibrationSummary, type DecisionMapping, type GateResults } from "./grade.ts";
import { assembleReport, writeGrade, writeQuestions, writeRootIssueRegister } from "./report.ts";
import { validateFinding, type Workstream } from "./schema.ts";
import { cycleDir, readRegister, readRubricScore, registerForCycle } from "./store.ts";
import type { GatesFile } from "../verify/gates.ts";

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

function role(): LoopRole {
	return parseRole(process.env.PI_LOOP_ROLE);
}

function requireWrite(): ToolResult | null {
	if (role() === "human") return null;
	const decision = roleGate("auditkit_write", role());
	return decision.allowed ? null : fail(decision.reason ?? "role may not write audit records");
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

function readGates(root: string, cycle: number): GatesFile | null {
	const file = path.join(cycleDir(root, cycle), "gates.json");
	if (!fs.existsSync(file)) return null;
	return JSON.parse(fs.readFileSync(file, "utf8")) as GatesFile;
}

function toGateResults(gates: GatesFile | null): GateResults | null {
	if (!gates) return null;
	return {
		ok: gates.ok,
		cycle: gates.cycle,
		snapshot: gates.snapshot,
		gates: gates.gates.map((gate) => ({
			id: gate.id,
			status: gate.status,
			detail: gate.detail,
			hash_ok: gate.hash_ok,
		})),
	};
}

function readCalibration(root: string, explicit?: string): CalibrationSummary | null {
	const file = explicit
		? path.resolve(root, explicit)
		: path.join(root, ".pi", "audit", "calibration", "latest", "recall.json");
	if (!fs.existsSync(file)) return null;
	if (fs.existsSync(path.join(path.dirname(file), "UNCALIBRATED"))) {
		return { recall: 0, rejected_confirmed: 1, calibrated: false };
	}
	const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as {
		recall?: number;
		rejected_confirmed?: number;
		calibrated?: boolean;
	};
	if (typeof parsed.recall !== "number") return null;
	return {
		recall: parsed.recall,
		rejected_confirmed: parsed.rejected_confirmed ?? 0,
		calibrated: parsed.calibrated === true,
	};
}

function readDecisionMapping(root: string): DecisionMapping | undefined {
	const file = path.join(root, ".pi", "rubrics", "decision.json");
	if (!fs.existsSync(file)) return undefined;
	const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as DecisionMapping;
	return parsed;
}

function traceabilityFromGates(gates: GatesFile | null): { ok: boolean; dangling?: string[]; uncited?: string[] } | null {
	if (!gates) return null;
	const gate = gates.gates.find((entry) => entry.id === "traceability");
	if (!gate) return null;
	if (gate.status === "pass") return { ok: true };
	if (gate.status === "fail") return { ok: false };
	return null;
}

export default function (pi: ExtensionAPI) {
	pi.registerTool(
		defineTool({
			name: "finding_validate",
			label: "Validate finding",
			description:
				"Validate one finding against the nine-field evidence schema plus severity/confidence/status/reproducible. Rejects a finding whose command is not executable (no command → not a finding).",
			parameters: Type.Object({
				finding: Type.Record(Type.String(), Type.Unknown()),
				workstream: Type.Optional(Type.String({ description: "Expected workstream letter (M/S/P/D/I/R/H/E/C/L)" })),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, _ctx) {
				return guard((args: { finding: Record<string, unknown>; workstream?: string }) => {
					const validation = validateFinding(args.finding, {
						expectedWorkstream: args.workstream as Workstream | undefined,
					});
					const text = validation.ok
						? `VALID — ${String(args.finding.id)}; ${validation.warnings.length} warning(s)`
						: `INVALID — ${validation.errors.length} error(s):\n- ${validation.errors.join("\n- ")}`;
					return validation.ok ? ok(text, { ...validation }) : fail(text, { ...validation });
				})(params as { finding: Record<string, unknown>; workstream?: string });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "finding_register",
			label: "Register findings",
			description:
				"Validate and merge findings into .pi/audit/cycle-<n>/<ws>/findings.json and the cycle register. Dedupes by normalized location+title (a duplicate becomes a cross-ref, never a second body).",
			parameters: Type.Object({
				cycle: Type.Number(),
				workstream: Type.String({ description: "M/S/P/D/I/R/H/E/C/L" }),
				findings: Type.Array(Type.Record(Type.String(), Type.Unknown())),
				phase: Type.Optional(Type.String({ description: "finder|verify|synth|adjudicate|canary" })),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(
					(args: { cycle: number; workstream: string; findings: Array<Record<string, unknown>>; phase?: string }) => {
						if (!/^[MSPDIRHECL]$/.test(args.workstream)) return fail(`unknown workstream ${args.workstream}`);
						if (!Number.isInteger(args.cycle) || args.cycle < 1) return fail("cycle must be a positive integer");
						const blocked = requireWrite();
						if (blocked) return blocked;
						const root = rootFor(ctx);
						const outcome = registerForCycle(
							root,
							args.cycle,
							args.workstream as Workstream,
							args.findings,
							{ phase: args.phase as never, author_agent: process.env.PI_AGENT_NAME },
						);
						const text = [
							`added ${outcome.added.length}, updated ${outcome.updated.length}, duplicates ${outcome.duplicates.length}, rejected ${outcome.rejected.length}`,
							outcome.duplicates.length > 0
								? `cross-refs: ${outcome.duplicates.map((entry) => `${entry.id}→${entry.duplicate_of}`).join(", ")}`
								: "",
							...outcome.rejected.map((entry) => `rejected[${entry.index}]: ${entry.errors.join("; ")}`),
							`register: ${outcome.files.register}`,
						]
							.filter(Boolean)
							.join("\n");
						return outcome.rejected.length === 0 ? ok(text, { ...outcome }) : fail(text, { ...outcome });
					},
				)(params as { cycle: number; workstream: string; findings: Array<Record<string, unknown>>; phase?: string });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "findings_render",
			label: "Render findings",
			description: "Render the cycle register (or one workstream) as FINDINGS.md plus the machine-readable findings.json.",
			parameters: Type.Object({
				cycle: Type.Number(),
				workstream: Type.Optional(Type.String()),
				format: Type.Optional(Type.String({ description: "md (default) or json" })),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard((args: { cycle: number; workstream?: string; format?: string }) => {
					const root = rootFor(ctx);
					const findings = readRegister(root, args.cycle).filter(
						(finding) => !args.workstream || finding.id.replace(/^WS-/, "").charAt(0) === args.workstream,
					);
					if (args.format === "json") return ok(`${JSON.stringify(findings, null, 2)}`, { findings });
					const file = path.join(cycleDir(root, args.cycle), args.workstream ? `${args.workstream.toLowerCase()}/FINDINGS.md` : "FINDINGS.md");
					if (!fs.existsSync(file)) return fail(`no rendered findings at ${file}; register the findings first`);
					return ok(fs.readFileSync(file, "utf8"), { path: file, count: findings.length });
				})(params as { cycle: number; workstream?: string; format?: string });
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "audit_grade",
			label: "Audit grade",
			description:
				"Apply the blocking rules of plan/07 §7 to the verified findings, gates, rubric, and calibration: S0 or core S1 caps at weak reject, an unfixed other S1 caps at borderline, a failed gate caps at weak reject, and accept needs green gates + traceability + limitations. Writes GRADE.md.",
			parameters: Type.Object({
				cycle: Type.Number(),
				rubric_path: Type.Optional(Type.String({ description: "Default: the cycle's rubric.json" })),
				calibration_path: Type.Optional(Type.String({ description: "Default: .pi/audit/calibration/latest/recall.json" })),
				traceability_ok: Type.Optional(Type.Boolean()),
				limitations_disclosed: Type.Optional(Type.Boolean()),
				inapplicable_gates: Type.Optional(Type.Array(Type.String())),
				waived_gates: Type.Optional(Type.Array(Type.String())),
				write: Type.Optional(Type.Boolean({ description: "Write GRADE.md (default true)" })),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(
					(args: {
						cycle: number;
						rubric_path?: string;
						calibration_path?: string;
						traceability_ok?: boolean;
						limitations_disclosed?: boolean;
						inapplicable_gates?: string[];
						waived_gates?: string[];
						write?: boolean;
					}) => {
						const blocked = requireWrite();
						if (blocked) return blocked;
						const root = rootFor(ctx);
						const findings = readRegister(root, args.cycle);
						const gates = readGates(root, args.cycle);
						if (!gates) return fail(`no gates.json for cycle ${args.cycle}; run run_gates first (the phase order is gates → find → verify → synthesise → adjudicate)`);
						const rubric = readRubricScore(root, args.cycle);
						const calibration = readCalibration(root, args.calibration_path);
						const decision = readDecisionMapping(root);
						const traceability =
							args.traceability_ok !== undefined ? { ok: args.traceability_ok } : traceabilityFromGates(gates);
						const result = auditGrade({
							findings,
							gates: toGateResults(gates),
							rubric,
							calibration,
							traceability,
							limitations_disclosed: args.limitations_disclosed === true,
							inapplicable_gates: args.inapplicable_gates,
							waived_gates: args.waived_gates,
							cycle: args.cycle,
							max_cycles: decision?.max_cycles ?? 5,
						});
						if (args.write !== false) writeGrade(root, args.cycle, result);
						const decisionResult = applyDecision(result.grade, result.blocking_fixes, decision, args.cycle);
						const text = [
							`GRADE: ${result.grade}${result.usable ? "" : " (not usable)"}`,
							`journal: ${result.journal}; base: ${result.base_grade ?? "none"}; capped: ${result.capped}`,
							`reason: ${result.reason}`,
							`decision: ${decisionResult.status} — ${decisionResult.action} (reaudit=${decisionResult.reaudit}, done=${decisionResult.done})`,
							result.gate_failures.length > 0 ? `failing gates: ${result.gate_failures.join(", ")}` : "",
						]
							.filter(Boolean)
							.join("\n");
						return result.usable ? ok(text, { ...result, decision: decisionResult }) : fail(text, { ...result, decision: decisionResult });
					},
				)(params as {
					cycle: number;
					rubric_path?: string;
					calibration_path?: string;
					traceability_ok?: boolean;
					limitations_disclosed?: boolean;
					inapplicable_gates?: string[];
					waived_gates?: string[];
					write?: boolean;
				});
			},
		}),
	);

	pi.registerTool(
		defineTool({
			name: "audit_report_assemble",
			label: "Assemble audit report",
			description:
				"Assemble FINAL_REPORT.md from verified finder findings + gates + rubric, and write ROOT_ISSUE_REGISTER.csv and QUESTIONS_FOR_AUTHORS.md. A synthesizer-created finding without a verified id is rejected.",
			parameters: Type.Object({
				cycle: Type.Number(),
				snapshot: Type.Optional(Type.String()),
				meta_coverage_path: Type.Optional(Type.String()),
				limitations_disclosed: Type.Optional(Type.Boolean()),
				traceability_ok: Type.Optional(Type.Boolean()),
			}),
			executionMode: "sequential",
			async execute(_id, params, _signal, _update, ctx) {
				return guard(
					(args: {
						cycle: number;
						snapshot?: string;
						meta_coverage_path?: string;
						limitations_disclosed?: boolean;
						traceability_ok?: boolean;
					}) => {
						const blocked = requireWrite();
						if (blocked) return blocked;
						const root = rootFor(ctx);
						const findings = readRegister(root, args.cycle);
						const gates = readGates(root, args.cycle);
						if (!gates) return fail(`no gates.json for cycle ${args.cycle}; run run_gates first`);
						const rubric = readRubricScore(root, args.cycle);
						const calibration = readCalibration(root);
						const traceability =
							args.traceability_ok !== undefined ? { ok: args.traceability_ok } : traceabilityFromGates(gates);
						const grade = auditGrade({
							findings,
							gates: toGateResults(gates),
							rubric,
							calibration,
							traceability,
							limitations_disclosed: args.limitations_disclosed === true,
							cycle: args.cycle,
						});
						const assembled = assembleReport({
							root,
							cycle: args.cycle,
							snapshot: args.snapshot ?? gates?.snapshot ?? "HEAD",
							findings,
							gates,
							grade,
							fixList: grade.fix_list,
							metaCoveragePath: args.meta_coverage_path,
						});
						writeGrade(root, args.cycle, grade);
						writeRootIssueRegister(root, args.cycle, findings);
						writeQuestions(root, args.cycle, findings);
						const text = [
							`assembled ${assembled.final_report}`,
							`reported: ${assembled.reported.length} verified finding(s)`,
							assembled.rejected.length > 0 ? `rejected synthesis records: ${assembled.synth_added.join(", ")}` : "",
							`grade: ${grade.grade}`,
						]
							.filter(Boolean)
							.join("\n");
						return assembled.rejected.length === 0 ? ok(text, { ...assembled, grade }) : fail(text, { ...assembled, grade });
					},
				)(params as {
					cycle: number;
					snapshot?: string;
					meta_coverage_path?: string;
					limitations_disclosed?: boolean;
					traceability_ok?: boolean;
				});
			},
		}),
	);
}
