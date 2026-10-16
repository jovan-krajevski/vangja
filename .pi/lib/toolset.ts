/**
 * The tool registry: `.pi/config/tools.json`.
 *
 * Agent frontmatter (and the manifest in `.pi/config/agents.json`) may name a
 * concrete tool or a *family* (dispatch, ledger, hpc, verify, task, paper,
 * rubric, auditkit, merge). Families expand to concrete names for the child
 * `--tools` allowlist. The registry is versioned so the dispatch extension and
 * the tests share one source of truth, and a typo in an agent file fails the
 * agent test instead of silently disabling a tool.
 *
 * No pi imports: usable from `node` tests and from extensions (jiti).
 */

import * as fs from "node:fs";
import { configInvalid, configMissing, VangjaLoopError } from "./errors.ts";
import { configPath } from "./paths.ts";
import { readJsonFile } from "./io.ts";

export type ToolKind = "builtin" | "user-package" | "extension" | "planned";

export const TOOL_KINDS: readonly ToolKind[] = ["builtin", "user-package", "extension", "planned"];

export interface ToolEntry {
	name: string;
	kind: ToolKind;
	/** Plan stage that delivers the tool: S0, S3, S5, S6, … */
	stage: string;
	/** Extension directory under `.pi/extensions/` for extension/planned tools. */
	extension?: string;
}

export interface ToolRegistry {
	schema: string;
	/** family name -> concrete tool names */
	families: Record<string, string[]>;
	tools: ToolEntry[];
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireString(file: string, obj: Record<string, unknown>, key: string): string {
	const value = obj[key];
	if (typeof value !== "string" || value.trim() === "") {
		throw configInvalid(file, `"${key}" must be a non-empty string`);
	}
	return value;
}

export function parseToolRegistry(raw: unknown, file: string): ToolRegistry {
	if (!isPlainObject(raw)) throw configInvalid(file, "top level must be a JSON object");
	const schema = requireString(file, raw, "schema");
	if (schema !== "pi-tools/1") throw configInvalid(file, `schema must be "pi-tools/1", got "${schema}"`);

	const rawFamilies = raw.families;
	if (!isPlainObject(rawFamilies)) throw configInvalid(file, `"families" must be an object`);
	const families: Record<string, string[]> = {};
	for (const [family, members] of Object.entries(rawFamilies)) {
		if (!Array.isArray(members) || members.some((m) => typeof m !== "string" || m.trim() === "")) {
			throw configInvalid(file, `families.${family} must be an array of non-empty strings`);
		}
		if (members.length === 0) throw configInvalid(file, `families.${family} must not be empty`);
		families[family] = [...(members as string[])];
	}

	const rawTools = raw.tools;
	if (!Array.isArray(rawTools) || rawTools.length === 0) {
		throw configInvalid(file, `"tools" must be a non-empty array`);
	}
	const tools: ToolEntry[] = [];
	const seen = new Set<string>();
	for (let i = 0; i < rawTools.length; i += 1) {
		const entry = rawTools[i];
		if (!isPlainObject(entry)) throw configInvalid(file, `tools[${i}] must be an object`);
		const name = requireString(file, entry, "name");
		if (seen.has(name)) throw configInvalid(file, `duplicate tool name "${name}"`);
		seen.add(name);
		const kind = requireString(file, entry, "kind") as ToolKind;
		if (!TOOL_KINDS.includes(kind)) {
			throw configInvalid(file, `tools[${i}].kind must be one of ${TOOL_KINDS.join(", ")}, got "${kind}"`);
		}
		const stage = requireString(file, entry, "stage");
		if (!/^S\d+$/.test(stage)) throw configInvalid(file, `tools[${i}].stage must look like "S3", got "${stage}"`);
		if ((kind === "extension" || kind === "planned") && typeof entry.extension !== "string") {
			throw configInvalid(file, `tools[${i}] ("${name}") needs an "extension" directory name`);
		}
		tools.push({
			name,
			kind,
			stage,
			extension: typeof entry.extension === "string" ? entry.extension : undefined,
		});
	}

	const names = new Set(tools.map((tool) => tool.name));
	for (const [family, members] of Object.entries(families)) {
		if (names.has(family)) {
			throw configInvalid(file, `family "${family}" collides with a concrete tool name`);
		}
		for (const member of members) {
			if (!names.has(member)) {
				throw configInvalid(file, `families.${family} names unknown tool "${member}"`);
			}
		}
	}

	return { schema, families, tools };
}

export function loadToolRegistry(root: string): ToolRegistry {
	const file = configPath(root, "tools.json");
	if (!fs.existsSync(file)) throw configMissing(file);
	return parseToolRegistry(readJsonFile<unknown>(file), file);
}

export function knownToolNames(registry: ToolRegistry): string[] {
	return registry.tools.map((tool) => tool.name);
}

export function isKnownToolName(registry: ToolRegistry, name: string): boolean {
	return name in registry.families || registry.tools.some((tool) => tool.name === name);
}

export interface ToolExpansion {
	/** Concrete names, in first-seen order, with duplicates removed. */
	names: string[];
	/** Names that are neither a family nor a concrete tool. */
	unknown: string[];
}

/** Expand family names to concrete tool names; typos land in `unknown`. */
export function expandTools(registry: ToolRegistry, names: readonly string[]): ToolExpansion {
	const expanded: string[] = [];
	const seen = new Set<string>();
	const unknown: string[] = [];
	for (const name of names) {
		const members = registry.families[name] ?? (isKnownToolName(registry, name) ? [name] : undefined);
		if (!members) {
			unknown.push(name);
			continue;
		}
		for (const member of members) {
			if (seen.has(member)) continue;
			seen.add(member);
			expanded.push(member);
		}
	}
	return { names: expanded, unknown };
}

export function toolEntry(registry: ToolRegistry, name: string): ToolEntry | undefined {
	return registry.tools.find((tool) => tool.name === name);
}

/** Concrete tools delivered by `stage` (e.g. "S3") or earlier. */
export function toolsUpToStage(registry: ToolRegistry, stage: string): ToolEntry[] {
	const rank = (value: string): number => Number(value.slice(1));
	return registry.tools.filter((tool) => rank(tool.stage) <= rank(stage));
}

export function assertNoUnknownTools(registry: ToolRegistry, names: readonly string[], context: string): string[] {
	const { names: expanded, unknown } = expandTools(registry, names);
	if (unknown.length > 0) {
		throw new VangjaLoopError(
			"CONFIG_INVALID",
			`${context} names unknown tool(s) ${unknown.join(", ")}. Known tools/families: ` +
				`${[...Object.keys(registry.families), ...knownToolNames(registry)].join(", ")} ` +
				`(registry: .pi/config/tools.json).`,
		);
	}
	return expanded;
}
