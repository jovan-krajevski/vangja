/**
 * Agent definitions (`.pi/agents/*.md`) and the dispatch allowlist.
 *
 * One Markdown file per agent: YAML-ish frontmatter `name`, `description`,
 * `tools`, `model`, then the system-prompt body. The parser is deliberately
 * minimal and dependency-free (pi's `parseFrontmatter` is not importable from
 * `node --test`), and it fails loudly on anything it does not understand so a
 * broken agent cannot be dispatched silently.
 *
 * The dispatch allowlist is the S3 enforcement of plan/03_AGENTS.md §2 and
 * plan/05_EXTENSIONS_AND_TOOLS.md §7: the guard blocks roles without the
 * `dispatch` capability; this table additionally says *which* workers a role
 * may spawn (only phd_candidate may spawn experimenter; only audit roles may
 * spawn audit workers; primaries are conductor-spawned only).
 */

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import { agentInvalid, agentNotFound, VangjaLoopError } from "./errors.ts";
import { agentsDir } from "./paths.ts";
import { loadAgentsManifest, type AgentManifestEntry } from "./config.ts";
import type { LoopRole } from "./roles.ts";

export const AGENT_FRONTMATTER_KEYS = ["name", "description", "tools", "model"] as const;

export interface AgentDefinition {
	name: string;
	description: string;
	tools: string[];
	model: string;
	body: string;
	/** Body line count (the ≤120-line contract in research_agent/prompts/03). */
	lines: number;
	/** Repo-relative path, e.g. `.pi/agents/scout.md`. */
	file: string;
	absPath: string;
	/** SHA-256 of the whole file (recorded in every dispatch event). */
	sha256: string;
}

export const DISPATCH_CONCURRENCY = 4;
export const MAX_DISPATCH_DEPTH = 2;

export const PRIMARY_AGENTS: readonly string[] = ["phd_candidate", "collector", "supervisor"];

/**
 * Roles that may dispatch, and the workers each may spawn. Roles absent from
 * this table may not dispatch at all.
 */
export const DISPATCH_ALLOWLIST: Partial<Record<LoopRole, readonly string[]>> = {
	phd_candidate: [
		"scout",
		"planner",
		"methodologist",
		"experimenter",
		"repro_engineer",
		"literature_scout",
		"paper_writer",
		"critic",
	],
	supervisor: ["critic"],
	audit_finder: ["scout", "methodologist"],
	audit_verifier: [],
	audit_synthesizer: ["meta_review"],
	audit_adjudicator: [],
	meta_review: [],
};

function invalid(file: string, detail: string): VangjaLoopError {
	return agentInvalid(file, detail);
}

function frontmatterValue(raw: string, file: string, key: string): string {
	const value = raw.trim();
	if (value === "") throw invalid(file, `frontmatter "${key}" must not be empty`);
	if (
		(value.startsWith('"') && value.endsWith('"')) ||
		(value.startsWith("'") && value.endsWith("'"))
	) {
		return value.slice(1, -1);
	}
	return value;
}

function parseToolsValue(raw: string, file: string): string[] {
	const value = raw.trim();
	const inner = value.startsWith("[") && value.endsWith("]") ? value.slice(1, -1) : value;
	const tools = inner
		.split(",")
		.map((tool) => tool.trim())
		.filter((tool) => tool !== "");
	if (tools.length === 0) throw invalid(file, `frontmatter "tools" must name at least one tool`);
	return tools;
}

/** Parse one agent Markdown file. Throws AGENT_INVALID on any malformed input. */
export function parseAgentMarkdown(content: string, file: string): AgentDefinition {
	const lines = content.split("\n");
	if (lines[0]?.trim() !== "---") {
		throw invalid(file, `must start with a "---" frontmatter fence`);
	}
	let end = -1;
	for (let i = 1; i < lines.length; i += 1) {
		if (lines[i].trim() === "---") {
			end = i;
			break;
		}
	}
	if (end === -1) throw invalid(file, `frontmatter fence is never closed`);

	const fields = new Map<string, string>();
	for (let i = 1; i < end; i += 1) {
		const line = lines[i];
		if (line.trim() === "" || line.trimStart().startsWith("#")) continue;
		const match = /^([A-Za-z_][A-Za-z0-9_]*):(.*)$/.exec(line);
		if (!match) throw invalid(file, `cannot parse frontmatter line ${i + 1}: "${line.trim()}"`);
		const key = match[1];
		if (!(AGENT_FRONTMATTER_KEYS as readonly string[]).includes(key)) {
			throw invalid(
				file,
				`unknown frontmatter key "${key}"; allowed: ${AGENT_FRONTMATTER_KEYS.join(", ")} ` +
					`(the model and thinking level come from .pi/config/model.json, never from an agent file)`,
			);
		}
		if (fields.has(key)) throw invalid(file, `duplicate frontmatter key "${key}"`);
		fields.set(key, match[2]);
	}
	for (const key of AGENT_FRONTMATTER_KEYS) {
		if (!fields.has(key)) throw invalid(file, `missing frontmatter key "${key}"`);
	}

	const name = frontmatterValue(fields.get("name") ?? "", file, "name");
	if (!/^[a-z][a-z0-9_]*$/.test(name)) {
		throw invalid(file, `"name" must be snake_case, got "${name}"`);
	}
	const description = frontmatterValue(fields.get("description") ?? "", file, "description");
	const tools = parseToolsValue(fields.get("tools") ?? "", file);
	const model = frontmatterValue(fields.get("model") ?? "", file, "model");

	const body = lines.slice(end + 1).join("\n").replace(/^\n+/, "").replace(/\s+$/, "");
	if (body === "") throw invalid(file, `system-prompt body must not be empty`);
	const bodyLines = body.split("\n").length;

	return {
		name,
		description,
		tools,
		model,
		body,
		lines: bodyLines,
		file,
		absPath: file,
		sha256: crypto.createHash("sha256").update(content, "utf8").digest("hex"),
	};
}

function readAgentFile(root: string, absPath: string, rel: string): AgentDefinition {
	const content = fs.readFileSync(absPath, "utf8");
	const parsed = parseAgentMarkdown(content, rel);
	const expected = `${parsed.name}.md`;
	if (path.basename(absPath) !== expected) {
		throw invalid(rel, `file name must match the frontmatter name ("${parsed.name}")`);
	}
	return { ...parsed, absPath };
}

/** Load every `.pi/agents/*.md` definition, sorted by name. Throws on any invalid file. */
export function loadAgentDefinitions(root: string): AgentDefinition[] {
	const dir = agentsDir(root);
	if (!fs.existsSync(dir)) {
		throw agentNotFound(
			`no agent directory at ${dir}; expected one Markdown file per agent (plan/03_AGENTS.md)`,
		);
	}
	return fs
		.readdirSync(dir, { withFileTypes: true })
		.filter((entry) => entry.isFile() && entry.name.endsWith(".md"))
		.map((entry) => {
			const absPath = path.join(dir, entry.name);
			return readAgentFile(root, absPath, `.pi/agents/${entry.name}`);
		})
		.sort((a, b) => a.name.localeCompare(b.name));
}

export function loadAgentDefinition(root: string, name: string): AgentDefinition | undefined {
	return loadAgentDefinitions(root).find((agent) => agent.name === name);
}

export function availableAgentNames(root: string): string[] {
	return loadAgentDefinitions(root).map((agent) => agent.name);
}

export function missingAgentError(root: string, name: string): VangjaLoopError {
	let available: string[] = [];
	try {
		available = availableAgentNames(root);
	} catch {
		// The directory itself is broken; say so below.
	}
	return agentNotFound(
		`unknown agent "${name}". Available agents: ${available.length > 0 ? available.join(", ") : "none"} ` +
			`(definitions: .pi/agents/, roster: .pi/config/agents.json).`,
	);
}

export interface DispatchTarget {
	definition: AgentDefinition;
	manifest: AgentManifestEntry;
}

/**
 * Decide whether `parentRole` may dispatch `targetName`, and return both the
 * definition and the manifest entry when it may. Throws ROLE_DENIED / AGENT_NOT_FOUND.
 */
export function resolveDispatchTarget(root: string, parentRole: LoopRole, targetName: string): DispatchTarget {
	// Unknown agents fail as unknown regardless of the caller's allowlist, so a
	// typo is never reported as a role violation.
	const definition = loadAgentDefinition(root, targetName);
	if (!definition) throw missingAgentError(root, targetName);

	const allowed = DISPATCH_ALLOWLIST[parentRole] ?? [];
	if (PRIMARY_AGENTS.includes(targetName)) {
		throw new VangjaLoopError(
			"ROLE_DENIED",
			`"${parentRole}" may not dispatch primary agent "${targetName}"; primary agents are spawned by the conductor, ` +
				`one fresh process per work unit (plan/02_ARCHITECTURE.md §1, plan/03_AGENTS.md §1).`,
		);
	}
	if (!allowed.includes(targetName)) {
		throw new VangjaLoopError(
			"ROLE_DENIED",
			`"${parentRole}" may not dispatch "${targetName}". Allowed for this role: ` +
				`${allowed.length > 0 ? allowed.join(", ") : "none"}. ` +
				`The dispatch allowlist is in .pi/lib/agents.ts (plan/05_EXTENSIONS_AND_TOOLS.md §7); ` +
				`only audit roles may dispatch audit workers, and only phd_candidate may dispatch experimenter.`,
		);
	}

	const manifest = loadAgentsManifest(root);
	const entry = manifest.agents.find((agent) => agent.name === targetName);
	if (!entry) {
		throw new VangjaLoopError(
			"AGENT_NOT_FOUND",
			`agent file .pi/agents/${targetName}.md exists but "${targetName}" is not in `.pi/config/agents.json`. ` +
				`Add the roster entry or remove the orphan file.`,
		);
	}
	return { definition, manifest: entry };
}

/** Safe path/name segment: keep [A-Za-z0-9._-], replace everything else. */
export function sanitizeSegment(value: string): string {
	const cleaned = value
		.trim()
		.replace(/[^A-Za-z0-9._-]+/g, "-")
		.replace(/^-+|-+$/g, "");
	return cleaned === "" ? "unknown" : cleaned.slice(0, 96);
}

/** `<agent>__<task>__<attempt>` — the session name / actor id contract. */
export function dispatchSessionName(agent: string, taskId: string, attempt: number): string {
	return `${sanitizeSegment(agent)}__${sanitizeSegment(taskId)}__${attempt}`;
}
