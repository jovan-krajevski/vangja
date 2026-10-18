/**
 * Loaders and validators for `.pi/config/*.json`.
 *
 * Every loader throws a `VangjaLoopError` naming the file when something is
 * missing or malformed. Nothing has a default that could silently change the
 * model pin, the budget, or the project policy.
 */

import * as fs from "node:fs";
import { configInvalid, configMissing, VangjaLoopError } from "./errors.ts";
import { configPath, piDir } from "./paths.ts";
import { readJsonFile } from "./io.ts";

export const THINKING_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"] as const;
export type ThinkingLevel = (typeof THINKING_LEVELS)[number];

export interface ModelPin {
	provider: string;
	model: string;
	thinking: ThinkingLevel;
	contextWindow: number;
	enforce: boolean;
}

export interface AgentManifestEntry {
	name: string;
	file: string;
	role: string;
	tools: string[];
	skills: string[];
	effort: "small" | "medium" | "large";
	maxParallel: number;
	model: string;
}

export interface AgentsManifest {
	schema: string;
	pinnedModel: string;
	pinnedThinking: ThinkingLevel;
	agents: AgentManifestEntry[];
}

export interface BudgetConfig {
	maxSessionsPerDay: number;
	maxParallelWorkers: number;
	maxParallelHpcJobs: number;
	maxTaskWallClockMinutes: number;
	maxHpcWallClockHours: number;
	maxConsecutiveFailures: number;
	maxNoProgressCycles: number;
	maxAuditCycles: number;
	maxCostUsdPerRun: number;
}

export interface ProjectConfig {
	schema: string;
	repoRoot: string;
	baseBranch: string;
	venue: string;
	requireHumanMerge: boolean;
	dryRun: boolean;
	sandboxPath: string;
	sessionDir: string;
	artifactDir: string;
	stateDir: string;
	worktreeDir: string;
}

export interface HpcConfig {
	host: string;
	user: string;
	port: number;
	remoteRepoDir: string;
	remoteScratchDir: string;
	hpcKey: string;
	gitKey: string;
	forwardGitAgent: boolean;
	container: string;
	partition: string;
	cpus: number;
	memGb: number;
	gpus: number;
	wallClockHours: number;
	pollSeconds: number;
	maxPolls: number;
	maxAttemptsPerExperiment: number;
	maxParallelJobs: number;
}

/**
 * `project.json` baseBranch value: merge into whatever branch the checkout is on
 * when the run boots. The resolved name is pinned in LOOP_STATE.base_branch, so
 * a later branch switch cannot silently retarget a run in flight.
 */
export const CURRENT_BRANCH_SENTINEL = "@current";

/**
 * Resolve the configured base branch against the checkout's current branch.
 * A literal branch name is returned unchanged; `@current` requires a real branch
 * (a detached HEAD has no merge target and is refused loudly).
 */
export function resolveBaseBranch(configured: string, current: string | null): string {
	const value = configured.trim();
	if (value !== CURRENT_BRANCH_SENTINEL) return value;
	const branch = (current ?? "").trim();
	if (branch === "" || branch === "HEAD") {
		throw new VangjaLoopError(
			"CONFIG_INVALID",
			`.pi/config/project.json baseBranch is "${CURRENT_BRANCH_SENTINEL}" but the checkout is on a detached HEAD ` +
				`(no current branch). Check out the branch the loop should merge into, or set "baseBranch" to a literal branch name.`,
		);
	}
	return branch;
}

/** plan/02_ARCHITECTURE.md §6 — exact defaults. */
export const DEFAULT_BUDGETS: BudgetConfig = {
	maxSessionsPerDay: 120,
	maxParallelWorkers: 4,
	maxParallelHpcJobs: 1,
	maxTaskWallClockMinutes: 360,
	maxHpcWallClockHours: 48,
	maxConsecutiveFailures: 3,
	maxNoProgressCycles: 2,
	maxAuditCycles: 5,
	maxCostUsdPerRun: 250,
};

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

function requireBoolean(file: string, obj: Record<string, unknown>, key: string): boolean {
	const value = obj[key];
	if (typeof value !== "boolean") throw configInvalid(file, `"${key}" must be a boolean`);
	return value;
}

function requireNumber(file: string, obj: Record<string, unknown>, key: string, min = 0): number {
	const value = obj[key];
	if (typeof value !== "number" || !Number.isFinite(value) || value < min) {
		throw configInvalid(file, `"${key}" must be a number >= ${min}`);
	}
	return value;
}

function parseJsonObject(file: string, raw: unknown): Record<string, unknown> {
	if (!isPlainObject(raw)) throw configInvalid(file, "top level must be a JSON object");
	return raw;
}

export function parseModelPin(raw: unknown, file: string): ModelPin {
	const obj = parseJsonObject(file, raw);
	const pin: ModelPin = {
		provider: requireString(file, obj, "provider"),
		model: requireString(file, obj, "model"),
		thinking: requireString(file, obj, "thinking") as ThinkingLevel,
		contextWindow: requireNumber(file, obj, "contextWindow", 1),
		enforce: requireBoolean(file, obj, "enforce"),
	};
	if (!(THINKING_LEVELS as readonly string[]).includes(pin.thinking)) {
		throw configInvalid(file, `"thinking" must be one of ${THINKING_LEVELS.join(", ")}`);
	}
	if (obj.contextWindow !== 1_000_000 && pin.enforce) {
		throw configInvalid(
			file,
			`"contextWindow" must be 1000000 while enforce=true (plan/02_ARCHITECTURE.md §7)`,
		);
	}
	return pin;
}

/** The model id used by `--model` (provider/id). */
export function pinnedModelId(pin: ModelPin): string {
	return `${pin.provider}/${pin.model}`;
}

export function loadModelPin(root: string): ModelPin {
	const file = configPath(root, "model.json");
	if (!fs.existsSync(file)) throw configMissing(file);
	return parseModelPin(readJsonFile<unknown>(file), file);
}

export function parseBudgetConfig(raw: unknown, file: string): BudgetConfig {
	const obj = parseJsonObject(file, raw);
	const budgets = {} as BudgetConfig;
	for (const key of Object.keys(DEFAULT_BUDGETS) as (keyof BudgetConfig)[]) {
		budgets[key] = requireNumber(file, obj, key, 1);
	}
	return budgets;
}

export function loadBudgetConfig(root: string): BudgetConfig {
	const file = configPath(root, "budgets.json");
	if (!fs.existsSync(file)) throw configMissing(file);
	return parseBudgetConfig(readJsonFile<unknown>(file), file);
}

export function parseProjectConfig(raw: unknown, file: string): ProjectConfig {
	const obj = parseJsonObject(file, raw);
	const config: ProjectConfig = {
		schema: requireString(file, obj, "schema"),
		repoRoot: requireString(file, obj, "repoRoot"),
		baseBranch: requireString(file, obj, "baseBranch"),
		venue: requireString(file, obj, "venue"),
		requireHumanMerge: requireBoolean(file, obj, "requireHumanMerge"),
		dryRun: requireBoolean(file, obj, "dryRun"),
		sandboxPath: requireString(file, obj, "sandboxPath"),
		sessionDir: requireString(file, obj, "sessionDir"),
		artifactDir: requireString(file, obj, "artifactDir"),
		stateDir: requireString(file, obj, "stateDir"),
		worktreeDir: requireString(file, obj, "worktreeDir"),
	};
	return config;
}

export function loadProjectConfig(root: string): ProjectConfig {
	const file = configPath(root, "project.json");
	if (!fs.existsSync(file)) throw configMissing(file);
	return parseProjectConfig(readJsonFile<unknown>(file), file);
}

const EFFORT_CLASSES = ["small", "medium", "large"] as const;

export function parseAgentsManifest(raw: unknown, file: string): AgentsManifest {
	const obj = parseJsonObject(file, raw);
	const pinnedModel = requireString(file, obj, "pinnedModel");
	const pinnedThinking = requireString(file, obj, "pinnedThinking") as ThinkingLevel;
	if (!(THINKING_LEVELS as readonly string[]).includes(pinnedThinking)) {
		throw configInvalid(file, `"pinnedThinking" must be one of ${THINKING_LEVELS.join(", ")}`);
	}
	const rawAgents = obj.agents;
	if (!Array.isArray(rawAgents) || rawAgents.length === 0) {
		throw configInvalid(file, `"agents" must be a non-empty array`);
	}
	const seen = new Set<string>();
	const agents: AgentManifestEntry[] = rawAgents.map((entry, index) => {
		if (!isPlainObject(entry)) throw configInvalid(file, `agents[${index}] must be an object`);
		const name = requireString(file, entry, "name");
		if (seen.has(name)) throw configInvalid(file, `duplicate agent name "${name}"`);
		seen.add(name);
		const tools = entry.tools;
		const skills = entry.skills;
		const maxParallel = requireNumber(file, entry, "maxParallel", 1);
		const effort = requireString(file, entry, "effort");
		if (!(EFFORT_CLASSES as readonly string[]).includes(effort)) {
			throw configInvalid(file, `agents[${index}].effort must be one of ${EFFORT_CLASSES.join(", ")}`);
		}
		if (!Array.isArray(tools) || tools.some((t) => typeof t !== "string")) {
			throw configInvalid(file, `agents[${index}].tools must be an array of strings`);
		}
		if (!Array.isArray(skills) || skills.some((s) => typeof s !== "string")) {
			throw configInvalid(file, `agents[${index}].skills must be an array of strings`);
		}
		const model = requireString(file, entry, "model");
		return {
			name,
			file: requireString(file, entry, "file"),
			role: requireString(file, entry, "role"),
			tools: tools as string[],
			skills: skills as string[],
			effort: effort as AgentManifestEntry["effort"],
			maxParallel,
			model,
		};
	});
	return { schema: requireString(file, obj, "schema"), pinnedModel, pinnedThinking, agents };
}

export function loadAgentsManifest(root: string): AgentsManifest {
	const file = configPath(root, "agents.json");
	if (!fs.existsSync(file)) throw configMissing(file);
	return parseAgentsManifest(readJsonFile<unknown>(file), file);
}

export function parseHpcConfig(raw: unknown, file: string): HpcConfig {
	const obj = parseJsonObject(file, raw);
	const config: HpcConfig = {
		host: typeof obj.host === "string" ? obj.host : "",
		user: typeof obj.user === "string" ? obj.user : "",
		port: requireNumber(file, obj, "port", 1),
		remoteRepoDir: typeof obj.remoteRepoDir === "string" ? obj.remoteRepoDir : "",
		remoteScratchDir: typeof obj.remoteScratchDir === "string" ? obj.remoteScratchDir : "",
		hpcKey: typeof obj.hpcKey === "string" ? obj.hpcKey : "",
		gitKey: typeof obj.gitKey === "string" ? obj.gitKey : "",
		forwardGitAgent: requireBoolean(file, obj, "forwardGitAgent"),
		container: typeof obj.container === "string" ? obj.container : "",
		partition: typeof obj.partition === "string" ? obj.partition : "",
		cpus: requireNumber(file, obj, "cpus", 1),
		memGb: requireNumber(file, obj, "memGb", 1),
		gpus: requireNumber(file, obj, "gpus", 0),
		wallClockHours: requireNumber(file, obj, "wallClockHours", 1),
		pollSeconds: requireNumber(file, obj, "pollSeconds", 1),
		maxPolls: requireNumber(file, obj, "maxPolls", 1),
		maxAttemptsPerExperiment: requireNumber(file, obj, "maxAttemptsPerExperiment", 1),
		maxParallelJobs: requireNumber(file, obj, "maxParallelJobs", 1),
	};
	return config;
}

/** Amendment records are append-only; read them without importing the ledger module. */
function readAmendmentRecords(file: string): Array<Record<string, unknown>> {
	try {
		const raw = fs.readFileSync(file, "utf8");
		return raw
			.split("\n")
			.filter((line) => line.trim() !== "")
			.map((line) => JSON.parse(line) as Record<string, unknown>);
	} catch {
		return [];
	}
}

/**
 * Load `.pi/config/hpc.json`. A missing file is a valid "disabled" state and is
 * reported as such — it never enables SSH and never throws.
 */
export function loadHpcConfig(root: string): { enabled: boolean; config: HpcConfig | null; reason: string } {
	const file = configPath(root, "hpc.json");
	if (!fs.existsSync(file)) {
		return { enabled: false, config: null, reason: `no ${file}; the HPC path is disabled` };
	}
	const config = parseHpcConfig(readJsonFile<unknown>(file), file);
	if (config.host.trim() === "") {
		return { enabled: false, config, reason: `${file} has an empty "host"; the HPC path is disabled` };
	}
	if (config.maxParallelJobs > 1) {
		const amendmentsFile = `${piDir(root)}/state/AMENDMENTS.jsonl`;
		const covering = readAmendmentRecords(amendmentsFile).some(
			(record) => /hpc/i.test(String(record.path ?? "")) && /maxParallelJobs/i.test(String(record.change ?? "")),
		);
		if (!covering) {
			throw new VangjaLoopError(
				"CONFIG_INVALID",
				`${file}: maxParallelJobs=${config.maxParallelJobs} > 1 is rejected unless an amendment covering ` +
					`maxParallelJobs is recorded in ${amendmentsFile} (amendment_record with path ".pi/config/hpc.json"). ` +
					`The single-slot rule is a hard project constraint (plan/06_HPC_AND_COLLECTOR.md).`,
			);
		}
	}
	return { enabled: true, config, reason: `${file} is configured for host ${config.host}` };
}

export function loadHpcExample(root: string): HpcConfig {
	const file = configPath(root, "hpc.example.json");
	if (!fs.existsSync(file)) throw configMissing(file);
	return parseHpcConfig(readJsonFile<unknown>(file), file);
}
