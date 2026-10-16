/**
 * Shared helpers for the orchestrator tests.
 *
 * Tests never touch the real repository: every run happens in a scratch clone
 * under `.pi/state/orchestrator-tests/`, driven through the real CLI with the
 * deterministic child stub (`PI_LOOP_COMMAND`). The clone's `.pi/state` starts
 * empty because it is gitignored.
 */

import { execFileSync, spawnSync } from "node:child_process";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";
import { REPO_ROOT } from "../helpers.ts";
import type { LedgerView, LoopOptions } from "../../extensions/orchestrator/state-machine.ts";
import { emptyBudget } from "../../extensions/orchestrator/budget.ts";
import { defaultLoopState, type LoopState } from "../../lib/state.ts";
import { DEFAULT_BUDGETS } from "../../lib/config.ts";

export const CLI = path.join(REPO_ROOT, ".pi", "extensions", "orchestrator", "cli.ts");
export const STUB_CHILD = path.join(REPO_ROOT, ".pi", "tests", "fixtures", "orchestrator_stub_child.mjs");
export const TESTS_ROOT = path.join(REPO_ROOT, ".pi", "state", "orchestrator-tests");

export interface Sandbox {
	root: string;
	cleanup: () => void;
}

export function createSandbox(label: string): Sandbox {
	fs.mkdirSync(TESTS_ROOT, { recursive: true });
	const dir = fs.mkdtempSync(path.join(TESTS_ROOT, `${label}-`));
	const result = spawnSync("git", ["clone", "--quiet", "--local", REPO_ROOT, dir], { encoding: "utf8" });
	if (result.status !== 0) {
		throw new Error(`git clone into ${dir} failed: ${result.stderr || result.stdout}`);
	}
	execFileSync("git", ["-C", dir, "checkout", "--quiet", "update_pymc"], { encoding: "utf8" });
	fs.mkdirSync(path.join(dir, ".pi", "state", "tasks"), { recursive: true });
	return {
		root: dir,
		cleanup: () => {
			try {
				fs.rmSync(dir, { recursive: true, force: true });
			} catch {
				// Best effort: the directory is gitignored test residue.
			}
		},
	};
}

export interface CliRun {
	status: number;
	stdout: string;
	stderr: string;
}

export function runCli(root: string, args: string[], env: NodeJS.ProcessEnv = {}): CliRun {
	const result = spawnSync("node", [CLI, ...args, "--sandbox", root, "--headless", "--command", STUB_CHILD], {
		cwd: REPO_ROOT,
		encoding: "utf8",
		env: { ...process.env, ...env },
		timeout: 180_000,
	});
	return { status: result.status ?? -1, stdout: String(result.stdout ?? ""), stderr: String(result.stderr ?? "") };
}

export function seedTask(root: string, overrides: Record<string, unknown> = {}): void {
	const script = `
import * as fs from "node:fs";
import { appendRecord } from ${JSON.stringify(pathToFileURL(path.join(REPO_ROOT, ".pi", "extensions", "ledger", "store.ts")).href)};
const root = process.argv[2];
const overrides = JSON.parse(process.argv[3]);
const record = appendRecord(root, "task", {
  id: "T-001",
  title: "stub task for the sandbox cycle",
  origin: "PLAN:stub",
  priority: "P2",
  objective: "Exercise the orchestrator end to end in a scratch clone.",
  acceptance_criteria: ["STUB_TASK.md exists in the worktree"],
  deliverables: ["STUB_TASK.md"],
  compute: "none",
  status: "open",
  effort: "small",
  ...overrides,
}, { root, role: "conductor", actor: "test" });
fs.writeFileSync(root + "/.pi/state/tasks/" + String(record.id) + ".md", "## " + String(record.id) + " stub brief\\n");
`;
	const scriptFile = path.join(root, ".seed_task.mjs");
	fs.writeFileSync(scriptFile, script);
	const result = spawnSync("node", [scriptFile, root, JSON.stringify(overrides)], { encoding: "utf8" });
	fs.rmSync(scriptFile, { force: true });
	if (result.status !== 0) throw new Error(`seedTask failed: ${result.stderr || result.stdout}`);
}

/** Keep the pre-merge gate fast and deterministic: the tests command is `true`. */
export function writeGatesConfig(root: string, testsCommand: string[] = ["true"]): void {
	fs.writeFileSync(path.join(root, ".pi", "config", "gates.json"), `${JSON.stringify({ testsCommand }, null, 2)}\n`);
}

export function readLedger<T = Record<string, unknown>>(root: string, file: string): T[] {
	const target = path.join(root, ".pi", "state", file);
	if (!fs.existsSync(target)) return [];
	return fs
		.readFileSync(target, "utf8")
		.split("\n")
		.filter((line) => line.trim() !== "")
		.map((line) => JSON.parse(line) as T);
}

export function latestById<T extends { id?: string }>(records: T[]): T[] {
	const latest = new Map<string, T>();
	for (const record of records) {
		if (typeof record.id === "string") latest.set(record.id, record);
	}
	return [...latest.values()];
}

export function readState(root: string): LoopState {
	const file = path.join(root, ".pi", "state", "LOOP_STATE.json");
	if (!fs.existsSync(file)) return defaultLoopState();
	return { ...defaultLoopState(), ...(JSON.parse(fs.readFileSync(file, "utf8")) as Partial<LoopState>) };
}

export function readEvents(root: string): Array<Record<string, unknown>> {
	return readLedger<Record<string, unknown>>(root, "EVENTS.jsonl");
}

export function readBudgetFile(root: string): Record<string, unknown> {
	const file = path.join(root, ".pi", "state", "BUDGET.json");
	return fs.existsSync(file) ? (JSON.parse(fs.readFileSync(file, "utf8")) as Record<string, unknown>) : {};
}

export function git(root: string, args: string[]): string {
	return execFileSync("git", ["-C", root, ...args], { encoding: "utf8" }).trim();
}

export function fileHashes(dir: string): Map<string, string> {
	const hashes = new Map<string, string>();
	const walk = (current: string): void => {
		for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
			const full = path.join(current, entry.name);
			if (entry.isDirectory()) walk(full);
			else if (entry.isFile()) {
				hashes.set(path.relative(dir, full), crypto.createHash("sha256").update(fs.readFileSync(full)).digest("hex"));
			}
		}
	};
	walk(dir);
	return hashes;
}

// ---------------------------------------------------------------------------
// Pure-snapshot builders for the state-machine table tests
// ---------------------------------------------------------------------------

export function loopState(overrides: Partial<LoopState> = {}): LoopState {
	return { ...defaultLoopState(), state: "WORK", ...overrides };
}

export function loopOptions(overrides: Partial<LoopOptions> = {}): LoopOptions {
	return { requireHumanMerge: true, budgets: { ...DEFAULT_BUDGETS }, sandbox: "/tmp/sandbox", ...overrides };
}

export function view(overrides: Partial<LedgerView> = {}): LedgerView {
	return {
		tasks: [],
		prs: [],
		reviews: [],
		experiments: [],
		gates: [],
		events: [],
		findings: [],
		waivers: [],
		budget: emptyBudget(),
		readiness: null,
		decision: null,
		audit: null,
		stopFlag: false,
		baseSha: null,
		repoClean: true,
		...overrides,
	};
}

export function task(overrides: Partial<LedgerView["tasks"][number]> = {}): LedgerView["tasks"][number] {
	return { id: "T-001", title: "task", priority: "P2", status: "open", origin: "PLAN:stub", ...overrides };
}

export function pr(overrides: Partial<LedgerView["prs"][number]> = {}): LedgerView["prs"][number] {
	return {
		id: "PR-001",
		task_id: "T-001",
		status: "open",
		branch: "work/T-001-stub",
		head_sha: "a".repeat(40),
		base_sha: "b".repeat(40),
		kind: "work",
		summary: "stub PR",
		...overrides,
	};
}

export function review(overrides: Partial<LedgerView["reviews"][number]> = {}): LedgerView["reviews"][number] {
	return { id: "RV-001", pr_id: "PR-001", decision: "accept", actor: "supervisor", ...overrides };
}

export function experiment(overrides: Partial<LedgerView["experiments"][number]> = {}): LedgerView["experiments"][number] {
	return { id: "E-0001", status: "running", attempt: 1, task_id: "T-001", ...overrides };
}

export function gate(overrides: Partial<LedgerView["gates"][number]> = {}): LedgerView["gates"][number] {
	return { id: "G-001", gate: "merge", status: "pending", note: "merge PR-001", pr_id: "PR-001", ...overrides };
}

export function event(overrides: Partial<LedgerView["events"][number]> = {}): LedgerView["events"][number] {
	return { seq: 1, ts: "2026-01-01T00:00:00.000Z", actor: "conductor", summary: "event", ...overrides };
}

export function budget(overrides: Partial<LedgerView["budget"]> = {}): LedgerView["budget"] {
	return { ...emptyBudget(), ...overrides };
}
