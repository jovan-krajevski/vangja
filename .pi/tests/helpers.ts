/**
 * Test helpers. Tests run under `node --test` with Node's built-in TypeScript
 * type stripping, so there is no build step and no dependency.
 */

import { execFileSync, spawn, spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

export const TESTS_DIR = path.dirname(fileURLToPath(import.meta.url));
export const PI_DIR = path.dirname(TESTS_DIR);
export const REPO_ROOT = path.dirname(PI_DIR);
export const FIXTURES_DIR = path.join(TESTS_DIR, "fixtures");

export const DEFAULT_PIN = {
	provider: "opencode-go",
	model: "deepseek-v4.1-flash",
	thinking: "max",
	contextWindow: 1_000_000,
	enforce: true,
};

/** A disposable repo-like root: .pi/config, .pi/state, AGENTS.md. */
export function scratchRoot(prefix = "vangja-pi-"): string {
	const root = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
	fs.mkdirSync(path.join(root, ".pi", "config"), { recursive: true });
	fs.mkdirSync(path.join(root, ".pi", "state"), { recursive: true });
	fs.writeFileSync(path.join(root, "AGENTS.md"), "# scratch project\n");
	writeJson(path.join(root, ".pi", "config", "model.json"), DEFAULT_PIN);
	fs.copyFileSync(path.join(PI_DIR, "config", "budgets.json"), path.join(root, ".pi", "config", "budgets.json"));
	return root;
}

export function writeJson(file: string, value: unknown): void {
	fs.mkdirSync(path.dirname(file), { recursive: true });
	fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
}

export function readRepoConfig<T = unknown>(name: string): T {
	return JSON.parse(fs.readFileSync(path.join(PI_DIR, "config", name), "utf8")) as T;
}

export function readRepoJson<T = unknown>(relative: string): T {
	return JSON.parse(fs.readFileSync(path.join(REPO_ROOT, relative), "utf8")) as T;
}

/** Minimal stand-in for `ExtensionAPI` that records handlers. */
export function fakePi() {
	const handlers = new Map<string, Array<(...args: unknown[]) => unknown>>();
	const api = {
		on(event: string, handler: (...args: unknown[]) => unknown) {
			const list = handlers.get(event) ?? [];
			list.push(handler);
			handlers.set(event, list);
			return () => {};
		},
		registerTool() {},
		registerCommand() {},
		registerToolRenderer() {},
		registerShortcut() {},
		registerFlag() {},
		registerMessageRenderer() {},
		registerEntryRenderer() {},
		registerMarkdownTransformer() {},
		appendEntry() {},
		sendMessage() {},
		sendUserMessage() {},
		getActiveTools: () => [],
		setActiveTools() {},
		getThinkingLevel: () => "max",
		setModel: async () => true,
		exec: async () => ({ stdout: "", stderr: "", code: 0 }),
	};
	async function fire(event: string, ...args: unknown[]): Promise<unknown> {
		const list = handlers.get(event) ?? [];
		const results: unknown[] = [];
		for (const handler of list) {
			const result = await handler(...args);
			if (result !== undefined) results.push(result);
		}
		return results.length <= 1 ? results[0] : results;
	}
	function count(event: string): number {
		return (handlers.get(event) ?? []).length;
	}
	return { api, fire, count, handlers };
}

/** Minimal stand-in for `ExtensionContext`. */
export function fakeCtx(overrides: Record<string, unknown> = {}) {
	return {
		model: { provider: "opencode-go", id: "deepseek-v4.1-flash", name: "deepseek-v4.1-flash" },
		thinkingLevel: "max",
		cwd: REPO_ROOT,
		mode: "print",
		hasUI: false,
		ui: {
			notify: () => {},
			setStatus: () => {},
			setWidget: () => {},
			confirm: async () => true,
			select: async () => undefined,
		},
		sessionManager: { getBranch: () => [] },
		abort: () => {},
		isProjectTrusted: () => true,
		...overrides,
	};
}

export function resetExitCode(): void {
	process.exitCode = 0;
}

export function findPiPackage(): string | null {
	try {
		const resolved = execFileSync("bash", ["-lc", 'command -v pi && readlink -f "$(command -v pi)"'], {
			encoding: "utf8",
		}).trim();
		const bin = resolved.split("\n").pop() ?? "";
		let dir = path.dirname(bin);
		for (;;) {
			const packageFile = path.join(dir, "package.json");
			if (fs.existsSync(packageFile)) {
				const pkg = JSON.parse(fs.readFileSync(packageFile, "utf8")) as { name?: string; version?: string };
				if (pkg.name === "@earendil-works/pi-coding-agent") return dir;
			}
			const parent = path.dirname(dir);
			if (parent === dir) return null;
			dir = parent;
		}
	} catch {
		return null;
	}
}

export function piVersion(): string | null {
	const pkgDir = findPiPackage();
	if (!pkgDir) return null;
	const pkg = JSON.parse(fs.readFileSync(path.join(pkgDir, "package.json"), "utf8")) as { version?: string };
	return pkg.version ?? null;
}

export interface SpawnResult {
	status: number | null;
	stdout: string;
	stderr: string;
}

export function spawnSyncNode(
	file: string,
	args: string[],
	options: { cwd?: string; env?: NodeJS.ProcessEnv; timeoutMs?: number } = {},
): SpawnResult {
	const result = spawnSync(file, args, {
		cwd: options.cwd ?? REPO_ROOT,
		env: { ...process.env, ...options.env },
		timeout: options.timeoutMs ?? 180_000,
		encoding: "utf8",
	});
	return { status: result.status, stdout: String(result.stdout ?? ""), stderr: String(result.stderr ?? "") };
}

/** Launch a child without blocking; used when a test must run processes concurrently. */
export function spawnAsyncNode(
	file: string,
	args: string[],
	options: { cwd?: string; env?: NodeJS.ProcessEnv; timeoutMs?: number } = {},
): Promise<SpawnResult> {
	return new Promise((resolve, reject) => {
		const child = spawn(file, args, {
			cwd: options.cwd ?? REPO_ROOT,
			env: { ...process.env, ...options.env },
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stdout = "";
		let stderr = "";
		let timer: NodeJS.Timeout | undefined;
		if (options.timeoutMs) {
			timer = setTimeout(() => {
				child.kill("SIGKILL");
				reject(new Error(`spawnAsyncNode timed out after ${options.timeoutMs}ms: ${file}`));
			}, options.timeoutMs);
		}
		child.stdout.on("data", (chunk) => {
			stdout += String(chunk);
		});
		child.stderr.on("data", (chunk) => {
			stderr += String(chunk);
		});
		child.on("error", (error) => {
			if (timer) clearTimeout(timer);
			reject(error);
		});
		child.on("close", (status) => {
			if (timer) clearTimeout(timer);
			resolve({ status, stdout, stderr });
		});
	});
}
