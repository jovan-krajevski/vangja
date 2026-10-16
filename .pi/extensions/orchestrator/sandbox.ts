/**
 * Run context: repo root, `--sandbox`, `--dry-run`, the child `pi` executable,
 * and flag parsing shared by the CLI and the `/loop:*` commands.
 *
 * `--sandbox <path>` redirects every read and write of the loop to a scratch
 * clone: the child processes get `PI_LOOP_ROOT=<sandbox>` and nothing under the
 * real repository is touched. The safety rule is absolute: the sandbox may not
 * be the real repository root, and it must look like a checkout (`.pi/` +
 * `AGENTS.md`) — otherwise the flag is refused by name.
 *
 * `--dry-run` writes exactly one file, `state/DRYRUN.jsonl` (the action log),
 * and no ledgers, worktrees, or branches.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { appendLineDurable } from "../../lib/io.ts";
import { resolveRepoRoot, statePath } from "../../lib/paths.ts";
import { VangjaLoopError } from "../../lib/errors.ts";

export interface CliFlags {
	sandbox?: string;
	dryRun: boolean;
	headless: boolean;
	json: boolean;
	forever: boolean;
	maxSteps: number;
	command?: string;
	/** Positional arguments after the command (e.g. gate id / note). */
	positional: string[];
}

export function parseFlags(argv: string[]): CliFlags {
	const flags: CliFlags = {
		dryRun: false,
		headless: false,
		json: false,
		forever: false,
		maxSteps: 60,
		positional: [],
	};
	for (let i = 0; i < argv.length; i += 1) {
		const arg = argv[i];
		if (arg === "--sandbox") {
			flags.sandbox = argv[++i];
			if (!flags.sandbox) throw new VangjaLoopError("PATH_INVALID", "--sandbox requires a path");
			continue;
		}
		if (arg === "--dry-run") {
			flags.dryRun = true;
			continue;
		}
		if (arg === "--headless") {
			flags.headless = true;
			continue;
		}
		if (arg === "--json") {
			flags.json = true;
			continue;
		}
		if (arg === "--forever") {
			flags.forever = true;
			continue;
		}
		if (arg === "--max-steps") {
			const value = Number(argv[++i]);
			if (!Number.isFinite(value) || value < 1) {
				throw new VangjaLoopError("PATH_INVALID", "--max-steps requires a positive integer");
			}
			flags.maxSteps = Math.floor(value);
			continue;
		}
		if (arg === "--command") {
			flags.command = argv[++i];
			if (!flags.command) throw new VangjaLoopError("PATH_INVALID", "--command requires an executable");
			continue;
		}
		if (arg.startsWith("-")) {
			throw new VangjaLoopError("PATH_INVALID", `unknown flag "${arg}" (see .pi/loop/README.md)`);
		}
		flags.positional.push(arg);
	}
	return flags;
}

export interface RunContext {
	/** Root all loop operations use (the sandbox when set). */
	root: string;
	/** The real repository root, never modified when a sandbox is set. */
	realRoot: string;
	sandbox: string | null;
	dryRun: boolean;
	headless: boolean;
	/** The `pi` executable used for child processes. */
	command: string;
}

export interface ResolveOptions {
	/** Override for tests; defaults to the module's repository root. */
	realRoot?: string;
}

export function resolveRunContext(flags: CliFlags, options: ResolveOptions = {}): RunContext {
	const realRoot = options.realRoot ?? resolveRepoRoot(process.env.PI_LOOP_REAL_ROOT ?? undefined);
	const command = flags.command ?? process.env.PI_LOOP_COMMAND ?? "pi";
	if (flags.sandbox) {
		const sandbox = path.resolve(flags.sandbox);
		assertSandboxIsSafe(realRoot, sandbox);
		return { root: sandbox, realRoot, sandbox, dryRun: flags.dryRun, headless: flags.headless, command };
	}
	// A child session inherits PI_LOOP_ROOT (the conductor sets it). When it points
	// away from the real root it is a sandbox and gets the same safety check.
	const fromEnv = process.env.PI_LOOP_ROOT?.trim() ? path.resolve(process.env.PI_LOOP_ROOT) : null;
	if (fromEnv) {
		let isRealRoot = false;
		try {
			isRealRoot = fs.realpathSync(fromEnv) === fs.realpathSync(realRoot);
		} catch {
			isRealRoot = false;
		}
		if (!isRealRoot) {
			assertSandboxIsSafe(realRoot, fromEnv);
			return { root: fromEnv, realRoot, sandbox: fromEnv, dryRun: flags.dryRun, headless: flags.headless, command };
		}
		return { root: fromEnv, realRoot, sandbox: null, dryRun: flags.dryRun, headless: flags.headless, command };
	}
	return { root: realRoot, realRoot, sandbox: null, dryRun: flags.dryRun, headless: flags.headless, command };
}

/** Refuse a sandbox that is the real root, missing, or not a checkout. */
export function assertSandboxIsSafe(realRoot: string, sandbox: string): void {
	const real = fs.realpathSync(realRoot);
	let resolvedSandbox: string;
	try {
		resolvedSandbox = fs.realpathSync(sandbox);
	} catch {
		throw new VangjaLoopError(
			"PATH_INVALID",
			`--sandbox ${sandbox} does not exist. Create a scratch clone first ` +
				`(git clone --local <repo> <path>) — the loop never invents a repository.`,
		);
	}
	if (resolvedSandbox === real) {
		throw new VangjaLoopError(
			"PATH_INVALID",
			`--sandbox ${sandbox} resolves to the real repository root; refusing to run the loop against it. ` +
				`Use a scratch clone (tests and dry runs must never touch the real checkout).`,
		);
	}
	if (!fs.existsSync(path.join(resolvedSandbox, ".pi"))) {
		throw new VangjaLoopError("PATH_INVALID", `--sandbox ${sandbox} has no .pi directory; not a loop checkout`);
	}
	if (!fs.existsSync(path.join(resolvedSandbox, "AGENTS.md"))) {
		throw new VangjaLoopError("PATH_INVALID", `--sandbox ${sandbox} has no AGENTS.md; not a loop checkout`);
	}
}

export function dryRunLogPath(root: string): string {
	return statePath(root, "DRYRUN.jsonl");
}

/** The only write a dry run performs. */
export function logDryRun(root: string, entry: Record<string, unknown>): void {
	const line = JSON.stringify({ ts: new Date().toISOString(), ...entry });
	appendLineDurable(dryRunLogPath(root), line);
}

/** Environment for a child process (documented in the S4 contract). */
export function childEnvironment(context: RunContext, extra: NodeJS.ProcessEnv = {}): NodeJS.ProcessEnv {
	const env: NodeJS.ProcessEnv = { ...process.env, ...extra };
	env.PI_LOOP_ROOT = context.root;
	if (context.sandbox) env.PI_LOOP_SANDBOX = context.sandbox;
	else delete env.PI_LOOP_SANDBOX;
	env.PI_LOOP_REAL_ROOT = context.realRoot;
	return env;
}
