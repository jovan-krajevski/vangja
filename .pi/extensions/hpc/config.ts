/**
 * HPC config resolution for the `hpc_*` tools.
 *
 * `.pi/config/hpc.json` is runtime-only (gitignored). `host: ""` disables the
 * whole path: every tool returns `{enabled:false}` before doing any I/O.
 * Validation on load (plan/06_HPC_AND_COLLECTOR.md §1): required fields present
 * when enabled, key files exist and are readable, and `maxParallelJobs > 1` is
 * rejected unless a matching amendment is recorded.
 *
 * No pi imports: `node --test` drives this module directly.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { loadHpcConfig, type HpcConfig } from "../../lib/config.ts";
import { VangjaLoopError } from "../../lib/errors.ts";
import { scanForSecrets } from "../../lib/secrets.ts";

export interface HpcRuntime {
	root: string;
	config: HpcConfig;
	/** Why the config is considered usable (goes into `hpc_config`). */
	reason: string;
	/** `~` expanded; the path is not secret, its contents are. */
	hpcKeyPath: string;
	/** `~` expanded, or "" when unset. Never read into memory except by ssh-add. */
	gitKeyPath: string;
}

export type HpcResolution = { enabled: true; runtime: HpcRuntime } | { enabled: false; reason: string; config: HpcConfig | null };

export function expandHome(value: string): string {
	if (value === "~") return os.homedir();
	if (value.startsWith("~/")) return path.join(os.homedir(), value.slice(2));
	return value;
}

/** Readability probe that never reads the file contents. */
export function checkReadableKey(file: string): { ok: boolean; reason: string } {
	if (file === "") return { ok: false, reason: "path is empty" };
	try {
		fs.accessSync(file, fs.constants.R_OK);
	} catch (error) {
		return { ok: false, reason: `${file} is not readable (${(error as Error).message})` };
	}
	const stat = fs.statSync(file);
	if (!stat.isFile()) return { ok: false, reason: `${file} is not a regular file` };
	return { ok: true, reason: `${file} is readable` };
}

/**
 * Resolve the HPC runtime. Throws `VangjaLoopError` when the config is present
 * but unusable; returns `{enabled:false}` when the host is empty (a valid,
 * silent state that must never touch the network).
 */
export function resolveHpc(root: string): HpcResolution {
	const loaded = loadHpcConfig(root);
	if (!loaded.enabled || loaded.config === null) {
		return { enabled: false, reason: loaded.reason, config: loaded.config };
	}
	const config = loaded.config;
	const missing: string[] = [];
	if (config.user.trim() === "") missing.push('"user"');
	if (config.remoteRepoDir.trim() === "") missing.push('"remoteRepoDir"');
	if (config.remoteScratchDir.trim() === "") missing.push('"remoteScratchDir"');
	if (config.hpcKey.trim() === "") missing.push('"hpcKey"');
	if (missing.length > 0) {
		throw new VangjaLoopError(
			"CONFIG_INVALID",
			`.pi/config/hpc.json has host "${config.host}" but is missing ${missing.join(", ")}; ` +
				`fill every required field or set "host" to "" to disable HPC deliberately.`,
		);
	}
	const hpcKeyPath = expandHome(config.hpcKey);
	const key = checkReadableKey(hpcKeyPath);
	if (!key.ok) {
		throw new VangjaLoopError(
			"CONFIG_INVALID",
			`.pi/config/hpc.json hpcKey ${key.reason}; the HPC path is enabled (host "${config.host}") ` +
				`but no SSH key can be used. Fix the path or set "host" to "".`,
		);
	}
	const gitKeyPath = config.gitKey.trim() === "" ? "" : expandHome(config.gitKey);
	return {
		enabled: true,
		runtime: {
			root,
			config,
			reason: `host ${config.host}:${config.port} as ${config.user}; repo ${config.remoteRepoDir}; scratch ${config.remoteScratchDir}`,
			hpcKeyPath,
			gitKeyPath,
		},
	};
}

/** The host as shown in tool output: present enough to identify, not to connect. */
export function redactHost(host: string): string {
	if (host.length <= 2) return "*".repeat(host.length);
	if (host.length <= 6) return `${host[0]}${"*".repeat(host.length - 2)}${host[host.length - 1]}`;
	return `${host.slice(0, 3)}${"*".repeat(Math.min(6, host.length - 6))}${host.slice(-3)}`;
}

/** Strip any credential-shaped substring from text before it is logged/returned. */
export function redact(text: string): string {
	return scanForSecrets(text).redacted;
}

/** Truncate long command output for a ledger evidence field. */
export function trimOutput(text: string, maxChars = 4000): string {
	const clean = redact(text.trim());
	if (clean.length <= maxChars) return clean;
	return `${clean.slice(0, maxChars)}\n… [trimmed at ${maxChars} chars]`;
}

export interface ConfigSummary {
	enabled: boolean;
	reason: string;
	host?: string;
	hostRedacted?: string;
	user?: string;
	port?: number;
	remoteRepoDir?: string;
	remoteScratchDir?: string;
	hpcKey?: string;
	gitKey?: string;
	forwardGitAgent?: boolean;
	container?: string;
	partition?: string;
	cpus?: number;
	memGb?: number;
	gpus?: number;
	wallClockHours?: number;
	pollSeconds?: number;
	maxPolls?: number;
	maxAttemptsPerExperiment?: number;
	maxParallelJobs?: number;
}

/** The `hpc_config` payload. Never includes key material, only key paths. */
export function configSummary(resolution: HpcResolution): ConfigSummary {
	if (!resolution.enabled) return { enabled: false, reason: resolution.reason };
	const runtime = resolution.runtime;
	const config = runtime.config;
	return {
		enabled: true,
		reason: runtime.reason,
		host: config.host,
		hostRedacted: redactHost(config.host),
		user: config.user,
		port: config.port,
		remoteRepoDir: config.remoteRepoDir,
		remoteScratchDir: config.remoteScratchDir,
		hpcKey: runtime.hpcKeyPath,
		gitKey: runtime.gitKeyPath,
		forwardGitAgent: config.forwardGitAgent,
		container: config.container,
		partition: config.partition,
		cpus: config.cpus,
		memGb: config.memGb,
		gpus: config.gpus,
		wallClockHours: config.wallClockHours,
		pollSeconds: config.pollSeconds,
		maxPolls: config.maxPolls,
		maxAttemptsPerExperiment: config.maxAttemptsPerExperiment,
		maxParallelJobs: config.maxParallelJobs,
	};
}

/** Remote run directory for one experiment: `remoteRepoDir/runs/<id>`. */
export function remoteRunDir(runtime: HpcRuntime, experimentId: string): string {
	return `${runtime.config.remoteRepoDir.replace(/\/+$/, "")}/runs/${experimentId}`;
}

/** Remote scratch directory for one experiment: `remoteScratchDir/<id>`. */
export function remoteScratchDir(runtime: HpcRuntime, experimentId: string): string {
	return `${runtime.config.remoteScratchDir.replace(/\/+$/, "")}/${experimentId}`;
}

/** Quote a value for a POSIX shell command executed over ssh. */
export function shq(value: string): string {
	return `'${value.replace(/'/g, `'\\''`)}'`;
}
