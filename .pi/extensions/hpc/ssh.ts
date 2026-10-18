/**
 * SSH transport for the `hpc_*` tools: a private `ssh-agent` per loop run and
 * every remote command executed with an explicit timeout.
 *
 * Rules (plan/06_HPC_AND_COLLECTOR.md §2):
 * - the extension starts its own `ssh-agent -s` and adds **only** the configured
 *   `gitKey`; the user's agent is never touched or inherited (`SSH_AUTH_SOCK`
 *   is deleted from every child env unless it is our private socket);
 * - if `gitKey` is missing or cannot be added, `forwardGitAgent` is reported
 *   unavailable and remote git is skipped — never an implicit fallback;
 * - every command is recorded (argv minus credentials, status, trimmed output)
 *   so the collector can attach raw evidence;
 * - no key material is ever read into the process: `ssh-add` reads the file.
 *
 * No pi imports: usable from `node --test`.
 */

import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import { checkReadableKey, redact, trimOutput, type HpcRuntime } from "./config.ts";

export interface CommandRecord {
	tool: string;
	argv: string[];
	status: number;
	stdout: string;
	stderr: string;
	timedOut: boolean;
	ms: number;
	remote?: string;
}

export interface GitAgentStatus {
	ok: boolean;
	/** True when a private agent holds the git key. */
	forwarding: boolean;
	reason: string;
	socket: string | null;
	key: string | null;
}

interface AgentHandle {
	socket: string;
	pid: string | null;
	env: NodeJS.ProcessEnv;
	addedKey: string | null;
	identities: string[];
}

let agentSingleton: AgentHandle | null = null;
let agentFailure: string | null = null;
let cleanupRegistered = false;

function baseEnv(extra: NodeJS.ProcessEnv): NodeJS.ProcessEnv {
	const env: NodeJS.ProcessEnv = { ...process.env };
	// Never inherit the user's agent: the private one is the only one we use.
	delete env.SSH_AUTH_SOCK;
	delete env.SSH_AGENT_PID;
	return { ...env, ...extra };
}

function registerAgentCleanup(): void {
	if (cleanupRegistered) return;
	cleanupRegistered = true;
	process.once("exit", () => {
		if (!agentSingleton) return;
		try {
			spawnSync("ssh-agent", ["-k"], { env: agentSingleton.env, encoding: "utf8", timeout: 10_000 });
		} catch {
			// The agent dies with the process anyway; this only keeps /tmp tidy.
		}
	});
}

function parseAgentOutput(text: string): { socket: string; pid: string | null } {
	const socket = /SSH_AUTH_SOCK=([^;\n]+)/.exec(text)?.[1] ?? "";
	const pid = /SSH_AGENT_PID=([^;\n]+)/.exec(text)?.[1] ?? null;
	if (socket === "") throw new Error(`ssh-agent -s printed no SSH_AUTH_SOCK; output: ${trimOutput(text, 400)}`);
	return { socket, pid };
}

/** Start (once per process) the private agent and add only the git key. */
export function startPrivateAgent(runtime: HpcRuntime): GitAgentStatus {
	if (agentSingleton) {
		return {
			ok: true,
			forwarding: agentSingleton.addedKey !== null,
			reason: `private ssh-agent (${agentSingleton.socket}) holds ${agentSingleton.identities.length} identity/ies`,
			socket: agentSingleton.socket,
			key: agentSingleton.addedKey,
		};
	}
	if (agentFailure) return { ok: false, forwarding: false, reason: agentFailure, socket: null, key: null };

	const started = spawnSync("ssh-agent", ["-s"], { env: baseEnv({}), encoding: "utf8", timeout: 20_000 });
	if (started.error || started.status !== 0) {
		agentFailure =
			`ssh-agent is unavailable (${started.error?.message ?? `exit ${started.status}`}); ` +
			`remote git cannot be authenticated and forwardGitAgent is off.`;
		return { ok: false, forwarding: false, reason: agentFailure, socket: null, key: null };
	}
	let parsed: { socket: string; pid: string | null };
	try {
		parsed = parseAgentOutput(String(started.stdout ?? ""));
	} catch (error) {
		agentFailure = (error as Error).message;
		return { ok: false, forwarding: false, reason: agentFailure, socket: null, key: null };
	}
	const agentEnv = baseEnv({ SSH_AUTH_SOCK: parsed.socket, SSH_AGENT_PID: parsed.pid ?? "" });
	const handle: AgentHandle = { socket: parsed.socket, pid: parsed.pid, env: agentEnv, addedKey: null, identities: [] };

	if (!runtime.config.forwardGitAgent) {
		agentSingleton = handle;
		registerAgentCleanup();
		return {
			ok: true,
			forwarding: false,
			reason: `forwardGitAgent=false in .pi/config/hpc.json; remote git is skipped (rsync-only mode)`,
			socket: parsed.socket,
			key: null,
		};
	}
	if (runtime.gitKeyPath === "") {
		agentSingleton = handle;
		registerAgentCleanup();
		return {
			ok: false,
			forwarding: false,
			reason: `git-forwarding-off: .pi/config/hpc.json has no "gitKey"; remote git is skipped, never the user's agent`,
			socket: parsed.socket,
			key: null,
		};
	}
	const readable = checkReadableKey(runtime.gitKeyPath);
	if (!readable.ok) {
		agentSingleton = handle;
		registerAgentCleanup();
		return {
			ok: false,
			forwarding: false,
			reason: `git-forwarding-off: ${readable.reason}; remote git is skipped, never the user's agent`,
			socket: parsed.socket,
			key: null,
		};
	}
	const added = spawnSync("ssh-add", [runtime.gitKeyPath], { env: agentEnv, encoding: "utf8", timeout: 30_000 });
	if (added.error || added.status !== 0) {
		agentSingleton = handle;
		registerAgentCleanup();
		return {
			ok: false,
			forwarding: false,
			reason:
				`git-forwarding-off: ssh-add ${runtime.gitKeyPath} failed ` +
				`(${trimOutput(`${added.stderr ?? ""} ${added.stdout ?? ""}`.trim() || `exit ${added.status}`, 300)}); ` +
				`remote git is skipped, never the user's agent`,
			socket: parsed.socket,
			key: null,
		};
	}
	const listed = spawnSync("ssh-add", ["-l"], { env: agentEnv, encoding: "utf8", timeout: 15_000 });
	handle.addedKey = runtime.gitKeyPath;
	handle.identities = String(listed.stdout ?? "")
		.split("\n")
		.filter((line) => line.trim() !== "");
	agentSingleton = handle;
	registerAgentCleanup();
	return {
		ok: handle.identities.length > 0,
		forwarding: handle.identities.length > 0,
		reason:
			handle.identities.length > 0
				? `private ssh-agent holds exactly ${handle.identities.length} identity/ies (${runtime.gitKeyPath})`
				: `ssh-add reported success but the agent lists no identity; remote git is skipped`,
		socket: parsed.socket,
		key: runtime.gitKeyPath,
	};
}

/** Test/reset hook: forget the in-process agent (does not kill it). */
export function resetPrivateAgent(): void {
	agentSingleton = null;
	agentFailure = null;
}

export function agentStatus(runtime: HpcRuntime | null): GitAgentStatus | null {
	if (runtime === null) return null;
	if (!agentSingleton && !agentFailure) return null;
	if (agentFailure) return { ok: false, forwarding: false, reason: agentFailure, socket: null, key: null };
	if (!agentSingleton) return null;
	return {
		ok: agentSingleton.addedKey !== null,
		forwarding: agentSingleton.addedKey !== null && agentSingleton.identities.length > 0,
		reason:
			agentSingleton.addedKey !== null
				? `private ssh-agent holds ${agentSingleton.identities.length} identity/ies`
				: "private ssh-agent started but holds no git key",
		socket: agentSingleton.socket,
		key: agentSingleton.addedKey,
	};
}

export interface SessionOptions {
	timeoutMs?: number;
	/** Extra env for the ssh/scp/rsync children (tests). */
	env?: NodeJS.ProcessEnv;
}

/**
 * One HPC session: the private agent plus command execution. Every method
 * records the command; `transcripts` is the evidence the collector attaches.
 */
export class HpcSession {
	readonly runtime: HpcRuntime;
	readonly transcripts: CommandRecord[] = [];
	private readonly timeoutMs: number;
	private readonly extraEnv: NodeJS.ProcessEnv;
	private gitAgent: GitAgentStatus | null = null;

	constructor(runtime: HpcRuntime, options: SessionOptions = {}) {
		this.runtime = runtime;
		this.timeoutMs = options.timeoutMs ?? Number(process.env.PI_HPC_SSH_TIMEOUT_MS ?? 300_000);
		this.extraEnv = options.env ?? {};
	}

	/** Start the private agent lazily: no SSH work means no agent at all. */
	gitForwarding(): GitAgentStatus {
		if (!this.gitAgent) this.gitAgent = startPrivateAgent(this.runtime);
		return this.gitAgent;
	}

	private childEnv(): NodeJS.ProcessEnv {
		const forwarding = this.gitForwarding();
		const env: NodeJS.ProcessEnv = { ...process.env, ...this.extraEnv };
		// The only agent we ever pass on is our private one. If the private agent
		// is unavailable, SSH_AUTH_SOCK is absent — never the user's agent.
		if (forwarding.socket !== null && forwarding.forwarding) env.SSH_AUTH_SOCK = forwarding.socket;
		else delete env.SSH_AUTH_SOCK;
		delete env.SSH_AGENT_PID;
		return env;
	}

	private run(tool: string, argv: string[], options: { timeoutMs?: number; remote?: string } = {}): CommandRecord {
		const started = Date.now();
		const result = spawnSync(tool, argv, {
			encoding: "utf8",
			env: this.childEnv(),
			timeout: options.timeoutMs ?? this.timeoutMs,
			maxBuffer: 64 * 1024 * 1024,
		});
		const record: CommandRecord = {
			tool,
			argv: argv.map((value) => redact(value)),
			status: result.status ?? -1,
			stdout: redact(String(result.stdout ?? "")),
			stderr: redact(String(result.stderr ?? "")),
			timedOut: (result.error as NodeJS.ErrnoException | undefined)?.code === "ETIMEDOUT",
			ms: Date.now() - started,
			remote: options.remote ? redact(options.remote) : undefined,
		};
		this.transcripts.push(record);
		return record;
	}

	/** Identity options shared by ssh/scp/rsync (no `-A` when nothing is forwarded). */
	private sshIdentityOptions(): string[] {
		const config = this.runtime.config;
		return [
			"-o",
			"IdentitiesOnly=yes",
			"-o",
			"StrictHostKeyChecking=accept-new",
			"-o",
			"ConnectTimeout=15",
			"-o",
			"BatchMode=yes",
			"-i",
			this.runtime.hpcKeyPath,
		];
	}

	private target(): string {
		const config = this.runtime.config;
		return `${config.user}@${config.host}`;
	}

	/** Run a remote shell command. `-A` forwards only our private agent. */
	ssh(remoteCommand: string, options: { timeoutMs?: number } = {}): CommandRecord {
		const config = this.runtime.config;
		const argv = [
			...(this.gitForwarding().forwarding ? ["-A"] : []),
			"-o",
			"ForwardAgent=yes",
			"-p",
			String(config.port),
			...this.sshIdentityOptions(),
			this.target(),
			remoteCommand,
		];
		return this.run("ssh", argv, { timeoutMs: options.timeoutMs, remote: remoteCommand });
	}

	/** Copy a file/directory between local and remote paths; exactly one side has `host:`. */
	scp(from: string, to: string, options: { timeoutMs?: number } = {}): CommandRecord {
		const config = this.runtime.config;
		const argv = ["-q", "-P", String(config.port), ...this.sshIdentityOptions(), from, to];
		return this.run("scp", argv, { timeoutMs: options.timeoutMs, remote: `${from} -> ${to}` });
	}

	/** rsync over ssh; used for declared outputs and checkpoints. */
	rsync(from: string, to: string, extra: string[] = [], options: { timeoutMs?: number } = {}): CommandRecord {
		const config = this.runtime.config;
		const e = `ssh -p ${config.port} ${this.sshIdentityOptions().join(" ")}`;
		const argv = ["-az", "--partial", "-e", e, ...extra, from, to];
		return this.run("rsync", argv, { timeoutMs: options.timeoutMs, remote: `${from} -> ${to}` });
	}
}

/** True when the local binary exists (ssh, scp, rsync, ssh-agent, ssh-add). */
export function binaryAvailable(name: string, pathValue = process.env.PATH ?? ""): boolean {
	for (const dir of pathValue.split(":")) {
		if (dir === "") continue;
		const candidate = `${dir}/${name}`;
		try {
			fs.accessSync(candidate, fs.constants.X_OK);
			return true;
		} catch {
			// keep looking
		}
	}
	return false;
}
