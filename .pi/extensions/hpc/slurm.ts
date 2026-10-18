/**
 * SLURM glue: render the job template, submit with `sbatch --parsable`, read
 * `squeue`, and classify `sacct --json` exactly like
 * `.pi/skills/hpc-slurm-and-checkpointing/scripts/parse_sacct.py` (the
 * reference implementation; a test cross-checks the two on the same fixtures).
 *
 * No pi imports: usable from `node --test`.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { remoteRunDir, remoteScratchDir, shq, type HpcRuntime } from "./config.ts";

// ---------------------------------------------------------------------------
// State mapping (references/slurm-fields.md §2)
// ---------------------------------------------------------------------------

export const COMPLETED_STATES = ["COMPLETED"] as const;
export const QUEUED_STATES = ["PENDING", "CONFIGURING", "REQUEUED", "RESIZING"] as const;
export const RUNNING_STATES = ["RUNNING", "COMPLETING", "SUSPENDED", "STAGE_OUT"] as const;
export const TIMED_OUT_STATES = ["TIMEOUT", "TIME_LIMIT", "DEADLINE"] as const;
export const OOM_STATES = ["OUT_OF_MEMORY"] as const;
export const PREEMPTED_STATES = ["NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "NODE_DOWN"] as const;
export const CANCELLED_STATES = ["CANCELLED"] as const;
export const FAILED_STATES = ["FAILED", "ERROR", "SPECIAL_EXIT", "REVOKED", "LAUNCH_FAILED"] as const;

export type VangjaJobState =
	| "queued"
	| "running"
	| "completed"
	| "interrupted_clean"
	| "timed_out"
	| "oom"
	| "preempted"
	| "cancelled"
	| "failed"
	| "unknown";

export type SacctAction =
	| "wait"
	| "collect"
	| "requeue"
	| "requeue-no-progress"
	| "retry-once-doubled-memory"
	| "failed-final"
	| "stop";

export interface SacctJob {
	job_id: string;
	name: string;
	raw_state: string;
	state: VangjaJobState;
	exit_code: number | null;
	signal: number | null;
	elapsed_seconds: number | null;
	node: string;
	action: SacctAction;
	reason: string;
	progress_hint: string;
}

export interface SacctClassifyOptions {
	progressAdvanced?: boolean;
	attempt?: number;
	maxAttempts?: number;
	memoryCanDouble?: boolean;
}

function asList(value: unknown): unknown[] {
	if (value === undefined || value === null) return [];
	return Array.isArray(value) ? value : [value];
}

function extractState(job: Record<string, unknown>): string {
	const state = job.state;
	if (typeof state === "object" && state !== null) {
		const current = (state as Record<string, unknown>).current;
		const values = asList(current);
		return values.length > 0 ? String(values[0]).toUpperCase() : "UNKNOWN";
	}
	if (Array.isArray(state)) return state.length > 0 ? String(state[0]).toUpperCase() : "UNKNOWN";
	return String(state ?? "UNKNOWN").toUpperCase();
}

function extractExitCode(job: Record<string, unknown>): number | null {
	for (const key of ["derived_exit_code", "exit_code"]) {
		const value = job[key];
		if (value === undefined || value === null) continue;
		if (typeof value === "string") {
			const parts = value.split(":");
			const last = Number(parts[parts.length - 1]);
			if (Number.isFinite(last)) return last;
			continue;
		}
		if (typeof value === "number") return value;
		if (typeof value === "object") {
			const record = value as Record<string, unknown>;
			for (const field of ["return_code", "number", "code"]) {
				if (field in record) {
					const parsed = Number(record[field]);
					return Number.isFinite(parsed) ? parsed : null;
				}
			}
		}
	}
	return null;
}

function extractSignal(job: Record<string, unknown>): number | null {
	const value = (job.derived_exit_code ?? job.exit_code) as unknown;
	if (typeof value === "object" && value !== null && "signal" in (value as Record<string, unknown>)) {
		const parsed = Number((value as Record<string, unknown>).signal);
		return Number.isFinite(parsed) ? parsed : null;
	}
	return null;
}

export function elapsedSeconds(job: Record<string, unknown>): number | null {
	const time = job.time;
	if (typeof time !== "object" || time === null) return null;
	const elapsed = (time as Record<string, unknown>).elapsed;
	if (typeof elapsed === "number") return elapsed;
	if (typeof elapsed === "object" && elapsed !== null && "seconds" in (elapsed as Record<string, unknown>)) {
		const seconds = Number((elapsed as Record<string, unknown>).seconds);
		return Number.isFinite(seconds) ? seconds : null;
	}
	if (typeof elapsed === "string") {
		const match = /^(?:(\d+)-)?(\d+):(\d+):(\d+)$/.exec(elapsed);
		if (match) {
			const [, days = "0", hours, minutes, seconds] = match;
			return (
				Number(days) * 86400 + Number(hours) * 3600 + Number(minutes) * 60 + Number(seconds)
			);
		}
	}
	return null;
}

function nodeList(job: Record<string, unknown>): string {
	let nodes: unknown = job.nodes;
	if (typeof nodes === "object" && nodes !== null && !Array.isArray(nodes)) {
		nodes = (nodes as Record<string, unknown>).list;
	}
	if (Array.isArray(nodes)) return nodes.map((node) => String(node)).join(",");
	return String(nodes ?? "");
}

function inSet(value: string, set: readonly string[]): boolean {
	return set.includes(value);
}

/** Port of `parse_sacct.py:classify` — keep the two in lockstep (cross-checked). */
export function classifySacctJob(
	job: Record<string, unknown>,
	options: SacctClassifyOptions = {},
): SacctJob {
	const progressAdvanced = options.progressAdvanced === true;
	const attempt = options.attempt ?? 1;
	const maxAttempts = options.maxAttempts ?? 6;
	const memoryCanDouble = options.memoryCanDouble === true;

	const rawState = extractState(job);
	const exitCode = extractExitCode(job);
	const signal = extractSignal(job);
	let state: VangjaJobState = "unknown";
	let action: SacctAction = "failed-final";
	let reason = "";

	if (inSet(rawState, QUEUED_STATES)) {
		state = "queued";
		action = "wait";
	} else if (inSet(rawState, RUNNING_STATES)) {
		state = "running";
		action = "wait";
	} else if (inSet(rawState, COMPLETED_STATES)) {
		if (exitCode === 99) {
			state = "interrupted_clean";
			action = "requeue";
		} else if (exitCode === 0 || exitCode === null) {
			state = "completed";
			action = "collect";
		} else {
			state = "failed";
			action = "failed-final";
		}
	} else if (inSet(rawState, TIMED_OUT_STATES)) {
		state = "timed_out";
		action = progressAdvanced ? "requeue" : "requeue-no-progress";
	} else if (inSet(rawState, OOM_STATES)) {
		state = "oom";
		action = memoryCanDouble ? "retry-once-doubled-memory" : "failed-final";
		if (attempt > 1 && progressAdvanced) {
			reason = "OOM after a previous attempt; doubling memory is allowed once";
		} else if (!memoryCanDouble) {
			reason = "partition headroom unknown: memory cannot be doubled from the record";
		}
	} else if (inSet(rawState, PREEMPTED_STATES)) {
		state = "preempted";
		action = progressAdvanced ? "requeue" : "requeue-no-progress";
	} else if (inSet(rawState, CANCELLED_STATES)) {
		state = "cancelled";
		action = "stop";
	} else if (inSet(rawState, FAILED_STATES)) {
		if (exitCode === 99) {
			state = "interrupted_clean";
			action = "requeue";
		} else if (exitCode === 137 || signal === 9) {
			state = "oom";
			action = memoryCanDouble ? "retry-once-doubled-memory" : "failed-final";
		} else {
			state = "failed";
			action = "failed-final";
		}
	} else {
		state = "unknown";
		action = "failed-final";
		reason = `unrecognised state ${rawState}`;
	}

	if (action === "requeue" || action === "requeue-no-progress" || action === "retry-once-doubled-memory") {
		if (attempt >= maxAttempts) {
			action = "failed-final";
			reason = `attempt ${attempt} >= maxAttempts ${maxAttempts}`;
		} else if (action === "requeue-no-progress" && attempt > 1) {
			reason = reason || "no checkpoint progress; a second such attempt is failed-final";
		}
	}

	return {
		job_id: String(job.job_id ?? job.job_id_with_sched ?? ""),
		name: String(job.name ?? ""),
		raw_state: rawState,
		state,
		exit_code: exitCode,
		signal,
		elapsed_seconds: elapsedSeconds(job),
		node: nodeList(job),
		action,
		reason,
		progress_hint: progressAdvanced ? "advanced" : "unknown-or-none",
	};
}

export function parseSacctJson(raw: string): SacctJob[] {
	let payload: unknown;
	try {
		payload = JSON.parse(raw);
	} catch (error) {
		throw new Error(`sacct --json output is not JSON: ${(error as Error).message}`);
	}
	const jobs = (payload as Record<string, unknown> | null)?.jobs;
	if (!Array.isArray(jobs)) throw new Error(`sacct --json output has no "jobs" array`);
	return jobs.map((job) => classifySacctJob(job as Record<string, unknown>));
}

/**
 * The accounting fields the manifest must carry (plan/06 §6.4), verbatim from
 * the raw `sacct --json` job object.
 */
export interface SacctAccounting {
	State: string;
	ExitCode: string;
	Elapsed: string;
	MaxRSS: string;
	NodeList: string;
	AllocCPUS: string;
	Start: string;
	End: string;
	JobID: string;
}

export function sacctAccounting(rawJob: Record<string, unknown>, classified?: SacctJob): SacctAccounting {
	const time = (rawJob.time ?? {}) as Record<string, unknown>;
	const exitCode = (rawJob.exit_code ?? {}) as Record<string, unknown>;
	const derived = (rawJob.derived_exit_code ?? {}) as Record<string, unknown>;
	const returnCode = exitCode.return_code ?? derived.return_code ?? "";
	const signal = exitCode.signal ?? derived.signal ?? "";
	const exitText =
		typeof rawJob.exit_code === "string"
			? String(rawJob.exit_code)
			: `${String(returnCode)}:${String(signal)}`;
	return {
		State: classified?.state ?? String(rawJob.state ?? ""),
		ExitCode: exitText,
		Elapsed: String(time.elapsed ?? classified?.elapsed_seconds ?? ""),
		MaxRSS: String(rawJob.max_rss ?? rawJob.maxRSS ?? ""),
		NodeList: classified?.node ?? nodeList(rawJob),
		AllocCPUS: String(rawJob.alloc_cpus ?? rawJob.allocCPUS ?? ""),
		Start: String(time.start ?? ""),
		End: String(time.end ?? ""),
		JobID: String(rawJob.job_id ?? ""),
	};
}

// ---------------------------------------------------------------------------
// squeue / sbatch
// ---------------------------------------------------------------------------

export interface SqueueRow {
	jobId: string;
	state: string;
	elapsed: string;
	reason: string;
}

/** `%i|%T|%M|%R` — the collector's format (plan/06 §6.1). */
export function parseSqueueRows(stdout: string): SqueueRow[] {
	const rows: SqueueRow[] = [];
	for (const line of stdout.split("\n")) {
		const trimmed = line.trim();
		if (trimmed === "") continue;
		const parts = trimmed.split("|");
		if (parts.length < 2) continue;
		const [jobId, state, elapsed = "", reason = ""] = parts;
		if (!/^\d+([._].*)?$/.test(jobId.trim())) continue;
		rows.push({ jobId: jobId.trim(), state: state.trim().toUpperCase(), elapsed: elapsed.trim(), reason: reason.trim() });
	}
	return rows;
}

export function squeueStates(rows: SqueueRow[], jobId: string | null): SqueueRow | null {
	if (jobId === null) return rows[0] ?? null;
	return rows.find((row) => row.jobId === jobId || row.jobId.startsWith(`${jobId}.`)) ?? null;
}

export function parseSbatchJobId(stdout: string): string {
	const first = stdout.trim().split(/\s+/)[0] ?? "";
	const match = /^(\d+)(?:[.;].*)?$/.exec(first);
	if (!match) {
		throw new Error(
			`sbatch --parsable did not print a job id (got "${stdout.trim().slice(0, 200)}"); ` +
				`submission cannot be recorded without one.`,
		);
	}
	return match[1];
}

// ---------------------------------------------------------------------------
// Job script rendering
// ---------------------------------------------------------------------------

export function renderTemplate(template: string, vars: Record<string, string>): string {
	const lines: string[] = [];
	for (const line of template.split("\n")) {
		const placeholders = [...line.matchAll(/\{\{([A-Z0-9_]+)\}\}/g)].map((match) => match[1]);
		// Optional SBATCH lines / optional commands vanish when their value is
		// empty. A line is optional only when every placeholder on it is empty,
		// so `#SBATCH --partition=` (one empty placeholder) is dropped while
		// `{{CONTAINER_PREFIX}}{{RESUME_COMMAND}}` survives a non-empty command.
		if (placeholders.length > 0 && placeholders.every((key) => (vars[key] ?? "").trim() === "")) continue;
		const rendered = line.replace(/\{\{([A-Z0-9_]+)\}\}/g, (_match, key: string) => vars[key] ?? "");
		lines.push(rendered);
	}
	return `${lines.join("\n").replace(/\n{3,}/g, "\n\n").trimEnd()}\n`;
}

export function jobTemplatePath(root: string): string {
	return path.join(root, ".pi", "skills", "hpc-slurm-and-checkpointing", "templates", "job.slurm");
}

export function loadJobTemplate(root: string): string {
	return fs.readFileSync(jobTemplatePath(root), "utf8");
}

export interface RenderedJob {
	script: string;
	jobName: string;
	remoteScriptPath: string;
	logDir: string;
	vars: Record<string, string>;
}

/** Render the final job script for one attempt (template: skills/.../job.slurm). */
export function renderJobScript(options: {
	runtime: HpcRuntime;
	experimentId: string;
	attempt: number;
	template: string;
	resumeCommand: string;
	runDir: string;
	kind?: string;
	kind?: string;
	memGb?: number;
	cpus?: number;
	gpus?: number;
	partition?: string;
}): RenderedJob {
	const { runtime, experimentId, attempt } = options;
	const config = runtime.config;
	const scratch = remoteScratchDir(runtime, experimentId);
	const runDir = options.runDir;
	const jobName = `${experimentId}-a${attempt}`;
	const logDir = scratch;
	const container = config.container.trim();
	const vars: Record<string, string> = {
		JOB_NAME: jobName,
		EXPERIMENT_ID: experimentId,
		ATTEMPT: String(attempt),
		RUN_DIR: runDir,
		SCRATCH: scratch,
		CHECKPOINT_DIR: `${scratch}/checkpoints`,
		LOG_DIR: logDir,
		CPUS: String(options.cpus ?? config.cpus),
		MEM_GB: String(options.memGb ?? config.memGb),
		GPUS: String(options.gpus ?? config.gpus),
		PARTITION: options.partition ?? config.partition,
		GRES: (options.gpus ?? config.gpus) > 0 ? `gpu:${options.gpus ?? config.gpus}` : "",
		CONTAINER: container,
		CONTAINER_PREFIX: container === "" ? "" : `apptainer exec ${shq(container)} `,
		RESUME_COMMAND: options.resumeCommand,
		RESUME_DIR: `${scratch}/checkpoints`,
		KIND: options.kind ?? "hpc",
	};
	const script = renderTemplate(options.template, vars);
	return { script, jobName, remoteScriptPath: `${scratch}/job.slurm`, logDir, vars };
}

/** The remote command that submits the rendered script and prints the job id. */
export function sbatchCommand(options: {
	runtime: HpcRuntime;
	runDir: string;
	remoteScriptPath: string;
	jobName: string;
	partition?: string;
	memGb?: number;
	cpus?: number;
	gpus?: number;
	stage?: number;
}): string {
	const config = options.runtime.config;
	const args = [
		"sbatch",
		"--parsable",
		`--job-name=${shq(options.jobName)}`,
		"--time=24:00:00",
		`--cpus-per-task=${options.cpus ?? config.cpus}`,
		`--mem=${options.memGb ?? config.memGb}G`,
	];
	const partition = options.partition ?? config.partition;
	if (partition.trim() !== "") args.push(`--partition=${shq(partition)}`);
	const gpus = options.gpus ?? config.gpus;
	if (gpus > 0) args.push(`--gres=gpu:${gpus}`);
	args.push(shq(options.remoteScriptPath));
	return `cd ${shq(options.runDir)} && ${args.join(" ")}`;
}

/**
 * Preflight script: a clean detached checkout of the record's commit and the
 * `uv.lock` hash, all in one remote command whose output is parsed locally.
 * The last line is always `PREFLIGHT-HEAD <sha>` / `PREFLIGHT-LOCK <sha256>` so
 * a failure names which check failed.
 */
export function preflightCommand(options: {
	runtime: HpcRuntime;
	experimentId: string;
	commitSha: string;
	remoteRepoDir?: string;
	create?: boolean;
}): string {
	const runDir = options.remoteRepoDir ?? remoteRunDir(options.runtime, options.experimentId);
	const repoRoot = options.runtime.config.remoteRepoDir.replace(/\/+$/, "");
	const create = options.create === false ? "" : "true";
	return [
		"set -e",
		`cd ${shq(repoRoot)}`,
		"git fetch --quiet origin",
		`if [ "${create}" = "true" ] && [ ! -e ${shq(`${runDir}/.git`)} ]; then mkdir -p ${shq(path.dirname(runDir))} && git worktree add --force --detach ${shq(runDir)} ${shq(options.commitSha)} >/dev/null; fi`,
		`cd ${shq(runDir)}`,
		`git checkout --quiet --detach ${shq(options.commitSha)}`,
		'echo "PREFLIGHT-STATUS-BEGIN"',
		// Untracked files are expected after the first attempt: the job writes
		// declared outputs into the run dir. Only modified tracked files make the
		// checkout dirty — those would invalidate the record's commit_sha.
		"git status --porcelain --untracked-files=no",
		'echo "PREFLIGHT-STATUS-END"',
		`echo "PREFLIGHT-HEAD $(git rev-parse HEAD)"`,
		'echo "PREFLIGHT-LOCK $(sha256sum uv.lock | cut -d" " -f1)"',
		'echo "PREFLIGHT-DONE"',
	].join("\n");
}

export interface PreflightResult {
	statusLines: string[];
	head: string;
	uvLockSha: string;
	ok: boolean;
	reason: string;
}

export function parsePreflight(stdout: string): PreflightResult {
	const statusLines: string[] = [];
	let insideStatus = false;
	let head = "";
	let uvLockSha = "";
	for (const line of stdout.split("\n")) {
		const trimmed = line.trimEnd();
		if (trimmed === "PREFLIGHT-STATUS-BEGIN") {
			insideStatus = true;
			continue;
		}
		if (trimmed === "PREFLIGHT-STATUS-END") {
			insideStatus = false;
			continue;
		}
		if (insideStatus) statusLines.push(trimmed);
		if (trimmed.startsWith("PREFLIGHT-HEAD ")) head = trimmed.slice("PREFLIGHT-HEAD ".length).trim();
		if (trimmed.startsWith("PREFLIGHT-LOCK ")) uvLockSha = trimmed.slice("PREFLIGHT-LOCK ".length).trim();
	}
	const dirty = statusLines.filter((line) => line.trim() !== "");
	if (head === "" || uvLockSha === "") {
		return { statusLines, head, uvLockSha, ok: false, reason: "preflight output did not report HEAD and uv.lock (see raw output)" };
	}
	if (dirty.length > 0) {
		return { statusLines, head, uvLockSha, ok: false, reason: `remote checkout is dirty: ${dirty.join(" | ")}` };
	}
	return { statusLines, head, uvLockSha, ok: true, reason: "clean detached checkout" };
}
