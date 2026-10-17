/**
 * `hpc_*` tool handlers — the behaviour behind `.pi/extensions/hpc/index.ts`.
 *
 * All behaviour lives here (no pi imports) so `node --test` and the mock-cluster
 * harness drive exactly the code the tools run. The tools never talk to a host
 * when `host` is empty: `resolveHpc` returns `{enabled:false}` and the handler
 * returns before any ssh/scp/rsync call.
 *
 * Evidence discipline (plan/06 §2/§6): every remote command is recorded (argv
 * minus credentials + trimmed output), raw `squeue`/`sacct` output is persisted
 * under `.pi/state/hpc/`, and every fetched file is SHA-256'd into the artifact
 * manifest.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { readJsonFile, writeJsonAtomic } from "../../lib/io.ts";
import { statePath } from "../../lib/paths.ts";
import { appendRecord, readLedger } from "../ledger/store.ts";
import { activeExperiments, projectLatest } from "../../lib/records.ts";
import { roleGate, type Capability, type LoopRole } from "../../lib/roles.ts";
import { VangjaLoopError } from "../../lib/errors.ts";
import {
	configSummary,
	remoteRunDir,
	remoteScratchDir,
	redact,
	resolveHpc,
	shq,
	trimOutput,
	type HpcResolution,
	type HpcRuntime,
} from "./config.ts";
import { runSubmitGate, type GateCheck } from "./gate.ts";
import {
	artifactDirFor,
	hashTree,
	missingOutputs,
	parseProgress,
	writeManifest,
	type ArtifactManifest,
	type HashedFile,
} from "./manifest.ts";
import {
	HpcSession,
	agentStatus,
	type CommandRecord,
	type GitAgentStatus,
} from "./ssh.ts";
import {
	loadJobTemplate,
	parseSacctJson,
	parseSbatchJobId,
	parseSqueueRows,
	preflightCommand,
	renderJobScript,
	sacctAccounting,
	sbatchCommand,
	type SacctAccounting,
	type SqueueRow,
	type SacctJob,
} from "./slurm.ts";

export interface HpcContext {
	root: string;
	role: LoopRole;
	actor: string;
	/** Injected in tests; production builds the real session lazily. */
	session?: HpcSession;
}

export interface ToolTextResult {
	content: Array<{ type: "text"; text: string }>;
	details: Record<string, unknown>;
	isError?: boolean;
}

export function ok(text: string, details: Record<string, unknown> = {}): ToolTextResult {
	return { content: [{ type: "text", text }], details };
}

export function fail(message: string, details: Record<string, unknown> = {}): ToolTextResult {
	return { content: [{ type: "text", text: message }], details: { ...details, error: message }, isError: true };
}

function checkCapability(ctx: HpcContext, capability: Capability): void {
	const decision = roleGate(capability, ctx.role);
	if (!decision.allowed) throw new VangjaLoopError("ROLE_DENIED", decision.reason ?? `role ${ctx.role} may not ${capability}`);
}

function disabled(reason: string): ToolTextResult {
	return ok(`hpc: disabled — ${reason}`, { enabled: false, reason });
}

export interface RuntimeResult {
	resolution: HpcResolution;
	error: VangjaLoopError | null;
}

function runtimeFor(root: string): RuntimeResult {
	try {
		return { resolution: resolveHpc(root), error: null };
	} catch (error) {
		return { resolution: { enabled: false, reason: (error as Error).message, config: null }, error: error as VangjaLoopError };
	}
}

function sessionFor(ctx: HpcContext, runtime: HpcRuntime): HpcSession {
	return ctx.session ?? new HpcSession(runtime);
}

function latestRecord(root: string, experimentId: string): Record<string, unknown> | null {
	const records = readLedger(root, "experiment");
	const latest = projectLatest(records as Array<{ id?: string; seq: number }>).find(
		(record) => record.id === experimentId,
	);
	return (latest as Record<string, unknown> | undefined) ?? null;
}

function experimentRecords(root: string): Array<Record<string, unknown>> {
	return readLedger(root, "experiment");
}

function numberField(record: Record<string, unknown>, key: string, fallback: number): number {
	const value = Number(record[key]);
	return Number.isFinite(value) ? value : fallback;
}


function writeEvidence(root: string, name: string, payload: Record<string, unknown>): string {
	const dir = statePath(root, "hpc");
	fs.mkdirSync(dir, { recursive: true });
	const stamp = new Date().toISOString().replace(/[:.]/g, "-");
	const file = path.join(dir, `${name}-${stamp}.json`);
	writeJsonAtomic(file, payload);
	return path.relative(root, file).split(path.sep).join("/");
}

function transcriptSummary(records: CommandRecord[]): Array<Record<string, unknown>> {
	return records.map((record) => ({
		tool: record.tool,
		status: record.status,
		timedOut: record.timedOut,
		ms: record.ms,
		argv: record.argv,
		remote: record.remote,
		stdout: trimOutput(record.stdout, 2000),
		stderr: trimOutput(record.stderr, 1000),
	}));
}

// ---------------------------------------------------------------------------
// hpc_config
// ---------------------------------------------------------------------------

export function handleHpcConfig(_args: Record<string, unknown>, ctx: HpcContext): ToolTextResult {
	checkCapability(ctx, "hpc_read");
	const { resolution, error } = runtimeFor(ctx.root);
	if (error) return fail(error.message, { enabled: false });
	const summary = configSummary(resolution);
	if (!resolution.enabled) return disabled(resolution.reason);
	const records = experimentRecords(ctx.root);
	const active = activeExperiments(records);
	const forwarding: GitAgentStatus | null = agentStatus(resolution.runtime);
	return ok(
		`hpc enabled: ${summary.hostRedacted}:${summary.port} as ${summary.user} (${active.length} active experiment(s))`,
		{
			enabled: true,
			config: summary,
			active: active.map((job) => ({ id: job.id, status: job.status, attempt: job.attempt })),
			git_forwarding: forwarding
				? { forwarding: forwarding.forwarding, reason: forwarding.reason }
				: { forwarding: null, reason: "private ssh-agent not started in this process yet" },
		},
	);
}

// ---------------------------------------------------------------------------
// hpc_status
// ---------------------------------------------------------------------------

export interface HpcStatusArgs {
	jobs?: string[];
}

export interface JobStatusView {
	jobId: string;
	name: string;
	experiment_id: string | null;
	squeue: SqueueRow | null;
	sacct: SacctJob | null;
	accounting: SacctAccounting | null;
}

const SQUEUE_FORMAT = "%i|%T|%M|%R";

export function handleHpcStatus(args: HpcStatusArgs, ctx: HpcContext): ToolTextResult {
	checkCapability(ctx, "hpc_read");
	const { resolution, error } = runtimeFor(ctx.root);
	if (error) return fail(error.message, { enabled: false });
	if (!resolution.enabled) return disabled(resolution.reason);
	const runtime = resolution.runtime;
	const session = sessionFor(ctx, runtime);

	const records = experimentRecords(ctx.root);
	// Only the latest revision of each record names the current job: older
	// revisions carry terminal job ids that must not be polled as if active.
	const latestRecords = projectLatest(records as Array<{ id?: string; seq: number }>) as Array<Record<string, unknown>>;
	const withJobs = latestRecords.filter((record) => typeof record.job_id === "string" && record.job_id !== "");
	const ids = (args.jobs ?? withJobs.map((record) => String(record.job_id))).filter(
		(id, index, all) => /^\d+([._].*)?$/.test(id) && all.indexOf(id) === index,
	);

	const squeueCommand =
		ids.length > 0
			? `squeue -u "$USER" -j ${ids.join(",")} -h -o ${shq(SQUEUE_FORMAT)}`
			: `squeue -u "$USER" -h -o ${shq(SQUEUE_FORMAT)}`;
	const squeue = session.ssh(squeueCommand);
	const sacct = ids.length > 0 ? session.ssh(`sacct -j ${ids.join(",")} --json`) : null;

	const squeueRows = squeue.status === 0 ? parseSqueueRows(squeue.stdout) : [];
	let sacctJobs: SacctJob[] = [];
	let sacctError: string | null = null;
	if (sacct) {
		if (sacct.status === 0) {
			try {
				sacctJobs = parseSacctJson(sacct.stdout);
			} catch (parseError) {
				sacctError = `sacct output could not be parsed: ${(parseError as Error).message}`;
			}
		} else {
			sacctError = `sacct exited ${sacct.status}: ${trimOutput(sacct.stderr, 300)}`;
		}
	}

	const views: JobStatusView[] = ids.map((id) => {
		const experiment = withJobs.find((record) => String(record.job_id) === id);
		const row = squeueRows.find((candidate) => candidate.jobId === id || candidate.jobId.startsWith(`${id}.`)) ?? null;
		const classified = sacctJobs.find((job) => job.job_id === id || job.job_id.startsWith(`${id}.`)) ?? null;
		return {
			jobId: id,
			name: classified?.name ?? String(experiment?.id ?? ""),
			experiment_id: typeof experiment?.id === "string" ? experiment.id : null,
			squeue: row,
			sacct: classified,
			accounting: null,
		};
	});

	const evidence = writeEvidence(ctx.root, "status", {
		at: new Date().toISOString(),
		actor: ctx.actor,
		job_ids: ids,
		squeue: { command: redact(squeueCommand), status: squeue.status, stdout: trimOutput(squeue.stdout, 8000), stderr: trimOutput(squeue.stderr, 2000) },
		sacct: sacct
			? { command: `sacct -j ${ids.join(",")} --json`, status: sacct.status, stdout: trimOutput(sacct.stdout, 20000), stderr: trimOutput(sacct.stderr, 2000) }
			: null,
		sacct_error: sacctError,
	});

	const lines = views.map((view) => {
		const state = view.squeue?.state ?? view.sacct?.raw_state ?? "UNKNOWN";
		return `${view.jobId} ${view.experiment_id ?? "-"}: ${state}` + (view.sacct ? ` (exit ${view.sacct.exit_code ?? "?"}, action ${view.sacct.action})` : "");
	});
	return ok(
		lines.length > 0 ? lines.join("\n") : "no jobs to report (no experiment carries a job_id yet)",
		{
			enabled: true,
			jobs: views,
			squeue_raw: trimOutput(squeue.stdout, 4000),
			sacct_raw: sacct ? trimOutput(sacct.stdout, 8000) : null,
			sacct_error: sacctError,
			squeue_status: squeue.status,
			sacct_status: sacct?.status ?? null,
			evidence,
			transcripts: transcriptSummary(session.transcripts),
		},
	);
}

// ---------------------------------------------------------------------------
// hpc_submit
// ---------------------------------------------------------------------------

interface SubmitMarker {
	status: "submitting" | "submitted";
	attempt: number;
	jobId?: string;
	command?: string;
	at: string;
	rawOutput?: string;
}

function markerPath(root: string, experimentId: string, attempt: number): string {
	return statePath(root, `hpc/submits/${experimentId}-a${attempt}.json`);
}

function readMarker(root: string, experimentId: string, attempt: number): SubmitMarker | null {
	return readJsonFile<SubmitMarker>(markerPath(root, experimentId, attempt)) ?? null;
}

export interface HpcSubmitArgs {
	experiment_id: string;
	stage?: number;
}

export function handleHpcSubmit(args: HpcSubmitArgs, ctx: HpcContext): ToolTextResult {
	checkCapability(ctx, "hpc_submit");
	const experimentId = String(args.experiment_id ?? "").trim();
	if (experimentId === "") return fail('argument "experiment_id" must be a non-empty string; e.g. "E-0007"');
	const { resolution, error } = runtimeFor(ctx.root);
	if (error) return fail(error.message, { enabled: false });
	if (!resolution.enabled) return disabled(resolution.reason);
	const runtime = resolution.runtime;
	const session = sessionFor(ctx, runtime);

	const record = latestRecord(ctx.root, experimentId);
	if (!record) {
		return fail(`unknown experiment ${experimentId}; no record in .pi/state/EXPERIMENTS.jsonl`, { enabled: true });
	}
	const attempt = numberField(record, "attempt", 1);

	// Crash-resume safety: a marker from a previous step either names a live job
	// (adopt it) or was never submitted (submit now).
	const existing = readMarker(ctx.root, experimentId, attempt);
	if (existing?.status === "submitted" && existing.jobId) {
		return fail(
			`experiment ${experimentId} attempt ${attempt} was already submitted as job ${existing.jobId}; ` +
				`a second submission is refused (single-slot rule)`,
			{ enabled: true, job_id: existing.jobId, adopted: true },
		);
	}
	if (existing?.status === "submitting") {
		const adopted = adoptExistingJob(session, runtime, `${experimentId}-a${attempt}`);
		if (adopted) {
			const appended = recordSubmission(ctx, record, attempt, adopted, "(adopted from squeue after an interrupted submit)");
			return ok(
				`adopted an already-running job ${adopted} for ${experimentId} attempt ${attempt}; ` +
					`no second sbatch was issued (single-slot rule)`,
				{ enabled: true, job_id: adopted, adopted: true, record: appended, transcripts: transcriptSummary(session.transcripts) },
			);
		}
	}

	const gate = runSubmitGate({
		root: ctx.root,
		session,
		runtime,
		experimentId,
		stage: args.stage,
	});
	if (!gate.ok) {
		return fail(`${gate.reason}\n- ${gate.missing.join("\n- ")}`, {
			enabled: true,
			gate: { checks: gate.checks, missing: gate.missing },
			transcripts: transcriptSummary(session.transcripts),
		});
	}
	const valid = gate.record as Record<string, unknown>;
	const runDir = remoteRunDir(runtime, experimentId);
	const scratchDir = remoteScratchDir(runtime, experimentId);
	const resources = (valid.resources ?? {}) as Record<string, unknown>;
	const memGb = Number(resources.memGb ?? runtime.config.memGb);
	const cpus = Number(resources.cpus ?? runtime.config.cpus);
	const gpus = Number(resources.gpus ?? runtime.config.gpus);
	const partition = typeof resources.partition === "string" && resources.partition !== "" ? resources.partition : runtime.config.partition;
	const checkpoints = valid.checkpoints as { resumeCommand: string; progressFile: string };
	const template = loadJobTemplate(ctx.root);

	const rendered = renderJobScript({
		runtime,
		experimentId,
		attempt,
		template,
		resumeCommand: checkpoints.resumeCommand,
		runDir,
		memGb,
		cpus,
		gpus,
		partition,
	});
	const localScript = statePath(ctx.root, `hpc/jobs/${experimentId}-a${attempt}.slurm`);
	fs.mkdirSync(path.dirname(localScript), { recursive: true });
	fs.writeFileSync(localScript, rendered.script);

	writeJsonAtomic(markerPath(ctx.root, experimentId, attempt), {
		status: "submitting",
		attempt,
		command: redact(sbatchCommand({ runtime, runDir, remoteScriptPath: rendered.remoteScriptPath, jobName: rendered.jobName, memGb, cpus, gpus, partition })),
		at: new Date().toISOString(),
	} satisfies SubmitMarker);

	const prepare = session.ssh(`mkdir -p ${shq(`${scratchDir}/checkpoints`)} ${shq(scratchDir)}`);
	if (prepare.status !== 0) {
		return fail(
			`could not prepare ${scratchDir} on the remote host (exit ${prepare.status}): ${trimOutput(prepare.stderr, 400)}`,
			{ enabled: true, transcripts: transcriptSummary(session.transcripts) },
		);
	}
	const upload = session.scp(localScript, `${runtime.config.user}@${runtime.config.host}:${rendered.remoteScriptPath}`);
	if (upload.status !== 0) {
		return fail(
			`could not upload the job script to ${rendered.remoteScriptPath} (exit ${upload.status}): ${trimOutput(upload.stderr, 400)}`,
			{ enabled: true, transcripts: transcriptSummary(session.transcripts) },
		);
	}

	const submitCommand = sbatchCommand({
		runtime,
		runDir,
		remoteScriptPath: rendered.remoteScriptPath,
		jobName: rendered.jobName,
		memGb,
		cpus,
		gpus,
		partition,
	});
	const submitted = session.ssh(submitCommand);
	if (submitted.status !== 0) {
		return fail(
			`sbatch failed (exit ${submitted.status}): ${trimOutput(submitted.stderr || submitted.stdout, 500)}\n` +
				`command: ${redact(submitCommand)}`,
			{ enabled: true, transcripts: transcriptSummary(session.transcripts) },
		);
	}
	let jobId: string;
	try {
		jobId = parseSbatchJobId(submitted.stdout);
	} catch (parseError) {
		return fail((parseError as Error).message, { enabled: true, stdout: trimOutput(submitted.stdout, 400) });
	}

	writeJsonAtomic(markerPath(ctx.root, experimentId, attempt), {
		status: "submitted",
		attempt,
		jobId,
		command: redact(submitCommand),
		rawOutput: trimOutput(submitted.stdout, 200),
		at: new Date().toISOString(),
	} satisfies SubmitMarker);

	const appended = recordSubmission(ctx, valid, attempt, jobId, submitted.stdout);
	const forwarding = session.gitForwarding();
	const evidence = writeEvidence(ctx.root, `submit-${experimentId}`, {
		at: new Date().toISOString(),
		experiment_id: experimentId,
		attempt,
		job_id: jobId,
		command: redact(submitCommand),
		stdout: trimOutput(submitted.stdout, 2000),
		stderr: trimOutput(submitted.stderr, 1000),
		gate: gate.checks,
		git_forwarding: { forwarding: forwarding.forwarding, reason: forwarding.reason },
		transcripts: transcriptSummary(session.transcripts),
	});
	return ok(
		`experiment ${experimentId} attempt ${attempt} submitted as job ${jobId} ` +
			`(${runtime.config.maxParallelJobs} job slot; no further submission until it is collected)` +
			(forwarding.forwarding ? "" : ` [git-forwarding-off: ${forwarding.reason}]`),
		{
			enabled: true,
			job_id: jobId,
			attempt,
			run_dir: runDir,
			scratch_dir: scratchDir,
			local_script: path.relative(ctx.root, localScript),
			remote_script: rendered.remoteScriptPath,
			gate: gate.checks,
			git_forwarding: { forwarding: forwarding.forwarding, reason: forwarding.reason },
			record: appended,
			evidence,
			transcripts: transcriptSummary(session.transcripts),
		},
	);
}

function adoptExistingJob(session: HpcSession, runtime: HpcRuntime, jobName: string): string | null {
	const result = session.ssh(`squeue -u "$USER" -h -o ${shq("%i|%T|%M|%R")} --name=${shq(jobName)}`);
	if (result.status !== 0) return null;
	const rows = parseSqueueRows(result.stdout);
	const running = rows.find((row) => !["PENDING", "CONFIGURING"].includes(row.state)) ?? rows[0];
	return running ? running.jobId : null;
}

function recordSubmission(
	ctx: HpcContext,
	record: Record<string, unknown>,
	attempt: number,
	jobId: string,
	rawOutput: string,
): Record<string, unknown> {
	return appendRecord(
		ctx.root,
		"experiment",
		{
			...record,
			id: record.id,
			status: record.status,
			job_id: jobId,
			submit: {
				job_id: jobId,
				attempt,
				at: new Date().toISOString(),
				raw_output: trimOutput(rawOutput, 200),
			},
		},
		{ root: ctx.root, role: "conductor", actor: ctx.actor },
	);
}

// ---------------------------------------------------------------------------
// hpc_checkpoints
// ---------------------------------------------------------------------------

export interface CheckpointListing {
	exists: boolean;
	dir: string;
	progress: { completed: string[]; version: number } | null;
	progressRaw: string | null;
	files: Array<{ path: string; size: number; mtime: number }>;
	hashes: Record<string, string>;
	raw: string;
}

const CHECKPOINT_SCRIPT_HEAD = [
	'echo "CKPT-BEGIN"',
	'echo "CKPT-DIR <dir>"',
	'if [ ! -d <dir> ]; then echo "CKPT-MISSING"; echo "CKPT-END"; exit 0; fi',
	'cd <dir>',
	'echo "PROGRESS-BEGIN"',
	"cat <progressFile> 2>/dev/null || true",
	'echo',
	'echo "PROGRESS-END"',
	'echo "FILES-BEGIN"',
	"find . -type f -printf '%p\\t%s\\t%T@\\n' | sort",
	'echo "FILES-END"',
	'echo "HASHES-BEGIN"',
	"find . -type f -print0 | sort -z | xargs -0 -r sha256sum",
	'echo "HASHES-END"',
	'echo "CKPT-END"',
].join("\n");

export function parseCheckpointListing(stdout: string, dir: string): CheckpointListing {
	const missing = stdout.includes("CKPT-MISSING");
	const progressLines: string[] = [];
	const fileLines: string[] = [];
	const hashLines: string[] = [];
	let section = "";
	for (const line of stdout.split("\n")) {
		const trimmed = line.trimEnd();
		if (trimmed === "PROGRESS-BEGIN") {
			section = "progress";
			continue;
		}
		if (trimmed === "PROGRESS-END") {
			section = "";
			continue;
		}
		if (trimmed === "FILES-BEGIN") {
			section = "files";
			continue;
		}
		if (trimmed === "FILES-END") {
			section = "";
			continue;
		}
		if (trimmed === "HASHES-BEGIN") {
			section = "hashes";
			continue;
		}
		if (trimmed === "HASHES-END") {
			section = "";
			continue;
		}
		if (section === "progress") progressLines.push(trimmed);
		if (section === "files" && trimmed !== "") fileLines.push(trimmed);
		if (section === "hashes" && trimmed !== "") hashLines.push(trimmed);
	}
	const files = fileLines.map((line) => {
		const [file, size, mtime] = line.split("\t");
		return { path: file.replace(/^\.\//, ""), size: Number(size ?? 0), mtime: Number(mtime ?? 0) };
	});
	const hashes: Record<string, string> = {};
	for (const line of hashLines) {
		const match = /^([0-9a-f]{64})\s+\*?(.+)$/.exec(line.trim());
		if (match) hashes[match[2].replace(/^\.\//, "")] = match[1];
	}
	const progressRaw = progressLines.join("\n").trim();
	return {
		exists: !missing,
		dir,
		progress: progressRaw === "" ? null : parseProgress(progressRaw),
		progressRaw: progressRaw === "" ? null : progressRaw,
		files,
		hashes,
		raw: trimOutput(stdout, 8000),
	};
}

export interface HpcCheckpointsArgs {
	experiment_id: string;
}

export function handleHpcCheckpoints(args: HpcCheckpointsArgs, ctx: HpcContext): ToolTextResult {
	checkCapability(ctx, "hpc_read");
	const experimentId = String(args.experiment_id ?? "").trim();
	if (experimentId === "") return fail('argument "experiment_id" must be a non-empty string');
	const { resolution, error } = runtimeFor(ctx.root);
	if (error) return fail(error.message, { enabled: false });
	if (!resolution.enabled) return disabled(resolution.reason);
	const runtime = resolution.runtime;
	const session = sessionFor(ctx, runtime);
	const record = latestRecord(ctx.root, experimentId);
	if (!record) return fail(`unknown experiment ${experimentId}`, { enabled: true });
	const checkpoints = (record.checkpoints ?? {}) as { dir?: string; progressFile?: string };
	const remoteDir = checkpoints.dir ?? "";
	if (remoteDir === "") return fail(`${experimentId} has no checkpoints.dir`, { enabled: true });

	const progressFile = checkpoints.progressFile ?? "progress.json";
	const script = CHECKPOINT_SCRIPT_HEAD.replaceAll("<dir>", shq(remoteDir)).replaceAll(
		"<progressFile>",
		shq(`${remoteDir.replace(/\/+$/, "")}/${progressFile}`),
	);
	const result = session.ssh(script);
	const listing = parseCheckpointListing(result.stdout, remoteDir);
	const evidence = writeEvidence(ctx.root, `checkpoints-${experimentId}`, {
		at: new Date().toISOString(),
		experiment_id: experimentId,
		command: redact(script),
		status: result.status,
		stdout: trimOutput(result.stdout, 8000),
		stderr: trimOutput(result.stderr, 2000),
		listing: { exists: listing.exists, files: listing.files, progress: listing.progress },
	});
	return ok(
		listing.exists
			? `${listing.files.length} checkpoint file(s) under ${remoteDir}; progress ${listing.progress?.completed.length ?? 0} unit(s)`
			: `no checkpoint directory ${remoteDir} yet`,
		{
			enabled: true,
			exists: listing.exists,
			dir: remoteDir,
			files: listing.files,
			hashes: listing.hashes,
			progress: listing.progress,
			progress_raw: listing.progressRaw,
			status: result.status,
			evidence,
			transcripts: transcriptSummary(session.transcripts),
		},
	);
}

// ---------------------------------------------------------------------------
// hpc_fetch
// ---------------------------------------------------------------------------

export interface HpcFetchArgs {
	experiment_id: string;
	/** Fetch only the checkpoint directory (used on timeout, before requeueing). */
	checkpointsOnly?: boolean;
	/** Explicit remote paths (relative to the run dir); defaults to expectedOutputs. */
	paths?: string[];
}

export function handleHpcFetch(args: HpcFetchArgs, ctx: HpcContext): ToolTextResult {
	checkCapability(ctx, "hpc_read");
	const experimentId = String(args.experiment_id ?? "").trim();
	if (experimentId === "") return fail('argument "experiment_id" must be a non-empty string');
	const { resolution, error } = runtimeFor(ctx.root);
	if (error) return fail(error.message, { enabled: false });
	if (!resolution.enabled) return disabled(resolution.reason);
	const runtime = resolution.runtime;
	const session = sessionFor(ctx, runtime);
	const record = latestRecord(ctx.root, experimentId);
	if (!record) return fail(`unknown experiment ${experimentId}`, { enabled: true });

	const attempt = numberField(record, "attempt", 1);
	const jobId = typeof record.job_id === "string" && record.job_id !== "" ? record.job_id : null;
	const checkpoints = (record.checkpoints ?? {}) as { dir?: string; progressFile?: string };
	const runDir = remoteRunDir(runtime, experimentId);
	const artifactDir = artifactDirFor(ctx.root, experimentId);
	const checkpointsLocal = path.join(artifactDir, "checkpoints");
	fs.mkdirSync(checkpointsLocal, { recursive: true });
	const host = `${runtime.config.user}@${runtime.config.host}`;
	const commands: Array<Record<string, unknown>> = [];

	const fetchItems: string[] = [];
	if (args.checkpointsOnly === true) {
		if (checkpoints.dir) fetchItems.push("checkpoints");
	} else {
		const declared = args.paths ?? (Array.isArray(record.expectedOutputs) ? (record.expectedOutputs as string[]) : []);
		for (const item of declared) fetchItems.push(item);
		if (checkpoints.dir) fetchItems.push("checkpoints");
	}

	for (const item of fetchItems) {
		if (item === "checkpoints") {
			const source = `${host}:${checkpoints.dir?.replace(/\/+$/, "")}/`;
			const result = session.rsync(source, `${checkpointsLocal}/`);
			commands.push({ item, kind: "checkpoints", from: `${host}:${checkpoints.dir}`, status: result.status, stderr: trimOutput(result.stderr, 400) });
			continue;
		}
		const target = path.join(artifactDir, item);
		fs.mkdirSync(path.dirname(target), { recursive: true });
		const source = `${host}:${runDir}/./${item}`;
		const result = session.rsync(source, `${artifactDir}/`, ["--relative"]);
		commands.push({ item, kind: "output", from: `${host}:${runDir}/${item}`, status: result.status, stderr: trimOutput(result.stderr, 400) });
	}

	const allFiles: HashedFile[] = hashTree(artifactDir);
	const checkpointFiles = allFiles.filter((file) => file.path.startsWith("checkpoints/"));
	const outputFiles = allFiles.filter((file) => !file.path.startsWith("checkpoints/"));

	const progressFile = checkpoints.progressFile ?? "progress.json";
	const localProgress = path.join(checkpointsLocal, progressFile);
	const progress = fs.existsSync(localProgress) ? parseProgress(fs.readFileSync(localProgress, "utf8")) : null;

	const expectedOutputs = Array.isArray(record.expectedOutputs) ? (record.expectedOutputs as string[]) : [];
	const missing = args.checkpointsOnly === true ? [] : missingOutputs(expectedOutputs, outputFiles);

	let accounting: SacctAccounting | null = null;
	if (jobId) {
		const sacct = session.ssh(`sacct -j ${jobId} --json`);
		if (sacct.status === 0) {
			try {
				const jobs = parseSacctJson(sacct.stdout);
				const classified = jobs.find((job) => job.job_id === jobId) ?? jobs[0] ?? null;
				const raw = JSON.parse(sacct.stdout) as { jobs?: Array<Record<string, unknown>> };
				const rawJob = (raw.jobs ?? []).find((job) => String(job.job_id ?? "") === jobId) ?? (raw.jobs ?? [])[0];
				if (rawJob) accounting = sacctAccounting(rawJob, classified ?? undefined);
			} catch {
				accounting = null;
			}
		}
	}

	const status: ArtifactManifest["status"] =
		args.checkpointsOnly === true ? "checkpoints-only" : missing.length === 0 ? "collected" : "collection-failed";
	const manifest: ArtifactManifest = {
		schema: "pi-artifact/1",
		experiment_id: experimentId,
		task_id: String(record.task_id ?? ""),
		attempt,
		status,
		idempotencyKey: String(record.idempotencyKey ?? ""),
		commit_sha: String(record.commit_sha ?? ""),
		job_id: jobId,
		collected_at: new Date().toISOString(),
		collected_by: ctx.actor,
		expectedOutputs: args.checkpointsOnly === true ? [] : expectedOutputs,
		files: outputFiles,
		missing,
		checkpoints: checkpointFiles,
		sacct: accounting as unknown as Record<string, unknown> | null,
	};
	const written = writeManifest(ctx.root, manifest);

	const evidence = writeEvidence(ctx.root, `fetch-${experimentId}`, {
		at: new Date().toISOString(),
		experiment_id: experimentId,
		attempt,
		job_id: jobId,
		checkpoints_only: args.checkpointsOnly === true,
		commands,
		missing,
		manifest: written,
		progress,
		sacct: accounting,
	});
	const details: Record<string, unknown> = {
		enabled: true,
		manifest: path.relative(ctx.root, written.path).split(path.sep).join("/"),
		manifest_sha256: written.sha256,
		status,
		files: outputFiles,
		checkpoints: checkpointFiles,
		missing,
		progress,
		sacct: accounting,
		commands,
		evidence,
		transcripts: transcriptSummary(session.transcripts),
	};
	if (missing.length > 0) {
		return fail(
			`collection failed for ${experimentId}: declared output(s) missing after fetch: ${missing.join(", ")}. ` +
				`The manifest records the missing list; nothing is fabricated (plan/06 §6.4).`,
			details,
		);
	}
	return ok(
		args.checkpointsOnly === true
			? `fetched ${checkpointFiles.length} checkpoint file(s) for ${experimentId} (progress ${progress?.completed.length ?? 0} unit(s))`
			: `collected ${outputFiles.length} output file(s) and ${checkpointFiles.length} checkpoint file(s) for ${experimentId}; ` +
				`manifest ${path.relative(ctx.root, written.path)}`,
		details,
	);
}

// ---------------------------------------------------------------------------
// hpc_tail
// ---------------------------------------------------------------------------

export interface HpcTailArgs {
	experiment_id: string;
	stream?: "out" | "err";
	lines?: number;
}

export function handleHpcTail(args: HpcTailArgs, ctx: HpcContext): ToolTextResult {
	checkCapability(ctx, "hpc_read");
	const experimentId = String(args.experiment_id ?? "").trim();
	if (experimentId === "") return fail('argument "experiment_id" must be a non-empty string');
	const { resolution, error } = runtimeFor(ctx.root);
	if (error) return fail(error.message, { enabled: false });
	if (!resolution.enabled) return disabled(resolution.reason);
	const runtime = resolution.runtime;
	const session = sessionFor(ctx, runtime);
	const record = latestRecord(ctx.root, experimentId);
	if (!record) return fail(`unknown experiment ${experimentId}`, { enabled: true });
	const jobId = typeof record.job_id === "string" && record.job_id !== "" ? record.job_id : null;
	if (jobId === null) {
		return fail(`${experimentId} has no job_id yet; nothing to tail (queue it and submit first)`, { enabled: true });
	}
	const stream = args.stream === "err" ? "err" : "out";
	const lines = Math.min(Math.max(Math.floor(args.lines ?? 50), 1), 2000);
	const scratch = remoteScratchDir(runtime, experimentId);
	const result = session.ssh(`tail -n ${lines} ${shq(`${scratch}/slurm-${jobId}.${stream}`)}`);
	return ok(trimOutput(result.stdout, 4000) || `(empty ${stream} stream for job ${jobId})`, {
		enabled: true,
		job_id: jobId,
		stream,
		status: result.status,
		text: trimOutput(result.stdout, 4000),
		stderr: trimOutput(result.stderr, 500),
		transcripts: transcriptSummary(session.transcripts),
	});
}

// ---------------------------------------------------------------------------
// hpc_cancel
// ---------------------------------------------------------------------------

export interface HpcCancelArgs {
	experiment_id: string;
	reason: string;
}

/** A cancellation needs an explicit instruction record (plan/06 §4). */
export function hasCancelInstruction(root: string, experimentId: string): boolean {
	const decisions = readLedger(root, "decision");
	const instructed = decisions.some((record) => {
		const text = `${String(record.title ?? "")} ${String(record.decision ?? "")} ${String(record.context ?? "")}`;
		return text.includes(experimentId) && /cancel/i.test(text);
	});
	if (instructed) return true;
	const events = readLedger(root, "event");
	return events.some(
		(record) =>
			String(record.experiment_id ?? "") === experimentId &&
			(record.instruction === "cancel" || /^cancel\b/i.test(String(record.summary ?? ""))),
	);
}

export function handleHpcCancel(args: HpcCancelArgs, ctx: HpcContext): ToolTextResult {
	checkCapability(ctx, "hpc_cancel");
	const experimentId = String(args.experiment_id ?? "").trim();
	const reason = String(args.reason ?? "").trim();
	if (experimentId === "") return fail('argument "experiment_id" must be a non-empty string');
	if (reason === "") {
		return fail(
			'argument "reason" is required: hpc_cancel is destructive and must state the instruction it executes ' +
				'(plan/06 §4: no cancelling without an instruction record)',
		);
	}
	const { resolution, error } = runtimeFor(ctx.root);
	if (error) return fail(error.message, { enabled: false });
	if (!resolution.enabled) return disabled(resolution.reason);
	const runtime = resolution.runtime;
	const session = sessionFor(ctx, runtime);
	const record = latestRecord(ctx.root, experimentId);
	if (!record) return fail(`unknown experiment ${experimentId}`, { enabled: true });
	if (!hasCancelInstruction(ctx.root, experimentId)) {
		return fail(
			`refused: no cancellation instruction record names ${experimentId}. ` +
				`Record a decision ("cancel ${experimentId}: …") or an event with instruction="cancel" first; ` +
				`the collector/human must not cancel silently (plan/06 §4).`,
			{ enabled: true },
		);
	}
	const jobId = typeof record.job_id === "string" && record.job_id !== "" ? record.job_id : null;
	if (jobId === null) return fail(`${experimentId} has no job_id; there is nothing to cancel`, { enabled: true });
	const result = session.ssh(`scancel ${jobId}`);
	if (result.status !== 0) {
		return fail(`scancel ${jobId} failed (exit ${result.status}): ${trimOutput(result.stderr, 400)}`, {
			enabled: true,
			transcripts: transcriptSummary(session.transcripts),
		});
	}
	const appended = appendRecord(
		ctx.root,
		"experiment",
		{ ...record, id: experimentId, status: "cancelled", cancel_reason: reason },
		{ root: ctx.root, role: "conductor", actor: ctx.actor },
	);
	appendRecord(
		ctx.root,
		"event",
		{ summary: `cancelled experiment ${experimentId} (job ${jobId}): ${reason}`, experiment_id: experimentId, instruction: "cancel" },
		{ root: ctx.root, role: "conductor", actor: ctx.actor },
	);
	return ok(`cancelled ${experimentId} (job ${jobId}): ${reason}`, {
		enabled: true,
		record: appended,
		transcripts: transcriptSummary(session.transcripts),
	});
}

// ---------------------------------------------------------------------------
// hpc_requeue
// ---------------------------------------------------------------------------

export interface HpcRequeueArgs {
	experiment_id: string;
	reason: string;
	/** Set by the conductor's OOM retry (doubles the job memory). */
	memoryGb?: number;
}

export function handleHpcRequeue(args: HpcRequeueArgs, ctx: HpcContext): ToolTextResult {
	checkCapability(ctx, "hpc_requeue");
	const experimentId = String(args.experiment_id ?? "").trim();
	const reason = String(args.reason ?? "").trim();
	if (experimentId === "") return fail('argument "experiment_id" must be a non-empty string');
	if (reason === "") return fail('argument "reason" is required: a requeue must carry the collector/conductor decision');
	const { resolution, error } = runtimeFor(ctx.root);
	if (error) return fail(error.message, { enabled: false });
	if (!resolution.enabled) return disabled(resolution.reason);
	const runtime = resolution.runtime;
	const session = sessionFor(ctx, runtime);
	const record = latestRecord(ctx.root, experimentId);
	if (!record) return fail(`unknown experiment ${experimentId}`, { enabled: true });

	// Single-slot: no other experiment may be active.
	const others = activeExperiments(experimentRecords(ctx.root)).filter((job) => String(job.id) !== experimentId);
	if (others.length > 0) {
		return fail(
			`refused: experiment ${others.map((job) => `${String(job.id)}(${String(job.status)})`).join(", ")} is active; ` +
				`only one job may be active at a time (plan/06_HPC_AND_COLLECTOR.md)`,
			{ enabled: true, active: others.map((job) => job.id) },
		);
	}
	const status = String(record.status ?? "");
	if (status !== "requeue_requested") {
		return fail(
			`refused: ${experimentId} is "${status}", not "requeue_requested". The collector records the evidence and the ` +
				`requeue_requested status first; hpc_requeue only executes that decision.`,
			{ enabled: true, status },
		);
	}
	const attempt = numberField(record, "attempt", 1);
	const maxAttempts = runtime.config.maxAttemptsPerExperiment;
	if (attempt >= maxAttempts) {
		return fail(
			`refused: attempt ${attempt} >= maxAttemptsPerExperiment ${maxAttempts}; the conductor must mark ${experimentId} ` +
				`failed-final (plan/06 §6)`,
			{ enabled: true },
		);
	}

	// The previous attempt must be confirmed ended before a new one is queued.
	const jobId = typeof record.job_id === "string" && record.job_id !== "" ? record.job_id : null;
	if (jobId !== null) {
		const check = session.ssh(`squeue -u "$USER" -j ${jobId} -h -o ${shq(SQUEUE_FORMAT)}`);
		if (check.status === 0) {
			const row = parseSqueueRows(check.stdout).find(
				(candidate) => candidate.jobId === jobId || candidate.jobId.startsWith(`${jobId}.`),
			);
			if (row && !["PENDING", "CONFIGURING"].includes(row.state)) {
				return fail(
					`refused: job ${jobId} is still ${row.state} (${row.elapsed}); the previous attempt must be confirmed ` +
						`ended before requeueing (single-slot rule)`,
					{ enabled: true, job_id: jobId, state: row.state },
				);
			}
		}
	}

	const resources = (record.resources ?? {}) as Record<string, unknown>;
	const memoryGb = Number(args.memoryGb ?? resources.memGb ?? runtime.config.memGb);
	const updated = appendRecord(
		ctx.root,
		"experiment",
		{
			...record,
			id: experimentId,
			status: "queued",
			attempt: attempt + 1,
			job_id: null,
			previous_job_id: jobId,
			memoryDoubled: args.memoryGb !== undefined ? true : record.memoryDoubled === true,
			resources: { ...resources, memGb: memoryGb },
			requeue: { from_job: jobId, from_attempt: attempt, reason, at: new Date().toISOString(), actor: ctx.actor },
		},
		{ root: ctx.root, role: "conductor", actor: ctx.actor },
	);
	appendRecord(
		ctx.root,
		"event",
		{
			summary: `requeued ${experimentId}: attempt ${attempt} → ${attempt + 1} (job ${jobId ?? "?"} ended): ${reason}`,
			experiment_id: experimentId,
			from: "requeue_requested",
			to: "queued",
		},
		{ root: ctx.root, role: "conductor", actor: ctx.actor },
	);
	return ok(
		`requeued ${experimentId}: attempt ${attempt + 1} queued with the same idempotency key ` +
			`(${String(record.idempotencyKey ?? "").slice(0, 16)}…)` +
			(args.memoryGb !== undefined ? ` and memory doubled to ${memoryGb}G` : ""),
		{ enabled: true, record: updated, attempt: attempt + 1, memory_gb: memoryGb, transcripts: transcriptSummary(session.transcripts) },
	);
}

// ---------------------------------------------------------------------------
// Collector housekeeping
// ---------------------------------------------------------------------------

/** Append one `heartbeat` event for an unchanged poll (plan/06 §6.5). */
export function recordHeartbeat(root: string, experimentId: string, pollIndex: number, evidence: string, actor: string): Record<string, unknown> {
	return appendRecord(
		root,
		"event",
		{
			summary: `heartbeat ${pollIndex} for ${experimentId}: no transition`,
			experiment_id: experimentId,
			heartbeat: true,
			poll: pollIndex,
			evidence: trimOutput(evidence, 2000),
		},
		{ root, role: "collector", actor },
	);
}

/** Append one `hpc-disabled` event when the collector runs without a host. */
export function recordHpcDisabled(root: string, actor: string, reason: string): Record<string, unknown> {
	return appendRecord(
		root,
		"event",
		{ summary: `hpc-disabled: ${reason}`, hpc_disabled: true },
		{ root, role: "collector", actor },
	);
}

export { runSubmitGate, type GateCheck };
