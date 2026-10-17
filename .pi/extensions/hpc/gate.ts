/**
 * The `hpc_submit` gate (plan/06_HPC_AND_COLLECTOR.md §3, prompt S5 §1).
 *
 * Every check is named, and every failure names the missing item. The gate is
 * pure with respect to the cluster except for the preflight ssh command, which
 * proves the remote work tree is a clean detached checkout of the record's
 * commit and that `uv.lock` matches the record.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { validateRecord, activeExperiments, projectLatest } from "../../lib/records.ts";
import { redact, trimOutput, type HpcRuntime } from "./config.ts";
import { sha256File } from "./manifest.ts";
import { parsePreflight, preflightCommand } from "./slurm.ts";
import type { HpcSession } from "./ssh.ts";

export interface GateCheck {
	name: string;
	ok: boolean;
	detail: string;
}

export interface SubmitGateResult {
	enabled: boolean;
	ok: boolean;
	checks: GateCheck[];
	missing: string[];
	record: Record<string, unknown> | null;
	reason: string;
}

function check(name: string, ok: boolean, detail: string): GateCheck {
	return { name, ok, detail };
}

export interface SubmitGateOptions {
	root: string;
	session: HpcSession;
	runtime: HpcRuntime;
	experimentId: string;
	/** Stage index for a record explicitly split into stages. */
	stage?: number;
	/** Skip the remote preflight (unit tests that only exercise the local rules). */
	skipRemote?: boolean;
}

export function runSubmitGate(options: SubmitGateOptions): SubmitGateResult {
	const { root, session, runtime, experimentId } = options;
	const checks: GateCheck[] = [];

	const records = readJsonl(root);
	const latest = projectLatest(records as Array<{ id?: string; seq: number }>).find(
		(record) => record.id === experimentId,
	) as Record<string, unknown> | undefined;
	if (!latest) {
		return {
			enabled: true,
			ok: false,
			checks: [check("record", false, `no experiment record ${experimentId} in .pi/state/EXPERIMENTS.jsonl`)],
			missing: [`experiment record ${experimentId}`],
			record: null,
			reason: `unknown experiment ${experimentId}`,
		};
	}

	// 1. Single-slot rule: no *other* experiment may be active.
	const active = activeExperiments(records);
	const others = active.filter((job) => String(job.id) !== experimentId);
	checks.push(
		check(
			"single-slot",
			others.length === 0,
			others.length === 0
				? "no other experiment is queued/running/collecting/requeue_requested"
				: `experiment ${others.map((job) => `${String(job.id)}(${String(job.status)})`).join(", ")} is active`,
		),
	);

	// 2. The record validates against the experiment schema.
	try {
		validateRecord("experiment", latest, `${root}/.pi/state/EXPERIMENTS.jsonl`);
		checks.push(check("record-schema", true, "record validates against the experiment schema"));
	} catch (error) {
		checks.push(check("record-schema", false, (error as Error).message));
	}

	const status = String(latest.status ?? "");
	checks.push(
		check(
			"record-status",
			status === "queued",
			status === "queued" ? 'status "queued"' : `status is "${status}"; only a queued record may be submitted`,
		),
	);
	if (typeof latest.job_id === "string" && latest.job_id !== "") {
		checks.push(
			check(
				"not-already-submitted",
				false,
				`record already carries job_id ${latest.job_id}; a second submission is refused (use hpc_requeue after the previous attempt ended)`,
			),
		);
	} else {
		checks.push(check("not-already-submitted", true, "no job_id recorded yet"));
	}

	// 3. Checkpoint block: dir, progressFile, resumeCommand, flushOnSignal.
	const checkpoints = (latest.checkpoints ?? {}) as Record<string, unknown>;
	const checkpointProblem = checkpointBlockProblem(checkpoints);
	checks.push(
		check(
			"checkpoint-block",
			checkpointProblem === null,
			checkpointProblem ?? "checkpoint dir, progressFile, resumeCommand and flushOnSignal=true are present",
		),
	);

	// 4. Local resume smoke-test evidence.
	const smoke = smokeTestProblem(latest);
	checks.push(check("resume-smoke-test", smoke === null, smoke ?? "local resume smoke-test evidence is recorded"));

	// 5. Local `uv.lock` matches the record (a stale record must be rebuilt).
	const recordLock = String(latest.uv_lock_sha ?? "");
	const localLock = path.join(root, "uv.lock");
	if (fs.existsSync(localLock)) {
		const localSha = sha256File(localLock);
		checks.push(
			check(
				"uv-lock",
				recordLock !== "" && localSha === recordLock,
				recordLock === ""
					? "record has no uv_lock_sha"
					: localSha === recordLock
						? `local uv.lock matches ${recordLock.slice(0, 12)}…`
						: `record uv_lock_sha ${recordLock.slice(0, 12)}… != local uv.lock ${localSha.slice(0, 12)}… (the record is stale)`,
			),
		);
	} else {
		checks.push(check("uv-lock", recordLock !== "", recordLock !== "" ? `record uv_lock_sha ${recordLock.slice(0, 12)}…` : "record has no uv_lock_sha"));
	}

	// 6. Container hash (only when the config names one).
	const container = runtime.config.container.trim();
	const recordContainer = String(latest.container_sha ?? "");
	if (container === "") {
		checks.push(
			check(
				"container",
				recordContainer === "" || recordContainer === "-",
				recordContainer === "" || recordContainer === "-"
					? "no container configured"
					: `config has no container but the record names container_sha ${recordContainer.slice(0, 12)}…`,
			),
		);
	} else if (recordContainer === "" || recordContainer === "-") {
		checks.push(check("container", false, `config names container ${container} but the record has no container_sha`));
	} else if (!options.skipRemote) {
		const result = session.ssh(`sha256sum ${JSON.stringify(container)} | cut -d" " -f1`);
		const remoteSha = result.stdout.trim().split(/\s+/)[0] ?? "";
		checks.push(
			check(
				"container",
				result.status === 0 && remoteSha === recordContainer,
				result.status !== 0
					? `cannot hash ${container} remotely: ${trimOutput(result.stderr, 300)}`
					: remoteSha === recordContainer
						? `remote container matches ${recordContainer.slice(0, 12)}…`
						: `remote container ${remoteSha.slice(0, 12)}… != record ${recordContainer.slice(0, 12)}…`,
			),
		);
	} else {
		checks.push(check("container", true, `container hash check skipped (remote disabled in this run)`));
	}

	// 7. Wall clock: the job assumes a 24h kill; a longer run must be staged.
	const resources = (latest.resources ?? {}) as Record<string, unknown>;
	const hours = Number(resources.wallClockHours ?? runtime.config.wallClockHours);
	const limit = Math.min(runtime.config.wallClockHours, 24);
	const stages = Number(latest.stages ?? 1);
	const staged = stages > 1 && options.stage !== undefined;
	checks.push(
		check(
			"wall-clock",
			hours <= limit || staged,
			hours <= limit
				? `${hours}h <= ${limit}h`
				: staged
					? `${hours}h > ${limit}h but the record is split into ${stages} stages and stage ${options.stage} was requested`
					: `${hours}h > ${limit}h and the record is not staged (stages>1 plus an explicit stage index is required)`,
		),
	);

	// 8. Remote preflight: clean detached checkout + uv.lock at the record's SHA.
	const commitSha = String(latest.commit_sha ?? "");
	if (options.skipRemote) {
		checks.push(check("remote-checkout", true, "remote preflight skipped (remote disabled in this run)"));
	} else if (commitSha === "") {
		checks.push(check("remote-checkout", false, "record has no commit_sha"));
	} else {
		const remote = session.ssh(preflightCommand({ runtime, experimentId, commitSha }));
		if (remote.status !== 0) {
			checks.push(
				check(
					"remote-checkout",
					false,
					`remote preflight failed (exit ${remote.status}): ${trimOutput(remote.stderr || remote.stdout, 400)}`,
				),
			);
		} else {
			const parsed = parsePreflight(remote.stdout);
			const headOk = parsed.head === commitSha;
			const lockOk = parsed.uvLockSha === recordLock;
			const ok = parsed.ok && headOk && lockOk;
			checks.push(
				check(
					"remote-checkout",
					ok,
					ok
						? `clean detached checkout at ${commitSha.slice(0, 12)}…; remote uv.lock matches`
						: !parsed.ok
							? parsed.reason
							: !headOk
								? `remote HEAD ${parsed.head.slice(0, 12)}… != record commit_sha ${commitSha.slice(0, 12)}…`
								: `remote uv.lock ${parsed.uvLockSha.slice(0, 12)}… != record ${recordLock.slice(0, 12)}…`,
				),
			);
		}
	}

	const missing = checks.filter((item) => !item.ok).map((item) => `${item.name}: ${item.detail}`);
	return {
		enabled: true,
		ok: missing.length === 0,
		checks,
		missing,
		record: latest,
		reason:
			missing.length === 0
				? `all ${checks.length} submission checks passed for ${experimentId}`
				: `submission refused for ${experimentId}: ${missing.length} check(s) failed`,
	};
}

export function checkpointBlockProblem(checkpoints: Record<string, unknown>): string | null {
	for (const field of ["dir", "progressFile", "resumeCommand"]) {
		const value = checkpoints[field];
		if (typeof value !== "string" || value.trim() === "") {
			return `checkpoint metadata is missing "${field}" (need dir, progressFile, resumeCommand, flushOnSignal)`;
		}
	}
	if (checkpoints.flushOnSignal !== true) {
		return `checkpoint metadata must set "flushOnSignal": true (the USR1 trap flushes and exits 99)`;
	}
	return null;
}

export function smokeTestProblem(record: Record<string, unknown>): string | null {
	const smoke = record.smokeTest as Record<string, unknown> | undefined;
	if (smoke && smoke.ok === true) return null;
	if (typeof record.resumeSmokeTest === "string" && record.resumeSmokeTest.trim() !== "") return null;
	return (
		`local resume smoke-test evidence is missing: record "resumeSmokeTest" (or a "smokeTest" object with ` +
		`"ok": true) must prove that --resume skips completed units (plan/06 §5.4)`
	);
}

function readJsonl(root: string): Array<Record<string, unknown>> {
	const file = path.join(root, ".pi", "state", "EXPERIMENTS.jsonl");
	const raw = fs.existsSync(file) ? fs.readFileSync(file, "utf8") : "";
	return raw
		.split("\n")
		.filter((line) => line.trim() !== "")
		.map((line) => JSON.parse(line) as Record<string, unknown>);
}

export function redactedChecks(checks: GateCheck[]): GateCheck[] {
	return checks.map((item) => ({ ...item, detail: redact(item.detail) }));
}
