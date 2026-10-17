/**
 * The conductor's retry decision — the retry table of
 * `plan/06_HPC_AND_COLLECTOR.md` §6, as a pure function.
 *
 * The collector only records facts (raw squeue/sacct output, checkpoint progress,
 * hashes) and appends `requeue_requested`. The conductor owns the decision:
 * requeue (attempt+1), requeue once with doubled memory after an OOM, or
 * `failed-final` + `BLOCKED`. No guess is made when the retry table does not
 * cover the situation: the function returns `blocked` with a question.
 *
 * No pi imports: usable from `node --test`.
 */

export type RetryAction = "wait" | "collect" | "requeue" | "failed-final" | "blocked" | "open-task";

export interface ProgressHistoryEntry {
	attempt: number;
	completed: number;
}

export interface ProgressStats {
	advanced: boolean;
	noProgressAttempts: number;
	current: number;
	previous: number;
}

/** Compare the per-attempt checkpoint progress snapshots. */
export function progressStats(history: ProgressHistoryEntry[], currentAttempt: number): ProgressStats {
	const byAttempt = new Map<number, number>();
	for (const entry of history) {
		byAttempt.set(entry.attempt, Math.max(byAttempt.get(entry.attempt) ?? 0, entry.completed));
	}
	const current = byAttempt.get(currentAttempt) ?? 0;
	const previous = byAttempt.get(currentAttempt - 1) ?? 0;
	const advanced = current > previous;
	// Trailing attempts that did not raise the high-water mark of completed units
	// (attempt 1 with 0 units counts: nothing was checkpointed).
	let best = 0;
	let noProgressAttempts = 0;
	for (let attempt = 1; attempt <= currentAttempt; attempt += 1) {
		const completed = byAttempt.get(attempt) ?? 0;
		if (completed > best) {
			best = completed;
			noProgressAttempts = 0;
		} else {
			noProgressAttempts += 1;
		}
	}
	return { advanced, noProgressAttempts, current, previous };
}

/** Statuses that still need the collector (any status that is not terminal). */
export const TERMINAL_EXPERIMENT_STATUSES = ["collected", "failed-final", "cancelled"] as const;

export function isTerminalExperimentStatus(status: string): boolean {
	return (TERMINAL_EXPERIMENT_STATUSES as readonly string[]).includes(status);
}

/** Statuses the collector must still process (checkpoints, transitions, evidence). */
export const PENDING_COLLECTOR_STATUSES = ["running", "collecting", "timed_out", "interrupted_clean", "failed"] as const;

export interface RetryInput {
	status: string;
	attempt: number;
	maxAttempts: number;
	/** Per-attempt checkpoint progress snapshots recorded by the collector. */
	progressHistory: ProgressHistoryEntry[];
	/** Memory already doubled once for this experiment (OOM retry used). */
	memoryDoubled: boolean;
	memoryGb: number;
	/** Partition memory limit from `resources.partitionMemGbLimit`, or null (unknown). */
	partitionMemGbLimit: number | null;
	/** Last classified sacct state: queued|running|completed|oom|timed_out|… */
	sacctState: string | null;
	exitCode: number | null;
	stderrTail: string | null;
	/** Heartbeat events recorded for this experiment so far. */
	heartbeatCount: number;
	/** From `.pi/config/hpc.json` maxPolls. */
	maxPolls: number;
	/** Declared outputs that the last collection found missing. */
	missingOutputs: string[];
	taskId: string;
	jobId: string | null;
}

export interface RetryDecision {
	action: RetryAction;
	reason: string;
	/** Set when action === "requeue" after an OOM. */
	memoryGb?: number;
	/** Set when action === "blocked" / "open-task": the smallest human/task decision. */
	question?: string;
	/** Set when action === "open-task". */
	title?: string;
}

const STATUS_COLLECT = new Set<string>([...PENDING_COLLECTOR_STATUSES, "queued"]);

export function decideRetry(input: RetryInput): RetryDecision {
	const stats = progressStats(input.progressHistory, input.attempt);
	const attempt = input.attempt;

	if (input.status === "collected") {
		return { action: "wait", reason: `experiment already collected (job ${input.jobId ?? "?"})` };
	}
	if (input.status === "cancelled") {
		return {
			action: "blocked",
			reason: `experiment was cancelled (job ${input.jobId ?? "?"})`,
			question: "record what should happen next: resubmit as a new experiment, or close the task",
		};
	}
	if (input.status === "failed-final") {
		return {
			action: "blocked",
			reason: `experiment is failed-final after attempt ${attempt}`,
			question: "read the failure evidence in .pi/state/EXPERIMENTS.jsonl and decide whether to open a fix task",
		};
	}
	if (input.status === "collection-failed") {
		const missing = input.missingOutputs.length > 0 ? input.missingOutputs.join(", ") : "(see the manifest)";
		return {
			action: "open-task",
			reason: `collection failed for ${input.taskId}: declared outputs missing: ${missing}`,
			title: `repair missing outputs for experiment (${input.taskId})`,
			question:
				`declared outputs are missing after a completed job: ${missing}. ` +
				`Decide whether the runner must write them incrementally or the record was wrong; do not fabricate results.`,
		};
	}

	// Anything the collector has not yet turned into a decision: hand it to the
	// collector, unless the poll budget is exhausted.
	if (STATUS_COLLECT.has(input.status)) {
		if (input.heartbeatCount >= input.maxPolls) {
			return {
				action: "blocked",
				reason: `no transition for ${input.heartbeatCount} polls (maxPolls ${input.maxPolls}); job ${input.jobId ?? "?"} stuck in ${input.status}`,
				question: "inspect squeue/sacct evidence and decide whether to cancel, requeue, or fix the collector",
			};
		}
		return {
			action: "collect",
			reason: `collector pass needed for ${input.status} (job ${input.jobId ?? "?"}, attempt ${attempt})`,
		};
	}

	// requeue_requested (and any collector-completed failure status left behind):
	// apply the retry table.
	if (attempt >= input.maxAttempts) {
		return {
			action: "failed-final",
			reason: `attempt ${attempt} >= maxAttemptsPerExperiment ${input.maxAttempts}`,
			question: "raise maxAttemptsPerExperiment by an explicit decision, or fix the underlying failure",
		};
	}
	if (input.sacctState === "cancelled") {
		return {
			action: "failed-final",
			reason: "job was cancelled; only an explicit instruction may resubmit",
			question: "resubmit as a new experiment only after recording the cancellation instruction",
		};
	}
	if (input.sacctState === "oom" || input.exitCode === 137) {
		const limit = input.partitionMemGbLimit;
		const doubled = input.memoryGb * 2;
		if (!input.memoryDoubled && limit !== null && doubled <= limit) {
			return {
				action: "requeue",
				reason: `OOM at ${input.memoryGb}G; retrying once with ${doubled}G (partition limit ${limit}G)`,
				memoryGb: doubled,
			};
		}
		return {
			action: "failed-final",
			reason: input.memoryDoubled
				? `OOM again at ${input.memoryGb}G after the memory doubling; no further retry`
				: limit === null
					? `OOM at ${input.memoryGb}G but resources.partitionMemGbLimit is unknown; refusing to guess a larger memory`
					: `OOM at ${input.memoryGb}G; ${doubled}G exceeds the partition limit ${limit}G`,
			question: "raise resources.partitionMemGbLimit / memGb by an explicit decision, or fix the memory use",
		};
	}
	if (input.sacctState === "failed" && (input.exitCode ?? 0) !== 99) {
		// A real (non-99) failure is a diagnosis task, not a blind retry.
		return {
			action: "open-task",
			reason: `non-zero exit ${input.exitCode ?? "?"} (exit code ≠ 99 is not a clean interruption)`,
			title: `diagnose experiment failure (exit ${input.exitCode ?? "?"}, ${input.taskId})`,
			question: `the job failed with exit ${input.exitCode ?? "?"}; stderr tail: ${input.stderrTail ?? "(none recorded)"}`,
		};
	}
	if (stats.noProgressAttempts >= 2) {
		return {
			action: "failed-final",
			reason: `no checkpoint progress across ${stats.noProgressAttempts} consecutive attempts (completed units: ${stats.current})`,
			question: "the runner is not reaching a checkpoint; diagnose it locally before any further submission",
		};
	}
	if (stats.advanced) {
		return {
			action: "requeue",
			reason: `attempt ${attempt} ended (${input.sacctState ?? "unknown"}) but checkpoints advanced ${stats.previous} → ${stats.current} units`,
		};
	}
	if (attempt < input.maxAttempts && stats.current === 0) {
		// No progress yet, but this is the first such attempt: requeue once with a
		// warning. Two consecutive no-progress attempts are failed-final above, so
		// "resubmit from scratch" never becomes the default policy.
		return {
			action: "requeue",
			reason: `attempt ${attempt} ended (${input.sacctState ?? "unknown"}) with no completed unit yet; requeueing once — a second such attempt is failed-final`,
		};
	}
	return {
		action: "failed-final",
		reason: `attempt ${attempt} ended (${input.sacctState ?? "unknown"}) with no checkpoint progress`,
		question: "no completed units were checkpointed; run the smoke test locally and fix the runner before resubmitting",
	};
}
