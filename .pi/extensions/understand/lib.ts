/**
 * Pure logic for the `understand_loop` tutorial extension.
 *
 * No pi imports: `node --test` drives this module directly, and the extension
 * index is only a thin pi-facing shell around these functions.
 *
 * Progress is durable so a tutorial survives a session restart:
 * `.pi/state/UNDERSTAND_LOOP.json` records which chapters were delivered, which
 * the reader marked understood, and the questions asked along the way.
 */

import * as fs from "node:fs";
import { VangjaLoopError } from "../../lib/errors.ts";
import { readJsonFile, withLock, writeJsonAtomic } from "../../lib/io.ts";
import { statePath } from "../../lib/paths.ts";
import { CURRICULUM, chapterById, chapterTitle, type Chapter } from "./curriculum.ts";

export const UNDERSTAND_SCHEMA = "pi-understand/1";
/** Marks a message this extension injected, so it is never recorded as a question. */
export const PROMPT_SENTINEL = "[[understand_loop]]";
export const MAX_QUESTIONS = 100;
export const PROGRESS_FILE = "UNDERSTAND_LOOP.json";

export type UnderstandingStatus = "not-started" | "awaiting" | "complete";

export interface QuestionRecord {
	at: string;
	chapter: number;
	text: string;
}

export interface UnderstandingProgress {
	schema: typeof UNDERSTAND_SCHEMA;
	started_at: string;
	updated_at: string;
	/** 1-based chapter currently in front of the reader; 0 before the tour starts. */
	current: number;
	delivered: number[];
	completed: number[];
	questions: QuestionRecord[];
	status: UnderstandingStatus;
	/** How many times the tour was restarted from chapter 1. */
	restarts: number;
}

export interface UnderstandCommand {
	action: "next" | "start" | "back" | "status" | "reset" | "help" | "show" | "ask";
	chapter?: number;
	question?: string;
}

const USAGE = [
	"usage: /understand_loop [start|next|back|status|reset|help]",
	"       /understand_loop show <chapter>",
	"       /understand_loop ask <question>",
].join("\n");

export function defaultProgress(startedAt = new Date().toISOString()): UnderstandingProgress {
	return {
		schema: UNDERSTAND_SCHEMA,
		started_at: startedAt,
		updated_at: new Date().toISOString(),
		current: 0,
		delivered: [],
		completed: [],
		questions: [],
		status: "not-started",
		restarts: 0,
	};
}

export function progressPath(root: string): string {
	return statePath(root, PROGRESS_FILE);
}

function asIntegerArray(value: unknown, total: number, field: string, file: string): number[] {
	if (value === undefined) return [];
	if (!Array.isArray(value)) throw new VangjaLoopError("CONFIG_INVALID", `${file}: "${field}" must be an array`);
	const ids = value.map((id) => Number(id));
	for (const id of ids) {
		if (!Number.isInteger(id) || id < 1 || id > total) {
			throw new VangjaLoopError("CONFIG_INVALID", `${file}: "${field}" contains ${String(id)}, outside 1..${total}`);
		}
	}
	return [...new Set(ids)].sort((a, b) => a - b);
}

/** Load progress, or a fresh record when the tour has never started. */
export function loadProgress(root: string): UnderstandingProgress {
	const file = progressPath(root);
	const raw = readJsonFile<Record<string, unknown>>(file);
	if (!raw) return defaultProgress();
	if (raw.schema !== UNDERSTAND_SCHEMA) {
		throw new VangjaLoopError(
			"CONFIG_INVALID",
			`${file}: expected schema "${UNDERSTAND_SCHEMA}", got ${String(raw.schema)}. ` +
				`Delete the file (or /understand_loop reset) to start the tour again.`,
		);
	}
	const status = String(raw.status ?? "not-started");
	if (!["not-started", "awaiting", "complete"].includes(status)) {
		throw new VangjaLoopError("CONFIG_INVALID", `${file}: unknown status "${status}"`);
	}
	const current = Number(raw.current ?? 0);
	if (!Number.isInteger(current) || current < 0 || current > CURRICULUM.length) {
		throw new VangjaLoopError("CONFIG_INVALID", `${file}: "current" must be 0..${CURRICULUM.length}`);
	}
	const questions = Array.isArray(raw.questions)
		? (raw.questions as QuestionRecord[]).filter((item) => item && typeof item.text === "string")
		: [];
	return {
		...defaultProgress(String(raw.started_at ?? new Date().toISOString())),
		updated_at: String(raw.updated_at ?? ""),
		current,
		delivered: asIntegerArray(raw.delivered, CURRICULUM.length, "delivered", file),
		completed: asIntegerArray(raw.completed, CURRICULUM.length, "completed", file),
		questions: questions.slice(-MAX_QUESTIONS),
		status: status as UnderstandingStatus,
		restarts: Number.isInteger(Number(raw.restarts)) ? Number(raw.restarts) : 0,
	};
}

/** Persist progress atomically under the standard loop lock. */
export function saveProgress(root: string, progress: UnderstandingProgress): UnderstandingProgress {
	const file = progressPath(root);
	const next: UnderstandingProgress = { ...progress, updated_at: new Date().toISOString() };
	withLock(`${file}.lock`, () => {
		writeJsonAtomic(file, next);
	});
	return next;
}

function clampChapter(id: number): number {
	return Math.min(Math.max(Math.trunc(id), 1), CURRICULUM.length);
}

function deliver(progress: UnderstandingProgress, id: number): UnderstandingProgress {
	const chapter = clampChapter(id);
	const delivered = new Set([...progress.delivered, chapter]);
	return {
		...progress,
		current: chapter,
		delivered: [...delivered].sort((a, b) => a - b),
		status: "awaiting",
	};
}

/** Start (or restart) the tour at a chapter, resetting the understood set. */
export function beginTutorial(progress: UnderstandingProgress, id = 1): UnderstandingProgress {
	const chapter = clampChapter(id);
	const restarted = progress.status !== "not-started";
	return deliver(
		{
			...progress,
			completed: [],
			restarts: restarted ? progress.restarts + 1 : progress.restarts,
			started_at: restarted ? new Date().toISOString() : progress.started_at,
		},
		chapter,
	);
}

/** Re-show a chapter without changing what the reader has marked understood. */
export function showChapter(progress: UnderstandingProgress, id: number): UnderstandingProgress {
	return deliver(progress, id);
}

/** Re-show the previous chapter (stops at chapter 1). */
export function goBack(progress: UnderstandingProgress): UnderstandingProgress {
	const previous = progress.current > 1 ? progress.current - 1 : 1;
	return deliver(progress, previous);
}

/**
 * Mark the current chapter understood and deliver the next one. Returns a
 * `complete` progress (current stays on the last chapter) at the end.
 */
export function nextChapter(progress: UnderstandingProgress): UnderstandingProgress {
	if (progress.status === "not-started") return deliver(progress, 1);
	if (progress.status === "complete") return progress;
	const completed = new Set([...progress.completed, progress.current]);
	const done = [...completed].sort((a, b) => a - b);
	if (progress.current >= CURRICULUM.length) {
		return { ...progress, completed: done, status: "complete" };
	}
	return deliver({ ...progress, completed: done }, progress.current + 1);
}

/** Record a question against the current chapter (bounded, newest kept). */
export function recordQuestion(
	progress: UnderstandingProgress,
	text: string,
	at = new Date().toISOString(),
): UnderstandingProgress {
	const clean = text.trim();
	if (clean === "") return progress;
	const questions = [...progress.questions, { at, chapter: progress.current, text: clean.slice(0, 1000) }];
	return {
		...progress,
		questions: questions.slice(-MAX_QUESTIONS),
		status: progress.status === "complete" ? "complete" : "awaiting",
	};
}

export function resetProgress(): UnderstandingProgress {
	return defaultProgress();
}

export function completedCount(progress: UnderstandingProgress): number {
	return Math.min(progress.completed.length, CURRICULUM.length);
}

export function progressBar(progress: UnderstandingProgress, width = 12): string {
	const position =
		progress.status === "not-started"
			? 0
			: Math.min(Math.max(progress.current, completedCount(progress)), CURRICULUM.length);
	const filled = Math.round((position / CURRICULUM.length) * width);
	return `[${"#".repeat(filled)}${"-".repeat(width - filled)}]`;
}

/** One compact line for `ctx.ui.setStatus`; the bar shows tour position. */
export function formatProgressLine(progress: UnderstandingProgress): string {
	if (progress.status === "not-started" || progress.current < 1) {
		return `${progressBar(progress)} 0/${CURRICULUM.length} · not started`;
	}
	const position = Math.min(progress.current, CURRICULUM.length);
	const percent = Math.round((position / CURRICULUM.length) * 100);
	return `${progressBar(progress)} ch. ${position}/${CURRICULUM.length} (${percent}%) · ${chapterTitle(position)}`;
}

/** Widget lines: progress, what to do next, and the recorded interaction count. */
export function formatStatusLines(progress: UnderstandingProgress, busy = false): string[] {
	if (progress.status === "not-started") {
		return ["understand_loop — not started", "begin: /understand_loop start"];
	}
	const next =
		progress.status === "complete"
			? "tour complete — /understand_loop reset to run it again"
			: busy
				? "explaining this chapter…"
				: `ask questions in chat, then /understand_loop next`;
	return [
		`understand_loop ${formatProgressLine(progress)}`,
		`${next} · ${completedCount(progress)}/${CURRICULUM.length} understood · ${progress.questions.length} question(s)`,
	];
}

/** Multi-line text for the `/understand_loop status` notification. */
export function formatStatusText(progress: UnderstandingProgress): string {
	if (progress.status === "not-started") {
		return `understand_loop — not started (${CURRICULUM.length} chapters).\nRun /understand_loop start.`;
	}
	const lines = [
		`understand_loop ${formatProgressLine(progress)}`,
		progress.status === "complete"
			? "status: tour complete"
			: `status: chapter ${progress.current} delivered, awaiting your questions`,
		`questions recorded: ${progress.questions.length}${progress.questions.length > 0 ? ` (latest: "${progress.questions[progress.questions.length - 1].text.slice(0, 80)}")` : ""}`,
		"commands: /understand_loop next | back | show <n> | ask <question> | status | reset",
	];
	return lines.join("\n");
}

/**
 * Parse the command tail. Bare `/understand_loop` means "continue": start when
 * the tour has not begun, otherwise deliver the next chapter.
 */
export function parseCommand(args: string): UnderstandCommand {
	const trimmed = args.trim();
	if (trimmed === "") return { action: "next" };
	const [word, ...rest] = trimmed.split(/\s+/);
	switch (word) {
		case "start":
			return { action: "start" };
		case "next":
			return { action: "next" };
		case "back":
			return { action: "back" };
		case "status":
			return { action: "status" };
		case "reset":
			return { action: "reset" };
		case "help":
			return { action: "help" };
		case "show": {
			if (rest.length !== 1) throw new VangjaLoopError("STATE_INVALID", `show needs exactly one chapter number\n${USAGE}`);
			const id = Number(rest[0]);
			if (!Number.isInteger(id) || id < 1 || id > CURRICULUM.length) {
				throw new VangjaLoopError(
					"STATE_INVALID",
					`chapter must be an integer in 1..${CURRICULUM.length}, got "${rest[0]}"\n${USAGE}`,
				);
			}
			return { action: "show", chapter: id };
		}
		case "ask": {
			const question = rest.join(" ").trim();
			if (question === "") throw new VangjaLoopError("STATE_INVALID", `ask needs a question\n${USAGE}`);
			return { action: "ask", question };
		}
		default:
			throw new VangjaLoopError("STATE_INVALID", `unknown action "${word}"\n${USAGE}`);
	}
}

export function usageText(): string {
	return USAGE;
}

function chapterOrThrow(id: number): Chapter {
	const chapter = chapterById(id);
	if (!chapter) throw new VangjaLoopError("STATE_INVALID", `no chapter ${id} in the curriculum`);
	return chapter;
}

/**
 * The user message that makes the tutor present one chapter. The chapter text is
 * included verbatim so the explanation (and any follow-up Q&A) is grounded in
 * the versioned material even if the model never opens a file.
 */
export function buildChapterPrompt(progress: UnderstandingProgress): string {
	const chapter = chapterOrThrow(progress.current);
	const total = CURRICULUM.length;
	return [
		`${PROMPT_SENTINEL} chapter ${chapter.id}/${total} — ${chapter.title}`,
		"",
		"Explain this chapter to me in the chat as one compact tutorial chunk (roughly 150–250 words).",
		"Rules:",
		"- Present only this chapter. Do not preview, start, or summarise the next chapter.",
		"- Be faithful to the material below; you may open the referenced files to add exact file:line detail, but do not invent facts.",
		"- Do not modify any files and do not call ledger or HPC tools.",
		"- End with the checkpoint questions (verbatim) and exactly one closing line:",
		`  "Progress: ${completedCount(progress)}/${total} chapters complete. Ask me anything about this chapter; when you are ready, run /understand_loop next."`,
		"- If I ask a question instead, answer it from this chapter and do not advance the tour.",
		"",
		"--- chapter material ---",
		chapter.body,
		"",
		"--- references (repo-relative) ---",
		...chapter.refs.map((ref) => `- ${ref}`),
		"",
		"--- checkpoint questions ---",
		...chapter.checks.map((check, index) => `${index + 1}. ${check}`),
	].join("\n");
}

/** The user message for `/understand_loop ask <question>`: answer in chapter context. */
export function buildQuestionPrompt(progress: UnderstandingProgress, question: string): string {
	const chapter = chapterOrThrow(progress.current);
	const total = CURRICULUM.length;
	return [
		`${PROMPT_SENTINEL} question about chapter ${chapter.id}/${total} — ${chapter.title}`,
		"",
		`I am studying this chapter and ask: ${question}`,
		"",
		"Answer in the chat using the chapter material and the referenced files. Keep it compact.",
		"Do not advance the tour and do not present the next chapter; end with:",
		`  "Run /understand_loop next when you are ready to continue."`,
		"",
		"--- chapter material ---",
		chapter.body,
		"",
		"--- references (repo-relative) ---",
		...chapter.refs.map((ref) => `- ${ref}`),
	].join("\n");
}

/** First chapter id, used by tests and the "start" handler. */
export const FIRST_CHAPTER = CURRICULUM[0]?.id ?? 1;
export const TOTAL_CHAPTERS = CURRICULUM.length;

/** True when a state file exists (used by the session-start widget refresh). */
export function progressExists(root: string): boolean {
	return fs.existsSync(progressPath(root));
}
