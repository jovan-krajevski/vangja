/**
 * understand_loop: pure progress logic, prompt content, and the pi wiring.
 *
 * The wiring test drives the real extension factory with a fake `ExtensionAPI`
 * and a fake TUI context, so it proves command registration, message injection,
 * progress persistence, question recording, and the status/widget updates
 * without a model call.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import { test } from "node:test";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
	beginTutorial,
	buildChapterPrompt,
	buildQuestionPrompt,
	defaultProgress,
	formatProgressLine,
	formatStatusLines,
	formatStatusText,
	goBack,
	loadProgress,
	MAX_QUESTIONS,
	nextChapter,
	parseCommand,
	progressBar,
	progressExists,
	progressPath,
	PROMPT_SENTINEL,
	recordQuestion,
	resetProgress,
	saveProgress,
	showChapter,
	UNDERSTAND_SCHEMA,
} from "../extensions/understand/lib.ts";
import { CURRICULUM } from "../extensions/understand/curriculum.ts";
import understandLoop from "../extensions/understand/index.ts";
import { scratchRoot } from "./helpers.ts";

test("the curriculum is twelve ordered, non-empty chapters with refs and checks", () => {
	assert.equal(CURRICULUM.length, 12);
	for (const [index, chapter] of CURRICULUM.entries()) {
		assert.equal(chapter.id, index + 1, `chapter ${chapter.slug} has id ${chapter.id}`);
		assert.ok(chapter.title.trim() !== "", `chapter ${chapter.id} has no title`);
		assert.ok(chapter.body.trim().length > 300, `chapter ${chapter.id} body is too thin`);
		assert.ok(chapter.refs.length >= 1, `chapter ${chapter.id} has no refs`);
		assert.ok(chapter.checks.length >= 1, `chapter ${chapter.id} has no checks`);
	}
});

test("the tour begins at chapter 1, advances by one, and completes at the end", () => {
	const fresh = defaultProgress();
	assert.equal(fresh.status, "not-started");
	assert.equal(fresh.current, 0);

	let progress = beginTutorial(fresh);
	assert.equal(progress.status, "awaiting");
	assert.equal(progress.current, 1);
	assert.deepEqual(progress.delivered, [1]);
	assert.deepEqual(progress.completed, []);

	progress = nextChapter(progress);
	assert.equal(progress.current, 2);
	assert.deepEqual(progress.completed, [1]);
	assert.deepEqual(progress.delivered, [1, 2]);

	for (let step = 2; step < CURRICULUM.length; step += 1) progress = nextChapter(progress);
	assert.equal(progress.current, CURRICULUM.length);
	assert.equal(progress.status, "awaiting");
	assert.equal(progress.completed.length, CURRICULUM.length - 1);

	progress = nextChapter(progress);
	assert.equal(progress.status, "complete");
	assert.equal(progress.current, CURRICULUM.length);
	assert.equal(progress.completed.length, CURRICULUM.length);
	assert.deepEqual(nextChapter(progress), progress, "a complete tour is a no-op");
});

test("back and show move the pointer without falsifying completion", () => {
	let progress = beginTutorial(defaultProgress());
	progress = nextChapter(nextChapter(progress));
	assert.equal(progress.current, 3);
	progress = goBack(progress);
	assert.equal(progress.current, 2);
	assert.deepEqual(progress.completed, [1, 2]);
	progress = goBack(progress);
	assert.equal(progress.current, 1);
	progress = goBack(progress);
	assert.equal(progress.current, 1, "back stops at the first chapter");

	progress = showChapter(progress, 7);
	assert.equal(progress.current, 7);
	assert.deepEqual(progress.completed, [1, 2]);
	assert.ok(progress.delivered.includes(7));
});

test("starting again resets understanding and counts the restart", () => {
	let progress = beginTutorial(defaultProgress());
	progress = nextChapter(progress);
	progress = beginTutorial(progress, 1);
	assert.deepEqual(progress.completed, []);
	assert.equal(progress.current, 1);
	assert.equal(progress.restarts, 1);
});

test("questions are recorded against the current chapter and bounded", () => {
	let progress = beginTutorial(defaultProgress());
	progress = nextChapter(progress);
	progress = recordQuestion(progress, "  why pin the base branch?  ", "2026-01-01T00:00:00.000Z");
	assert.equal(progress.questions.length, 1);
	assert.equal(progress.questions[0].text, "why pin the base branch?");
	assert.equal(progress.questions[0].chapter, 2);
	for (let index = 1; index <= MAX_QUESTIONS + 5; index += 1) progress = recordQuestion(progress, `q${index}`);
	assert.equal(progress.questions.length, MAX_QUESTIONS);
	assert.equal(progress.questions[progress.questions.length - 1].text, `q${MAX_QUESTIONS + 5}`);
	assert.equal(recordQuestion(progress, "   "), progress, "blank questions are ignored");
});

test("reset returns the untouched default state", () => {
	const reset = resetProgress();
	assert.equal(reset.status, "not-started");
	assert.equal(reset.current, 0);
	assert.deepEqual(reset.completed, []);
	assert.deepEqual(reset.questions, []);
});

test("parseCommand maps the documented surface and refuses bad input", () => {
	assert.deepEqual(parseCommand(""), { action: "next" });
	assert.deepEqual(parseCommand("next"), { action: "next" });
	assert.deepEqual(parseCommand("start"), { action: "start" });
	assert.deepEqual(parseCommand("back"), { action: "back" });
	assert.deepEqual(parseCommand("status"), { action: "status" });
	assert.deepEqual(parseCommand("reset"), { action: "reset" });
	assert.deepEqual(parseCommand("help"), { action: "help" });
	assert.deepEqual(parseCommand("show 4"), { action: "show", chapter: 4 });
	assert.deepEqual(parseCommand("ask why does BOOT pin the branch?"), {
		action: "ask",
		question: "why does BOOT pin the branch?",
	});
	assert.throws(() => parseCommand("show 99"), /1\.\.12/);
	assert.throws(() => parseCommand("show x"), /integer/);
	assert.throws(() => parseCommand("ask"), /needs a question/);
	assert.throws(() => parseCommand("dance"), /unknown action/);
});

test("progress persists atomically and a corrupt file fails loudly", () => {
	const root = scratchRoot("vangja-understand-");
	try {
		assert.equal(progressExists(root), false);
		assert.equal(loadProgress(root).status, "not-started");

		const saved = saveProgress(root, nextChapter(beginTutorial(defaultProgress())));
		assert.equal(progressExists(root), true);
		const raw = JSON.parse(fs.readFileSync(progressPath(root), "utf8")) as Record<string, unknown>;
		assert.equal(raw.schema, UNDERSTAND_SCHEMA);
		const loaded = loadProgress(root);
		assert.equal(loaded.current, saved.current);
		assert.deepEqual(loaded.completed, saved.completed);
		assert.ok(Date.parse(loaded.updated_at) > 0);

		fs.writeFileSync(progressPath(root), JSON.stringify({ schema: "something-else" }));
		assert.throws(() => loadProgress(root), /expected schema/);
	} finally {
		fs.rmSync(root, { recursive: true, force: true });
	}
});

test("the chapter prompt is self-contained, includes checks, and does not leak the next chapter", () => {
	let progress = beginTutorial(defaultProgress());
	progress = nextChapter(progress);
	const chapter = CURRICULUM[1];
	const next = CURRICULUM[2];
	const prompt = buildChapterPrompt(progress);
	assert.ok(prompt.startsWith(PROMPT_SENTINEL));
	assert.ok(prompt.includes(`chapter 2/${CURRICULUM.length}`));
	assert.ok(prompt.includes(chapter.body));
	for (const ref of chapter.refs) assert.ok(prompt.includes(ref), `prompt is missing ref ${ref}`);
	for (const check of chapter.checks) assert.ok(prompt.includes(check));
	assert.ok(prompt.includes("/understand_loop next"));
	assert.equal(prompt.includes(next.body), false, "the next chapter's body leaked");
	assert.equal(prompt.includes(next.checks[0]), false, "the next chapter's check leaked");
});

test("the question prompt keeps the current chapter in context", () => {
	let progress = beginTutorial(defaultProgress());
	progress = showChapter(progress, 6);
	const prompt = buildQuestionPrompt(progress, "Does a merge ever rebase?");
	assert.ok(prompt.startsWith(PROMPT_SENTINEL));
	assert.ok(prompt.includes("Does a merge ever rebase?"));
	assert.ok(prompt.includes(CURRICULUM[5].body));
	assert.ok(prompt.includes("/understand_loop next"));
	assert.equal(prompt.includes(CURRICULUM[6].body), false);
});

test("progress formatting is compact and names the current chapter", () => {
	let progress = beginTutorial(defaultProgress());
	progress = nextChapter(nextChapter(progress));
	const line = formatProgressLine(progress);
	assert.match(line, /3\/12/);
	assert.match(line, /ch\. 3/);
	assert.match(line, new RegExp(CURRICULUM[2].title.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")));
	assert.equal(progressBar(progress).length, 14);
	const widget = formatStatusLines(progress);
	assert.equal(widget.length, 2);
	assert.match(widget[1], /\/understand_loop next/);
	assert.match(widget[1], /2\/12 understood/);
	assert.match(formatStatusText(progress), /questions recorded: 0/);
	assert.match(formatStatusText(defaultProgress()), /not started/);
});

// ---------------------------------------------------------------------------
// pi wiring
// ---------------------------------------------------------------------------

interface FakeUi {
	statuses: Array<[string, string | undefined]>;
	widgets: Array<[string, string[] | undefined]>;
	notices: string[];
}

function fakeRuntime(root: string, idle = true): {
	api: ExtensionAPI;
	command: { handler: (args: string, ctx: unknown) => Promise<void> };
	handlers: Map<string, Array<(event: unknown, ctx: unknown) => Promise<unknown>>>;
	ui: FakeUi;
	sent: Array<{ text: string; options?: unknown }>;
	ctx: Record<string, unknown>;
} {
	const ui: FakeUi = { statuses: [], widgets: [], notices: [] };
	const sent: Array<{ text: string; options?: unknown }> = [];
	const commands = new Map<string, { handler: (args: string, ctx: unknown) => Promise<void> }>();
	const handlers = new Map<string, Array<(event: unknown, ctx: unknown) => Promise<unknown>>>();
	const api = {
		registerCommand(name: string, options: { handler: (args: string, ctx: unknown) => Promise<void> }) {
			commands.set(name, options);
		},
		on(event: string, handler: (event: unknown, ctx: unknown) => Promise<unknown>) {
			handlers.set(event, [...(handlers.get(event) ?? []), handler]);
			return () => {};
		},
		sendUserMessage(text: string, options?: unknown) {
			sent.push({ text, options });
		},
	} as unknown as ExtensionAPI;
	const ctx = {
		cwd: root,
		mode: "tui",
		hasUI: true,
		isIdle: () => idle,
		ui: {
			notify: (text: string) => ui.notices.push(text),
			setStatus: (key: string, text?: string) => ui.statuses.push([key, text]),
			setWidget: (key: string, lines?: string[]) => ui.widgets.push([key, lines]),
		},
	};
	understandLoop(api);
	const command = commands.get("understand_loop");
	assert.ok(command, "the extension did not register /understand_loop");
	return { api, command, handlers, ui, sent, ctx };
}

test("wiring: /understand_loop starts chapter 1, records progress, and shows it", async () => {
	const root = scratchRoot("vangja-understand-wire-");
	try {
		const runtime = fakeRuntime(root);
		await runtime.command.handler("", runtime.ctx);
		assert.equal(runtime.sent.length, 1);
		assert.ok(runtime.sent[0].text.startsWith(PROMPT_SENTINEL));
		assert.ok(runtime.sent[0].text.includes(CURRICULUM[0].body));

		const progress = loadProgress(root);
		assert.equal(progress.current, 1);
		assert.equal(progress.status, "awaiting");

		const status = runtime.ui.statuses.find(([key, text]) => key === "understand-loop" && text !== undefined);
		assert.ok(status, "no status line was set");
		assert.match(String(status?.[1]), /1\/12/);
		const widget = runtime.ui.widgets.find(([key, lines]) => key === "understand-loop-progress" && lines !== undefined);
		assert.ok(widget, "no progress widget was set");
	} finally {
		fs.rmSync(root, { recursive: true, force: true });
	}
});

test("wiring: next advances, questions are recorded, and injected prompts are ignored", async () => {
	const root = scratchRoot("vangja-understand-wire-");
	try {
		const runtime = fakeRuntime(root);
		await runtime.command.handler("start", runtime.ctx);
		await runtime.command.handler("next", runtime.ctx);
		assert.equal(loadProgress(root).current, 2);
		assert.deepEqual(loadProgress(root).completed, [1]);
		assert.ok(runtime.sent[1].text.includes(CURRICULUM[1].body));

		const before = runtime.handlers.get("before_agent_start")?.[0];
		assert.ok(before, "no before_agent_start handler");
		await before({ messageType: "user", prompt: "Why pin the base branch?" }, runtime.ctx);
		const recorded = loadProgress(root);
		assert.equal(recorded.questions.length, 1);
		assert.equal(recorded.questions[0].chapter, 2);
		assert.match(recorded.questions[0].text, /base branch/);

		await before({ messageType: "user", prompt: `${PROMPT_SENTINEL} chapter 3/12` }, runtime.ctx);
		assert.equal(loadProgress(root).questions.length, 1, "the injected chapter prompt must not be recorded");

		await runtime.command.handler("ask and what about tags?", runtime.ctx);
		assert.equal(loadProgress(root).questions.length, 2);
		assert.ok(runtime.sent[2].text.includes("and what about tags?"));
		assert.ok(runtime.sent[2].text.includes(CURRICULUM[1].body));
	} finally {
		fs.rmSync(root, { recursive: true, force: true });
	}
});

test("wiring: a busy agent gets a follow-up, back re-shows, and reset clears the widget", async () => {
	const root = scratchRoot("vangja-understand-wire-");
	try {
		const runtime = fakeRuntime(root);
		await runtime.command.handler("start", runtime.ctx);
		await runtime.command.handler("next", runtime.ctx);
		await runtime.command.handler("back", runtime.ctx);
		assert.equal(loadProgress(root).current, 1);

		const busyRuntime = fakeRuntime(root, false);
		await busyRuntime.command.handler("next", busyRuntime.ctx);
		assert.deepEqual(busyRuntime.sent[0].options, { deliverAs: "followUp" });

		await runtime.command.handler("reset", runtime.ctx);
		assert.equal(loadProgress(root).status, "not-started");
		const cleared = runtime.ui.statuses[runtime.ui.statuses.length - 1];
		assert.deepEqual(cleared, ["understand-loop", undefined]);
		const clearedWidget = runtime.ui.widgets[runtime.ui.widgets.length - 1];
		assert.deepEqual(clearedWidget, ["understand-loop-progress", undefined]);
	} finally {
		fs.rmSync(root, { recursive: true, force: true });
	}
});
