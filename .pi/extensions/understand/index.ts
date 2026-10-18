/**
 * understand_loop — a guided, chapter-by-chapter tour of the Vangja research
 * loop, driven by `/understand_loop`.
 *
 * Design:
 * - one small chapter per delivery; the extension then stops and lets the
 *   reader ask questions in the normal chat;
 * - the next delivery is explicit (`/understand_loop next`), so the reader
 *   controls the pace;
 * - progress (delivered, understood, questions) is durable in
 *   `.pi/state/UNDERSTAND_LOOP.json` and visible in the TUI status/widget;
 * - the chapter text is injected into the prompt verbatim, so answers are
 *   grounded in versioned material instead of model memory.
 *
 * All pi-free logic lives in `./lib.ts` and `./curriculum.ts`.
 */

import type { ExtensionAPI, ExtensionCommandContext, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { resolveRepoRoot } from "../../lib/paths.ts";
import {
	beginTutorial,
	buildChapterPrompt,
	buildQuestionPrompt,
	formatProgressLine,
	formatStatusLines,
	formatStatusText,
	goBack,
	loadProgress,
	nextChapter,
	parseCommand,
	PROMPT_SENTINEL,
	recordQuestion,
	resetProgress,
	saveProgress,
	showChapter,
	usageText,
	type UnderstandCommand,
	type UnderstandingProgress,
} from "./lib.ts";

const STATUS_KEY = "understand-loop";
const WIDGET_KEY = "understand-loop-progress";

function rootFor(ctx: ExtensionContext): string {
	try {
		return resolveRepoRoot(process.env.PI_LOOP_ROOT ?? ctx.cwd);
	} catch {
		return ctx.cwd;
	}
}

export default function (pi: ExtensionAPI) {
	const paint = (ctx: ExtensionContext, progress: UnderstandingProgress, busy = false): void => {
		if (ctx.mode !== "tui") return;
		if (progress.status === "not-started") {
			ctx.ui.setStatus(STATUS_KEY, undefined);
			ctx.ui.setWidget(WIDGET_KEY, undefined);
			return;
		}
		const suffix = busy ? " · explaining…" : progress.status === "complete" ? " · complete" : "";
		ctx.ui.setStatus(STATUS_KEY, `UNDERSTAND ${formatProgressLine(progress)}${suffix}`);
		ctx.ui.setWidget(WIDGET_KEY, formatStatusLines(progress, busy), { placement: "belowEditor" });
	};

	const send = (ctx: ExtensionCommandContext, text: string): void => {
		if (ctx.isIdle()) pi.sendUserMessage(text);
		else pi.sendUserMessage(text, { deliverAs: "followUp" });
	};

	const deliverChapter = (ctx: ExtensionCommandContext, progress: UnderstandingProgress): void => {
		paint(ctx, progress, true);
		send(ctx, buildChapterPrompt(progress));
		ctx.ui.notify(
			`understand_loop ${formatProgressLine(progress)} — ask questions in chat; /understand_loop next when ready.`,
			"info",
		);
	};

	const finish = (ctx: ExtensionCommandContext, progress: UnderstandingProgress): void => {
		paint(ctx, progress);
		ctx.ui.notify(
			`understand_loop complete (${progress.completed.length} chapters). ` +
				`Questions recorded: ${progress.questions.length}. /understand_loop reset to run it again.`,
			"info",
		);
	};

	pi.registerCommand("understand_loop", {
		description: "Guided tour of the Vangja research loop, one small chapter at a time",
		handler: async (args: string, ctx: ExtensionCommandContext) => {
			const root = rootFor(ctx);
			let progress: UnderstandingProgress;
			let command: UnderstandCommand;
			try {
				progress = loadProgress(root);
				command = parseCommand(args);
			} catch (error) {
				ctx.ui.notify((error as Error).message, "error");
				return;
			}

			switch (command.action) {
				case "help":
					ctx.ui.notify(usageText(), "info");
					return;
				case "status":
					ctx.ui.notify(formatStatusText(progress), "info");
					paint(ctx, progress);
					return;
				case "reset":
					progress = saveProgress(root, resetProgress());
					paint(ctx, progress);
					ctx.ui.notify("understand_loop progress reset; /understand_loop start to begin again.", "info");
					return;
				case "ask": {
					if (progress.status === "not-started" || progress.current < 1) {
						ctx.ui.notify("Start the tour first with /understand_loop start.", "warning");
						return;
					}
					const question = command.question ?? "";
					progress = saveProgress(root, recordQuestion(progress, question));
					paint(ctx, progress, true);
					send(ctx, buildQuestionPrompt(progress, question));
					return;
				}
				case "back": {
					if (progress.status === "not-started") {
						ctx.ui.notify("The tour has not started; /understand_loop start.", "warning");
						return;
					}
					progress = saveProgress(root, goBack(progress));
					deliverChapter(ctx, progress);
					return;
				}
				case "show": {
					if (command.chapter === undefined) return;
					progress = saveProgress(root, showChapter(progress, command.chapter));
					deliverChapter(ctx, progress);
					return;
				}
				case "start": {
					// Re-show the current chapter when resuming; restart only when idle or complete.
					progress =
						progress.status === "awaiting"
							? saveProgress(root, showChapter(progress, progress.current))
							: saveProgress(root, beginTutorial(progress, 1));
					deliverChapter(ctx, progress);
					return;
				}
				case "next": {
					if (progress.status === "complete") {
						ctx.ui.notify(
							"The tour is complete. /understand_loop reset to run it again, or /understand_loop show <n> to revisit a chapter.",
							"info",
						);
						return;
					}
					progress =
						progress.status === "not-started"
							? saveProgress(root, beginTutorial(progress, 1))
							: saveProgress(root, nextChapter(progress));
					if (progress.status === "complete") {
						finish(ctx, progress);
						return;
					}
					deliverChapter(ctx, progress);
					return;
				}
			}
		},
	});

	pi.on("session_start", async (_event, ctx) => {
		try {
			paint(ctx, loadProgress(rootFor(ctx)));
		} catch {
			// A corrupt progress file must not block the session; the command reports it.
		}
	});

	pi.on("agent_end", async (_event, ctx) => {
		if (ctx.mode !== "tui") return;
		try {
			paint(ctx, loadProgress(rootFor(ctx)));
		} catch {
			// Same: the widget is best-effort.
		}
	});

	// While a chapter is awaiting questions, any real user prompt is a question
	// worth recording. Prompts this extension injected are recognised by sentinel.
	pi.on("before_agent_start", async (event, ctx) => {
		if (event.messageType !== "user") return;
		const prompt = typeof event.prompt === "string" ? event.prompt : "";
		if (prompt.trim() === "" || prompt.startsWith(PROMPT_SENTINEL)) return;
		try {
			const root = rootFor(ctx);
			const progress = loadProgress(root);
			if (progress.status !== "awaiting") return;
			paint(ctx, saveProgress(root, recordQuestion(progress, prompt)));
		} catch {
			// Recording a question is best-effort; never block the user's prompt.
		}
		return undefined;
	});

	pi.on("session_shutdown", async (_event, ctx) => {
		if (ctx.mode !== "tui") return;
		ctx.ui.setStatus(STATUS_KEY, undefined);
		ctx.ui.setWidget(WIDGET_KEY, undefined);
	});
}
