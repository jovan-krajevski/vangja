/**
 * status — TUI status line + widget, and `renderResult` for the ledger tools so
 * every state write is visible in the transcript.
 *
 * Non-TUI modes are unaffected: `composeStatusLines` is also what `status.sh`
 * prints, so the headless and TUI views cannot diverge.
 */

import { Text } from "@earendil-works/pi-tui";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { resolveRepoRoot } from "../../lib/paths.ts";
import { composeStatusLine, composeStatusLines, firstLine, readStatus } from "./status.ts";

const LEDGER_TOOL_PREFIXES = [
	"ledger_",
	"task_",
	"pr_open",
	"review_write",
	"experiment_",
	"decision_record",
	"waiver_record",
	"amendment_record",
	"state_get",
	"state_set",
];

function isLedgerTool(name: string): boolean {
	return LEDGER_TOOL_PREFIXES.some((prefix) => name === prefix || name.startsWith(prefix));
}

export default function (pi: ExtensionAPI) {
	const rootFor = (ctx: ExtensionContext): string => {
		try {
			return resolveRepoRoot(process.env.PI_LOOP_ROOT ?? ctx.cwd);
		} catch {
			return ctx.cwd;
		}
	};

	const refresh = (ctx: ExtensionContext): void => {
		if (ctx.mode !== "tui") return;
		try {
			const snapshot = readStatus(rootFor(ctx));
			ctx.ui.setStatus("loop", composeStatusLine(snapshot));
			ctx.ui.setWidget("loop-status", composeStatusLines(snapshot), { placement: "belowEditor" });
		} catch (error) {
			ctx.ui.setStatus("loop", `LOOP status error: ${(error as Error).message}`);
			ctx.ui.setWidget("loop-status", [`LOOP status error: ${(error as Error).message}`], {
				placement: "belowEditor",
			});
		}
	};

	pi.on("session_start", async (_event, ctx) => refresh(ctx));
	pi.on("agent_end", async (_event, ctx) => refresh(ctx));
	pi.on("agent_settled", async (_event, ctx) => refresh(ctx));
	pi.on("tool_execution_end", async (event, ctx) => {
		if (isLedgerTool(event.toolName)) refresh(ctx);
	});
	pi.on("session_shutdown", async (_event, ctx) => {
		if (ctx.mode !== "tui") return;
		ctx.ui.setStatus("loop", undefined);
		ctx.ui.setWidget("loop-status", undefined);
	});

	// Make ledger writes visible in the transcript even when the tool's own
	// output is long: show the first content line (or the error) and a delta hint.
	pi.registerToolRenderer((toolName, next) => {
		if (!isLedgerTool(toolName)) return next();
		return {
			renderResult(result, { expanded }, theme) {
				const details = result.details as { error?: string } | undefined;
				const text = result.content
					.map((part) => (part.type === "text" ? part.text : ""))
					.join("\n")
					.trim();
				if (details?.error) {
					return new Text(theme.fg("error", `✗ ${firstLine(details.error)}`), 0, 0);
				}
				const lines = text.split("\n");
				const head = theme.fg("success", "✓ ") + theme.fg("muted", firstLine(text));
				if (expanded || lines.length <= 1) return new Text(head, 0, 0);
				return new Text(`${head}${theme.fg("dim", ` … +${lines.length - 1} line(s)`)}`, 0, 0);
			},
		};
	});
}
