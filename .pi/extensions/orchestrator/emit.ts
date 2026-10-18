/**
 * Command-result rendering for the `/loop:*` commands.
 *
 * Interactive pi sessions get a toast (+ a status widget for long output).
 * Print/JSON/RPC modes have no toast, so the same text is written to stdout —
 * without this, `pi --print "/loop:verify-env"` prints nothing and the
 * acceptance evidence cannot be captured.
 *
 * No pi imports: unit-testable with `node --test`.
 */

export interface EmitUi {
	notify(text: string, kind: "info" | "error"): void;
	setWidget?(id: string, lines: string[]): void;
}

export interface EmitContext {
	/** pi sets this to "tui" in interactive mode. */
	mode?: string;
	ui: EmitUi;
	/** Where non-TUI output goes; injectable for tests. */
	write?: (text: string) => void;
}

export function emitResult(
	ctx: EmitContext,
	text: string,
	lines?: string[],
	kind: "info" | "error" = "info",
): void {
	if (ctx.mode !== "tui") {
		const write = ctx.write ?? ((chunk: string) => void process.stdout.write(chunk));
		write(`${text}\n`);
		return;
	}
	if (lines && ctx.ui.setWidget) ctx.ui.setWidget("loop-status", lines);
	ctx.ui.notify(text, kind);
}
