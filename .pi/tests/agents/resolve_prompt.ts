/**
 * Offline prompt-template resolver (acceptance test 1, "or equivalent").
 *
 * Parses a `.pi/prompts/*.md` template and applies pi's documented
 * substitutions — `$1`, `$@`, `$ARGUMENTS`, `${1:-default}`, `${@:-default}`,
 * `${@:N}`, `${@:N:L}` — then prints the resolved text without calling a model.
 *
 * Usage: node .pi/tests/agents/resolve_prompt.ts .pi/prompts/research-next-task.md T-099
 */

import * as fs from "node:fs";
import * as path from "node:path";

export interface PromptTemplate {
	name: string;
	description: string;
	argumentHint: string;
	body: string;
}

export function parsePromptTemplate(content: string, file: string): PromptTemplate {
	const lines = content.split("\n");
	if (lines[0]?.trim() !== "---") {
		throw new Error(`${file}: template must start with a "---" frontmatter fence (pi prompt-template format)`);
	}
	let end = -1;
	for (let i = 1; i < lines.length; i += 1) {
		if (lines[i].trim() === "---") {
			end = i;
			break;
		}
	}
	if (end === -1) throw new Error(`${file}: unterminated frontmatter fence`);
	const frontmatter: Record<string, string> = {};
	for (const line of lines.slice(1, end)) {
		const match = /^([A-Za-z_-]+):\s*(.*)$/.exec(line);
		if (match) frontmatter[match[1]] = match[2].replace(/^["']|["']$/g, "");
	}
	const body = lines
		.slice(end + 1)
		.join("\n")
		.replace(/^\n+/, "")
		.replace(/\s+$/, "");
	if (body === "") throw new Error(`${file}: empty template body`);
	const name = path.basename(file, ".md");
	return {
		name,
		description: frontmatter.description ?? "",
		argumentHint: frontmatter["argument-hint"] ?? "",
		body,
	};
}

/** Apply pi's documented positional-argument substitutions. */
export function substituteArguments(body: string, args: string[]): string {
	const arg = (index: number, fallback = ""): string => args[index - 1] ?? fallback;
	let out = body;
	// ${@:N:L}, ${@:N}, ${@:-default}, ${1:-default} first (longest forms).
	out = out.replace(/\$\{@:(\d+):(\d+)\}/g, (_m, start: string, count: string) =>
		args.slice(Number(start) - 1, Number(start) - 1 + Number(count)).join(" "),
	);
	out = out.replace(/\$\{@:(\d+)\}/g, (_m, start: string) => args.slice(Number(start) - 1).join(" "));
	out = out.replace(/\$\{@:-([^}]*)\}/g, (_m, fallback: string) => (args.length > 0 ? args.join(" ") : fallback));
	out = out.replace(/\$\{(\d+):-([^}]*)\}/g, (_m, index: string, fallback: string) => arg(Number(index), fallback));
	out = out.replace(/\$ARGUMENTS/g, args.join(" "));
	out = out.replace(/\$@/g, args.join(" "));
	out = out.replace(/\$(\d+)/g, (_m, index: string) => arg(Number(index)));
	return out;
}

function main(): void {
	const [file, ...args] = process.argv.slice(2);
	if (!file) {
		process.stderr.write("usage: node .pi/tests/agents/resolve_prompt.ts <template.md> [args...]\n");
		process.exit(2);
	}
	const template = parsePromptTemplate(fs.readFileSync(file, "utf8"), file);
	process.stdout.write(`TEMPLATE: ${template.name}\nARGS: ${args.join(" ")}\n--- resolved ---\n`);
	process.stdout.write(`${substituteArguments(template.body, args)}\n`);
}

if (process.argv[1] && import.meta.url === `file://${path.resolve(process.argv[1])}`) main();
