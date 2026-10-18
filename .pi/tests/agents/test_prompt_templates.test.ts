/**
 * Prompt templates: every `.pi/prompts/*.md` parses as a pi template, stays
 * under 30 lines, names the agent it drives, states its acceptance evidence,
 * and the argument substitutions behave like pi's documented syntax.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { loadAgentsManifest } from "../../lib/config.ts";
import { promptsDir } from "../../lib/paths.ts";
import { parsePromptTemplate, substituteArguments } from "./resolve_prompt.ts";
import { REPO_ROOT } from "../helpers.ts";

const EXPECTED = [
	"research-next-task",
	"research-implement",
	"research-review",
	"research-queue-experiment",
	"research-collect",
	"research-write-paper",
	"audit-cycle",
	"audit-workstream",
	"audit-verify",
	"audit-synthesize",
	"loop-status",
];

const DIR = promptsDir(REPO_ROOT);
const FILES = fs
	.readdirSync(DIR)
	.filter((file) => file.endsWith(".md"))
	.sort();
const agentNames = new Set(loadAgentsManifest(REPO_ROOT).agents.map((agent) => agent.name));

test("the 11 templates exist and parse", () => {
	assert.deepEqual(
		FILES.map((file) => path.basename(file, ".md")),
		[...EXPECTED].sort(),
	);
	for (const file of FILES) {
		const template = parsePromptTemplate(fs.readFileSync(path.join(DIR, file), "utf8"), file);
		assert.ok(template.description.length >= 10, `${file} needs a routing description`);
	}
});

test("templates stay under 30 lines", () => {
	for (const file of FILES) {
		const lines = fs.readFileSync(path.join(DIR, file), "utf8").split("\n").length;
		assert.ok(lines <= 30, `${file} is ${lines} lines (cap 30)`);
	}
});

test("templates name an agent (or are the status view) and their acceptance evidence", () => {
	for (const file of FILES) {
		const name = path.basename(file, ".md");
		const body = parsePromptTemplate(fs.readFileSync(path.join(DIR, file), "utf8"), file).body;
		if (name !== "loop-status") {
			const named = [...agentNames].some((agent) => body.includes(agent));
			assert.ok(named, `${file} must name the agent it drives`);
		}
		assert.match(body, /Acceptance evidence/i, `${file} must state the acceptance evidence`);
	}
});

test("argument substitution follows pi's documented syntax", () => {
	assert.equal(substituteArguments("run ${1:-<T-###>}", []), "run <T-###>");
	assert.equal(substituteArguments("run ${1:-<T-###>}", ["T-012"]), "run T-012");
	assert.equal(substituteArguments("a $1 b $2", ["x", "y"]), "a x b y");
	assert.equal(substituteArguments("all: $@", ["x", "y"]), "all: x y");
	assert.equal(substituteArguments("all: $ARGUMENTS", ["x", "y"]), "all: x y");
	assert.equal(substituteArguments("from 2: ${@:2}", ["a", "b", "c"]), "from 2: b c");
	assert.equal(substituteArguments("one: ${@:2:1}", ["a", "b", "c"]), "one: b");
	assert.equal(substituteArguments("fallback: ${@:-none}", []), "fallback: none");
});
