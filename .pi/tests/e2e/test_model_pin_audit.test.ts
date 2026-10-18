/**
 * Model-pin audit tests (S7 acceptance test 4).
 *
 * The audit is the recorded-history half of `plan/02_ARCHITECTURE.md` §7: the
 * guard blocks drift at spawn time, the audit proves the ledgers agree.
 */

import { strict as assert } from "node:assert";
import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { REPO_ROOT } from "../helpers.ts";
import { auditArgv, auditFiles, renderPinAudit } from "./audit_model_pin.ts";

const SCRIPT = path.join(path.dirname(fileURLToPath(import.meta.url)), "audit_model_pin.ts");
const PIN = { provider: "opencode-go", model: "deepseek-v4.1-flash", thinking: "max" };

function writeEvents(lines: Array<Record<string, unknown>>): string {
	const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vangja-pin-audit-"));
	const file = path.join(dir, "EVENTS.jsonl");
	fs.writeFileSync(file, `${lines.map((line) => JSON.stringify(line)).join("\n")}\n`);
	return file;
}

test("a pinned argv passes and a drifted argv is reported with its reason", () => {
	const file = writeEvents([
		{ seq: 1, actor: "conductor", argv: ["pi", "--provider", "opencode-go", "--model", "opencode-go/deepseek-v4.1-flash", "--thinking", "max", "TASK: x"] },
		{ seq: 2, actor: "conductor", argv: ["pi", "--model", "opencode-go/gpt-5.6-luna", "--thinking", "max", "TASK: x"] },
		{ seq: 3, actor: "conductor", argv: ["pi", "--model", "opencode-go/deepseek-v4.1-flash", "--thinking", "high", "TASK: x"] },
		{ seq: 4, actor: "conductor", argv: ["pi", "--model", "opencode-go/deepseek-v4.1-flash", "TASK: x"] },
		{ seq: 5, actor: "conductor", summary: "no argv on this event" },
	]);
	const violations = auditArgv(REPO_ROOT, file, PIN);
	assert.equal(violations.length, 3);
	assert.match(violations[0].reason, /--model is "opencode-go\/gpt-5\.6-luna"/);
	assert.match(violations[1].reason, /--thinking is "high"/);
	assert.match(violations[2].reason, /no --thinking flag/);
});

test("a missing events file is vacuously clean (a fresh clone records nothing)", () => {
	const dir = fs.mkdtempSync(path.join(os.tmpdir(), "vangja-pin-audit-"));
	const result = auditFiles(REPO_ROOT, [path.join(dir, "EVENTS.jsonl")]);
	assert.equal(result.invocations, 0);
	assert.equal(result.violations.length, 0);
	const rendered = renderPinAudit(REPO_ROOT, result).join("\n");
	assert.match(rendered, /file absent — nothing recorded yet/);
	assert.match(rendered, /0 violation\(s\)/);
});

test("the CLI exits 1 on a violation and 0 on a pinned file", () => {
	const bad = writeEvents([{ seq: 1, actor: "conductor", argv: ["pi", "--model", "other/model", "--thinking", "low"] }]);
	const failed = spawnSync("node", [SCRIPT, bad], { cwd: REPO_ROOT, encoding: "utf8" });
	assert.equal(failed.status, 1);
	assert.match(failed.stdout, /VIOLATION/);

	const good = writeEvents([{ seq: 1, actor: "conductor", argv: ["pi", "--provider", "opencode-go", "--model", "opencode-go/deepseek-v4.1-flash", "--thinking", "max"] }]);
	const passed = spawnSync("node", [SCRIPT, good], { cwd: REPO_ROOT, encoding: "utf8" });
	assert.equal(passed.status, 0, passed.stderr);
	assert.match(passed.stdout, /1\/1 recorded invocation\(s\) use the pin/);
});

test("the repository's own EVENTS.jsonl has no model-pin violation", () => {
	const result = auditFiles(REPO_ROOT, [path.join(REPO_ROOT, ".pi", "state", "EVENTS.jsonl")]);
	assert.equal(result.violations.length, 0, result.violations.map((violation) => violation.reason).join("; "));
});
