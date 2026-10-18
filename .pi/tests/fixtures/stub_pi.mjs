#!/usr/bin/env node
/**
 * Deterministic stand-in for a `pi --mode json` child, used by the offline
 * dispatch tests (`PI_DISPATCH_COMMAND`). It emits a minimal JSONL stream and
 * records argv/timing facts the tests assert on. It never talks to a model.
 *
 * STUB_PI_MODE=ok|big|fail|slow     (default: ok)
 * STUB_PI_SLEEP_MS=<ms>             (slow mode, default 3000)
 * STUB_PI_CONCURRENCY_FILE=<path>   append "<ts> start|end ..." lines
 * STUB_PI_TASK_FILE=<path>          append the final TASK argument per step
 */
import * as fs from "node:fs";
import * as path from "node:path";

const args = process.argv.slice(2);
const mode = process.env.STUB_PI_MODE ?? "ok";
const sleepMs = Number(process.env.STUB_PI_SLEEP_MS ?? "3000");

function flag(name) {
	const index = args.indexOf(name);
	return index >= 0 ? args[index + 1] : undefined;
}

const sessionDir = flag("--session-dir");
const sessionName = flag("--name") ?? "stub";
const task = args.length > 0 ? args[args.length - 1] : "";
const sessionId = `stub-${process.pid}-${Date.now()}`;

if (sessionDir) {
	fs.mkdirSync(sessionDir, { recursive: true });
	fs.writeFileSync(
		path.join(sessionDir, `${Date.now()}_${sessionId}.jsonl`),
		`${JSON.stringify({ type: "session", version: 3, id: sessionId, cwd: process.cwd() })}\n`,
	);
}
if (process.env.STUB_PI_TASK_FILE) {
	fs.appendFileSync(process.env.STUB_PI_TASK_FILE, `${sessionName}\t${task}\n`);
}

function record(phase) {
	if (!process.env.STUB_PI_CONCURRENCY_FILE) return;
	fs.appendFileSync(
		process.env.STUB_PI_CONCURRENCY_FILE,
		`${Date.now()} ${phase} ${process.pid} ${sessionName}\n`,
	);
}

function emit(event) {
	process.stdout.write(`${JSON.stringify(event)}\n`);
}

function assistant(text, usage, stopReason = "end", errorMessage) {
	emit({
		type: "message_end",
		message: {
			role: "assistant",
			model: "opencode-go/deepseek-v4-flash",
			stopReason,
			...(errorMessage ? { errorMessage } : {}),
			content: [{ type: "text", text }],
			usage,
		},
	});
}

const usage = { input: 10, output: 5, cacheRead: 1, cacheWrite: 2, totalTokens: 18, cost: { total: 0.001 } };

function sleep(ms) {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

record("start");
emit({ type: "agent_start" });
emit({ type: "message_start", message: { role: "user", content: task, timestamp: Date.now() } });
emit({ type: "message_end", message: { role: "user", content: task, timestamp: Date.now() } });

if (mode === "slow") {
	await sleep(sleepMs);
	assistant(`STUB SLOW ${sessionName} :: ${task}`, usage);
	record("end");
	process.exit(0);
}

if (mode === "fail") {
	emitFailure();
}
function emitFailure() {
	assistant("PARTIAL-LOOKING-TEXT", usage, "error", "stub failure after partial output");
	process.stderr.write("stub stderr detail: simulated provider failure\n");
	record("end");
	process.exit(3);
}
if (process.env.STUB_PI_FAIL_TASK_CONTAINS && task.includes(process.env.STUB_PI_FAIL_TASK_CONTAINS)) {
	emitFailure();
}

if (mode === "big") {
	assistant("x".repeat(60 * 1024), usage);
	record("end");
	process.exit(0);
}

assistant(`STUB OK ${sessionName} :: ${task}`, usage);
record("end");
process.exit(0);
