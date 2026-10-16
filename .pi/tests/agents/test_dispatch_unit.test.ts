/**
 * Dispatch engine, offline: the real spawn path against a stub `pi` that
 * emits canned JSONL. Proves argv pinning, session isolation, usage + ledger
 * accounting, output capping, failure surfacing, abort handling, the ≤4
 * concurrency cap, and chain `{previous}` substitution without any model call.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { test } from "node:test";
import { readJsonlFile } from "../../lib/io.ts";
import { loadAgentDefinitions, DISPATCH_CONCURRENCY } from "../../lib/agents.ts";
import {
	DEFAULT_OUTPUT_CAP_BYTES,
	buildChildInvocation,
	isFailedOutcome,
	mapWithConcurrencyLimit,
	runDispatchAgent,
	runDispatchChain,
	visibleText,
	type DispatchRequest,
} from "../../extensions/dispatch/lib.ts";
import { PI_DIR } from "../helpers.ts";

const STUB = path.join(PI_DIR, "tests", "fixtures", "stub_pi.mjs");

/** Scratch repo with the real configs/agents/tools and empty skill dirs. */
function fixtureRoot(prefix = "vangja-dispatch-"): string {
	const root = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
	fs.mkdirSync(path.join(root, ".pi", "config"), { recursive: true });
	fs.mkdirSync(path.join(root, ".pi", "state"), { recursive: true });
	fs.writeFileSync(path.join(root, "AGENTS.md"), "# dispatch fixture\n");
	for (const file of ["model.json", "agents.json", "tools.json", "budgets.json", "project.json"]) {
		fs.copyFileSync(path.join(PI_DIR, "config", file), path.join(root, ".pi", "config", file));
	}
	fs.cpSync(path.join(PI_DIR, "agents"), path.join(root, ".pi", "agents"), { recursive: true });
	const manifest = JSON.parse(fs.readFileSync(path.join(PI_DIR, "config", "agents.json"), "utf8")) as {
		agents: Array<{ skills: string[] }>;
	};
	for (const agent of manifest.agents) {
		for (const skill of agent.skills) fs.mkdirSync(path.join(root, ".pi", "skills", skill), { recursive: true });
	}
	return root;
}

function eventsOf(root: string): Array<Record<string, unknown>> {
	return readJsonlFile<Record<string, unknown>>(path.join(root, ".pi", "state", "EVENTS.jsonl"));
}

function stubOptions(root: string, extraEnv: NodeJS.ProcessEnv = {}, overrides: Record<string, unknown> = {}) {
	return {
		root,
		parentRole: "phd_candidate" as const,
		parentAgent: "phd_candidate__T-001__1",
		runId: "run-test",
		command: STUB,
		extraEnv: { STUB_PI_MODE: "ok", ...extraEnv },
		...overrides,
	};
}

test("buildChildInvocation pins the model, tools, session dir, and name", () => {
	const root = fixtureRoot();
	const { invocation, role, agentFile, agentFileHash } = buildChildInvocation(
		{ agent: "scout", task: "map .pi/skills", taskId: "T-001", attempt: 2 },
		{ root, parentRole: "phd_candidate", runId: "run-test" },
	);
	const args = invocation.args;
	assert.equal(role, "scout");
	assert.equal(args[args.indexOf("--mode") + 1], "json");
	assert.equal(args[args.indexOf("--provider") + 1], "opencode-go");
	assert.equal(args[args.indexOf("--model") + 1], "opencode-go/deepseek-v4.1-flash");
	assert.equal(args[args.indexOf("--thinking") + 1], "max");
	assert.deepEqual(args[args.indexOf("--tools") + 1].split(","), ["read", "grep", "find", "ls"]);
	assert.ok(args.includes("-na"), "children must not load trust-gated project resources implicitly");
	assert.ok(!args.includes("--continue") && !args.includes("--resume") && !args.includes("--session"));
	assert.equal(args[args.indexOf("--name") + 1], "scout__T-001__2");
	assert.equal(args[args.length - 1], "TASK: map .pi/skills");
	assert.equal(
		args[args.indexOf("--session-dir") + 1],
		path.join(root, ".pi", "state", "sessions", "run-test", "scout", "T-001", "a2"),
	);
	for (const skill of ["vangja-codebase", "research-memory-and-handoff"]) {
		assert.ok(args.includes(path.join(root, ".pi", "skills", skill)), `--skill ${skill}`);
	}
	assert.equal(invocation.env.PI_LOOP_ROLE, "scout");
	assert.equal(invocation.env.PI_AGENT_NAME, "scout__T-001__2");
	assert.equal(invocation.env.PI_LOOP_ROOT, root);
	assert.equal(invocation.env.PI_DISPATCH_DEPTH, "1");
	assert.equal(agentFile, ".pi/agents/scout.md");
	const definition = loadAgentDefinitions(root).find((agent) => agent.name === "scout");
	assert.equal(agentFileHash, definition?.sha256);
	const systemPrompt = fs.readFileSync(invocation.systemPromptFile, "utf8");
	assert.ok(systemPrompt.startsWith("# scout"));
	assert.ok(!systemPrompt.includes("tools: read"), "frontmatter must not leak into the system prompt");
	invocation.cleanup();
	assert.ok(!fs.existsSync(invocation.systemPromptFile), "temp system prompt cleaned up");
});

test("a successful run returns text + usage and writes exactly one ledger event", async () => {
	const root = fixtureRoot();
	const outcome = await runDispatchAgent(
		{ agent: "scout", task: "map skills", taskId: "T-001" },
		stubOptions(root),
	);
	assert.equal(outcome.exit_state, "ok");
	assert.match(outcome.final_text, /STUB OK scout__T-001__1/);
	assert.deepEqual(outcome.usage, {
		input: 10,
		output: 5,
		cacheRead: 1,
		cacheWrite: 2,
		cost: 0.001,
		contextTokens: 18,
		turns: 1,
	});
	assert.ok(outcome.session_file && fs.existsSync(outcome.session_file));
	assert.ok(outcome.session_file.startsWith(path.join(root, ".pi", "state", "sessions")));

	const events = eventsOf(root);
	assert.equal(events.length, 1, "one dispatch event per child");
	const event = events[0];
	assert.equal(event.agent, "scout");
	assert.equal(event.exit_state, "ok");
	assert.equal(event.dispatch_mode, "agent");
	assert.equal(event.agent_file, ".pi/agents/scout.md");
	assert.equal(event.agent_file_hash, outcome.agent_file_hash);
	assert.equal(event.session_path, outcome.session_file);
	assert.deepEqual(event.usage, outcome.usage);
	assert.ok(Array.isArray(event.argv) && (event.argv as string[]).includes("--mode"));
	assert.equal(event.task_id, "T-001");
});

test("output is capped at 50 KB and points at the persisted session", async () => {
	const root = fixtureRoot();
	const outcome = await runDispatchAgent(
		{ agent: "scout", task: "big output", taskId: "T-003" },
		stubOptions(root, { STUB_PI_MODE: "big" }),
	);
	assert.equal(outcome.truncated, true);
	assert.ok(outcome.output_bytes > DEFAULT_OUTPUT_CAP_BYTES);
	const text = visibleText(outcome);
	assert.ok(Buffer.byteLength(text, "utf8") < DEFAULT_OUTPUT_CAP_BYTES + 1000);
	assert.match(text, /output truncated/);
	assert.ok(text.includes(outcome.session_file ?? outcome.session_dir));
});

test("a failed child surfaces stderr + stopReason and its partial text is not the result", async () => {
	const root = fixtureRoot();
	const outcome = await runDispatchAgent(
		{ agent: "scout", task: "fail", taskId: "T-004" },
		stubOptions(root, { STUB_PI_MODE: "fail" }),
	);
	assert.equal(outcome.exit_state, "failed");
	assert.equal(outcome.stop_reason, "error");
	assert.match(outcome.error_message ?? "", /stub failure after partial output/);
	assert.match(outcome.stderr, /stub stderr detail/);
	assert.equal(outcome.final_text, "PARTIAL-LOOKING-TEXT");
	assert.equal(isFailedOutcome(outcome), true);
	const events = eventsOf(root);
	assert.equal(events.length, 1);
	assert.equal(events[0].exit_state, "failed");
});

test("killing a child mid-run yields aborted, never a final result", async () => {
	const root = fixtureRoot();
	const controller = new AbortController();
	const pending = runDispatchAgent(
		{ agent: "scout", task: "slow", taskId: "T-005" },
		stubOptions(root, { STUB_PI_MODE: "slow", STUB_PI_SLEEP_MS: "10000" }, { signal: controller.signal }),
	);
	setTimeout(() => controller.abort(), 250);
	const outcome = await pending;
	assert.equal(outcome.exit_state, "aborted");
	assert.equal(outcome.final_text, "");
	const events = eventsOf(root);
	assert.equal(events.length, 1);
	assert.equal(events[0].exit_state, "aborted");
});

test("parallel runs respect the ≤4 concurrency cap", async () => {
	const root = fixtureRoot();
	const concurrencyFile = path.join(root, "concurrency.txt");
	const tasks: DispatchRequest[] = Array.from({ length: 6 }, (_, index) => ({
		agent: "scout",
		task: `task ${index}`,
		taskId: `T-10${index}`,
	}));
	const outcomes = await mapWithConcurrencyLimit(tasks, DISPATCH_CONCURRENCY, (task) =>
		runDispatchAgent(
			task,
			stubOptions(root, {
				STUB_PI_MODE: "slow",
				STUB_PI_SLEEP_MS: "250",
				STUB_PI_CONCURRENCY_FILE: concurrencyFile,
			}),
		),
	);
	assert.equal(outcomes.length, 6);
	assert.ok(outcomes.every((outcome) => outcome.exit_state === "ok"));
	assert.equal(DISPATCH_CONCURRENCY, 4);

	const intervals: Array<{ start: number; end: number }> = [];
	const open = new Map<string, number>();
	for (const line of fs.readFileSync(concurrencyFile, "utf8").trim().split("\n")) {
		const [ts, phase, pid] = line.split(" ");
		if (phase === "start") open.set(pid, Number(ts));
		if (phase === "end" && open.has(pid)) {
			intervals.push({ start: open.get(pid) as number, end: Number(ts) });
			open.delete(pid);
		}
	}
	assert.equal(intervals.length, 6);
	const points = intervals.flatMap((interval) => [
		{ at: interval.start, delta: 1 },
		{ at: interval.end, delta: -1 },
	]);
	points.sort((a, b) => a.at - b.at || a.delta - b.delta);
	let running = 0;
	let peak = 0;
	for (const point of points) {
		running += point.delta;
		peak = Math.max(peak, running);
	}
	assert.ok(peak <= DISPATCH_CONCURRENCY, `peak concurrency ${peak} exceeded ${DISPATCH_CONCURRENCY}`);
	assert.ok(peak > 1, "the pool should actually parallelise the stub runs");
	assert.equal(eventsOf(root).length, 6);
});

test("parallel runs use distinct session directories", async () => {
	const root = fixtureRoot();
	const [first, second] = await Promise.all([
		runDispatchAgent({ agent: "scout", task: "a", taskId: "T-201" }, stubOptions(root)),
		runDispatchAgent({ agent: "scout", task: "b", taskId: "T-202" }, stubOptions(root)),
	]);
	assert.notEqual(first.session_dir, second.session_dir);
	assert.equal(path.dirname(first.session_file ?? ""), first.session_dir);
	assert.equal(path.dirname(second.session_file ?? ""), second.session_dir);
});

test("chain substitutes {previous} and stops at the first failure", async () => {
	const root = fixtureRoot();
	const taskFile = path.join(root, "tasks.txt");
	const { outcomes, failedAt } = await runDispatchChain(
		[
			{ agent: "scout", task: "step one", taskId: "T-300" },
			{ agent: "planner", task: "use {previous} now", taskId: "T-300" },
		],
		stubOptions(root, { STUB_PI_TASK_FILE: taskFile }),
	);
	assert.equal(failedAt, undefined);
	assert.equal(outcomes.length, 2);
	assert.match(outcomes[1].final_text, /use STUB OK scout__T-300__1 :: TASK: step one now/);
	const tasks = fs.readFileSync(taskFile, "utf8").trim().split("\n");
	assert.equal(tasks.length, 2);
	assert.match(tasks[1], /use STUB OK scout__T-300__1 :: TASK: step one now/);
	const events = eventsOf(root);
	assert.equal(events.filter((event) => event.dispatch_mode === "chain").length, 2);
	assert.deepEqual(
		events.map((event) => event.dispatch_step),
		[1, 2],
	);

	const broken = await runDispatchChain(
		[
			{ agent: "scout", task: "ok", taskId: "T-301" },
			{ agent: "planner", task: "fail", taskId: "T-301" },
			{ agent: "critic", task: "never", taskId: "T-301" },
		],
		stubOptions(root, { STUB_PI_TASK_FILE: path.join(root, "tasks2.txt"), STUB_PI_FAIL_TASK_CONTAINS: "TASK: fail" }),
	);
	assert.equal(broken.failedAt, 2);
	assert.equal(broken.outcomes.length, 2);
});

test("unknown agents and forbidden dispatches fail before any spawn", () => {
	const root = fixtureRoot();
	assert.throws(
		() => buildChildInvocation({ agent: "no_such_agent", task: "x" }, { root, parentRole: "phd_candidate" }),
		/Available agents: .*scout/,
	);
	assert.throws(
		() => buildChildInvocation({ agent: "scout", task: "x" }, { root, parentRole: "collector" }),
		/may not dispatch/,
	);
	assert.throws(
		() => buildChildInvocation({ agent: "audit_finder", task: "x" }, { root, parentRole: "supervisor" }),
		/may not dispatch/,
	);
});

test("dispatch depth is capped", () => {
	const root = fixtureRoot();
	const previous = process.env.PI_DISPATCH_DEPTH;
	process.env.PI_DISPATCH_DEPTH = "2";
	try {
		assert.throws(
			() => buildChildInvocation({ agent: "scout", task: "x" }, { root, parentRole: "phd_candidate" }),
			/dispatch depth 2 reached the cap/,
		);
	} finally {
		if (previous === undefined) delete process.env.PI_DISPATCH_DEPTH;
		else process.env.PI_DISPATCH_DEPTH = previous;
	}
});

test("a failed dispatch still records exactly one event (aborted or failed)", async () => {
	const root = fixtureRoot();
	await runDispatchAgent({ agent: "scout", task: "fail", taskId: "T-400" }, stubOptions(root, { STUB_PI_MODE: "fail" }));
	await runDispatchAgent(
		{ agent: "scout", task: "ok", taskId: "T-401" },
		stubOptions(root),
	);
	const events = eventsOf(root);
	assert.equal(events.length, 2);
	assert.deepEqual(
		events.map((event) => event.exit_state),
		["failed", "ok"],
	);
});
