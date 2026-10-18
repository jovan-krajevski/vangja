#!/usr/bin/env node
/**
 * Deterministic stand-in for a conductor-spawned `pi` child, used by the
 * orchestrator tests through `PI_LOOP_COMMAND`. It accepts the same argv shape
 * (`--mode json -p -na --provider … --session-dir … --name … "TASK: …"`),
 * ignores the model flags, and performs the ledger/git side effects the real
 * child is expected to perform — so the orchestrator's spawn path, JSON stream
 * parsing, budget accounting, state transitions, gates, and merge are all
 * exercised without a model call.
 *
 * PI_STUB_MODE=work    create the task worktree, commit a file, open a PR record
 * PI_STUB_MODE=review  append an accepted review for the open PR
 * PI_STUB_MODE=fail    exit 3 after a partial message (failure accounting)
 * PI_STUB_MODE=none    do nothing (a completed child that changed nothing)
 * PI_STUB_MODE=slow    sleep PI_STUB_SLEEP_MS then succeed (liveness/lock tests)
 *
 * The stub writes through the *sandbox's own* ledger module, so record
 * validation is the real one.
 */

import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const root = process.env.PI_LOOP_ROOT ?? process.cwd();
const mode = process.env.PI_STUB_MODE ?? "none";
const task = args[args.length - 1] ?? "";
const sessionDir = flag("--session-dir");
const sessionName = flag("--name") ?? "stub";

function flag(name) {
	const index = args.indexOf(name);
	return index >= 0 ? args[index + 1] : undefined;
}

function git(gitArgs, cwd = root) {
	return execFileSync("git", ["-C", cwd, ...gitArgs], { encoding: "utf8" }).trim();
}

async function ledger() {
	const modulePath = path.join(root, ".pi", "extensions", "ledger", "store.ts");
	return import(pathToFileURL(modulePath).href);
}

function emit(event) {
	process.stdout.write(`${JSON.stringify(event)}\n`);
}

function usage() {
	return { input: 10, output: 5, cacheRead: 1, cacheWrite: 2, totalTokens: 18, cost: { total: 0.002 } };
}

function assistant(text, stopReason = "end") {
	emit({
		type: "message_end",
		message: {
			role: "assistant",
			model: "opencode-go/deepseek-v4.1-flash",
			stopReason,
			content: [{ type: "text", text }],
			usage: usage(),
		},
	});
}

const sessionId = `stub-orch-${process.pid}-${Date.now()}`;
if (sessionDir) {
	fs.mkdirSync(sessionDir, { recursive: true });
	fs.writeFileSync(
		path.join(sessionDir, `${Date.now()}_${sessionId}.jsonl`),
		`${JSON.stringify({ type: "session", version: 3, id: sessionId, cwd: root })}\n`,
	);
}
emit({ type: "agent_start" });
emit({ type: "message_start", message: { role: "user", content: task, timestamp: Date.now() } });
emit({ type: "message_end", message: { role: "user", content: task, timestamp: Date.now() } });
emit({ type: "tool_execution_start", toolCallId: "call_1", toolName: "bash", args: { command: "stub" } });
emit({ type: "tool_execution_end", toolCallId: "call_1", toolName: "bash", result: { content: [] }, isError: false, durationMs: 1 });

async function doWork() {
	const project = JSON.parse(fs.readFileSync(path.join(root, ".pi", "config", "project.json"), "utf8"));
	const tasks = fs
		.readFileSync(path.join(root, ".pi", "state", "TASKS.jsonl"), "utf8")
		.split("\n")
		.filter((line) => line.trim() !== "")
		.map((line) => JSON.parse(line));
	const latestTask = tasks.filter((record) => record.id).pop();
	const taskId = latestTask?.id ?? "T-001";
	const slug = "stub";
	const worktree = path.join(root, ".pi", "state", "worktrees", taskId);
	const branch = `work/${taskId}-${slug}`;
	fs.mkdirSync(path.dirname(worktree), { recursive: true });
	if (!branchExists(branch)) {
		git(["worktree", "add", "-b", branch, worktree, project.baseBranch]);
	}
	fs.writeFileSync(path.join(worktree, "STUB_TASK.md"), `# ${taskId}\n\nImplemented by the orchestrator test stub.\n`);
	git(["add", "-A"], worktree);
	git(["-c", "user.name=stub", "-c", "user.email=stub@example.invalid", "commit", "-m", `${taskId}: stub implementation`], worktree);
	const head = git(["rev-parse", "HEAD"], worktree);
	const base = git(["rev-parse", project.baseBranch]);
	const { appendRecord } = await ledger();
	appendRecord(
		root,
		"pr",
		{
			task_id: taskId,
			head_sha: head,
			base_sha: base,
			branch,
			summary: `stub PR for ${taskId}`,
			artifacts: [],
			tests: [],
			claims: [],
			hpc_jobs: [],
			status: "open",
		},
		{ root, role: "phd_candidate", actor: sessionName },
	);
	return `opened PR for ${taskId} at ${head.slice(0, 12)}`;
}

function branchExists(branch) {
	try {
		git(["rev-parse", "--verify", `refs/heads/${branch}`]);
		return true;
	} catch {
		return false;
	}
}

async function doReview() {
	const prs = fs
		.readFileSync(path.join(root, ".pi", "state", "PRS.jsonl"), "utf8")
		.split("\n")
		.filter((line) => line.trim() !== "")
		.map((line) => JSON.parse(line));
	const pr = prs.filter((record) => record.id).pop();
	const { appendRecord } = await ledger();
	appendRecord(
		root,
		"review",
		{
			pr_id: pr.id,
			decision: "accept",
			rubric_scores: { tests_pass: 1, protected_paths: 1 },
			evidence: ["stub supervisor re-ran the acceptance criteria in the sandbox"],
			required_changes: [],
			residual_risk: "stub review: no independent verification",
		},
		{ root, role: "supervisor", actor: sessionName },
	);
	return `accepted ${pr.id}`;
}

if (mode === "slow") {
	await new Promise((resolve) => setTimeout(resolve, Number(process.env.PI_STUB_SLEEP_MS ?? "5000")));
	assistant(`STUB SLOW ${sessionName}`);
	process.exit(0);
}
if (mode === "fail") {
	assistant("STUB PARTIAL", "error");
	process.stderr.write("stub child failed on purpose\n");
	process.exit(3);
}
if (mode === "work") {
	const summary = await doWork();
	assistant(`STUB WORK OK ${summary}`);
	process.exit(0);
}
if (mode === "review") {
	const summary = await doReview();
	assistant(`STUB REVIEW OK ${summary}`);
	process.exit(0);
}
assistant(`STUB NONE ${sessionName}`);
process.exit(0);
