#!/usr/bin/env node
/**
 * Deterministic child for the S7 end-to-end sandbox smoke test.
 *
 * It accepts the same argv shape the conductor produces (`--mode json -p -na
 * --provider … --model … --thinking max … "TASK: <brief>"`), ignores the model
 * flags, and performs the ledger/git/file side effects a real agent would
 * perform for the synthetic task — so the whole loop (BOOT → WORK → REVIEW →
 * MERGE → READINESS → PAPER → PAPER_REVIEW → MERGE → AUDIT → DECIDE) runs
 * against the *real* conductor, guards, gates, ledger, audit wiring and rubric
 * code, with no model call and no network.
 *
 * The mode is detected from the brief (the last argv) so the driver can run one
 * plain `step` loop; `PI_E2E_MODE` overrides it for the resilience matrix.
 *
 * Modes:
 *   select       refuse: the smoke test seeds the task so SELECT_TASK never spawns
 *   work         queue the experiment, then (once collected) open the work PR
 *   local        run the experiment locally with a checkpoint + resume smoke test
 *   review       append an accepted review for the open PR (work or paper)
 *   readiness    verify the artifact hash before writing READINESS.json
 *   paper        open a kind=paper PR touching paper/paper.tex
 *   audit        run the phase named in the brief (gates, find:WS, verify, …)
 *   decide       apply rubrics/decision.json via the real applyAuditDecision
 *   work-kill-early / work-then-kill  fault injection for the resilience matrix
 */

import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { pathToFileURL } from "node:url";

const argv = process.argv.slice(2);
const root = process.env.PI_LOOP_ROOT ?? process.cwd();
const brief = argv[argv.length - 1] ?? "";
const sessionName = flag("--name") ?? "e2e-child";
const sessionDir = flag("--session-dir");

function flag(name) {
	const index = argv.indexOf(name);
	return index >= 0 ? argv[index + 1] : undefined;
}

function detectMode(text) {
	if (/## WORK \(local fallback\)/.test(text)) return "local";
	if (/## WORK ·/.test(text)) return "work";
	if (/## REVIEW ·/.test(text)) return "review";
	if (/## READINESS/.test(text)) return "readiness";
	if (/## PAPER/.test(text)) return "paper";
	if (/## AUDIT · cycle/.test(text)) return "audit";
	if (/## DECIDE/.test(text)) return "decide";
	if (/## SELECT_TASK/.test(text)) return "select";
	if (/## WAITING_HPC/.test(text)) return "collector";
	return "unknown";
}

const mode = process.env.PI_E2E_MODE ?? detectMode(brief);

function git(gitArgs, cwd = root) {
	return execFileSync("git", ["-C", cwd, ...gitArgs], { encoding: "utf8" }).trim();
}

function readJsonl(rel) {
	const file = path.join(root, rel);
	if (!fs.existsSync(file)) return [];
	return fs
		.readFileSync(file, "utf8")
		.split("\n")
		.filter((line) => line.trim() !== "")
		.map((line) => JSON.parse(line));
}

function latest(records) {
	const byId = new Map();
	for (const record of records) if (record.id) byId.set(record.id, record);
	return [...byId.values()];
}

const url = (rel) => pathToFileURL(path.join(root, rel)).href;

async function ledger() {
	return import(url(".pi/extensions/ledger/store.ts"));
}

async function append(kind, record, role = "phd_candidate") {
	const { appendRecord } = await ledger();
	return appendRecord(root, kind, record, { root, role, actor: sessionName });
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

function boot() {
	const sessionId = `e2e-${process.pid}-${Date.now()}`;
	if (sessionDir) {
		fs.mkdirSync(sessionDir, { recursive: true });
		fs.writeFileSync(
			path.join(sessionDir, `${Date.now()}_${sessionId}.jsonl`),
			`${JSON.stringify({ type: "session", version: 3, id: sessionId, cwd: root })}\n`,
		);
	}
	emit({ type: "agent_start" });
	emit({ type: "message_start", message: { role: "user", content: brief, timestamp: Date.now() } });
	emit({ type: "message_end", message: { role: "user", content: brief, timestamp: Date.now() } });
	emit({ type: "tool_execution_start", toolCallId: "call_1", toolName: "bash", args: { command: `e2e ${mode}` } });
	emit({ type: "tool_execution_end", toolCallId: "call_1", toolName: "bash", result: { content: [] }, isError: false, durationMs: 1 });
}

function fail(message) {
	process.stderr.write(`e2e child (${mode}) FAILED: ${message}\n`);
	assistant(`E2E FAIL ${message}`, "error");
	process.exit(3);
}

// ---------------------------------------------------------------------------
// WORK
// ---------------------------------------------------------------------------

function currentTask() {
	const tasks = latest(readJsonl(".pi/state/TASKS.jsonl")).filter((task) => ["open", "in_progress"].includes(task.status));
	return tasks.sort((a, b) => String(a.id).localeCompare(String(b.id)))[0] ?? null;
}

function prForTask(taskId) {
	return latest(readJsonl(".pi/state/PRS.jsonl")).find(
		(pr) => pr.task_id === taskId && ["open", "review"].includes(pr.status),
	);
}

function unfinishedExperiment() {
	return latest(readJsonl(".pi/state/EXPERIMENTS.jsonl")).find(
		(experiment) => !["collected", "failed-final", "cancelled"].includes(experiment.status),
	);
}

function branchExists(branch) {
	try {
		git(["rev-parse", "--verify", `refs/heads/${branch}`]);
		return true;
	} catch {
		return false;
	}
}

async function queueExperiment(task) {
	const project = JSON.parse(fs.readFileSync(path.join(root, ".pi", "config", "project.json"), "utf8"));
	const artifactDir = ".pi/artifacts/E-0001";
	const record = await append("experiment", {
		task_id: task.id,
		branch: `work/${task.id}-fixture-docstring`,
		commit_sha: git(["rev-parse", project.baseBranch]),
		data_hash: "sha256:fixture-synthetic-data",
		command:
			"python3 experiments/tiny_run.py --outputs .pi/artifacts/E-0001/outputs --checkpoint-dir .pi/artifacts/E-0001/checkpoints",
		resources: { kind: "local", cpus: 2, memGb: 2, partitionMemGbLimit: 64 },
		checkpoints: {
			dir: `${artifactDir}/checkpoints`,
			progressFile: "progress.json",
			resumeCommand:
				"python3 experiments/tiny_run.py --outputs .pi/artifacts/E-0001/outputs --checkpoint-dir .pi/artifacts/E-0001/checkpoints --resume",
			flushOnSignal: true,
		},
		expectedOutputs: ["outputs/numbers.json", "outputs/metrics.csv"],
		seeds: [20261018],
		idempotencyKey: `${task.id}-tiny-run-20261018`,
		status: "queued",
		attempt: 1,
	});
	return record;
}

async function runLocal() {
	const experiment = unfinishedExperiment();
	if (!experiment) fail("no unfinished experiment to run locally");
	const artifactDir = path.join(root, ".pi", "artifacts", experiment.id);
	const outputs = path.join(artifactDir, "outputs");
	const checkpoints = path.join(artifactDir, "checkpoints");
	fs.mkdirSync(artifactDir, { recursive: true });
	const run = (resume) => {
		const args = ["experiments/tiny_run.py", "--outputs", outputs, "--checkpoint-dir", checkpoints];
		if (resume) args.push("--resume");
		return execFileSync("python3", args, { cwd: root, encoding: "utf8" }).trim();
	};
	const progressExists = () => fs.existsSync(path.join(checkpoints, "progress.json"));
	run(progressExists());
	// The checkpoint contract: a resumed run must not lose completed units.
	fs.rmSync(path.join(outputs, "metrics.csv"), { force: true });
	const resumed = JSON.parse(run(true));
	if (!Array.isArray(resumed.completed) || resumed.completed.length !== 3) {
		fail(`resume smoke test lost progress: ${JSON.stringify(resumed)}`);
	}

	const manifest = await import(url(".pi/extensions/hpc/manifest.ts"));
	const files = manifest.hashTree(artifactDir).filter((file) => file.path !== "manifest.json");
	const missing = manifest.missingOutputs(experiment.expectedOutputs, files);
	const written = manifest.writeManifest(root, {
		schema: "pi-artifact/1",
		experiment_id: experiment.id,
		task_id: experiment.task_id,
		attempt: experiment.attempt,
		status: "collected",
		idempotencyKey: experiment.idempotencyKey,
		commit_sha: experiment.commit_sha,
		job_id: null,
		collected_at: new Date().toISOString(),
		collected_by: sessionName,
		expectedOutputs: experiment.expectedOutputs,
		files,
		missing,
		checkpoints: manifest.hashTree(checkpoints),
		sacct: null,
		seeds: experiment.seeds,
	});
	await append("experiment", {
		...experiment,
		status: "collected",
		resources: { ...experiment.resources, kind: "local" },
		progressHistory: [{ attempt: experiment.attempt, completed: resumed.completed.length }],
		checkpointsFetched: true,
		collected: { manifest: path.relative(root, written.path), sha256: written.sha256 },
		evidence: [`local run exit 0; resume preserved ${resumed.completed.length}/3 checkpointed units`],
	});
	return `collected ${experiment.id} (${files.length} hashed file(s), missing=${missing.length})`;
}

async function openWorkPr(task, options = {}) {
	const project = JSON.parse(fs.readFileSync(path.join(root, ".pi", "config", "project.json"), "utf8"));
	const worktree = path.join(root, ".pi", "state", "worktrees", task.id);
	const branch = `work/${task.id}-fixture-docstring`;
	if (!branchExists(branch)) {
		fs.mkdirSync(path.dirname(worktree), { recursive: true });
		git(["worktree", "add", "-b", branch, worktree, project.baseBranch]);
	}
	// The task: add a docstring and a passing test to the fixture module.
	fs.writeFileSync(
		path.join(worktree, "synthetic_fixture", "calculator.py"),
		'"""Tiny module used by the S7 end-to-end sandbox smoke test."""\n\n\ndef add(left, right):\n    """Return the sum of ``left`` and ``right``."""\n    return left + right\n',
	);
	fs.mkdirSync(path.join(worktree, "synthetic_fixture", "tests"), { recursive: true });
	fs.writeFileSync(path.join(worktree, "synthetic_fixture", "tests", "__init__.py"), "");
	fs.writeFileSync(
		path.join(worktree, "synthetic_fixture", "tests", "test_calculator.py"),
		'"""Tests for the synthetic fixture module."""\n\nimport unittest\n\nfrom synthetic_fixture.calculator import add\n\n\nclass TestAdd(unittest.TestCase):\n    def test_adds_integers(self):\n        self.assertEqual(add(2, 3), 5)\n\n    def test_adds_negative_numbers(self):\n        self.assertEqual(add(-1, 1), 0)\n\n\nif __name__ == "__main__":\n    unittest.main()\n',
	);
	execFileSync("python3", ["-m", "unittest", "discover", "-s", "synthetic_fixture/tests", "-t", "."], {
		cwd: worktree,
		encoding: "utf8",
	});
	git(["add", "-A"], worktree);
	// A resumed attempt adopts the commit the killed attempt already made.
	if (git(["status", "--porcelain"], worktree) !== "") {
		git(["-c", "user.name=e2e", "-c", "user.email=e2e@example.invalid", "commit", "-m", `${task.id}: document and test the fixture module`], worktree);
	}
	const head = git(["rev-parse", "HEAD"], worktree);
	const base = git(["rev-parse", project.baseBranch]);
	if (options.killBeforeRecord) {
		// Fault injection: the effect (branch + commit) exists, the ledger record does not.
		process.kill(process.pid, "SIGKILL");
	}
	const experiment = latest(readJsonl(".pi/state/EXPERIMENTS.jsonl"))[0];
	const pr = await append("pr", {
		task_id: task.id,
		head_sha: head,
		base_sha: base,
		branch,
		pr_kind: "work",
		summary: `document and test the fixture module (${task.id})`,
		artifacts: [`.pi/artifacts/${experiment.id}`],
		tests: ["synthetic_fixture/tests/test_calculator.py"],
		claims: [],
		hpc_jobs: [experiment.id],
		protected_paths: [],
		status: "open",
	});
	return `opened ${pr.id} for ${task.id} at ${head.slice(0, 12)}`;
}

async function doWork() {
	const task = currentTask();
	if (!task) fail("no open task to work on");
	if (prForTask(task.id)) return `PR already open for ${task.id}`;
	const experiment = unfinishedExperiment();
	if (!experiment) {
		const existing = latest(readJsonl(".pi/state/EXPERIMENTS.jsonl")).find((record) => record.task_id === task.id);
		if (!existing) {
			const record = await queueExperiment(task);
			return `queued experiment ${record.id} for ${task.id}`;
		}
		return openWorkPr(task, { killBeforeRecord: process.env.PI_E2E_KILL === "before-record" });
	}
	return `experiment ${experiment.id} is ${experiment.status}; waiting for the local run`;
}

// ---------------------------------------------------------------------------
// REVIEW
// ---------------------------------------------------------------------------

async function doReview() {
	const pr = latest(readJsonl(".pi/state/PRS.jsonl")).find((record) => ["open", "review"].includes(record.status));
	if (!pr) fail("no open PR to review");
	const review = await append(
		"review",
		{
			pr_id: pr.id,
			decision: "accept",
			rubric_scores: { acceptance_criteria: 1, tests: 1, protected_paths: 1, provenance: 1 },
			evidence: [
				`re-ran python3 -m unittest discover -s synthetic_fixture/tests -t . at ${pr.head_sha.slice(0, 12)}: 2 tests OK`,
				`artifact .pi/artifacts/E-0001/manifest.json verifies (hash_ok=true)`,
			],
			required_changes: [],
			residual_risk: "e2e stub review: the checks are deterministic, not a model's judgement",
		},
		"supervisor",
	);
	return `accepted ${pr.id} as ${review.id}`;
}

// ---------------------------------------------------------------------------
// READINESS
// ---------------------------------------------------------------------------

async function doReadiness() {
	const experiment = latest(readJsonl(".pi/state/EXPERIMENTS.jsonl"))[0];
	if (!experiment) fail("no experiment record for readiness");
	const artifactDir = path.join(root, ".pi", "artifacts", experiment.id);
	const verify = await import(url(".pi/extensions/verify/lib.ts"));
	const check = verify.verifyArtifactAgainstExpected(root, artifactDir, experiment.expectedOutputs);
	const state = {
		pass: check.ok,
		reason: check.ok
			? `artifact ${experiment.id} verifies: ${check.detail}`
			: `artifact ${experiment.id} failed verification: ${check.detail}`,
		evidence: [check.detail, `manifest=${path.relative(root, check.manifest_path)}`],
	};
	fs.writeFileSync(path.join(root, ".pi", "state", "READINESS.json"), `${JSON.stringify(state, null, 2)}\n`);
	if (!check.ok) {
		// The readiness rubric must not pass with a tampered artifact. The loop
		// returns to SELECT_TASK; the smoke test asserts PAPER is never entered.
		return `readiness FAILED (${check.detail})`;
	}
	return `readiness passed (${check.detail})`;
}

// ---------------------------------------------------------------------------
// PAPER
// ---------------------------------------------------------------------------

async function doPaper() {
	const project = JSON.parse(fs.readFileSync(path.join(root, ".pi", "config", "project.json"), "utf8"));
	const existing = latest(readJsonl(".pi/state/PRS.jsonl")).find((pr) => pr.kind === "paper" && ["open", "review"].includes(pr.status));
	if (existing) return `paper PR ${existing.id} already open`;
	const worktree = path.join(root, ".pi", "state", "worktrees", "PAPER");
	const branch = "paper/paper-update";
	if (!branchExists(branch)) {
		fs.mkdirSync(path.dirname(worktree), { recursive: true });
		git(["worktree", "add", "-b", branch, worktree, project.baseBranch]);
	}
	// The smoke paper phase deliberately does not touch paper/**: the seeded audit
	// tree keeps its uncited-number defect (D8) for the audit, and the pre-merge
	// claims gate is exercised by the S6 harness instead of a half-built
	// CLAIMS.json. The revision lands in docs/.
	fs.mkdirSync(path.join(worktree, "docs"), { recursive: true });
	fs.writeFileSync(
		path.join(worktree, "docs", "paper_smoke_note.md"),
		"# Paper smoke note\n\nThe S7 end-to-end run merged a paper-kind PR through the PAPER_REVIEW and MERGE states.\n",
	);
	git(["add", "-A"], worktree);
	git(["-c", "user.name=e2e", "-c", "user.email=e2e@example.invalid", "commit", "-m", "paper: record the sandbox smoke run"], worktree);
	const head = git(["rev-parse", "HEAD"], worktree);
	const base = git(["rev-parse", project.baseBranch]);
	const pr = await append("pr", {
		task_id: "PAPER",
		head_sha: head,
		base_sha: base,
		branch,
		pr_kind: "paper",
		summary: "paper revision for the sandbox smoke run (docs/ note; the manuscript is not rebuilt in the sandbox)",
		artifacts: [],
		tests: [],
		claims: [],
		hpc_jobs: [],
		protected_paths: [],
		status: "open",
	});
	return `opened ${pr.id} (kind=paper) at ${head.slice(0, 12)}`;
}

// ---------------------------------------------------------------------------
// AUDIT
// ---------------------------------------------------------------------------

function auditCycle() {
	const file = path.join(root, ".pi", "state", "AUDIT.json");
	if (fs.existsSync(file)) {
		const state = JSON.parse(fs.readFileSync(file, "utf8"));
		if (Number.isInteger(state.cycle) && state.cycle > 0) return state.cycle;
	}
	const match = /## AUDIT · cycle (\d+)/.exec(brief);
	return match ? Number(match[1]) : 1;
}

function auditPhase() {
	const match = /## AUDIT · cycle \d+ · phase (\S+)/.exec(brief);
	return match ? match[1] : "gates";
}

function canaryFindingsFor(workstream) {
	// The sandbox is the seeded-defect tree (the smoke test plants the S6 fixture
	// before the run), so the deterministic canary is the finder here.
	const file = path.join(root, ".pi", "audit", `cycle-${auditCycle()}`, "canary.json");
	if (!fs.existsSync(file)) fail("canary.json is missing; the gates phase must run first");
	const result = JSON.parse(fs.readFileSync(file, "utf8"));
	return result.findings.filter((finding) => String(finding.id).replace(/^WS-/, "").charAt(0) === workstream);
}

async function runCanary() {
	const canary = await import(url(".pi/extensions/auditkit/canary.ts"));
	const result = canary.runSeededAudit(root);
	const cycle = auditCycle();
	const dir = path.join(root, ".pi", "audit", `cycle-${cycle}`);
	fs.mkdirSync(dir, { recursive: true });
	fs.writeFileSync(path.join(dir, "canary.json"), `${JSON.stringify(result, null, 2)}\n`);
	fs.writeFileSync(path.join(dir, "CANARY.md"), canary.renderCanaryMatrix(result));
	return result;
}

async function doAudit() {
	const audit = await import(url(".pi/extensions/orchestrator/audit.ts"));
	const cycle = auditCycle();
	const phase = auditPhase();
	const dir = path.join(root, ".pi", "audit", `cycle-${cycle}`);
	fs.mkdirSync(dir, { recursive: true });

	if (phase === "gates") {
		audit.prepareAuditCycle(root, cycle, `sandbox@${git(["rev-parse", "HEAD"]).slice(0, 12)}`);
		const gates = audit.runAuditGates(root, cycle, `sandbox@${git(["rev-parse", "HEAD"]).slice(0, 12)}`);
		await runCanary();
		audit.syncAuditState(root, cycle);
		return `gates: ${gates.gates.length} gate(s), failed=${gates.failed.join(",") || "none"}`;
	}
	if (phase === "find" || phase.startsWith("find:")) {
		const store = await import(url(".pi/extensions/auditkit/store.ts"));
		const workstreams = phase.includes(":") ? [phase.split(":")[1]] : ["M", "S", "P", "D", "I", "R", "H", "E", "C", "L"];
		const results = [];
		for (const ws of workstreams) {
			const findings = canaryFindingsFor(ws);
			if (findings.length === 0) {
				results.push(`${ws}:0`);
				// An empty register still has to exist for the phase to advance.
				fs.mkdirSync(path.join(dir, ws.toLowerCase()), { recursive: true });
				fs.writeFileSync(path.join(dir, ws.toLowerCase(), "findings.json"), `${JSON.stringify({ schema: "vangja-findings/1", cycle, workstream: ws, summary: { count: 0 }, findings: [] }, null, 2)}\n`);
				continue;
			}
			store.registerForCycle(root, cycle, ws, findings, { phase: "canary", author_agent: sessionName });
			results.push(`${ws}:${findings.length}`);
		}
		audit.syncAuditState(root, cycle);
		return `find ${results.join(" ")}`;
	}
	if (phase === "verify") {
		const store = await import(url(".pi/extensions/auditkit/store.ts"));
		const findings = store.readRegister(root, cycle);
		const lines = ["# Verify — cycle " + cycle, "", "| Finding | Severity | Verdict | Evidence |", "|---|---|---|---|"];
		for (const finding of findings) {
			lines.push(`| ${finding.id} | ${finding.severity} | confirmed | re-ran \`${finding.command}\` |`);
		}
		fs.writeFileSync(path.join(dir, "VERIFY.md"), `${lines.join("\n")}\n`);
		fs.mkdirSync(path.join(dir, "verify"), { recursive: true });
		for (const ws of ["M", "S", "P", "D", "I", "R", "H", "E", "C", "L"]) {
			const own = findings.filter((finding) => String(finding.id).replace(/^WS-/, "").charAt(0) === ws);
			fs.writeFileSync(
				path.join(dir, "verify", `VERIFY_${ws}.md`),
				`# Verify — WS-${ws} — cycle ${cycle}\n\n${own.length === 0 ? "_No findings._" : own.map((finding) => `- ${finding.id}: confirmed (\`${finding.command}\`)`).join("\n")}\n`,
			);
		}
		audit.syncAuditState(root, cycle);
		return `verify: ${findings.length} finding(s) confirmed`;
	}
	if (phase === "synthesise") {
		const store = await import(url(".pi/extensions/auditkit/store.ts"));
		const report = await import(url(".pi/extensions/auditkit/report.ts"));
		const grade = await import(url(".pi/extensions/auditkit/grade.ts"));
		const findings = store.readRegister(root, cycle);
		const gates = JSON.parse(fs.readFileSync(path.join(dir, "gates.json"), "utf8"));
		fs.writeFileSync(path.join(dir, "rubric.json"), `${JSON.stringify({ total: 0.9, complete: true, unresolved: [] }, null, 2)}\n`);
		const result = grade.auditGrade({
			findings,
			gates: {
				ok: gates.ok,
				cycle,
				snapshot: gates.snapshot,
				gates: gates.gates.map((gate) => ({ id: gate.id, status: gate.status, detail: gate.detail, hash_ok: gate.hash_ok })),
			},
			rubric: { total: 0.9, complete: true },
			calibration: { recall: 1, rejected_confirmed: 0, calibrated: true },
			traceability: { ok: true },
			limitations_disclosed: true,
			cycle,
			max_cycles: 5,
		});
		report.writeGrade(root, cycle, result);
		report.assembleReport({
			root,
			cycle,
			snapshot: gates.snapshot,
			findings,
			gates,
			grade: result,
			fixList: result.fix_list,
		});
		fs.writeFileSync(path.join(dir, "META_COVERAGE.md"), `# Meta coverage — cycle ${cycle}\n\nAll ten workstreams produced a register.\n`);
		audit.syncAuditState(root, cycle);
		return `synthesise: grade=${result.grade} blocking_fixes=${result.blocking_fixes.length}`;
	}
	if (phase === "adjudicate") {
		const store = await import(url(".pi/extensions/auditkit/store.ts"));
		const report = await import(url(".pi/extensions/auditkit/report.ts"));
		const findings = store.readRegister(root, cycle);
		report.writeRootIssueRegister(root, cycle, findings);
		fs.writeFileSync(
			path.join(dir, "ADJUDICATED_REPORT.md"),
			`# Adjudicated report — cycle ${cycle}\n\nEvery dispute was resolved with the finder's command; the root-issue register maps each fix to an owner.\n`,
		);
		audit.syncAuditState(root, cycle);
		return `adjudicate: ${findings.length} finding(s) mapped`;
	}
	fail(`unknown audit phase "${phase}"`);
}

// ---------------------------------------------------------------------------
// DECIDE
// ---------------------------------------------------------------------------

async function doDecide() {
	const audit = await import(url(".pi/extensions/orchestrator/audit.ts"));
	const cycle = auditCycle();
	const result = audit.applyAuditDecision(root, cycle, { limitationsDisclosed: true });
	if (result.grade === "uncalibrated" || result.status === "BLOCKED") {
		fail(`decision refused: grade=${result.grade} status=${result.status} (${result.decision.reason})`);
	}
	if (result.blocking_fixes.length === 0) fail("a seeded-defect audit produced no blocking fix");
	return `decision grade=${result.grade} status=${result.status} fixes=${result.blocking_fixes.length}`;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

boot();

if (mode === "work-kill-early") {
	process.kill(process.pid, "SIGKILL");
}
if (mode === "work-then-kill") {
	const summary = await doWork();
	assistant(`E2E WORK BEFORE KILL ${summary}`);
	process.kill(process.pid, "SIGKILL");
}
if (mode === "select") {
	fail("SELECT_TASK spawned a child; the smoke test seeds the task so this must not happen");
}

let summary = "no-op";
try {
	switch (mode) {
		case "work":
			summary = await doWork();
			break;
		case "local":
			summary = await runLocal();
			break;
		case "review":
			summary = await doReview();
			break;
		case "readiness":
			summary = await doReadiness();
			break;
		case "paper":
			summary = await doPaper();
			break;
		case "audit":
			summary = await doAudit();
			break;
		case "decide":
			summary = await doDecide();
			break;
		case "collector":
			summary = "HPC is disabled; the conductor must not spawn a collector";
			break;
		default:
			fail(`unrecognised brief:\n${brief}`);
	}
} catch (error) {
	fail((error && error.stack) || String(error));
}
assistant(`E2E ${mode.toUpperCase()} OK ${summary}`);
process.exit(0);
