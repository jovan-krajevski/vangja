#!/usr/bin/env node
/**
 * S7 end-to-end sandbox smoke test (`make -C .pi smoke`).
 *
 * Creates a scratch clone of the repository, plants a synthetic task ("add a
 * docstring and a passing test to a fixture module") plus a mock experiment with
 * a checkpoint/resume contract and the S6 seeded-defect audit tree, then drives
 * the real conductor (`--sandbox`, deterministic child stub) from BOOT to the
 * audit and the decision rule. Every assertion prints one `CHECK <name>:
 * PASS|FAIL` line; the exit status is non-zero if any check fails.
 *
 * Nothing here touches the real repository: the sandbox is a `git clone --local`
 * under a temp dir, the loop only ever runs with `--sandbox <that clone>`, and
 * the final checks compare the real `update_pymc` SHA and refs before/after.
 *
 * Environment:
 *   E2E_KEEP=1     keep the sandbox for inspection (path is printed)
 *   E2E_TRANSCRIPT print every step's output
 */

import { spawnSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { auditFiles } from "./audit_model_pin.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.dirname(path.dirname(path.dirname(HERE)));
const CLI = path.join(REPO_ROOT, ".pi", "extensions", "orchestrator", "cli.ts");
const CHILD = path.join(REPO_ROOT, ".pi", "tests", "e2e", "e2e_child.mjs");
const FIXTURE = path.join(HERE, "fixture");
const SEEDED = path.join(REPO_ROOT, ".pi", "tests", "audit", "fixtures", "seeded");
const EVIDENCE_DIR = path.join(REPO_ROOT, ".pi", "state", "e2e-tests");

const keep = process.env.E2E_KEEP === "1";
const showTranscript = process.env.E2E_TRANSCRIPT === "1";

interface Check {
	name: string;
	ok: boolean;
	detail: string;
}
const checks: Check[] = [];

function check(name: string, ok: boolean, detail: string): void {
	checks.push({ name, ok, detail });
	process.stdout.write(`CHECK ${name}: ${ok ? "PASS" : "FAIL"} — ${detail}\n`);
}

function sh(cmd: string, args: string[], cwd = REPO_ROOT): { status: number; stdout: string; stderr: string } {
	const result = spawnSync(cmd, args, { cwd, encoding: "utf8", timeout: 600_000, env: { ...process.env } });
	return { status: result.status ?? -1, stdout: String(result.stdout ?? ""), stderr: String(result.stderr ?? "") };
}

function git(root: string, args: string[]): string {
	const result = sh("git", ["-C", root, ...args]);
	if (result.status !== 0) throw new Error(`git ${args.join(" ")} failed: ${result.stderr || result.stdout}`);
	return result.stdout.trim();
}

function readJsonl(file: string): Array<Record<string, unknown>> {
	if (!fs.existsSync(file)) return [];
	return fs
		.readFileSync(file, "utf8")
		.split("\n")
		.filter((line) => line.trim() !== "")
		.map((line) => JSON.parse(line) as Record<string, unknown>);
}

function latest(records: Array<Record<string, unknown>>): Array<Record<string, unknown>> {
	const byId = new Map<string, Record<string, unknown>>();
	for (const record of records) {
		if (typeof record.id === "string") byId.set(record.id, record);
	}
	return [...byId.values()];
}

function overlay(source: string, target: string): void {
	const stat = fs.statSync(source);
	if (stat.isDirectory()) {
		fs.mkdirSync(target, { recursive: true });
		for (const entry of fs.readdirSync(source)) overlay(path.join(source, entry), path.join(target, entry));
		return;
	}
	fs.mkdirSync(path.dirname(target), { recursive: true });
	fs.copyFileSync(source, target);
}

function writeJson(file: string, value: unknown): void {
	fs.mkdirSync(path.dirname(file), { recursive: true });
	fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
}

async function main(): Promise<number> {
	const startedAt = Date.now();
	// The clone lives under `.pi/state/` (gitignored): the same filesystem as the
	// repository, so `git clone --local` can hardlink, and the loop's own sandbox
	// safety check sees a checkout. E2E_KEEP=1 leaves it for inspection.
	const runRoot = path.join(REPO_ROOT, ".pi", "state", "e2e-tests", `run-${new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14)}`);
	fs.mkdirSync(runRoot, { recursive: true });
	const sandbox = path.join(runRoot, "sandbox");
	const transcript: string[] = [];

	// -----------------------------------------------------------------------
	// Setup: scratch clone + synthetic task + seeded audit tree
	// -----------------------------------------------------------------------
	const clone = sh("git", ["clone", "--quiet", "--local", REPO_ROOT, sandbox]);
	if (clone.status !== 0) throw new Error(`git clone failed: ${clone.stderr}`);
	git(sandbox, ["checkout", "--quiet", "update_pymc"]);

	// Keep the sandbox to the fixture's world: the full .pi environment plus the
	// synthetic fixture. The S6 seeded-defect tree is planted later, just before
	// the audit, so the pre-merge gates run against the clean tree (the seeded
	// defects are the audit's target, not a PR's diff).
	for (const dir of ["src", "tests", "paper", "docs", "fixed_case_studies", "case_studies", "images"]) {
		fs.rmSync(path.join(sandbox, dir), { recursive: true, force: true });
	}
	overlay(path.join(FIXTURE, "synthetic_fixture"), path.join(sandbox, "synthetic_fixture"));
	overlay(path.join(FIXTURE, "experiments"), path.join(sandbox, "experiments"));
	writeJson(path.join(sandbox, ".pi", "config", "gates.json"), {
		testsCommand: ["python3", "experiments/run_fixture_tests.py"],
	});
	// The sandbox's calibration is synthetic by design (the real calibration gate
	// is exercised by the S6 harness and refuses to grade when uncalibrated).
	writeJson(path.join(sandbox, ".pi", "audit", "calibration", "latest", "recall.json"), {
		calibrated: true,
		recall: 1,
		rejected_confirmed: 0,
	});

	const store = await import(pathToFileURL(path.join(sandbox, ".pi", "extensions", "ledger", "store.ts")).href);
	const task = (store as { appendRecord: (...args: unknown[]) => Record<string, unknown> }).appendRecord(
		sandbox,
		"task",
		{
			id: "T-001",
			title: "add a docstring and a passing test to a fixture module",
			origin: "PLAN:S7-smoke",
			priority: "P2",
			objective: "Add a docstring to synthetic_fixture.calculator.add and a passing unittest for it.",
			acceptance_criteria: [
				"synthetic_fixture/calculator.py documents add",
				"synthetic_fixture/tests/test_calculator.py passes with python3 -m unittest",
			],
			deliverables: ["synthetic_fixture/calculator.py", "synthetic_fixture/tests/test_calculator.py"],
			compute: "local",
			status: "open",
			effort: "small",
		},
		{ root: sandbox, role: "conductor", actor: "e2e-setup" },
	);
	fs.mkdirSync(path.join(sandbox, ".pi", "state", "tasks"), { recursive: true });
	fs.writeFileSync(
		path.join(sandbox, ".pi", "state", "tasks", `${String(task.id)}.md`),
		`# ${String(task.id)} — ${String(task.title)}\n\nAcceptance: see the task record.\n`,
	);

	git(sandbox, ["add", "-A"]);
	git(sandbox, ["-c", "user.name=e2e", "-c", "user.email=e2e@example.invalid", "commit", "-qm", "e2e setup: synthetic task, fixture module, mock experiment, seeded audit tree"]);
	const baseBefore = git(sandbox, ["rev-parse", "update_pymc"]);
	const realBefore = git(REPO_ROOT, ["rev-parse", "update_pymc"]);
	const realRefsBefore = git(REPO_ROOT, ["for-each-ref", "--format=%(refname) %(objectname)"]);

	const stateFile = path.join(sandbox, ".pi", "state", "LOOP_STATE.json");
	const readinessFile = path.join(sandbox, ".pi", "state", "READINESS.json");
	const decisionFile = path.join(sandbox, ".pi", "state", "DECISION.json");
	const eventsFile = path.join(sandbox, ".pi", "state", "EVENTS.jsonl");
	const artifactDir = path.join(sandbox, ".pi", "artifacts", "E-0001");
	const state = (): Record<string, unknown> =>
		fs.existsSync(stateFile) ? (JSON.parse(fs.readFileSync(stateFile, "utf8")) as Record<string, unknown>) : { state: "UNINITIALIZED" };

	// -----------------------------------------------------------------------
	// Drive the loop; tamper the artifact once, just before READINESS
	// -----------------------------------------------------------------------
	let tamperCaught = false;
	let tamperDetail = "not exercised";
	let paperEnteredWhileTampered = false;
	let artifactBackup: string | null = null;
	let restored = false;

	const stepOnce = (): { status: number; stdout: string; stderr: string } => {
		const result = sh("node", [CLI, "step", "--sandbox", sandbox, "--headless", "--command", CHILD], REPO_ROOT);
		transcript.push(`step ${transcript.length + 1}: ${result.stdout.trim() || result.stderr.trim()}`);
		if (showTranscript) process.stdout.write(`  step ${transcript.length}: ${result.stdout.trim()}\n`);
		return result;
	};
	const approveGates = (): number => {
		const pending = latest(readJsonl(path.join(sandbox, ".pi", "state", "GATES.jsonl"))).filter((gate) => gate.status === "pending");
		for (const gate of pending) {
			const result = sh("node", [CLI, "gate", "approve", String(gate.id), "e2e sandbox approval", "--sandbox", sandbox, "--headless"], REPO_ROOT);
			transcript.push(`gate approve ${String(gate.id)}: ${result.stdout.trim()}`);
		}
		return pending.length;
	};

	let steps = 0;
	let seededPlanted = false;
	let stopReason = "step cap";
	while (steps < 200) {
		const current = state();
		if (current.state === "PAUSED" || current.state === "BLOCKED") {
			stopReason = `${String(current.state)}: ${String(current.reason ?? "")}`;
			break;
		}
		// Plant the seeded-defect audit tree at the AUDIT boundary, so the audit
		// snapshot carries the known defects and the pre-merge gates never saw them.
		if (current.state === "AUDIT" && !seededPlanted) {
			for (const entry of fs.readdirSync(SEEDED)) {
				if (entry === "AGENTS.md" || entry === "README.md") continue; // keep the repo's own
				overlay(path.join(SEEDED, entry), path.join(sandbox, entry));
			}
			git(sandbox, ["add", "-A"]);
			git(sandbox, ["-c", "user.name=e2e", "-c", "user.email=e2e@example.invalid", "commit", "-qm", "audit snapshot: seeded defects"]);
			seededPlanted = true;
			transcript.push(`planted the seeded-defect audit tree at ${git(sandbox, ["rev-parse", "HEAD"]).slice(0, 12)}`);
		}
		// Tamper hook: the step after SELECT_TASK→READINESS spawns the readiness
		// check. Tamper the artifact first, prove it is caught, then restore it.
		if (current.state === "READINESS" && !fs.existsSync(readinessFile) && !tamperCaught) {
			artifactBackup = path.join(runRoot, "artifact-good");
			fs.cpSync(artifactDir, artifactBackup, { recursive: true });
			fs.writeFileSync(path.join(artifactDir, "outputs", "numbers.json"), `${JSON.stringify({ series: "synthetic", rel_mae: 0.0907 }, null, 2)}\n`);
			steps += 1;
			const result = stepOnce();
			if (result.status !== 0) {
				stopReason = `tamper step failed: ${result.stderr}`;
				break;
			}
			const readiness = fs.existsSync(readinessFile) ? (JSON.parse(fs.readFileSync(readinessFile, "utf8")) as Record<string, unknown>) : null;
			tamperCaught = readiness?.pass === false && /failed verification/.test(String(readiness?.reason ?? ""));
			tamperDetail = `READINESS.json pass=${String(readiness?.pass)} reason=${String(readiness?.reason ?? "absent")}`;
			// Step again: the failed readiness must send the loop back to SELECT_TASK.
			steps += 1;
			const back = stepOnce();
			if (back.status !== 0) {
				stopReason = `readiness-fail step failed: ${back.stderr}`;
				break;
			}
			paperEnteredWhileTampered = readJsonl(eventsFile).some((event) => event.to === "PAPER");
			// Restore the good artifact and continue.
			fs.rmSync(artifactDir, { recursive: true, force: true });
			fs.cpSync(artifactBackup, artifactDir, { recursive: true });
			restored = true;
			approveGates();
			continue;
		}
		steps += 1;
		const result = stepOnce();
		if (result.status !== 0) {
			stopReason = `step failed: ${result.stdout.trim() || result.stderr.trim()}`;
			break;
		}
		approveGates();
		if (fs.existsSync(decisionFile)) {
			// Apply the decision rule (the child wrote DECISION.json during the spawn).
			steps += 1;
			const applied = stepOnce();
			if (applied.status !== 0) {
				stopReason = `decision step failed: ${applied.stderr}`;
				break;
			}
			approveGates();
			stopReason = "DECISION.json written and applied";
			break;
		}
		if (state().state === "DONE") {
			stopReason = "DONE";
			break;
		}
	}

	// -----------------------------------------------------------------------
	// Assertions
	// -----------------------------------------------------------------------
	const events = readJsonl(eventsFile);
	const prs = latest(readJsonl(path.join(sandbox, ".pi", "state", "PRS.jsonl")));
	const reviews = latest(readJsonl(path.join(sandbox, ".pi", "state", "REVIEWS.jsonl")));
	const experiments = latest(readJsonl(path.join(sandbox, ".pi", "state", "EXPERIMENTS.jsonl")));
	const decision = fs.existsSync(decisionFile) ? (JSON.parse(fs.readFileSync(decisionFile, "utf8")) as Record<string, unknown>) : null;

	check("loop-reached-decision", decision !== null && stopReason === "DECISION.json written and applied", `stop=${stopReason}; steps=${steps}`);

	// Transition sequence (plan/02_ARCHITECTURE.md §2).
	const transitions = events
		.filter(
			(event) =>
				["bookkeep", "boot", "merge"].includes(String(event.action_kind)) &&
				event.action_phase === "complete" &&
				event.from &&
				event.to,
		)
		.map((event) => `${String(event.from)}->${String(event.to)}`);
	const expectedTransitions = [
		"UNINITIALIZED->SELECT_TASK",
		"SELECT_TASK->WORK",
		"WORK->WORK",
		"WORK->REVIEW",
		"REVIEW->MERGE",
		"MERGE->POST_MERGE",
		"POST_MERGE->SELECT_TASK",
		"SELECT_TASK->READINESS",
		"READINESS->SELECT_TASK",
		"SELECT_TASK->READINESS",
		"READINESS->PAPER",
		"PAPER->PAPER_REVIEW",
		"PAPER_REVIEW->MERGE",
		"MERGE->AUDIT",
		"AUDIT->DECIDE",
		"DECIDE->SELECT_TASK",
	];
	check(
		"transition-sequence",
		JSON.stringify(transitions) === JSON.stringify(expectedTransitions),
		`${transitions.join(" → ")}`,
	);

	// BOOT snapshot.
	const boot = events.find((event) => event.action_kind === "boot" && event.action_phase === "complete");
	check(
		"boot-snapshot",
		Boolean(boot) && boot?.base_sha === baseBefore && boot?.model === "opencode-go/deepseek-v4.1-flash" && boot?.thinking === "max",
		`base=${String(boot?.base_sha ?? "?").slice(0, 12)} model=${String(boot?.model)} thinking=${String(boot?.thinking)}`,
	);

	// Exactly one work branch / PR / review / merge commit.
	const workBranches = git(sandbox, ["branch", "--list", "work/*"]).split("\n").filter(Boolean);
	const workPrs = prs.filter((pr) => (pr.pr_kind ?? "work") === "work");
	const workPr = workPrs[0];
	const workReviews = workPr ? reviews.filter((review) => review.pr_id === workPr.id) : [];
	const log = git(sandbox, ["log", "--format=%s", `${baseBefore}..update_pymc`]).split("\n");
	const workMerges = log.filter((line) => /^merge PR-\d+ \(review RV-\d+\)/.test(line) && line.includes(String(workPr?.id ?? "PR-001")));
	check("exactly-one-work-branch", workBranches.length === 1, `${workBranches.length}: ${workBranches.join(", ")}`);
	check("exactly-one-work-pr", workPrs.length === 1, `${workPrs.length} kind=work PR record(s)`);
	check("exactly-one-work-review", workReviews.length === 1, `${workReviews.length} review(s) for ${String(workPr?.id)}`);
	check("exactly-one-work-merge-commit", workMerges.length === 1, `${workMerges.length} merge commit(s): ${workMerges.join("; ")}`);

	// The paper PR/review/merge is the route into AUDIT (kind=paper).
	const paperPrs = prs.filter((pr) => pr.pr_kind === "paper");
	const paperPr = paperPrs[0];
	const paperReviews = paperPr ? reviews.filter((review) => review.pr_id === paperPr.id) : [];
	const paperMerges = log.filter((line) => /^merge PR-\d+ \(review RV-\d+\)/.test(line) && line.includes(String(paperPr?.id ?? "PR-002")));
	check(
		"exactly-one-paper-pr-review-merge",
		paperPrs.length === 1 && paperReviews.length === 1 && paperMerges.length === 1 && log.filter((line) => line.startsWith("merge PR-")).length === 2,
		`${paperPrs.length} paper PR(s), ${paperReviews.length} review(s), ${paperMerges.length} merge commit(s), ${log.filter((line) => line.startsWith("merge PR-")).length} merge(s) total`,
	);

	// Exactly one experiment, collected locally, with checkpoint progress.
	const experiment = experiments[0];
	check(
		"exactly-one-experiment",
		experiments.length === 1 &&
			experiment?.status === "collected" &&
			experiment?.resources?.kind === "local" &&
			Array.isArray(experiment?.progressHistory) &&
			(experiment.progressHistory as unknown[]).length === 1,
		`${experiments.length} record(s); status=${String(experiment?.status)} kind=${String((experiment?.resources as Record<string, unknown>)?.kind)} progress=${JSON.stringify(experiment?.progressHistory ?? [])}`,
	);

	// Exactly one collected artifact, hashes verified.
	const verify = await import(pathToFileURL(path.join(sandbox, ".pi", "extensions", "verify", "lib.ts")).href);
	const artifactExists = fs.existsSync(path.join(artifactDir, "manifest.json"));
	const artifact = artifactExists
		? (verify as { verifyArtifact: (root: string, target: string) => Record<string, unknown> }).verifyArtifact(sandbox, artifactDir)
		: null;
	const manifest = artifactExists ? (JSON.parse(fs.readFileSync(path.join(artifactDir, "manifest.json"), "utf8")) as Record<string, unknown>) : null;
	const artifactDirs = fs.existsSync(path.join(sandbox, ".pi", "artifacts")) ? fs.readdirSync(path.join(sandbox, ".pi", "artifacts")) : [];
	check(
		"exactly-one-collected-artifact",
		artifactDirs.length === 1 &&
			artifact?.ok === true &&
			artifact?.hash_ok === true &&
			Array.isArray(manifest?.missing) &&
			(manifest?.missing as unknown[]).length === 0 &&
			(manifest?.files as unknown[]).length >= 2 &&
			(manifest?.checkpoints as unknown[]).length >= 1,
		`dirs=${artifactDirs.length}; ${String(artifact?.detail ?? "manifest absent")}; checkpoints=${(manifest?.checkpoints as unknown[])?.length ?? 0}`,
	);

	// Tampered copy caught before the paper step.
	check(
		"tampered-artifact-caught-before-paper",
		tamperCaught && restored && !paperEnteredWhileTampered,
		`${tamperDetail}; PAPER entered while tampered=${paperEnteredWhileTampered}; restored=${restored}`,
	);

	// The deterministic gates ran before any finder.
	const auditSpawns = events.filter((event) => event.action_kind === "spawn" && event.action_phase === "complete" && String(event.action_key ?? "").startsWith("audit:"));
	const gateIndex = auditSpawns.findIndex((event) => String(event.action_key).startsWith("audit:1:gates:"));
	const firstFinderIndex = auditSpawns.findIndex((event) => String(event.action_key).startsWith("audit:1:find"));
	const gatesFile = path.join(sandbox, ".pi", "audit", "cycle-1", "gates.json");
	check(
		"audit-gates-before-finders",
		gateIndex === 0 && firstFinderIndex > gateIndex && fs.existsSync(gatesFile),
		`first audit spawn=${String(auditSpawns[0]?.action_key)}; finders=${auditSpawns.filter((event) => String(event.action_key).includes(":find")).length}; gates.json=${fs.existsSync(gatesFile)}`,
	);
	// Seeded-defect audit: canary recall, grade, fix list.
	const canaryFile = path.join(sandbox, ".pi", "audit", "cycle-1", "canary.json");
	const canary = fs.existsSync(canaryFile) ? (JSON.parse(fs.readFileSync(canaryFile, "utf8")) as Record<string, unknown>) : null;
	const gradeFile = path.join(sandbox, ".pi", "audit", "cycle-1", "GRADE.md");
	const gradeText = fs.existsSync(gradeFile) ? fs.readFileSync(gradeFile, "utf8") : "";
	check(
		"seeded-defect-audit-recall",
		canary !== null && Number(canary.recall) >= 0.8 && (canary.false_positives as unknown[]).length === 0,
		`recall=${String(canary?.recall)} (${String(canary?.detected_s0s1)}/${String(canary?.expected_s0s1)}); false positives=${(canary?.false_positives as unknown[])?.length ?? "?"}`,
	);
	check(
		"audit-grade-and-fix-list",
		gradeText.includes("## Fix list") &&
			/- \*\*[A-Z]-9\d\d\*\*/.test(gradeText) &&
			Array.isArray(decision?.blocking_fixes) &&
			(decision.blocking_fixes as unknown[]).length > 0 &&
			["strong reject", "reject", "weak reject", "borderline"].includes(String(decision?.grade)),
		`grade=${String(decision?.grade)}; blocking_fixes=${(decision?.blocking_fixes as unknown[])?.length ?? 0}; GRADE.md fix list=${/## Fix list/.test(gradeText)}`,
	);
	check(
		"decision-rule-applied",
		decision?.done === false &&
			decision?.reaudit === true &&
			Array.isArray(decision?.rule_trace) &&
			(decision.rule_trace as unknown[]).length > 0 &&
			decision?.audit_usable === true,
		`done=${String(decision?.done)} reaudit=${String(decision?.reaudit)} action=${String(decision?.action)} trace=${(decision?.rule_trace as unknown[])?.length ?? 0}`,
	);

	// The sandbox moved; the real repository did not.
	check(
		"sandbox-update_pymc-moved",
		git(sandbox, ["rev-parse", "update_pymc"]) !== baseBefore,
		`${baseBefore.slice(0, 12)} → ${git(sandbox, ["rev-parse", "update_pymc"]).slice(0, 12)}`,
	);
	const realAfter = git(REPO_ROOT, ["rev-parse", "update_pymc"]);
	const realRefsAfter = git(REPO_ROOT, ["for-each-ref", "--format=%(refname) %(objectname)"]);
	check("real-update_pymc-unchanged", realAfter === realBefore, `${realAfter}`);
	check("real-repo-refs-unchanged", realRefsAfter === realRefsBefore, `${realRefsAfter.split("\n").length} ref(s) identical`);

	// Every child invocation recorded with argv, prompt hash, usage, session.
	const spawnCompletions = events.filter((event) => event.action_kind === "spawn" && event.action_phase === "complete");
	const incomplete = spawnCompletions.filter((event) => {
		const argv = event.argv;
		const usage = event.usage as Record<string, unknown> | undefined;
		return (
			!Array.isArray(argv) ||
			argv.length === 0 ||
			typeof event.prompt_hash !== "string" ||
			!/^[0-9a-f]{64}$/.test(String(event.prompt_hash)) ||
			typeof event.session_path !== "string" ||
			!usage ||
			typeof usage.cost !== "number"
		);
	});
	check(
		"child-records-complete",
		spawnCompletions.length >= 15 && incomplete.length === 0,
		`${spawnCompletions.length} child spawn(s), ${incomplete.length} incomplete record(s)`,
	);
	const pinAudit = auditFiles(REPO_ROOT, [eventsFile]);
	check(
		"model-pin",
		pinAudit.violations.length === 0 && pinAudit.invocations >= spawnCompletions.length,
		`${pinAudit.invocations} recorded argv record(s) (start+complete), ${pinAudit.violations.length} violation(s)`,
	);

	// -----------------------------------------------------------------------
	// Summary + evidence
	// -----------------------------------------------------------------------
	const failed = checks.filter((entry) => !entry.ok);
	process.stdout.write(`\nSMOKE ${failed.length === 0 ? "PASS" : "FAIL"}: ${checks.length - failed.length}/${checks.length} checks passed in ${((Date.now() - startedAt) / 1000).toFixed(1)}s (${steps} loop step(s))\n`);
	if (failed.length > 0) {
		for (const entry of failed) process.stdout.write(`  FAILED ${entry.name}: ${entry.detail}\n`);
	}
	fs.mkdirSync(EVIDENCE_DIR, { recursive: true });
	const stamp = new Date().toISOString().replace(/[-:T]/g, "").slice(0, 14);
	writeJson(path.join(EVIDENCE_DIR, `smoke-${stamp}.json`), { sandbox, steps, stopReason, checks, transcript, transitions });
	fs.writeFileSync(path.join(EVIDENCE_DIR, `smoke-${stamp}.log`), `${transcript.join("\n")}\n`);
	process.stdout.write(`evidence: .pi/state/e2e-tests/smoke-${stamp}.json\n`);
	if (keep) {
		process.stdout.write(`sandbox kept: ${sandbox}\n`);
	} else {
		fs.rmSync(runRoot, { recursive: true, force: true });
	}
	return failed.length === 0 ? 0 : 1;
}

main()
	.then((code) => {
		process.exitCode = code;
	})
	.catch((error) => {
		process.stderr.write(`${(error as Error).stack ?? String(error)}\n`);
		process.exitCode = 1;
	});
