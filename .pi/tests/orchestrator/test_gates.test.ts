/**
 * Pre-merge gates, git helpers, and environment guards.
 *
 * These exercise the deterministic checks directly (protected paths, test diff,
 * provenance, hygiene) plus the refusals: a dirty checkout must not be merged,
 * a missing branch must not be invented, and the conductor has no code path that
 * pushes anywhere.
 */

import { strict as assert } from "node:assert";
import { test } from "node:test";
import * as fs from "node:fs";
import * as path from "node:path";
import {
	DEFAULT_TESTS_COMMAND,
	hpcGate,
	openGate,
	runPreMergeGates,
	type PreMergeOptions,
} from "../../extensions/orchestrator/gates.ts";
import {
	assertCleanCheckout,
	assertNotOnSharedHistoryRewrite,
	commitAll,
	currentBranch,
	ensureWorktree,
	mergeNoFf,
	recordPr,
	worktreeDirty,
} from "../../extensions/orchestrator/git.ts";
import { execFileSync } from "node:child_process";
import { createSandbox, fileHashes, git, readLedger, runCli, seedTask, writeGatesConfig } from "./orchestrator_helpers.ts";

function makePr(sandbox: ReturnType<typeof createSandbox>, changes: (worktree: string) => void, footer: Record<string, unknown> = {}) {
	// Fast, deterministic `tests` check: the gate logic under test is the scan,
	// not pytest (which has its own suite and its own stage).
	writeGatesConfig(sandbox.root, ["true"]);
	git(sandbox.root, ["add", ".pi/config/gates.json"]);
	git(sandbox.root, ["-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-m", "test: gate config"]);
	const worktree = ensureWorktree(sandbox.root, { taskId: "T-001", slug: "gates", baseBranch: "update_pymc" });
	changes(worktree.path);
	const head = commitAll(worktree.path, { message: "T-001: gate fixture" });
	const base = git(sandbox.root, ["rev-parse", "update_pymc"]);
	const pr = recordPr(sandbox.root, {
		task_id: "T-001",
		head_sha: head,
		base_sha: base,
		branch: worktree.branch,
		summary: "gate fixture PR",
		artifacts: [],
		tests: [],
		claims: [],
		hpc_jobs: [],
		...(footer as Record<string, never>),
	});
	return { worktree, head, base, prId: String(pr.id) };
}

function options(sandbox: ReturnType<typeof createSandbox>, prId: string, head: string, base: string, overrides: Partial<PreMergeOptions> = {}): PreMergeOptions {
	return {
		root: sandbox.root,
		prId,
		taskId: "T-001",
		baseSha: base,
		headSha: head,
		branch: "work/T-001-gates",
		kind: "work",
		state: "WORK",
		artifacts: [],
		protectedPaths: [],
		reviewActor: "supervisor",
		runId: null,
		...overrides,
	};
}

function check(run: ReturnType<typeof runPreMergeGates>, id: string) {
	const found = run.checks.find((entry) => entry.id === id);
	assert.ok(found, `check ${id} missing from ${run.checks.map((entry) => entry.id).join(", ")}`);
	return found;
}

test("the default tests command is the plan's, and gates.json overrides it explicitly", () => {
	const sandbox = createSandbox("gates-config");
	try {
		assert.deepEqual(DEFAULT_TESTS_COMMAND, ["uv", "run", "pytest", "-q"]);
		assert.equal(fs.existsSync(path.join(sandbox.root, ".pi", "config", "gates.json")), false);
		writeGatesConfig(sandbox.root, ["true"]);
		const config = JSON.parse(fs.readFileSync(path.join(sandbox.root, ".pi", "config", "gates.json"), "utf8"));
		assert.deepEqual(config.testsCommand, ["true"]);
	} finally {
		sandbox.cleanup();
	}
});

test("protected paths fail the gate with the allowed alternative, and a waiver is the only way past", async () => {
	const sandbox = createSandbox("gates-protected");
	try {
		fs.mkdirSync(path.join(sandbox.root, "research_agent", "plan"), { recursive: true });
		const fixture = makePr(sandbox, (worktree) => {
			fs.mkdirSync(path.join(worktree, "research_agent", "plan"), { recursive: true });
			fs.writeFileSync(path.join(worktree, "research_agent", "plan", "PLAN.md"), "# plan\n");
		});
		const run = runPreMergeGates(options(sandbox, fixture.prId, fixture.head, fixture.base));
		assert.equal(run.ok, false);
		assert.match(check(run, "protected-paths").detail, /research_agent\/plan\/PLAN\.md/);
		assert.match(check(run, "protected-paths").detail, /amendment/);

		// A waiver naming the check (and optionally the PR) turns the failure into
		// `waived` — recorded, not hidden.
		const { appendRecord } = await import("../../extensions/ledger/store.ts");
		appendRecord(
			sandbox.root,
			"waiver",
			{ gate: "pre-merge:protected-paths", reason: "fixture: the plan file was needed for the test", pr_id: fixture.prId },
			{ root: sandbox.root, role: "supervisor", actor: "supervisor" },
		);
		const waived = runPreMergeGates(options(sandbox, fixture.prId, fixture.head, fixture.base));
		assert.equal(check(waived, "protected-paths").status, "waived");
		assert.match(check(waived, "protected-paths").detail, /WAIVED/);
		assert.equal(waived.ok, true);
	} finally {
		sandbox.cleanup();
	}
});

test("a removal of an assertion in a test file fails the test-diff check", () => {
	const sandbox = createSandbox("gates-testdiff");
	try {
		const fixture = makePr(sandbox, (worktree) => {
			fs.mkdirSync(path.join(worktree, "tests"), { recursive: true });
			fs.writeFileSync(path.join(worktree, "tests", "test_gate.py"), "def test_gate():\n    assert 1 == 1\n");
		});
		// Second commit: delete the assertion.
		fs.writeFileSync(path.join(fixture.worktree.path, "tests", "test_gate.py"), "def test_gate():\n    pass\n");
		const head = commitAll(fixture.worktree.path, { message: "weaken the test" });
		const run = runPreMergeGates(options(sandbox, fixture.prId, head, fixture.base));
		assert.equal(check(run, "test-diff").status, "fail");
		assert.match(check(run, "test-diff").detail, /removed 1 more assertion/);
		assert.match(check(run, "test-diff").detail, /waiver/);
	} finally {
		sandbox.cleanup();
	}
});

test("provenance fails for a missing artifact without a manifest and passes with one", () => {
	const sandbox = createSandbox("gates-provenance");
	try {
		const fixture = makePr(sandbox, (worktree) => fs.writeFileSync(path.join(worktree, "notes.md"), "notes\n"));
		const missing = runPreMergeGates(
			options(sandbox, fixture.prId, fixture.head, fixture.base, { artifacts: ["here/results.json"] }),
		);
		assert.equal(check(missing, "provenance").status, "fail");
		assert.match(check(missing, "provenance").detail, /does not exist/);
		fs.mkdirSync(path.join(sandbox.root, "here", "run"), { recursive: true });
		fs.writeFileSync(path.join(sandbox.root, "here", "run", "out.csv"), "x\n");
		const bad = runPreMergeGates(
			options(sandbox, fixture.prId, fixture.head, fixture.base, { artifacts: ["here/run"] }),
		);
		assert.equal(check(bad, "provenance").status, "fail");
		assert.match(check(bad, "provenance").detail, /no manifest/);
		fs.writeFileSync(
			path.join(sandbox.root, "here", "run", "manifest.json"),
			JSON.stringify({ commit_sha: fixture.head, seeds: [1], data_hash: "abc" }),
		);
		const proven = runPreMergeGates(
			options(sandbox, fixture.prId, fixture.head, fixture.base, { artifacts: ["here/run"] }),
		);
		assert.equal(check(proven, "provenance").status, "pass");
	} finally {
		sandbox.cleanup();
	}
});

test("hygiene fails when the head is not a descendant of the base (rewritten history)", () => {
	const sandbox = createSandbox("gates-hygiene");
	try {
		const fixture = makePr(sandbox, (worktree) => fs.writeFileSync(path.join(worktree, "notes.md"), "notes\n"));
		// An unrelated root commit: not an ancestor of the head.
		const tree = git(sandbox.root, ["rev-parse", "HEAD^{tree}"]);
		const other = execFileSync("git", ["-C", sandbox.root, "commit-tree", tree, "-m", "unrelated root"], { encoding: "utf8" }).trim();
		const run = runPreMergeGates(options(sandbox, fixture.prId, fixture.head, other));
		assert.equal(check(run, "hygiene").status, "fail");
		assert.match(check(run, "hygiene").detail, /not a descendant/);
	} finally {
		sandbox.cleanup();
	}
});

test("worktrees are created under .pi/state/worktrees on work/T-###-slug and reused", () => {
	const sandbox = createSandbox("gates-worktree");
	try {
		const first = ensureWorktree(sandbox.root, { taskId: "T-001", slug: "slug", baseBranch: "update_pymc" });
		assert.equal(first.created, true);
		assert.equal(first.branch, "work/T-001-slug");
		assert.equal(first.path, path.join(sandbox.root, ".pi", "state", "worktrees", "T-001"));
		assert.equal(currentBranch(first.path), "work/T-001-slug");
		const second = ensureWorktree(sandbox.root, { taskId: "T-001", slug: "slug", baseBranch: "update_pymc" });
		assert.equal(second.created, false);
		assert.equal(second.path, first.path);
		fs.writeFileSync(path.join(first.path, "file.md"), "x\n");
		// Untracked-only changes are not "dirty" for the tracked-file policy, but
		// the stricter query sees them (used by BOOT's cleanliness check).
		assert.equal(worktreeDirty(first.path), false);
		assert.equal(worktreeDirty(first.path, true), true);
		fs.writeFileSync(path.join(first.path, "AGENTS.md"), "# scratch\n");
		assert.equal(worktreeDirty(first.path), true);
	} finally {
		sandbox.cleanup();
	}
});

test("merge refuses a dirty checkout and a branch that does not exist", () => {
	const sandbox = createSandbox("gates-merge-refusals");
	try {
		const fixture = makePr(sandbox, (worktree) => fs.writeFileSync(path.join(worktree, "notes.md"), "notes\n"));
		assert.throws(
			() =>
				mergeNoFf(sandbox.root, {
					prId: fixture.prId,
					reviewId: "RV-001",
					branch: "work/does-not-exist",
					baseBranch: "update_pymc",
					summary: "nope",
					runId: "run-test",
				}),
			/does not exist/,
		);
		fs.writeFileSync(path.join(sandbox.root, "AGENTS.md"), "# uncommitted change\n");
		assert.throws(() => assertCleanCheckout(sandbox.root, "merge"), /uncommitted changes/);
		assert.throws(
			() =>
				mergeNoFf(sandbox.root, {
					prId: fixture.prId,
					reviewId: "RV-001",
					branch: fixture.worktree.branch,
					baseBranch: "update_pymc",
					summary: "nope",
					runId: "run-test",
				}),
			/uncommitted changes/,
		);
	} finally {
		sandbox.cleanup();
	}
});

test("the conductor has no push/force-push path in its git module", () => {
	const source = fs.readFileSync(path.join(process.cwd(), ".pi", "extensions", "orchestrator", "git.ts"), "utf8");
	assert.equal(/\[\s*"push"/.test(source), false, "git.ts must not invoke git push");
	assert.equal(/"--force-with-lease"|"--force", *"push"|"push", *"--force"/.test(source), false, "git.ts must not force-push");
	// History-rewriting commands are refused, not merely absent from a call site.
	assert.throws(() => assertNotOnSharedHistoryRewrite(["filter-branch"]), /never rewrites shared history/);
	assert.throws(() => assertNotOnSharedHistoryRewrite(["rebase"]), /never rewrites shared history/);
});

test("the HPC gate reports not-configured for an empty host and blocks on an invalid config", () => {
	const sandbox = createSandbox("gates-hpc");
	try {
		const gate = hpcGate(sandbox.root);
		assert.equal(gate.status, "not-configured");
		assert.match(gate.reason, /hpc\.json/);
		// No experiment to collect: the collector exits cleanly and says so (the
		// hpc-disabled event path is covered by .pi/tests/hpc/test_hpc_disabled.test.ts).
		const collect = runCli(sandbox.root, ["collect"]);
		assert.equal(collect.status, 0, collect.stderr);
		assert.match(collect.stdout, /nothing to collect/);
		// A configured-but-invalid hpc.json blocks; it is never silently "off".
		fs.writeFileSync(
			path.join(sandbox.root, ".pi", "config", "hpc.json"),
			JSON.stringify({ host: "cluster.example", user: "", port: 22, remoteRepoDir: "", remoteScratchDir: "", hpcKey: "", gitKey: "", forwardGitAgent: true, container: "", partition: "", cpus: 2, memGb: 8, gpus: 0, wallClockHours: 24, pollSeconds: 60, maxPolls: 40, maxAttemptsPerExperiment: 6, maxParallelJobs: 1 }),
		);
		const invalid = hpcGate(sandbox.root);
		assert.equal(invalid.status, "blocked");
		assert.match(invalid.reason, /missing "user"|hpcKey/);
	} finally {
		sandbox.cleanup();
	}
});

test("--dry-run writes only its own log", () => {
	const sandbox = createSandbox("gates-dry-run");
	try {
		seedTask(sandbox.root);
		writeGatesConfig(sandbox.root);
		const before = fileHashes(path.join(sandbox.root, ".pi", "state"));
		const result = runCli(sandbox.root, ["step", "--dry-run"]);
		assert.equal(result.status, 0, result.stderr);
		const after = fileHashes(path.join(sandbox.root, ".pi", "state"));
		const added = [...after.keys()].filter((file) => !before.has(file));
		const changed = [...after.keys()].filter((file) => before.has(file) && before.get(file) !== after.get(file));
		assert.deepEqual(added, ["DRYRUN.jsonl"]);
		assert.deepEqual(changed, []);
		assert.equal(fs.existsSync(path.join(sandbox.root, ".pi", "state", "LOOP_STATE.json")), false);
		const log = fs.readFileSync(path.join(sandbox.root, ".pi", "state", "DRYRUN.jsonl"), "utf8").trim();
		assert.match(log, /"kind":"boot"/);
	} finally {
		sandbox.cleanup();
	}
});

test("the human gate refuses to open twice and an undecided gate is never approved", async () => {
	const sandbox = createSandbox("gates-human");
	try {
		const gate = openGate(sandbox.root, { gate: "merge", note: "merge PR-001", pr_id: "PR-001" }, "conductor");
		assert.equal(String(gate.status), "pending");
		const { approveGate, rejectGate } = await import("../../extensions/orchestrator/gates.ts");
		approveGate(sandbox.root, String(gate.id), "verified", "human");
		assert.throws(() => rejectGate(sandbox.root, String(gate.id), "too late", "human"), /already approved/);
		const gates = readLedger<{ id: string; status: string }>(sandbox.root, "GATES.jsonl");
		assert.equal(gates.length, 2);
		assert.equal(gates[1].status, "approved");
	} finally {
		sandbox.cleanup();
	}
});

test("the merge_pr tool refuses without an accepted review, opens the gate, and merges once approved", async () => {
	const sandbox = createSandbox("merge-tool");
	try {
		fs.writeFileSync(path.join(sandbox.root, ".pi", "config", "project.json"), `${JSON.stringify({ schema: "pi-project/1", repoRoot: ".", baseBranch: "update_pymc", venue: "Journal of Forecasting", requireHumanMerge: true, dryRun: false, sandboxPath: ".pi/state/sandbox", sessionDir: ".pi/state/sessions", artifactDir: ".pi/artifacts", stateDir: ".pi/state", worktreeDir: ".pi/state/worktrees" }, null, 2)}\n`);
		const fixture = makePr(sandbox, (worktree) => fs.writeFileSync(path.join(worktree, "notes.md"), "notes\n"));
		// The supervisor's session runs in REVIEW; the tool must walk the loop to MERGE.
		const { saveLoopState } = await import("../../lib/state.ts");
		saveLoopState(sandbox.root, { state: "REVIEW", task_id: "T-001", pr_id: fixture.prId, reason: "acceptance test" }, { actor: "test", force: true });
		const { handleMergePr } = await import("../../extensions/orchestrator/merge-tool.ts");
		const { appendRecord } = await import("../../extensions/ledger/store.ts");
		const ctx = { root: sandbox.root, role: "supervisor" as const, actor: "supervisor__PR-001__1" };

		// No review yet: refused, and nothing recorded.
		const eventsBefore = readLedger(sandbox.root, "EVENTS.jsonl").length;
		const refused = await handleMergePr({ pr_id: fixture.prId }, ctx);
		assert.equal(refused.isError, true);
		assert.match(String(refused.content[0].text), /no review record exists/);
		assert.equal(readLedger(sandbox.root, "EVENTS.jsonl").length, eventsBefore);

		// Accepted review + requireHumanMerge → the gate opens and no merge happens.
		appendRecord(
			sandbox.root,
			"review",
			{ pr_id: fixture.prId, decision: "accept", evidence: ["acceptance test"], required_changes: [], residual_risk: "none" },
			{ root: sandbox.root, role: "supervisor", actor: "supervisor" },
		);
		const baseBefore = git(sandbox.root, ["rev-parse", "update_pymc"]);
		const gated = await handleMergePr({ pr_id: fixture.prId }, ctx);
		assert.equal(gated.isError, undefined);
		assert.equal((gated.details as Record<string, unknown>).merged, false);
		assert.equal((gated.details as Record<string, unknown>).status, "pending-human-gate");
		assert.equal(git(sandbox.root, ["rev-parse", "update_pymc"]), baseBefore);
		const gate = readLedger<{ id: string; status: string }>(sandbox.root, "GATES.jsonl").pop();
		assert.equal(gate?.status, "pending");

		// Approve, then the same call merges exactly once.
		const { approveGate } = await import("../../extensions/orchestrator/gates.ts");
		approveGate(sandbox.root, String(gate?.id), "verified", "human");
		const merged = await handleMergePr({ pr_id: fixture.prId }, ctx);
		assert.equal(merged.isError, undefined, String(merged.content[0].text));
		assert.equal((merged.details as Record<string, unknown>).merged, true);
		assert.match(git(sandbox.root, ["log", "--format=%s", "-1", "update_pymc"]), /^merge PR-001 \(review RV-001\)/);
		const again = await handleMergePr({ pr_id: fixture.prId }, ctx);
		assert.equal((again.details as Record<string, unknown>).status, "already-merged");
		assert.equal(git(sandbox.root, ["rev-list", "--count", `${baseBefore}..update_pymc`]).trim(), "2");
	} finally {
		sandbox.cleanup();
	}
});
