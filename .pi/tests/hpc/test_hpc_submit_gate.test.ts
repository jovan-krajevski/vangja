/**
 * Scenarios 7–9 — the submission gate and secret hygiene.
 *
 * 7. Missing checkpoint metadata at queue time → refused with the exact field.
 * 8. Missing checkpoint smoke-test evidence → refused.
 * 9. Fake key material never appears in tool output, ledger records, evidence
 *    files, or the mock cluster's transcripts.
 *
 * Plus the remote checks: dirty checkout, uv.lock mismatch, container hash,
 * wall-clock staging, and the single-slot refusal.
 */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { REPO_ROOT } from "../helpers.ts";
import { handleExperimentQueue } from "../../extensions/ledger/handlers.ts";
import { appendRecord, readLedger } from "../../extensions/ledger/store.ts";
import { projectLatest } from "../../lib/records.ts";
import { loadHpcExample } from "../../lib/config.ts";
import { handleHpcConfig, handleHpcStatus, handleHpcSubmit } from "../../extensions/hpc/handlers.ts";
import { loadJobTemplate, renderJobScript, renderTemplate } from "../../extensions/hpc/slurm.ts";
import { resetPrivateAgent } from "../../extensions/hpc/ssh.ts";
import { createHpcSandbox, seedExperiment, setUpCluster } from "./mock_cluster.ts";

const BASE_QUEUE = {
	task_id: "T-001",
	branch: "work/T-001",
	commit_sha: "c".repeat(40),
	data_hash: "sha256:mock",
	command: "uv run python mock_runner.py",
	resources: { kind: "hpc", cpus: 2, memGb: 8, gpus: 0, wallClockHours: 24 },
	checkpoints: { dir: "d", progressFile: "progress.json", resumeCommand: "r --resume $CKPT", flushOnSignal: true },
	expectedOutputs: ["results/units_a.csv"],
	seeds: [42],
	idempotencyKey: "sha256:key",
	resumeSmokeTest: "ran 1 unit, killed, --resume skipped it",
};

function latest(root: string, id: string): Record<string, unknown> {
	return projectLatest(readLedger(root, "experiment") as Array<{ id?: string; seq: number }>).find(
		(record) => record.id === id,
	) as Record<string, unknown>;
}

function mirror(root: string, id: string, patch: Record<string, unknown>): Record<string, unknown> {
	const current = latest(root, id);
	return appendRecord(root, "experiment", { ...current, ...patch }, { root, role: "conductor", actor: "test" });
}

test("scenario 7/8: queue-time refusals name the exact missing checkpoint field", () => {
	const sandbox = createHpcSandbox("hpc-gate-queue");
	const cluster = setUpCluster(sandbox.root, "cluster", { hpcEnabled: false });
	try {
		const ctx = { root: sandbox.root, role: "phd_candidate" as const, actor: "test" };
		const missingProgress = handleExperimentQueue(
			{ record: { ...BASE_QUEUE, checkpoints: { dir: "d", resumeCommand: "r", flushOnSignal: true } } },
			ctx,
		);
		assert.equal(missingProgress.isError, true);
		assert.match(missingProgress.content[0].text, /checkpoints\.progressFile/);

		const missingFlush = handleExperimentQueue(
			{ record: { ...BASE_QUEUE, checkpoints: { dir: "d", progressFile: "p", resumeCommand: "r" } } },
			ctx,
		);
		assert.equal(missingFlush.isError, true);
		assert.match(missingFlush.content[0].text, /flushOnSignal/);

		const missingDir = handleExperimentQueue({ record: { ...BASE_QUEUE, checkpoints: {} } }, ctx);
		assert.equal(missingDir.isError, true);
		assert.match(missingDir.content[0].text, /checkpoints\.dir/);

		const { resumeSmokeTest: _drop, ...noSmoke } = BASE_QUEUE;
		const smoke = handleExperimentQueue({ record: noSmoke }, ctx);
		assert.equal(smoke.isError, true);
		assert.match(smoke.content[0].text, /resumeSmokeTest/);
	} finally {
		cluster.cleanup();
		sandbox.cleanup();
	}
});

test("scenario 8: the submit gate refuses missing smoke-test evidence and missing flushOnSignal", () => {
	const sandbox = createHpcSandbox("hpc-gate-submit");
	const cluster = setUpCluster(sandbox.root, "cluster");
	try {
		// Bypass the queue handler so the gate itself is what refuses.
		const { resumeSmokeTest: _smoke, ...rest } = BASE_QUEUE;
		appendRecord(
			sandbox.root,
			"experiment",
			{
				...rest,
				id: "E-0001",
				status: "queued",
				attempt: 1,
				commit_sha: cluster.commitSha,
				uv_lock_sha: cluster.uvLockSha,
				checkpoints: {
					...BASE_QUEUE.checkpoints,
					dir: path.join(cluster.remoteScratch, "E-0001", "checkpoints"),
					resumeCommand: `uv run python mock_runner.py --resume $CKPT`,
				},
			},
			{ root: sandbox.root, role: "conductor", actor: "test" },
		);
		const ctx = { root: sandbox.root, role: "phd_candidate" as const, actor: "test" };
		const noSmoke = handleHpcSubmit({ experiment_id: "E-0001" }, ctx);
		assert.equal(noSmoke.isError, true);
		assert.match(noSmoke.content[0].text, /resume-smoke-test/);
		assert.match(noSmoke.content[0].text, /smoke[- ]test/i);

		mirror(sandbox.root, "E-0001", {
			resumeSmokeTest: "ok",
			checkpoints: { dir: "d", progressFile: "p", resumeCommand: "r" },
		});
		const noFlush = handleHpcSubmit({ experiment_id: "E-0001" }, ctx);
		assert.equal(noFlush.isError, true);
		assert.match(noFlush.content[0].text, /flushOnSignal/);
	} finally {
		cluster.cleanup();
		sandbox.cleanup();
	}
});

test("the submit gate checks the remote checkout, uv.lock, container, wall clock, and single slot", () => {
	const sandbox = createHpcSandbox("hpc-gate-checks");
	const cluster = setUpCluster(sandbox.root, "cluster");
	try {
		const ctx = { root: sandbox.root, role: "phd_candidate" as const, actor: "test" };
		const { id } = seedExperiment(sandbox.root, cluster);

		// A modified tracked file in the *run worktree* makes the checkout dirty
		// (untracked job outputs do not).
		const runDir = path.join(cluster.remoteRepo, "runs", id);
		execFileSync("git", ["-C", cluster.remoteRepo, "worktree", "add", "--force", "--detach", runDir, cluster.commitSha]);
		fs.writeFileSync(path.join(runDir, "README.mock.md"), "# mock remote\nmodified\n");
		const dirty = handleHpcSubmit({ experiment_id: id }, ctx);
		assert.equal(dirty.isError, true);
		assert.match(dirty.content[0].text, /remote-checkout/);
		assert.match(dirty.content[0].text, /dirty/);
		fs.writeFileSync(path.join(runDir, "README.mock.md"), "# mock remote\n");

		// uv.lock mismatch: a stale record must be rebuilt.
		mirror(sandbox.root, id, { uv_lock_sha: "0".repeat(64) });
		const stale = handleHpcSubmit({ experiment_id: id }, ctx);
		assert.equal(stale.isError, true);
		assert.match(stale.content[0].text, /uv-lock/);

		mirror(sandbox.root, id, { uv_lock_sha: cluster.uvLockSha });

		// Wall clock: 48h is refused unless the record is staged.
		mirror(sandbox.root, id, { resources: { ...BASE_QUEUE.resources, wallClockHours: 48 } });
		const unstaged = handleHpcSubmit({ experiment_id: id }, ctx);
		assert.equal(unstaged.isError, true);
		assert.match(unstaged.content[0].text, /wall-clock/);
		const stagedChecks = (unstaged.details.gate as { checks: Array<{ name: string; ok: boolean }> }).checks;
		assert.equal(stagedChecks.find((check) => check.name === "wall-clock")?.ok, false);
		mirror(sandbox.root, id, { resources: { ...BASE_QUEUE.resources, wallClockHours: 48 }, stages: 2 });
		const staged = handleHpcSubmit({ experiment_id: id, stage: 0 }, ctx);
		// Passing the staged wall-clock check lets the submission proceed.
		assert.equal(staged.isError, undefined, staged.content[0].text);
		assert.match(staged.content[0].text, /submitted as job \d+/);
		mirror(sandbox.root, id, { resources: BASE_QUEUE.resources, stages: 1 });

		// Single slot: a second record cannot be submitted while E-0001 is running.
		const second = handleExperimentQueue(
			{ record: { ...BASE_QUEUE, idempotencyKey: "sha256:other", task_id: "T-002" } },
			ctx,
		);
		assert.equal(second.isError, true);
		assert.match(second.content[0].text, /E-0001/);
		assert.match(second.content[0].text, /active|single-slot/i);
	} finally {
		cluster.cleanup();
		sandbox.cleanup();
	}
});

test("renderTemplate drops a line only when every placeholder on it is empty", () => {
	const rendered = renderTemplate(
		["A={{A}}", "DROP={{B}}", "SPACES={{C}}", "MIX={{A}}-{{B}}", "literal line"].join("\n"),
		{ A: "x", B: "", C: "   " },
	);
	assert.match(rendered, /^A=x$/m);
	assert.doesNotMatch(rendered, /^DROP=/m);
	assert.doesNotMatch(rendered, /^SPACES=/m);
	assert.match(rendered, /^MIX=x-$/m);
	assert.match(rendered, /^literal line$/m);
});

test("the job script submitted for the shipped config has no empty SBATCH directives", () => {
	const sandbox = createHpcSandbox("hpc-render-default");
	const cluster = setUpCluster(sandbox.root, "cluster");
	try {
		const { id } = seedExperiment(sandbox.root, cluster);
		const submitted = handleHpcSubmit(
			{ experiment_id: id },
			{ root: sandbox.root, role: "phd_candidate" as const, actor: "test" },
		);
		assert.equal(submitted.isError, undefined, submitted.content[0].text);
		const scriptPath = path.join(sandbox.root, String(submitted.details.local_script));
		assert.ok(fs.existsSync(scriptPath), `rendered script missing: ${scriptPath}`);
		const script = fs.readFileSync(scriptPath, "utf8");
		// The shipped hpc.example.json has partition="" and gpus=0: those directives
		// must vanish, not render as `#SBATCH --partition=` / `#SBATCH --gres=`.
		assert.doesNotMatch(script, /^#SBATCH --partition=\s*$/m);
		assert.doesNotMatch(script, /^#SBATCH --gres=\s*$/m);
		assert.doesNotMatch(script, /\{\{[A-Z0-9_]+\}\}/);
		assert.match(script, /^#SBATCH --job-name=E-0001-a1$/m);
		assert.match(script, /^#SBATCH --time=24:00:00$/m);
		assert.match(script, /^#SBATCH --signal=B:USR1@300$/m);
		assert.match(script, /^uv sync --frozen --no-dev$/m);
	} finally {
		cluster.cleanup();
		sandbox.cleanup();
	}
});

test("the rendered job script keeps partition and gres when they are configured", () => {
	const config = {
		...loadHpcExample(REPO_ROOT),
		host: "mock-cluster",
		user: "mockuser",
		remoteRepoDir: "/repo",
		remoteScratchDir: "/scratch",
		hpcKey: "/tmp/key",
		partition: "mypart",
		gpus: 2,
	};
	const rendered = renderJobScript({
		runtime: { root: REPO_ROOT, config, reason: "test", hpcKeyPath: "/tmp/key", gitKeyPath: "" },
		experimentId: "E-0001",
		attempt: 1,
		template: loadJobTemplate(REPO_ROOT),
		resumeCommand: "uv run python run.py --resume $CKPT",
		runDir: "/scratch/run",
		memGb: 8,
		cpus: 2,
		gpus: 2,
		partition: "mypart",
	});
	assert.match(rendered.script, /^#SBATCH --partition=mypart$/m);
	assert.match(rendered.script, /^#SBATCH --gres=gpu:2$/m);
	assert.doesNotMatch(rendered.script, /\{\{[A-Z0-9_]+\}\}/);
});

test("scenario 9: fake key material never reaches output, ledgers, evidence, or transcripts", () => {
	const sandbox = createHpcSandbox("hpc-secrets");
	const cluster = setUpCluster(sandbox.root, "cluster");
	const secret = "MOCKPRIVATEKEYMATERIAL".repeat(4);
	try {
		// A PEM-shaped fake key on disk; the config points at it, nothing may echo it.
		const keyBody = ["-----BEGIN OPENSSH PRIVATE KEY-----", secret, "-----END OPENSSH PRIVATE KEY-----"].join("\n");
		const keyPath = path.join(cluster.dir, "keys", "secret_mock_key");
		fs.writeFileSync(keyPath, `${keyBody}\n`);
		const config = JSON.parse(fs.readFileSync(path.join(sandbox.root, ".pi", "config", "hpc.json"), "utf8")) as Record<string, unknown>;
		config.hpcKey = keyPath;
		config.gitKey = keyPath;
		fs.writeFileSync(path.join(sandbox.root, ".pi", "config", "hpc.json"), `${JSON.stringify(config, null, 2)}\n`);
		// One private agent per loop run: this test starts a fresh run so the new key is the one added.
		resetPrivateAgent();

		const { id } = seedExperiment(sandbox.root, cluster);
		const ctx = { root: sandbox.root, role: "phd_candidate" as const, actor: "test" };
		const outputs: string[] = [];
		const push = (result: ReturnType<typeof handleHpcConfig>): void => {
			outputs.push(result.content[0].text, JSON.stringify(result.details));
		};
		push(handleHpcConfig({}, ctx));
		push(handleHpcStatus({}, { ...ctx, role: "collector" }));
		push(handleHpcSubmit({ experiment_id: id }, ctx));
		const forwarded = handleHpcSubmit({ experiment_id: id }, ctx);
		push(forwarded);
		// The second submission is refused (single slot) — and the refusal must not leak the key either.
		assert.equal(forwarded.isError, true);
		assert.match(forwarded.content[0].text, /already submitted/);
		push(handleHpcStatus({}, { ...ctx, role: "collector" }));

		for (const text of outputs) {
			assert.ok(!text.includes(secret), `key material leaked into a tool result: ${text.slice(0, 200)}`);
		}
		const files: string[] = [];
		const walk = (dir: string): void => {
			if (!fs.existsSync(dir)) return;
			for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
				const full = path.join(dir, entry.name);
				if (entry.isDirectory()) walk(full);
				else if (entry.isFile()) files.push(full);
			}
		};
		walk(path.join(sandbox.root, ".pi", "state"));
		walk(cluster.dir);
		walk(path.join(sandbox.root, ".pi", "artifacts"));
		for (const file of files) {
			if (file === keyPath) continue; // the credential store itself is not an output
			const content = fs.readFileSync(file, "utf8");
			assert.ok(!content.includes(secret), `key material leaked into ${file}`);
		}
		// The private agent holds only the configured key (the mock records add/list).
		const agentCalls = cluster.readCalls().filter((call) => call.tool === "ssh-add");
		assert.ok(agentCalls.some((call) => call.agent === "add" && String(call.key).includes("secret_mock_key")));
		assert.ok(agentCalls.every((call) => call.agent !== "add" || String(call.key).includes("secret_mock_key")));
	} finally {
		cluster.cleanup();
		sandbox.cleanup();
	}
});
