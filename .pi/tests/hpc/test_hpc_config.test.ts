/**
 * S5 config validation: `.pi/config/hpc.example.json` is the schema, `host: ""`
 * disables without touching anything, required fields are named, the single-slot
 * amendment is honoured, and the host is redacted in tool output.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { loadHpcConfig, loadHpcExample, parseHpcConfig } from "../../lib/config.ts";
import { configSummary, redact, redactHost, resolveHpc } from "../../extensions/hpc/config.ts";
import { scratchRoot, readRepoConfig, writeJson } from "../helpers.ts";

function example(overrides: Record<string, unknown> = {}): Record<string, unknown> {
	return { ...(readRepoConfig<Record<string, unknown>>("hpc.example.json")), ...overrides };
}

test("hpc.example.json parses and documents the single-slot schema", () => {
	const example = loadHpcExample(path.join(process.cwd()));
	assert.equal(example.host, "");
	assert.equal(example.maxParallelJobs, 1);
	assert.equal(example.wallClockHours, 24);
	assert.equal(example.pollSeconds, 60);
	assert.equal(example.maxPolls, 40);
	assert.equal(example.maxAttemptsPerExperiment, 6);
});

test("host: \"\" disables the whole path without requiring keys", () => {
	const root = scratchRoot();
	fs.writeFileSync(path.join(root, ".pi", "config", "hpc.json"), `${JSON.stringify(example(), null, 2)}\n`);
	const loaded = loadHpcConfig(root);
	assert.equal(loaded.enabled, false);
	const resolution = resolveHpc(root);
	assert.equal(resolution.enabled, false);
	assert.match(resolution.reason, /empty "host"/);
});

test("an enabled config names every missing required field", () => {
	const root = scratchRoot();
	writeJson(path.join(root, ".pi", "config", "hpc.json"), example({ host: "cluster.example", user: "", remoteRepoDir: "", remoteScratchDir: "", hpcKey: "" }));
	assert.throws(() => resolveHpc(root), /user.*remoteRepoDir.*remoteScratchDir.*hpcKey/);
});

test("keys must exist and be readable when the host is set", () => {
	const root = scratchRoot();
	const missingKey = path.join(root, "no-such-key");
	writeJson(
		path.join(root, ".pi", "config", "hpc.json"),
		example({ host: "cluster.example", user: "alice", remoteRepoDir: "/repo", remoteScratchDir: "/scratch", hpcKey: missingKey }),
	);
	assert.throws(() => resolveHpc(root), /hpcKey.*not readable/);
	const key = path.join(root, "id_mock");
	fs.writeFileSync(key, "placeholder\n");
	writeJson(
		path.join(root, ".pi", "config", "hpc.json"),
		example({ host: "cluster.example", user: "alice", remoteRepoDir: "/repo", remoteScratchDir: "/scratch", hpcKey: key }),
	);
	const resolved = resolveHpc(root);
	assert.equal(resolved.enabled, true);
});

test("maxParallelJobs > 1 needs a recorded amendment", () => {
	const root = scratchRoot();
	writeJson(
		path.join(root, ".pi", "config", "hpc.json"),
		example({ host: "cluster.example", user: "alice", remoteRepoDir: "/repo", remoteScratchDir: "/scratch", hpcKey: "/tmp/key", maxParallelJobs: 2 }),
	);
	fs.writeFileSync(path.join(root, "tmp-key"), "");
	assert.throws(() => loadHpcConfig(root), /maxParallelJobs=2 > 1 is rejected/);
	const amendment = {
		schema: "pi-ledger/1",
		kind: "amendment",
		seq: 1,
		ts: new Date().toISOString(),
		actor: "human",
		id: "A-001",
		path: ".pi/config/hpc.json",
		change: "maxParallelJobs: 2",
		justification: "test fixture only",
	};
	fs.writeFileSync(path.join(root, ".pi", "state", "AMENDMENTS.jsonl"), `${JSON.stringify(amendment)}\n`);
	const loaded = loadHpcConfig(root);
	assert.equal(loaded.enabled, true);
});

test("config summaries redact the host and never contain key material", () => {
	const root = scratchRoot();
	const key = path.join(root, "id_mock");
	const secret = "-----BEGIN OPENSSH PRIVATE KEY-----\nMOCKPRIVATEKEYMATERIALMOCKPRIVATEKEYMATERIAL\n-----END OPENSSH PRIVATE KEY-----";
	fs.writeFileSync(key, `${secret}\n`);
	writeJson(
		path.join(root, ".pi", "config", "hpc.json"),
		example({ host: "very-long-hostname.example.org", user: "alice", remoteRepoDir: "/repo", remoteScratchDir: "/scratch", hpcKey: key }),
	);
	const resolution = resolveHpc(root);
	assert.equal(resolution.enabled, true);
	if (!resolution.enabled) return;
	const summary = configSummary(resolution);
	assert.equal(summary.hostRedacted, "ver******org");
	assert.ok(!JSON.stringify(summary).includes(secret));
	assert.equal(redactHost("ab"), "**");
	assert.ok(!redact(`leak? ${secret}`).includes("MOCKPRIVATEKEYMATERIAL"));
	assert.equal(parseHpcConfig(example(), "fixture").maxParallelJobs, 1);
});
