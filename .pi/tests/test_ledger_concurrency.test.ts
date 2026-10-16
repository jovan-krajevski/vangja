/**
 * Concurrency: 200 appends from 4 processes must produce exactly 200 valid,
 * untorn JSONL lines; LOOP_STATE.json must survive 100 concurrent read/writes.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { FIXTURES_DIR, scratchRoot, spawnSyncNode } from "./helpers.ts";

test("200 concurrent appends from 4 processes yield 200 valid lines", () => {
	const root = scratchRoot("vangja-concurrency-");
	const workers = 4;
	const perWorker = 50;
	const results = [];
	for (let i = 0; i < workers; i += 1) {
		const worker = spawnSyncNode(
			process.execPath,
			[path.join(FIXTURES_DIR, "append_worker.ts"), root, "event", String(perWorker), `w${i}`],
			{ timeoutMs: 120_000 },
		);
		results.push(worker);
	}
	for (const worker of results) {
		assert.equal(worker.status, 0, `worker failed: ${worker.stderr}`);
	}

	const file = path.join(root, ".pi", "state", "EVENTS.jsonl");
	const raw = fs.readFileSync(file, "utf8");
	const lines = raw.split("\n").filter((line) => line.trim() !== "");
	assert.equal(lines.length, workers * perWorker, "line count must equal the number of appends");
	const seqs = new Set<number>();
	const actors = new Set<string>();
	for (const line of lines) {
		let record: { seq: number; actor: string; summary: string; schema: string; kind: string };
		try {
			record = JSON.parse(line);
		} catch (error) {
			assert.fail(`torn/invalid line: ${line.slice(0, 120)} (${(error as Error).message})`);
		}
		assert.equal(record.schema, "pi-ledger/1");
		assert.equal(record.kind, "event");
		assert.ok(Number.isInteger(record.seq) && record.seq >= 1, `bad seq ${record.seq}`);
		assert.ok(!seqs.has(record.seq), `duplicate seq ${record.seq}`);
		seqs.add(record.seq);
		actors.add(record.actor);
	}
	assert.equal(seqs.size, workers * perWorker, "every append must get a unique seq");
	assert.equal(actors.size, workers, "every worker must appear");
	assert.deepEqual(
		[...seqs].sort((a, b) => a - b),
		Array.from({ length: workers * perWorker }, (_, i) => i + 1),
		"seq numbers must be contiguous from 1",
	);
	assert.ok(!fs.existsSync(`${file}.lock`), "no lock directory may be left behind");
	assert.ok(fs.existsSync(path.join(root, ".pi", "state", "CONTEXT.md")), "the derived view is written");
});

test("LOOP_STATE.json survives 100 concurrent reads and writes", () => {
	const root = scratchRoot("vangja-state-race-");
	const workers = 4;
	const perWorker = 25;
	const results = [];
	for (let i = 0; i < workers; i += 1) {
		results.push(
			spawnSyncNode(
				process.execPath,
				[path.join(FIXTURES_DIR, "state_worker.ts"), root, String(perWorker), `s${i}`],
				{ timeoutMs: 120_000 },
			),
		);
	}
	for (const worker of results) {
		assert.equal(worker.status, 0, `state worker failed: ${worker.stderr}`);
	}

	const file = path.join(root, ".pi", "state", "LOOP_STATE.json");
	const parsed = JSON.parse(fs.readFileSync(file, "utf8")) as { state: string; schema: string; updated_by: string };
	assert.equal(parsed.schema, "pi-loop-state/1");
	assert.equal(parsed.state, "BOOT");
	assert.match(parsed.updated_by, /^worker-s\d$/);
	assert.ok(!fs.existsSync(`${file}.lock`), "no lock directory may be left behind");
});
