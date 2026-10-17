/**
 * Parallel finders share the cycle register. Ten child processes register one
 * workstream each at the same time; the lock must keep every finding.
 */

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import * as path from "node:path";
import { test } from "node:test";
import { AUDIT_WORKSTREAMS } from "../../extensions/orchestrator/audit.ts";
import { readRegister } from "../../extensions/auditkit/store.ts";
import { AUDIT_TESTS_DIR, scratchFixture } from "./helpers.ts";

function runWorker(root: string, workstream: string, id: string): Promise<void> {
	const worker = path.join(AUDIT_TESTS_DIR, "fixtures", "register_worker.ts");
	return new Promise((resolve, reject) => {
		const child = spawn(process.execPath, ["--experimental-strip-types", worker, root, workstream, id], {
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stderr = "";
		child.stderr.on("data", (chunk) => {
			stderr += String(chunk);
		});
		child.on("error", reject);
		child.on("exit", (code) => {
			if (code === 0) resolve();
			else reject(new Error(`worker ${workstream} exited ${code}: ${stderr}`));
		});
	});
}

test("ten parallel finders do not lose findings to a register race", async () => {
	const root = scratchFixture("vangja-register-race-");
	await Promise.all(
		AUDIT_WORKSTREAMS.map((workstream, index) => runWorker(root, workstream, `${workstream}-${String(index + 1).padStart(3, "0")}`)),
	);
	const register = readRegister(root, 1);
	const ids = register.map((finding) => finding.id).sort();
	assert.equal(register.length, AUDIT_WORKSTREAMS.length, `expected all ten, got ${ids.join(", ")}`);
	for (const workstream of AUDIT_WORKSTREAMS) {
		assert.ok(
			ids.some((id) => id === `${workstream}-${String(AUDIT_WORKSTREAMS.indexOf(workstream) + 1).padStart(3, "0")}`),
			`missing ${workstream} finding: ${ids.join(", ")}`,
		);
	}
});
