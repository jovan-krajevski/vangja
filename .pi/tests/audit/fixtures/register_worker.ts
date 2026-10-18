/**
 * Worker for the concurrent-register test: registers one finding for one
 * workstream in a shared cycle register. Run as a child process.
 *
 * Usage: node --experimental-strip-types register_worker.ts <root> <ws> <id>
 */

import { registerForCycle } from "../../../extensions/auditkit/store.ts";
import type { Workstream } from "../../../extensions/auditkit/schema.ts";

const [root, workstream, id] = process.argv.slice(2);
if (!root || !workstream || !id) {
	console.error("usage: register_worker.ts <root> <ws> <id>");
	process.exit(2);
}

const finding = {
	id,
	title: `Concurrent finding ${id}`,
	location: `src/concurrent_${id}.py:${id.slice(-3)}`,
	snapshot: "concurrency@test",
	command: `grep -n concurrent src/concurrent_${id}.py`,
	expected: "the register keeps every workstream's finding",
	observed: `registered ${id} from a parallel finder`,
	consequence: "a lost finding would hide a workstream's result",
	severity: "S2",
	confidence: "high",
	status: "verified",
	reproducible: "yes",
	evidence: [`src/concurrent_${id}.py:1`],
	cross_refs: [],
	owner: "code",
	found_in_cycle: 1,
	phase: "finder",
};

const result = registerForCycle(root, 1, workstream as Workstream, [finding]);
process.stdout.write(`${JSON.stringify({ id, added: result.added, duplicates: result.duplicates })}\n`);
