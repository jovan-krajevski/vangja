/**
 * Worker fixture for the LOOP_STATE concurrency test.
 * Usage: node state_worker.ts <root> <count> <tag>
 */

import { loadLoopState, saveLoopState } from "../../lib/state.ts";

const [root, countText, tag] = process.argv.slice(2);
if (!root || !countText || !tag) {
	process.stderr.write("usage: state_worker.ts <root> <count> <tag>\n");
	process.exit(2);
}
const count = Number(countText);
for (let i = 0; i < count; i += 1) {
	saveLoopState(root, { state: "BOOT", reason: `${tag}-${i}` }, { actor: `worker-${tag}` });
	const loaded = loadLoopState(root);
	// Parse the round-tripped value; a torn write would yield invalid JSON above.
	JSON.parse(JSON.stringify(loaded.state));
}
process.stdout.write(`worker ${tag}: wrote ${count}\n`);
