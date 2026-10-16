/**
 * Worker fixture for the ledger concurrency test.
 * Usage: node append_worker.ts <root> <kind> <count> <tag>
 */

import { appendRecord } from "../../extensions/ledger/handlers.ts";

const [root, kind, countText, tag] = process.argv.slice(2);
if (!root || !kind || !countText || !tag) {
	process.stderr.write("usage: append_worker.ts <root> <kind> <count> <tag>\n");
	process.exit(2);
}
const count = Number(countText);
for (let i = 0; i < count; i += 1) {
	appendRecord(root, kind as never, { summary: `${tag}-${i}`, note: `worker ${tag} record ${i}` }, {
		root,
		role: "conductor",
		actor: `worker-${tag}`,
	});
}
process.stdout.write(`worker ${tag}: appended ${count}\n`);
