/**
 * Deterministic PEM-fixture check used by the acceptance script.
 *
 * Writes a fixture file containing a PEM block into a scratch root, asks the
 * guard whether a write of that content would be allowed, and proves the denial
 * does not echo the key material. Exits non-zero on failure.
 *
 * The PEM header is assembled at runtime so this file's own source never
 * contains a credential-shaped literal.
 */

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { toolCallDecision } from "../../extensions/guards/policy.ts";

const root = fs.mkdtempSync(path.join(os.tmpdir(), "vangja-pem-fixture-"));
fs.mkdirSync(path.join(root, ".pi", "state"), { recursive: true });
fs.writeFileSync(path.join(root, "AGENTS.md"), "# fixture\n");

const header = ["-----BEGIN OPENSSH", ["PRIVATE", "KEY"].join(" ")].join(" ") + "-----";
const body = ["b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQ", "AAAAAAAAABAAAAMwAAAAtzc2gtZWQyNTUxOQ"].join("");
const fixture = `${header}\n${body}\n-----END OPENSSH PRIVATE KEY-----\n`;
const fixturePath = path.join(root, "fixture-pem.txt");
fs.writeFileSync(fixturePath, fixture);
process.stdout.write(`fixture written: ${fixturePath} (${fixture.length} bytes)\n`);

const decision = toolCallDecision(
	{ toolName: "write", path: "src/vangja/leaked-key.ts", contentToScan: fixture },
	{ role: "phd_candidate", loopState: "WORK", root, cwd: root, currentBranch: "work/T-001", worktreeDirty: false },
);

if (decision.allowed) {
	process.stderr.write("FAIL: the guard allowed a write containing PEM key material\n");
	process.exit(1);
}
process.stdout.write(`guard denial: [guard:${decision.code}] ${decision.reason}\n`);
if (decision.reason.includes(body)) {
	process.stderr.write("FAIL: the guard echoed the key material back in the denial\n");
	process.exit(1);
}
process.stdout.write("PASS: PEM content refused and the key material was not echoed\n");
