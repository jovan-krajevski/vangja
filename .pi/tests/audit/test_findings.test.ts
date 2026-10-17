/**
 * Finding schema, dedupe/register, and the synthesis no-new-findings rule.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { assembleReport, writeRootIssueRegister } from "../../extensions/auditkit/report.ts";
import { registerFindings, validateFinding, dedupeKey } from "../../extensions/auditkit/schema.ts";
import { readRegister, registerForCycle } from "../../extensions/auditkit/store.ts";
import { baseFinding, readText, scratchFixture } from "./helpers.ts";

test("a complete finding validates and is normalized", () => {
	const result = validateFinding(baseFinding());
	assert.equal(result.ok, true, result.errors.join("; "));
	assert.equal(result.normalized?.owner, "code");
	assert.equal(result.normalized?.phase, "finder");
});

test("a finding with no command is rejected (no command → not a finding)", () => {
	const missing = validateFinding(baseFinding({ command: "" }));
	assert.equal(missing.ok, false);
	assert.match(missing.errors.join(" "), /command: required/);

	const prose = validateFinding(baseFinding({ command: "static reading of lines 400-420" }));
	assert.equal(prose.ok, false);
	assert.match(prose.errors.join(" "), /not an executable command/);

	const placeholder = validateFinding(baseFinding({ command: "none" }));
	assert.equal(placeholder.ok, false);
});

test("an unknown severity, confidence, status, or owner is rejected", () => {
	assert.equal(validateFinding(baseFinding({ severity: "S9" as never })).ok, false);
	assert.equal(validateFinding(baseFinding({ confidence: "certain" as never })).ok, false);
	assert.equal(validateFinding(baseFinding({ status: "done" as never })).ok, false);
	assert.equal(validateFinding(baseFinding({ owner: "team" as never })).ok, false);
});

test("verified + confidence low is rejected; a no without a reason is rejected", () => {
	const low = validateFinding(baseFinding({ confidence: "low" }));
	assert.equal(low.ok, false);
	assert.match(low.errors.join(" "), /verified cannot carry confidence low/);

	const noReason = validateFinding(baseFinding({ reproducible: "no" }));
	assert.equal(noReason.ok, false);
	assert.match(noReason.errors.join(" "), /reproducible: must be/);

	const yes = validateFinding(
		baseFinding({ status: "to-verify", confidence: "medium", reproducible: "no: the data snapshot is unavailable" }),
	);
	assert.equal(yes.ok, true, yes.errors.join("; "));
});

test("verified with reproducible=no is rejected; unresolved needs a question tag", () => {
	const verifiedNo = validateFinding(baseFinding({ reproducible: "no: no data" }));
	assert.equal(verifiedNo.ok, false);
	assert.match(verifiedNo.errors.join(" "), /requires reproducible=yes/);

	const unresolved = validateFinding(
		baseFinding({ status: "unresolved", confidence: "medium", reproducible: "no: needs author input" }),
	);
	assert.equal(unresolved.ok, false);
	assert.match(unresolved.errors.join(" "), /question_tag/);

	const tagged = validateFinding(
		baseFinding({
			status: "unresolved",
			confidence: "medium",
			reproducible: "no: needs author input",
			question_tag: "blocking",
		}),
	);
	assert.equal(tagged.ok, true, tagged.errors.join("; "));
});

test("an id must name a known workstream and match the expected one", () => {
	assert.equal(validateFinding(baseFinding({ id: "X-001" })).ok, false);
	assert.equal(validateFinding(baseFinding({ id: "S-1" })).ok, false);
	assert.equal(validateFinding(baseFinding({ id: "S-001" }), { expectedWorkstream: "S" }).ok, true);
	assert.equal(validateFinding(baseFinding({ id: "M-001" }), { expectedWorkstream: "S" }).ok, false);
});

test("duplicate findings are deduped by normalized location+title with a cross-ref", () => {
	const first = baseFinding({ id: "S-001", location: "src/a.py:10" });
	const second = baseFinding({ id: "S-002", location: "src/a.py:25" });
	assert.equal(dedupeKey(first), dedupeKey(second));
	const outcome = registerFindings([], {
		cycle: 1,
		workstream: "S",
		findings: [first, second],
	});
	assert.equal(outcome.ok, true);
	assert.deepEqual(outcome.added, ["S-001"]);
	assert.equal(outcome.duplicates.length, 1);
	assert.equal(outcome.duplicates[0].duplicate_of, "S-001");
	assert.equal(outcome.register.length, 1);
	assert.deepEqual(outcome.register[0].cross_refs, ["S-002"]);
	assert.deepEqual(outcome.register[0].duplicates, ["S-002"]);
});

test("re-registering the same id records history instead of duplicating", () => {
	const first = baseFinding({ id: "S-001", status: "to-verify", confidence: "medium" });
	const verdict = baseFinding({ id: "S-001", status: "verified", observed: "reproduced with the exact command" });
	const outcome = registerFindings([first as never], {
		cycle: 1,
		workstream: "S",
		findings: [verdict],
	});
	assert.equal(outcome.register.length, 1);
	assert.equal(outcome.register[0].status, "verified");
	assert.equal(outcome.register[0].history?.length, 1);
	assert.equal(outcome.register[0].history?.[0].status, "to-verify");
});

test("registerForCycle writes the per-workstream and global registers", () => {
	const root = scratchFixture();
	const outcome = registerForCycle(root, 2, "S", [baseFinding()]);
	assert.ok(fs.existsSync(outcome.files.register));
	assert.ok(fs.existsSync(outcome.files.workstream));
	assert.ok(fs.existsSync(outcome.files.markdown));
	assert.equal(readRegister(root, 2).length, 1);
	const markdown = readText(outcome.files.markdown);
	assert.match(markdown, /S-001/);
	assert.match(markdown, /Command \/ test/);
});

test("the synthesizer may not add a primary finding without a verified id", () => {
	const root = scratchFixture();
	registerForCycle(root, 3, "S", [baseFinding()]);
	const synth = baseFinding({
		id: "C-999",
		title: "Synthesis says the paper is wrong about transfer",
		location: "paper/paper.tex:10",
		phase: "synth",
	});
	registerForCycle(root, 3, "C", [synth]);
	const findings = readRegister(root, 3);
	const assembled = assembleReport({
		root,
		cycle: 3,
		snapshot: "fixture@sha",
		findings,
		gates: null,
	});
	assert.equal(assembled.rejected.length, 1);
	assert.equal(assembled.rejected[0].id, "C-999");
	assert.deepEqual(assembled.synth_added, ["C-999"]);
	const report = readText(assembled.final_report);
	assert.doesNotMatch(report, /Synthesis says the paper is wrong/);
	assert.match(report, /Rejected synthesis records/);
});

test("a synthesizer record citing a verified id is allowed", () => {
	const root = scratchFixture();
	registerForCycle(root, 4, "S", [baseFinding()]);
	const synth = baseFinding({
		id: "C-998",
		title: "Restated verified finding for the author",
		location: "paper/paper.tex:10",
		phase: "synth",
		cites: ["S-001"],
	});
	registerForCycle(root, 4, "C", [synth]);
	const assembled = assembleReport({
		root,
		cycle: 4,
		snapshot: "fixture@sha",
		findings: readRegister(root, 4),
		gates: null,
	});
	assert.deepEqual(assembled.rejected, []);
	assert.ok(assembled.reported.includes("C-998"));
});

test("unverified and refuted findings never appear as confirmed report items", () => {
	const root = scratchFixture();
	registerForCycle(root, 5, "S", [
		baseFinding({ id: "S-001", title: "Verified finding", status: "verified" }),
		baseFinding({ id: "S-002", title: "Hypothesis only", status: "to-verify", confidence: "medium" }),
		baseFinding({ id: "S-003", title: "Refuted lead", status: "refuted" }),
	]);
	const assembled = assembleReport({
		root,
		cycle: 5,
		snapshot: "fixture@sha",
		findings: readRegister(root, 5),
		gates: null,
	});
	assert.deepEqual(assembled.reported, ["S-001"]);
	assert.deepEqual(assembled.unverified, ["S-002"]);
	assert.deepEqual(assembled.refuted, ["S-003"]);
	const report = readText(assembled.final_report);
	assert.match(report, /## Confirmed findings/);
	assert.match(report, /Hypothesis only|S-002/);
});

test("root issue register groups findings and maps owner/fix/status", () => {
	const root = scratchFixture();
	const a = baseFinding({ id: "S-001", title: "Root cause", root_issue: "transfer potential sign" });
	const b = baseFinding({
		id: "S-002",
		title: "Symptom",
		location: "src/b.py:5",
		root_issue: "transfer potential sign",
	});
	const file = writeRootIssueRegister(root, 6, [a, b]);
	const csv = readText(file).trim().split("\n");
	assert.equal(csv.length, 2);
	assert.match(csv[0], /^root_issue,title/);
	assert.match(csv[1], /RI-001/);
	assert.match(csv[1], /S-001 S-002/);
	assert.match(csv[1], /code/);
	assert.match(csv[1], /open/);
	assert.ok(fs.existsSync(path.join(root, ".pi", "audit", "cycle-6")));
});
