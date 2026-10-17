/**
 * verify primitives and the deterministic gate runner: tests, artifact hashes,
 * reproduction with numeric drift, traceability, guards, and gates.json.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import { execFileSync } from "node:child_process";
import * as path from "node:path";
import { test } from "node:test";
import {
	defaultRunner,
	verifyArtifact,
	verifyArtifactAgainstExpected,
	verifyGuards,
	verifyReproduce,
	verifyTests,
	verifyTraceability,
	type CommandRunner,
} from "../../extensions/verify/lib.ts";
import { GATE_ORDER, runGates } from "../../extensions/verify/gates.ts";
import { readText, scratchFixture } from "./helpers.ts";

const TAMPER_EXPECTED = 0.0607;
const TAMPER_ACTUAL = 0.0907;

test("verify_tests parses pass/fail counts from the runner output", () => {
	const runner: CommandRunner = (command) => ({
		status: 1,
		stdout: "FAILED tests/test_fit.py::test_a - AssertionError\n2 failed, 7 passed, 1 skipped in 1.2s\n",
		stderr: "",
		command: command.join(" "),
	});
	const result = verifyTests("/tmp", { command: ["uv", "run", "pytest", "-q"], runner });
	assert.equal(result.ok, false);
	assert.equal(result.failed, 2);
	assert.equal(result.passed, 7);
	assert.equal(result.skipped, 1);
	assert.equal(result.failures.length, 1);
	assert.match(result.failures[0], /test_a/);
});

test("verify_artifact accepts the seeded manifest and reports hash_ok", () => {
	const root = scratchFixture();
	const result = verifyArtifact(root, path.join(root, "artifacts", "E-0001"));
	assert.equal(result.ok, true, result.detail);
	assert.equal(result.hash_ok, true);
	assert.equal(result.schema_ok, true);
	assert.equal(result.manifest_sha256_ok, true);
	assert.equal(result.files, 2);
	assert.deepEqual(result.mismatched, []);
});

test("a tampered artifact fails hash_ok and names the mismatched files", () => {
	const root = scratchFixture();
	const dir = path.join(root, "artifacts", "E-0001");
	fs.writeFileSync(path.join(dir, "numbers.json"), `${JSON.stringify({ rel_mae: TAMPER_ACTUAL })}\n`);
	fs.writeFileSync(path.join(dir, "metrics.csv"), `series,rel_mae\nFridge,${TAMPER_ACTUAL}\n`);
	const result = verifyArtifact(root, dir);
	assert.equal(result.ok, false);
	assert.equal(result.hash_ok, false);
	assert.equal(result.mismatched.length, 2);
	assert.ok(result.mismatched.every((entry) => entry.actual !== null));
	assert.match(result.detail, /hash mismatches/);
});

test("verify_reproduce recomputes the numbers and reports per-number drift", () => {
	const root = scratchFixture();
	const good = verifyReproduce(root, path.join(root, "artifacts", "E-0001"));
	assert.equal(good.ok, true, `${good.drift} ${good.error ?? ""}`);
	assert.equal(good.diffs.length, 1);
	assert.equal(good.diffs[0].name, "rel_mae");
	assert.equal(good.diffs[0].actual, TAMPER_EXPECTED);
	assert.equal(good.diffs[0].within, true);

	const dir = path.join(root, "artifacts", "E-0001");
	fs.writeFileSync(path.join(dir, "numbers.json"), `${JSON.stringify({ rel_mae: TAMPER_ACTUAL })}\n`);
	const bad = verifyReproduce(root, dir);
	assert.equal(bad.ok, false);
	assert.equal(bad.diffs[0].expected, TAMPER_EXPECTED);
	assert.equal(bad.diffs[0].actual, TAMPER_ACTUAL);
	assert.ok(Math.abs((bad.diffs[0].delta ?? 0) - (TAMPER_ACTUAL - TAMPER_EXPECTED)) < 1e-12);
	assert.equal(bad.diffs[0].within, false);
	assert.match(bad.drift, /rel_mae: expected=0\.0607 actual=0\.0907/);
	assert.match(bad.drift, /❌/);
});

test("verify_reproduce refuses a manifest with no repro_command or no numbers", () => {
	const root = scratchFixture();
	const dir = path.join(root, "artifacts", "E-0001");
	const manifest = JSON.parse(readText(path.join(dir, "manifest.json")));
	delete manifest.repro_command;
	fs.writeFileSync(path.join(dir, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
	const result = verifyReproduce(root, dir);
	assert.equal(result.ok, false);
	assert.match(result.error ?? "", /no repro_command/);
});

test("verify_artifact checks declared expected outputs", () => {
	const root = scratchFixture();
	const result = verifyArtifactAgainstExpected(root, path.join(root, "artifacts", "E-0001"), [
		"numbers.json",
		"missing.csv",
	]);
	assert.equal(result.ok, false);
	assert.deepEqual(result.missing_expected, ["missing.csv"]);
});

test("verify_traceability finds the dangling claim and the uncited paper number", () => {
	const root = scratchFixture();
	const result = verifyTraceability(root, { requireAncestor: false });
	assert.equal(result.exists, true);
	assert.equal(result.ok, false);
	assert.ok(result.dangling.some((entry) => entry.claim === "C-002" && entry.artifact.includes("E-9999")));
	assert.ok(result.uncited.some((entry) => entry.number === "0.0758"));
	assert.ok(!result.uncited.some((entry) => entry.number === "0.0607"));
	assert.equal(result.claims, 2);
});

test("verify_guards flags a removed assertion and a protected paper write", () => {
	const root = fs.mkdtempSync(path.join(process.env.TMPDIR ?? "/tmp", "vangja-guards-"));
	fs.mkdirSync(path.join(root, "tests"), { recursive: true });
	fs.mkdirSync(path.join(root, "paper"), { recursive: true });
	fs.writeFileSync(path.join(root, "tests", "test_fit.py"), "def test_a():\n    assert 1 == 1\n    assert 2 == 2\n");
	fs.writeFileSync(path.join(root, "paper", "paper.tex"), "old\n");
	const git = (...args: string[]) => execFileSync("git", ["-C", root, ...args], { encoding: "utf8" });
	git("init", "-q");
	git("config", "user.email", "audit@example.invalid");
	git("config", "user.name", "audit");
	git("add", ".");
	git("commit", "-qm", "baseline");
	fs.writeFileSync(path.join(root, "tests", "test_fit.py"), "def test_a():\n    assert 1 == 1\n");
	fs.writeFileSync(path.join(root, "paper", "paper.tex"), "new number 0.5\n");
	git("add", ".");
	git("commit", "-qm", "weaken test and change paper");

	const result = verifyGuards(root, { range: "HEAD~1..HEAD", runner: defaultRunner });
	assert.equal(result.ok, false);
	assert.ok(result.protected_violations.some((entry) => entry.includes("paper/paper.tex")));
	assert.equal(result.test_diff.ok, false);
	assert.match(result.test_diff.problems.join(" "), /removed 1 more assertion/);
});

test("run_gates reports all thirteen gates and writes gates.json", () => {
	const root = scratchFixture();
	const gates = runGates(root, { cycle: 2, snapshot: "fixture@sha" });
	assert.equal(gates.schema, "vangja-gates/1");
	assert.equal(gates.gates.length, GATE_ORDER.length);
	assert.deepEqual(
		gates.gates.map((gate) => gate.id),
		[...GATE_ORDER],
	);
	for (const gate of gates.gates) {
		assert.ok(["pass", "fail", "skip", "error"].includes(gate.status), `${gate.id}: ${gate.status}`);
		assert.ok(gate.detail.length > 0, `${gate.id} has no detail`);
	}
	const byId = new Map(gates.gates.map((gate) => [gate.id, gate]));
	assert.equal(byId.get("tests")?.status, "pass");
	assert.equal(byId.get("protocol-hashes")?.status, "pass");
	assert.equal(byId.get("artifacts-complete")?.status, "pass");
	assert.equal(gates.hash_ok, true);
	assert.equal(byId.get("confirmation-frozen")?.status, "fail");
	assert.equal(byId.get("no-test-selection")?.status, "fail");
	assert.equal(byId.get("test-diff")?.status, "fail");
	assert.equal(byId.get("no-failures-hidden")?.status, "fail");
	assert.equal(byId.get("context-excluded")?.status, "fail");
	assert.equal(byId.get("traceability")?.status, "fail");
	assert.equal(byId.get("canary")?.status, "pass");
	const file = path.join(root, ".pi", "audit", "cycle-2", "gates.json");
	assert.ok(fs.existsSync(file));
	const onDisk = JSON.parse(readText(file));
	assert.equal(onDisk.gates.length, GATE_ORDER.length);
});

test("run_gates fails artifacts-complete and reproduce-headline on a tampered artifact", () => {
	const root = scratchFixture();
	const dir = path.join(root, "artifacts", "E-0001");
	fs.writeFileSync(path.join(dir, "numbers.json"), `${JSON.stringify({ rel_mae: TAMPER_ACTUAL })}\n`);
	const gates = runGates(root, { cycle: 3, snapshot: "fixture@sha" });
	const artifacts = gates.gates.find((gate) => gate.id === "artifacts-complete");
	assert.equal(artifacts?.status, "fail");
	assert.equal(artifacts?.hash_ok, false);
	assert.equal(gates.hash_ok, false);
	const reproduce = gates.gates.find((gate) => gate.id === "reproduce-headline");
	assert.equal(reproduce?.status, "fail");
	assert.match(reproduce?.output_tail ?? "", /rel_mae: expected=0\.0607 actual=0\.0907/);
});

test("a skipped gate reports why instead of passing silently", () => {
	const root = scratchFixture();
	const gates = runGates(root, { cycle: 4, snapshot: "fixture@sha", skip: ["tests"] });
	const tests = gates.gates.find((gate) => gate.id === "tests");
	assert.equal(tests?.status, "skip");
	assert.match(tests?.detail ?? "", /skipped/);
});
