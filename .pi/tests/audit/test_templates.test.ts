/**
 * Audit templates and finalised agent definitions (plan/07_AUDIT_HARNESS.md §3, §8).
 */

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { loadAgentsManifest } from "../../lib/config.ts";
import { loadToolRegistry, expandTools } from "../../lib/toolset.ts";
import { readText, REPO_ROOT, SEEDED_FIXTURE } from "./helpers.ts";

const TEMPLATES = path.join(REPO_ROOT, ".pi", "audit", "_templates");
const WORKSTREAMS = ["M", "S", "P", "D", "I", "R", "H", "E", "C", "L"];

/**
 * A fixture that is present in the working tree but ignored by .gitignore is
 * invisible in a fresh clone — the S6 fixture's `data/` and `paper/` files were
 * exactly that, and a fresh clone's audit tests failed (found by S7 acceptance
 * test 6). Every fixture file must be tracked.
 */
test("every seeded-fixture file is committed (a fresh clone is complete)", () => {
	const files: string[] = [];
	const walk = (dir: string): void => {
		for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
			const full = path.join(dir, entry.name);
			if (entry.isDirectory()) walk(full);
			else if (entry.isFile()) files.push(path.relative(REPO_ROOT, full));
		}
	};
	walk(SEEDED_FIXTURE);
	assert.ok(files.length >= 20, `expected the full fixture, found ${files.length} file(s)`);
	const untracked: string[] = [];
	for (const file of files) {
		try {
			execFileSync("git", ["ls-files", "--error-unmatch", "--", file], { cwd: REPO_ROOT, encoding: "utf8" });
		} catch {
			untracked.push(file);
		}
	}
	assert.deepEqual(untracked, [], `fixture files missing from the index (ignored or untracked): ${untracked.join(", ")}`);
});

test("every required audit template exists", () => {
	const required = [
		"SCOPE.md",
		"_PROTOCOL.md",
		"VERIFY.md",
		"FINAL_REPORT.md",
		"ADJUDICATED_REPORT.md",
		"ROOT_ISSUE_REGISTER.csv",
		"QUESTIONS_FOR_AUTHORS.md",
		"GRADE.md",
		"META_COVERAGE.md",
		"CALIBRATION_BRIEF.md",
		"verify/VERIFY_{{ws}}.md",
	];
	for (const file of required) {
		assert.ok(fs.existsSync(path.join(TEMPLATES, file)), `missing template ${file}`);
	}
});

test("all ten workstream briefs exist with the hard boundary, checklist, and 25-finding cap", () => {
	for (const ws of WORKSTREAMS) {
		const file = path.join(TEMPLATES, ws, "BRIEF.md");
		assert.ok(fs.existsSync(file), `missing WS-${ws} brief`);
		const text = readText(file);
		assert.match(text, new RegExp(`# WS-${ws}`));
		assert.match(text, /## Boundary \(hard\)/);
		assert.match(text, /## Coverage checklist/);
		assert.match(text, /- \[ \]/);
		assert.match(text, /At most \*\*25 findings\*\*/);
		assert.match(text, /No command → not a finding/);
		assert.match(text, /nine fields|nine-field/i);
		assert.match(text, /## Stop conditions/);
		assert.match(text, /FINDINGS\.md/);
		assert.match(text, /findings\.json/);
		assert.match(text, /\{\{cycle\}\}/);
		assert.match(text, /cross-ref/);
	}
});

test("the protocol encodes the four phases, independence, and append-only corrections", () => {
	const text = readText(path.join(TEMPLATES, "_PROTOCOL.md"));
	for (const phrase of [
		"## 1. Phases",
		"## 2. Finding schema",
		"## 3. Independence",
		"## 4. Anti-flooding",
		"## 6. Append-only within a cycle",
		"No command → not a finding",
		"severity (S0|S1|S2|S3)",
		"confidence (high|medium|low)",
		"status",
		"verified|to-verify|refuted|unresolved",
		"cross-ref",
		"25 findings",
	]) {
		assert.ok(text.includes(phrase), `protocol missing "${phrase}"`);
	}
});

test("SCOPE.md pins the snapshot, artifact index, protocol hashes, and out-of-scope paths", () => {
	const text = readText(path.join(TEMPLATES, "SCOPE.md"));
	for (const phrase of [
		"## Snapshot",
		"Audited commit",
		"paper/CLAIMS.json",
		"## Frozen protocol and environment",
		"CONFIRMATION_ORIGINS.csv",
		"## Artifact index",
		"Manifest SHA-256",
		"## Loop ledgers",
		"Vangja-project/**",
		"## What is explicitly out of scope",
	]) {
		assert.ok(text.includes(phrase), `SCOPE.md missing "${phrase}"`);
	}
});

test("the verification, synthesis, adjudication, and grade templates carry the required sections", () => {
	const verify = readText(path.join(TEMPLATES, "VERIFY.md"));
	assert.match(verify, /Default hypothesis: the finding is wrong/);
	assert.match(verify, /Re-opened location/);
	assert.match(verify, /Refutation attempts/);
	assert.match(verify, /Synthesizer may cite/);

	const final = readText(path.join(TEMPLATES, "FINAL_REPORT.md"));
	for (const phrase of ["## Deterministic gates", "## Confirmed findings", "## Coverage and limitations", "## Residual risk"]) {
		assert.ok(final.includes(phrase), `FINAL_REPORT missing ${phrase}`);
	}

	const adjudicated = readText(path.join(TEMPLATES, "ADJUDICATED_REPORT.md"));
	for (const phrase of ["## Disputes", "## Extremes checked", "## Final grade trace", "## Root issues", "## Residual risk"]) {
		assert.ok(adjudicated.includes(phrase), `ADJUDICATED_REPORT missing ${phrase}`);
	}

	const grade = readText(path.join(TEMPLATES, "GRADE.md"));
	for (const phrase of ["## Rule trace", "blocking-s0", "blocking-core-s1", "failed-gate", "## Failing gates", "## Fix list"]) {
		assert.ok(grade.includes(phrase), `GRADE missing ${phrase}`);
	}

	const questions = readText(path.join(TEMPLATES, "QUESTIONS_FOR_AUTHORS.md"));
	assert.match(questions, /\[blocking\]/);
	assert.match(questions, /\[non-blocking\]/);
});

test("the finder definition forbids reading other workstreams and the historical audit", () => {
	const text = readText(path.join(REPO_ROOT, ".pi", "agents", "audit_finder.md"));
	assert.match(text, /may not read\s*\n?\s*another workstream's/i);
	assert.match(text, /Never copy the historical audit/);
	assert.match(text, /No command → not a finding/);
	assert.match(text, /25 findings/);
	assert.match(text, /phase separation/i);
});

test("the verifier reopens lines and re-runs commands; the synthesizer is verified-only; the adjudicator maps root issues", () => {
	const verifier = readText(path.join(REPO_ROOT, ".pi", "agents", "audit_verifier.md"));
	assert.match(verifier, /reopen the cited/);
	assert.match(verifier, /re-run the exact command/);
	assert.match(verifier, /Never synthesise or grade/);

	const synthesizer = readText(path.join(REPO_ROOT, ".pi", "agents", "audit_synthesizer.md"));
	assert.match(synthesizer, /verified ids only/);
	assert.match(synthesizer, /No new primary inspection/);
	assert.match(synthesizer, /evidence pointer/);

	const adjudicator = readText(path.join(REPO_ROOT, ".pi", "agents", "audit_adjudicator.md"));
	assert.match(adjudicator, /Root-issue mapping/);
	assert.match(adjudicator, /code \| paper \| protocol \| docs \| env/);
	assert.match(adjudicator, /Never invent a resolution/);
});

test("every audit agent's tools resolve in the registry", () => {
	const registry = loadToolRegistry(REPO_ROOT);
	const manifest = loadAgentsManifest(REPO_ROOT);
	for (const agent of manifest.agents.filter((entry) => entry.name.startsWith("audit_") || entry.name === "meta_review")) {
		const { unknown } = expandTools(registry, agent.tools);
		assert.deepEqual(unknown, [], `${agent.name}: ${unknown.join(", ")}`);
	}
});
