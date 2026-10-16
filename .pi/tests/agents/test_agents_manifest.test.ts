/**
 * Agent definitions: every `.pi/agents/*.md` parses, matches the roster, names
 * real skills, stays under the 120-line contract, and carries the role
 * prohibitions the plan requires. Also pins the dispatch allowlist (plan/05 §7).
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { loadAgentsManifest, loadModelPin } from "../../lib/config.ts";
import {
	DISPATCH_ALLOWLIST,
	PRIMARY_AGENTS,
	availableAgentNames,
	loadAgentDefinitions,
	missingAgentError,
	parseAgentMarkdown,
	resolveDispatchTarget,
} from "../../lib/agents.ts";
import { roleGate } from "../../lib/roles.ts";
import { loadToolRegistry, expandTools } from "../../lib/toolset.ts";
import { PI_DIR, REPO_ROOT } from "../helpers.ts";

const manifest = loadAgentsManifest(REPO_ROOT);
const definitions = loadAgentDefinitions(REPO_ROOT);
const pin = loadModelPin(REPO_ROOT);
const registry = loadToolRegistry(REPO_ROOT);

const REQUIRED_SECTIONS = [
	"## Purpose",
	"## Inputs",
	"## Procedure",
	"## Output contract",
	"## Prohibitions",
	"## Stop conditions",
	"## Skills to load",
];

const NINE_FIELDS = [
	"location",
	"snapshot",
	"command",
	"expected",
	"observed",
	"consequence",
	"severity",
	"confidence",
	"status",
];

const AUDIT_AGENTS = ["audit_finder", "audit_verifier", "audit_synthesizer", "audit_adjudicator", "meta_review"];

test("every manifest agent has exactly one definition file and vice versa", () => {
	// The S3 acceptance line says "17 agent files"; the roster in
	// .pi/config/agents.json and plan/03_AGENTS.md define 16 (3 primary + 5 audit
	// + 8 workers). The test asserts file-vs-manifest equality, which is the
	// invariant that matters; the count discrepancy is recorded in S3's report.
	assert.equal(definitions.length, manifest.agents.length, "one file per roster entry");
	assert.deepEqual(
		definitions.map((agent) => agent.name).sort(),
		manifest.agents.map((agent) => agent.name).sort(),
	);
	for (const agent of manifest.agents) {
		assert.ok(fs.existsSync(path.join(REPO_ROOT, agent.file)), `${agent.file} is missing`);
	}
});

test("frontmatter matches the pin and the roster", () => {
	const pinnedId = `${pin.provider}/${pin.model}`;
	for (const definition of definitions) {
		const entry = manifest.agents.find((agent) => agent.name === definition.name);
		assert.ok(entry, `${definition.name} is not in agents.json`);
		assert.equal(definition.model, pinnedId, `${definition.name} must pin the canonical model`);
		assert.equal(entry.model, pinnedId);
		assert.equal(entry.role, definition.name, `${definition.name}.role must be its own role`);
		assert.equal(entry.file, definition.file);
		assert.deepEqual([...definition.tools].sort(), [...entry.tools].sort(), `${definition.name} tools`);
		assert.deepEqual([...skillsOf(definition)].sort(), [...entry.skills].sort(), `${definition.name} skills`);
		assert.ok(definition.description.length >= 20, `${definition.name} description too short`);
	}
});

function skillsOf(definition: { body: string }): string[] {
	return [...definition.body.matchAll(/`([a-z0-9-]+)`/g)]
		.map((match) => match[1])
		.filter((name) => fs.existsSync(path.join(REPO_ROOT, ".pi", "skills", name)));
}

test("bodies stay under 120 lines and keep the mandated sections", () => {
	for (const definition of definitions) {
		assert.ok(definition.lines <= 120, `${definition.name} body is ${definition.lines} lines (cap 120)`);
		for (const section of REQUIRED_SECTIONS) {
			assert.ok(definition.body.includes(section), `${definition.name} is missing "${section}"`);
		}
	}
});

test("every agent names the skills it should load, and the skill dirs exist", () => {
	for (const entry of manifest.agents) {
		const definition = definitions.find((agent) => agent.name === entry.name);
		assert.ok(definition, entry.name);
		for (const skill of entry.skills) {
			assert.ok(
				fs.existsSync(path.join(REPO_ROOT, ".pi", "skills", skill, "SKILL.md")),
				`skill ${skill} (named by ${entry.name}) has no SKILL.md`,
			);
			assert.ok(definition.body.includes(skill), `${entry.name} must name the skill "${skill}"`);
		}
	}
});

test("primaries carry their hard prohibitions", () => {
	const byName = new Map(definitions.map((agent) => [agent.name, agent.body]));
	const phd = byName.get("phd_candidate") ?? "";
	assert.match(phd, /Never merge/i, "phd_candidate must refuse to merge");
	assert.match(phd, /second HPC job/i, "phd_candidate must refuse a second concurrent job");
	assert.match(phd, /author ≠ reviewer|author != reviewer/i);

	const collector = byName.get("collector") ?? "";
	assert.match(collector, /Never touch code/i, "collector must refuse to touch code");

	const supervisor = byName.get("supervisor") ?? "";
	assert.match(supervisor, /run (?:the command|every acceptance criterion) yourself/i, "supervisor runs commands itself");
	assert.match(supervisor, /Never review work you authored/i);
});

test("audit agents use the nine-field schema and never copy the historical audit", () => {
	for (const name of AUDIT_AGENTS) {
		const body = definitions.find((agent) => agent.name === name)?.body ?? "";
		for (const field of NINE_FIELDS) {
			assert.ok(body.includes(field), `${name} must name the nine-field "${field}"`);
		}
		assert.match(body, /historical audit/i, `${name} must address the historical audit explicitly`);
		assert.match(body, /never copy|not a source|not evidence|never use/i, `${name} must forbid relying on the historical audit`);
	}
});

test("dispatch allowlist: only phd_candidate spawns experimenter; only audit roles spawn audit workers", () => {
	assert.ok(resolveDispatchTarget(REPO_ROOT, "phd_candidate", "experimenter").definition);
	assert.ok(resolveDispatchTarget(REPO_ROOT, "phd_candidate", "scout").definition);
	assert.ok(resolveDispatchTarget(REPO_ROOT, "supervisor", "critic").definition);
	assert.ok(resolveDispatchTarget(REPO_ROOT, "audit_synthesizer", "meta_review").definition);
	assert.ok(resolveDispatchTarget(REPO_ROOT, "audit_finder", "scout").definition);

	assert.throws(() => resolveDispatchTarget(REPO_ROOT, "phd_candidate", "audit_finder"), /may not dispatch/);
	assert.throws(() => resolveDispatchTarget(REPO_ROOT, "supervisor", "meta_review"), /may not dispatch/);
	assert.throws(() => resolveDispatchTarget(REPO_ROOT, "collector", "scout"), /may not dispatch/);
	assert.throws(() => resolveDispatchTarget(REPO_ROOT, "scout", "scout"), /may not dispatch/);
	for (const primary of PRIMARY_AGENTS) {
		assert.throws(() => resolveDispatchTarget(REPO_ROOT, "phd_candidate", primary), /primary agent/);
	}
	for (const role of ["collector", "scout", "planner", "methodologist", "experimenter"] as const) {
		assert.equal(DISPATCH_ALLOWLIST[role], undefined, `${role} must not dispatch`);
	}
});

test("the guard role matrix agrees with the dispatch boundary", () => {
	assert.equal(roleGate("dispatch", "phd_candidate").allowed, true);
	assert.equal(roleGate("dispatch", "supervisor").allowed, true);
	assert.equal(roleGate("dispatch", "collector").allowed, false);
	assert.equal(roleGate("merge_pr", "phd_candidate").allowed, false);
	assert.equal(roleGate("merge_pr", "supervisor").allowed, true);
	assert.equal(roleGate("hpc_submit", "collector").allowed, false);
});

test("an unknown agent is an explicit error listing the available agents", () => {
	assert.equal(availableAgentNames(REPO_ROOT).length, definitions.length);
	const error = missingAgentError(REPO_ROOT, "no_such_agent");
	assert.match(error.message, /unknown agent "no_such_agent"/);
	assert.match(error.message, /scout/);
	assert.throws(() => resolveDispatchTarget(REPO_ROOT, "phd_candidate", "no_such_agent"), /unknown agent/);
});

test("malformed agent files are rejected, not ignored", () => {
	const body = "\n## Purpose\n\nDo work.\n";
	assert.throws(() => parseAgentMarkdown(`---\nname: bad\ndescription: x\ntools: read\n---\n${body}`, "bad.md"), /missing frontmatter key "model"/);
	assert.throws(
		() => parseAgentMarkdown(`---\nname: bad\ndescription: x\ntools: read\nmodel: m\nthinking: max\n---\n${body}`, "bad.md"),
		/unknown frontmatter key "thinking"/,
	);
	assert.throws(() => parseAgentMarkdown(`name: no_fence\n`, "bad.md"), /must start with a "---"/);
	assert.throws(
		() => parseAgentMarkdown(`---\nname: bad\ndescription: x\ntools: read\nmodel: m\n---\n`, "bad.md"),
		/body must not be empty/,
	);
});

test("agent frontmatter tools are known registry names", () => {
	for (const definition of definitions) {
		const { unknown } = expandTools(registry, definition.tools);
		assert.deepEqual(unknown, [], `${definition.name} names unknown tools: ${unknown.join(", ")}`);
	}
});

test("registry file path is the versioned one", () => {
	assert.ok(fs.existsSync(path.join(PI_DIR, "config", "tools.json")));
});
