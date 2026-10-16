/**
 * Tool registry: `.pi/config/tools.json` is the single source of truth for
 * agent tools. This test cross-checks it against the extension sources so a
 * typo in a registry entry, or a tool registered but not declared, fails.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { loadAgentsManifest } from "../../lib/config.ts";
import { loadToolRegistry, expandTools, assertNoUnknownTools } from "../../lib/toolset.ts";
import { PI_DIR, REPO_ROOT } from "../helpers.ts";

const registry = loadToolRegistry(REPO_ROOT);
const manifest = loadAgentsManifest(REPO_ROOT);

/** `name: "<tool>"` declarations in every extension entry point. */
function registeredToolNames(): Set<string> {
	const entries = fs
		.readdirSync(path.join(PI_DIR, "extensions"), { withFileTypes: true })
		.filter((entry) => entry.isDirectory())
		.map((entry) => path.join(PI_DIR, "extensions", entry.name, "index.ts"))
		.filter((file) => fs.existsSync(file));
	const names = new Set<string>();
	for (const file of entries) {
		const source = fs.readFileSync(file, "utf8");
		for (const match of source.matchAll(/name:\s*"([a-z][a-z0-9_]*)"/g)) names.add(match[1]);
	}
	return names;
}

test("registry schema, unique names, and families point at concrete tools", () => {
	assert.equal(registry.schema, "pi-tools/1");
	assert.ok(registry.tools.length >= 40, `registry should cover every planned tool, has ${registry.tools.length}`);
	for (const [family, members] of Object.entries(registry.families)) {
		for (const member of members) {
			assert.ok(
				registry.tools.some((tool) => tool.name === member),
				`family ${family} names unknown tool ${member}`,
			);
		}
		assert.ok(!registry.tools.some((tool) => tool.name === family), `family ${family} collides with a tool`);
	}
});

test("every manifest tool resolves to concrete names", () => {
	for (const agent of manifest.agents) {
		const { names, unknown } = expandTools(registry, agent.tools);
		assert.deepEqual(unknown, [], `${agent.name} names unknown tools: ${unknown.join(", ")}`);
		assert.ok(names.length >= agent.tools.length, `${agent.name} expansion shrank`);
		for (const name of names) {
			assert.ok(
				registry.tools.some((tool) => tool.name === name),
				`${agent.name} expands to unregistered tool ${name}`,
			);
		}
	}
});

test("declared S0/S3 extension tools are actually registered; planned ones are not yet", () => {
	const registered = registeredToolNames();
	for (const tool of registry.tools) {
		if (tool.kind === "extension" && (tool.stage === "S0" || tool.stage === "S3")) {
			assert.ok(registered.has(tool.name), `registry declares ${tool.name} (${tool.stage}) but no extension registers it`);
		}
		if (tool.kind === "planned") {
			assert.ok(!registered.has(tool.name), `planned tool ${tool.name} is already registered — flip its kind to "extension"`);
		}
	}
	for (const name of registered) {
		assert.ok(
			registry.tools.some((tool) => tool.name === name),
			`extension registers "${name}" which is missing from .pi/config/tools.json`,
		);
	}
});

test("the dispatch family is exactly the three dispatch tools", () => {
	assert.deepEqual(registry.families.dispatch, ["dispatch_agent", "dispatch_parallel", "dispatch_chain"]);
	const scout = expandTools(registry, ["read", "grep", "find", "ls"]).names;
	assert.deepEqual(scout, ["read", "grep", "find", "ls"]);
	const phd = expandTools(registry, ["dispatch", "merge"]).names;
	assert.deepEqual(phd, ["dispatch_agent", "dispatch_parallel", "dispatch_chain", "merge_pr"]);
});

test("unknown tool names fail loudly", () => {
	const { unknown } = expandTools(registry, ["read", "not_a_real_tool"]);
	assert.deepEqual(unknown, ["not_a_real_tool"]);
	assert.throws(
		() => assertNoUnknownTools(registry, ["read", "not_a_real_tool"], 'agent "scout"'),
		/agent "scout" names unknown tool\(s\) not_a_real_tool/,
	);
});

test("merge and hpc tools are gated families, not free names", () => {
	assert.equal(registry.families.merge.length, 1);
	assert.equal(registry.families.merge[0], "merge_pr");
	assert.ok(registry.families.hpc.includes("hpc_submit"));
	assert.ok(registry.families.task.includes("task_enqueue"));
});
