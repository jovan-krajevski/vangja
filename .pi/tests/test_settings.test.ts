/**
 * Settings load: `.pi/settings.json` must pin the model, thinking level,
 * enabled models, tools, and compaction exactly as plan/02_ARCHITECTURE.md §7
 * requires, and every top-level key must exist in pi's own settings schema.
 */

import assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { test } from "node:test";
import { findPiPackage, PI_DIR, readRepoConfig, REPO_ROOT } from "./helpers.ts";

const settingsPath = path.join(PI_DIR, "settings.json");

test("settings.json parses and pins the model", () => {
	const raw = JSON.parse(fs.readFileSync(settingsPath, "utf8")) as Record<string, unknown>;
	assert.equal(raw.$schema, "https://pi.dev/schemas/settings.schema.json");
	assert.equal(raw.defaultProvider, "opencode-go");
	assert.equal(raw.defaultModel, "deepseek-v4.1-flash");
	assert.equal(raw.defaultThinkingLevel, "max");
	assert.deepEqual(raw.enabledModels, ["opencode-go/deepseek-v4.1-flash"]);
	assert.deepEqual(raw.modelThinkingLevels, { "opencode-go/deepseek-v4.1-flash": "max" });
});

test("settings.json enables codemode on top of the default tools", () => {
	const raw = JSON.parse(fs.readFileSync(settingsPath, "utf8")) as Record<string, unknown>;
	assert.deepEqual(raw.defaultTools, ["+codemode"]);
});

test("compaction keeps a large recent history and reserves the response budget", () => {
	const raw = JSON.parse(fs.readFileSync(settingsPath, "utf8")) as {
		compaction?: { reserveTokens?: number; keepRecentTokens?: number };
	};
	assert.equal(raw.compaction?.reserveTokens, 32768);
	assert.equal(raw.compaction?.keepRecentTokens, 120000);
});

test("settings.json uses conventional discovery (no explicit resource lists)", () => {
	const raw = JSON.parse(fs.readFileSync(settingsPath, "utf8")) as Record<string, unknown>;
	for (const key of ["extensions", "skills", "prompts", "themes", "packages"]) {
		assert.equal(raw[key], undefined, `${key} must not be set; .pi/{extensions,skills,prompts} are conventional`);
	}
});

test("every top-level settings key is known to pi's settings schema", (t) => {
	const pkgDir = findPiPackage();
	if (!pkgDir) {
		t.skip("pi package directory not found; run `make -C .pi test` with pi on PATH");
		return;
	}
	const schema = JSON.parse(fs.readFileSync(path.join(pkgDir, "schemas", "settings.schema.json"), "utf8")) as {
		properties: Record<string, unknown>;
	};
	const raw = JSON.parse(fs.readFileSync(settingsPath, "utf8")) as Record<string, unknown>;
	for (const key of Object.keys(raw)) {
		assert.ok(key in schema.properties, `settings key "${key}" is not in pi's settings schema`);
	}
});

test("the model pin file agrees with settings.json and the agents manifest", () => {
	const pin = JSON.parse(fs.readFileSync(path.join(PI_DIR, "config", "model.json"), "utf8")) as Record<string, unknown>;
	const manifest = readRepoConfig<{ pinnedModel: string; pinnedThinking: string; agents: Array<{ model: string }> }>(
		"agents.json",
	);
	assert.equal(pin.provider, "opencode-go");
	assert.equal(pin.model, "deepseek-v4.1-flash");
	assert.equal(pin.thinking, "max");
	assert.equal(pin.contextWindow, 1_000_000);
	assert.equal(pin.enforce, true);
	assert.equal(manifest.pinnedModel, `${pin.provider}/${pin.model}`);
	assert.equal(manifest.pinnedThinking, pin.thinking);
	for (const agent of manifest.agents) {
		assert.equal(agent.model, manifest.pinnedModel, `agent model must be the pinned model`);
	}
});

test("the repository root used by the tests is the real repo", () => {
	assert.ok(fs.existsSync(path.join(REPO_ROOT, ".pi", "settings.json")));
	assert.ok(fs.existsSync(path.join(REPO_ROOT, "AGENTS.md")));
});
