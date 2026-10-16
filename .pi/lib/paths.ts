/**
 * Repository- and state-path resolution.
 *
 * The environment lives at `<repo>/.pi`. Runtime state is `<repo>/.pi/state`.
 * `PI_LOOP_ROOT` overrides discovery for child processes and sandbox runs; it
 * must point at a directory that contains a `.pi` directory.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

export function dirOf(importMetaUrl: string): string {
	return path.dirname(fileURLToPath(importMetaUrl));
}

/** Walk up from `start` until a directory has both `.pi/` and `AGENTS.md`. */
export function findRepoRoot(start: string): string | null {
	let dir = path.resolve(start);
	for (;;) {
		if (fs.existsSync(path.join(dir, ".pi")) && fs.existsSync(path.join(dir, "AGENTS.md"))) {
			return dir;
		}
		const parent = path.dirname(dir);
		if (parent === dir) return null;
		dir = parent;
	}
}

export function resolveRepoRoot(explicit?: string): string {
	const fromEnv = explicit ?? process.env.PI_LOOP_ROOT;
	if (fromEnv && fromEnv.trim() !== "") {
		const abs = path.resolve(fromEnv);
		if (!fs.existsSync(path.join(abs, ".pi"))) {
			throw new Error(`PI_LOOP_ROOT=${abs} does not contain a .pi directory`);
		}
		return abs;
	}
	const fromCwd = findRepoRoot(process.cwd());
	if (fromCwd) return fromCwd;
	const fromModule = findRepoRoot(dirOf(import.meta.url));
	if (fromModule) return fromModule;
	throw new Error(
		"cannot locate the repository root (no directory containing .pi/ and AGENTS.md found from " +
			"the working directory or this module); set PI_LOOP_ROOT to the repository root",
	);
}

export function piDir(root: string): string {
	return path.join(root, ".pi");
}

export function configDir(root: string): string {
	return path.join(root, ".pi", "config");
}

export function configPath(root: string, name: string): string {
	return path.join(configDir(root), name);
}

export function stateDir(root: string): string {
	return path.join(root, ".pi", "state");
}

export function statePath(root: string, name: string): string {
	return path.join(stateDir(root), name);
}

export function ledgerPath(root: string, fileName: string): string {
	return statePath(root, fileName);
}

export function artifactDir(root: string): string {
	return path.join(root, ".pi", "artifacts");
}

/** Convert a possibly user-relative path into a repo-relative POSIX path. */
export function toRepoRelative(
	target: string,
	cwd: string,
	root: string,
): { abs: string; rel: string; outsideRoot: boolean } {
	const abs = path.resolve(cwd, target);
	const relNative = path.relative(root, abs);
	const outsideRoot = relNative === "" || relNative.startsWith("..") || path.isAbsolute(relNative);
	return { abs, rel: relNative.split(path.sep).join("/"), outsideRoot };
}

/** True when `rel` is exactly `prefix` or starts with `prefix + "/"`. */
export function underPath(rel: string, prefix: string): boolean {
	const clean = prefix.endsWith("/") ? prefix.slice(0, -1) : prefix;
	return rel === clean || rel.startsWith(`${clean}/`);
}
