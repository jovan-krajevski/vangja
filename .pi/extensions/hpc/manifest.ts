/**
 * Artifact collection: copy declared outputs and checkpoints, SHA-256 every
 * file, and write `.pi/artifacts/<experiment_id>/manifest.json` including the
 * `sacct` accounting fields (plan/06_HPC_AND_COLLECTOR.md §6.4).
 *
 * Fails loudly: a declared output that was not fetched is a `missing` entry and
 * an error, never a silently smaller manifest.
 *
 * No pi imports: usable from `node --test`.
 */

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";
import { writeJsonAtomic } from "../../lib/io.ts";
import { artifactDir } from "../../lib/paths.ts";

export interface HashedFile {
	path: string;
	sha256: string;
	size: number;
	mtime: string;
}

export function sha256File(file: string): string {
	const hash = crypto.createHash("sha256");
	hash.update(fs.readFileSync(file));
	return hash.digest("hex");
}

export function sha256Text(text: string): string {
	return crypto.createHash("sha256").update(text).digest("hex");
}

/** Relative POSIX paths of every regular file under `dir` (sorted). */
export function walkFiles(dir: string): string[] {
	if (!fs.existsSync(dir)) return [];
	const files: string[] = [];
	const walk = (current: string): void => {
		for (const entry of fs.readdirSync(current, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
			const full = path.join(current, entry.name);
			if (entry.isDirectory()) walk(full);
			else if (entry.isFile()) files.push(path.relative(dir, full).split(path.sep).join("/"));
		}
	};
	walk(dir);
	return files;
}

export function hashTree(dir: string): HashedFile[] {
	return walkFiles(dir).map((rel) => {
		const full = path.join(dir, rel);
		const stat = fs.statSync(full);
		return {
			path: rel,
			sha256: sha256File(full),
			size: stat.size,
			mtime: stat.mtime.toISOString(),
		};
	});
}

/** Which declared outputs are absent from the fetched set (glob-aware, POSIX). */
export function missingOutputs(expected: string[], files: HashedFile[]): string[] {
	const present = new Set(files.map((file) => file.path));
	const matches = (pattern: string): boolean => {
		if (present.has(pattern)) return true;
		if (!pattern.includes("*") && !pattern.includes("?") && !pattern.includes("[")) return false;
		const escaped = pattern
			.split("")
			.map((char) => (".+^${}()|[]\\".includes(char) ? `\\${char}` : char))
			.join("")
			.replace(/\\\*/g, "[^/]*")
			.replace(/\\\?/g, "[^/]");
		const regex = new RegExp(`^${escaped}$`);
		return [...present].some((file) => regex.test(file));
	};
	return expected.filter((pattern) => !matches(pattern));
}

export interface ArtifactManifest {
	schema: "pi-artifact/1";
	experiment_id: string;
	task_id: string;
	attempt: number;
	status: "collected" | "collection-failed" | "checkpoints-only";
	idempotencyKey: string;
	commit_sha: string;
	job_id: string | null;
	collected_at: string;
	collected_by: string;
	expectedOutputs: string[];
	files: HashedFile[];
	missing: string[];
	checkpoints: HashedFile[];
	sacct: Record<string, unknown> | null;
	manifest_sha256?: string;
}

export function artifactDirFor(root: string, experimentId: string): string {
	return path.join(artifactDir(root), experimentId);
}

export function manifestPath(root: string, experimentId: string): string {
	return path.join(artifactDirFor(root, experimentId), "manifest.json");
}

/**
 * Write the manifest atomically. The manifest's own SHA-256 is recorded in
 * `manifest_sha256` for the traceability check, computed over the JSON body
 * *without* that field.
 */
export function writeManifest(root: string, manifest: ArtifactManifest): { path: string; sha256: string } {
	const body: ArtifactManifest = { ...manifest };
	delete body.manifest_sha256;
	const serialized = `${JSON.stringify(body, null, 2)}\n`;
	const digest = sha256Text(serialized);
	const file = manifestPath(root, manifest.experiment_id);
	writeJsonAtomic(file, { ...body, manifest_sha256: digest });
	return { path: file, sha256: digest };
}

export function readManifest(root: string, experimentId: string): ArtifactManifest | null {
	const file = manifestPath(root, experimentId);
	if (!fs.existsSync(file)) return null;
	return JSON.parse(fs.readFileSync(file, "utf8")) as ArtifactManifest;
}

/** Parse the checkpoint progress marker: `{"version":1,"completed":[...]}`. */
export function parseProgress(text: string): { completed: string[]; version: number } | null {
	try {
		const parsed = JSON.parse(text) as Record<string, unknown>;
		const completed = Array.isArray(parsed.completed) ? parsed.completed.map(String) : null;
		if (completed === null) return null;
		return { completed, version: Number(parsed.version ?? 1) };
	} catch {
		return null;
	}
}

/** Compare two progress snapshots; `advanced` means new units completed. */
export function progressAdvanced(
	before: { completed: string[] } | null,
	after: { completed: string[] } | null,
): boolean {
	if (!after) return false;
	if (!before) return after.completed.length > 0;
	return after.completed.length > before.completed.length;
}
