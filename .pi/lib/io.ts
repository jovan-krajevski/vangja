/**
 * Durable, locked file IO for the loop ledgers.
 *
 * Guarantees:
 * - `withLock` takes an exclusive lock by `mkdir` (atomic on POSIX), retries
 *   with backoff, steals a lock older than `staleMs`, and throws
 *   LEDGER_LOCK_TIMEOUT with the lock path when it cannot proceed.
 * - `appendLineDurable` writes one line with a single `write()` on an O_APPEND
 *   descriptor and fsyncs it. Callers must hold the lock.
 * - `writeJsonAtomic`/`writeFileAtomic` write a sibling temp file, fsync it,
 *   rename it over the target, then fsync the directory (best effort).
 *
 * No dependencies; usable from `node` (tests) and from pi extensions (jiti).
 */

import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { lockTimeout } from "./errors.ts";

export interface LockOptions {
	timeoutMs?: number;
	staleMs?: number;
	retryMinMs?: number;
	retryMaxMs?: number;
}

const DEFAULT_LOCK: Required<LockOptions> = {
	timeoutMs: 30_000,
	staleMs: 15_000,
	retryMinMs: 3,
	retryMaxMs: 25,
};

export function ensureDir(dir: string): void {
	fs.mkdirSync(dir, { recursive: true });
}

function sleepSync(ms: number): void {
	const shared = new SharedArrayBuffer(4);
	Atomics.wait(new Int32Array(shared), 0, 0, Math.max(0, Math.ceil(ms)));
}

function randomDelay(min: number, max: number): number {
	const span = Math.max(1, max - min);
	return min + (crypto.randomInt(span * 1000) % (span * 1000)) / 1000;
}

/**
 * Run `fn` while holding the lock directory `lockPath`. The lock is released in
 * `finally`, also when `fn` throws.
 */
export function withLock<T>(lockPath: string, fn: () => T, options: LockOptions = {}): T {
	const opts = { ...DEFAULT_LOCK, ...options };
	ensureDir(path.dirname(lockPath));
	const deadline = Date.now() + opts.timeoutMs;

	for (;;) {
		try {
			fs.mkdirSync(lockPath);
			break;
		} catch (error) {
			const code = (error as NodeJS.ErrnoException).code;
			if (code !== "EEXIST") throw error;

			let stat: fs.Stats | undefined;
			try {
				stat = fs.statSync(lockPath);
			} catch {
				// Holder released between mkdir and stat: retry immediately.
				continue;
			}
			if (Date.now() - stat.mtimeMs > opts.staleMs) {
				// The owner is presumed dead. Record what we are removing.
				fs.rmSync(lockPath, { recursive: true, force: true });
				continue;
			}
			if (Date.now() >= deadline) throw lockTimeout(lockPath, opts.timeoutMs);
			sleepSync(randomDelay(opts.retryMinMs, opts.retryMaxMs));
		}
	}

	try {
		fs.writeFileSync(
			path.join(lockPath, "owner"),
			JSON.stringify({ pid: process.pid, host: os.hostname(), at: new Date().toISOString() }),
		);
	} catch {
		// Owner metadata is diagnostic only; the lock itself is the mkdir.
	}

	try {
		return fn();
	} finally {
		fs.rmSync(lockPath, { recursive: true, force: true });
	}
}

/** Append exactly one line (LF-terminated) and fsync. Caller holds the lock. */
export function appendLineDurable(file: string, line: string): void {
	ensureDir(path.dirname(file));
	const payload = line.endsWith("\n") ? line : `${line}\n`;
	const fd = fs.openSync(file, "a");
	try {
		let offset = 0;
		const buf = Buffer.from(payload, "utf8");
		while (offset < buf.length) {
			offset += fs.writeSync(fd, buf, offset, buf.length - offset);
		}
		fs.fsyncSync(fd);
	} finally {
		fs.closeSync(fd);
	}
}

export function writeFileAtomic(file: string, content: string): void {
	ensureDir(path.dirname(file));
	const tmp = path.join(
		path.dirname(file),
		`.${path.basename(file)}.tmp-${process.pid}-${crypto.randomBytes(4).toString("hex")}`,
	);
	const fd = fs.openSync(tmp, "w");
	try {
		fs.writeFileSync(fd, content);
		fs.fsyncSync(fd);
	} finally {
		fs.closeSync(fd);
	}
	fs.renameSync(tmp, file);
	fsyncDirBestEffort(path.dirname(file));
}

export function writeJsonAtomic(file: string, value: unknown): void {
	writeFileAtomic(file, `${JSON.stringify(value, null, 2)}\n`);
}

function fsyncDirBestEffort(dir: string): void {
	try {
		const fd = fs.openSync(dir, "r");
		try {
			fs.fsyncSync(fd);
		} finally {
			fs.closeSync(fd);
		}
	} catch {
		// Directory fsync is unsupported on some filesystems (and Windows).
		// The rename is still atomic; only crash-window durability is weaker.
	}
}

export function readJsonFile<T>(file: string): T | undefined {
	if (!fs.existsSync(file)) return undefined;
	const raw = fs.readFileSync(file, "utf8");
	try {
		return JSON.parse(raw) as T;
	} catch (error) {
		throw new Error(`cannot parse JSON in ${file}: ${(error as Error).message}`);
	}
}

export function readJsonlFile<T>(file: string): T[] {
	if (!fs.existsSync(file)) return [];
	const raw = fs.readFileSync(file, "utf8");
	const records: T[] = [];
	const lines = raw.split("\n");
	for (let i = 0; i < lines.length; i += 1) {
		const line = lines[i].trim();
		if (line === "") continue;
		try {
			records.push(JSON.parse(line) as T);
		} catch (error) {
			throw new Error(`cannot parse ${file} line ${i + 1}: ${(error as Error).message}`);
		}
	}
	return records;
}

export function countJsonlLines(file: string): number {
	return readJsonlFile<unknown>(file).length;
}
