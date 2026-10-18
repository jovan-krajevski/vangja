/**
 * Destructive-bash and shell-write policy.
 *
 * Conservative pattern matching, never shell execution: the guard refuses a
 * command it cannot rule safe, and every denial names an allowed alternative.
 * Splitting a command on `&&`, `||`, `;`, and `|` is deliberately simple — it
 * only needs to find an `rm`/`git` segment, not to parse shell semantics.
 */

import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { toRepoRelative, underPath } from "../../lib/paths.ts";
import type { LoopRole } from "../../lib/roles.ts";
import { describeFindings, scanForSecrets, secretPathDecision } from "../../lib/secrets.ts";
import { ALLOWED, deny, type GuardDecision } from "./decision.ts";
import { protectedPathDecision } from "./protected.ts";

// ---------------------------------------------------------------------------
// Destructive bash
// ---------------------------------------------------------------------------

export interface BashPolicyContext {
	root: string;
	cwd: string;
	currentBranch: string | null;
	worktreeDirty: boolean;
	sharedBranches?: string[];
	role: LoopRole;
	loopState: string | null;
}

const DEFAULT_SHARED_BRANCHES = ["update_pymc", "main", "master"];

/** Split a shell command into simple command segments. */
export function splitSegments(command: string): string[] {
	return command
		.split(/\n|&&|\|\||;|\|/)
		.map((segment) => segment.trim())
		.filter((segment) => segment !== "");
}

/** Quote-aware-ish tokenizer good enough for policy checks (never for execution). */
export function tokenize(segment: string): string[] {
	const tokens: string[] = [];
	const pattern = /"([^"]*)"|'([^']*)'|(\S+)/g;
	let match: RegExpExecArray | null;
	while ((match = pattern.exec(segment)) !== null) {
		tokens.push(match[1] ?? match[2] ?? match[3]);
	}
	return tokens;
}

const COMMAND_PREFIXES = new Set(["sudo", "doas", "command", "nice", "time", "env", "nohup", "xargs"]);

/** Index of the command word, skipping privilege/launcher prefixes. */
function commandIndex(tokens: string[]): number {
	let index = 0;
	while (index < tokens.length && COMMAND_PREFIXES.has(tokens[index])) index += 1;
	return index;
}

function isRmRecursiveForce(args: string[]): boolean {
	let recursive = false;
	let force = false;
	for (const arg of args) {
		if (arg === "--recursive") recursive = true;
		if (arg === "--force") force = true;
		if (arg.startsWith("-") && !arg.startsWith("--")) {
			if (arg.includes("r") || arg.includes("R")) recursive = true;
			if (arg.includes("f")) force = true;
		}
	}
	return recursive && force;
}

function rmTargetAllowed(target: string, ctx: BashPolicyContext): boolean {
	const { abs } = toRepoRelative(target, ctx.cwd, ctx.root);
	return underPath(abs, path.join(ctx.root, ".pi", "state")) || underPath(abs, path.join(ctx.root, ".pi", "artifacts"));
}

const SSH_PATH = /(?:^|[\s'"=:(])(?:~|\$HOME|\$\{HOME\}|\/home\/[^\s'":]+)\/\.ssh(?:\/|[\s'"]|$)/;
const WRITE_OP = /\b(?:tee|cp|mv|sed\s+-i|chmod|chown|ssh-keygen|install|rsync|truncate|dd|cat)\b|>>?/;

function mentionsSharedBranch(command: string, shared: string[]): boolean {
	return shared.some((branch) => new RegExp(`(^|[\\s/])${branch}(\\s|$|['"])`).test(command));
}

export function destructiveBashDecision(command: string, ctx: BashPolicyContext): GuardDecision {
	const shared = ctx.sharedBranches ?? DEFAULT_SHARED_BRANCHES;

	// 1. Force pushes and history rewrites.
	if (/\bgit\s+push\b[^\n]*?(?:--force(?:-with-lease)?\b|\s-f\b)/.test(command)) {
		return deny(
			"DESTRUCTIVE_BASH",
			`force-pushed history is forbidden: ${JSON.stringify(command)}`,
			"push a normal fast-forward update on a work/ branch; the supervisor merges into the base branch",
		);
	}
	if (/\bgit\s+filter-(?:branch|repo)\b/.test(command)) {
		return deny(
			"DESTRUCTIVE_BASH",
			`history rewriting is forbidden: ${JSON.stringify(command)}`,
			"add a new corrective commit; never rewrite published history",
		);
	}

	// 2. Rebase on shared branches.
	if (/\bgit\s+rebase\b/.test(command)) {
		const onSharedCurrent = ctx.currentBranch !== null && shared.includes(ctx.currentBranch);
		if (onSharedCurrent || mentionsSharedBranch(command, shared)) {
			return deny(
				"DESTRUCTIVE_BASH",
				`rebase refused on shared branch (current=${ctx.currentBranch ?? "unknown"}, shared=${shared.join(", ")}): ${JSON.stringify(command)}`,
				"rebase only work/T-### branches; use git merge for shared branches",
			);
		}
	}

	// 3. reset --hard with uncommitted work.
	if (/\bgit\s+reset\s+--hard\b/.test(command) && ctx.worktreeDirty) {
		return deny(
			"DESTRUCTIVE_BASH",
			`git reset --hard refused with uncommitted changes (git status is not clean)`,
			"commit or stash first (git stash push -m 'wip'), then reset",
		);
	}

	// 4. rm -rf outside the loop state/artifacts directories.
	for (const segment of splitSegments(command)) {
		const tokens = tokenize(segment);
		const index = commandIndex(tokens);
		if (tokens[index] !== "rm") continue;
		const args = tokens.slice(index + 1);
		const flagArgs = args.filter((t) => t.startsWith("-"));
		const targets = args.filter((t) => !t.startsWith("-"));
		if (!isRmRecursiveForce(flagArgs)) continue;
		for (const target of targets) {
			if (rmTargetAllowed(target, ctx)) continue;
			return deny(
				"DESTRUCTIVE_BASH",
				`recursive forced delete outside the allowed directories: rm -rf ${target}`,
				"delete only under .pi/state/ or .pi/artifacts/; move anything else aside with git mv and let a review authorize its removal",
			);
		}
	}

	// 5. Writes to ~/.ssh.
	if (SSH_PATH.test(command) && WRITE_OP.test(command)) {
		return deny(
			"DESTRUCTIVE_BASH",
			`refusing a write-like operation against ~/.ssh: ${JSON.stringify(command)}`,
			"manage keys manually outside the agent session; the loop reads key paths from .pi/config/hpc.json and never modifies them",
		);
	}

	// 6. curl | sh.
	if (/\b(?:curl|wget)\b[^\n|]*\|\s*(?:sudo\s+)?(?:ba|z|da|k)?sh\b/.test(command)) {
		return deny(
			"DESTRUCTIVE_BASH",
			`piping a download into a shell is forbidden: ${JSON.stringify(command)}`,
			"download to a file, inspect it, then run it explicitly (and never inside an experiment run)",
		);
	}

	// 7. Environment exfiltration.
	const envPipe = /\b(?:env|printenv|set)\b[^\n|]*\|\s*(?:curl|wget|nc|ncat|ssh|scp|rsync)\b/;
	if (envPipe.test(command) || /\b(?:curl|wget)\b[^\n]*\$\((?:\s*env|\s*printenv)/.test(command)) {
		return deny(
			"DESTRUCTIVE_BASH",
			`refusing to send the environment to the network: ${JSON.stringify(command)}`,
			"read the single variable you need; secrets come from the environment inside the process that needs them",
		);
	}

	// 8. World-writable permissions.
	if (/\bchmod\b[^\n]*\b777\b/.test(command)) {
		return deny(
			"DESTRUCTIVE_BASH",
			`chmod 777 is forbidden: ${JSON.stringify(command)}`,
			"grant the narrowest permission needed; if a container needs broader access, record an amendment",
		);
	}

	return ALLOWED;
}

// ---------------------------------------------------------------------------
// Write-like bash targeting protected paths
// ---------------------------------------------------------------------------

/** Best-effort extraction of paths a shell command would write to. */
export function extractWriteTargets(command: string): string[] {
	const targets: string[] = [];
	for (const segment of splitSegments(command)) {
		// Redirections: `> path`, `>> path`, `2> path`.
		const redirect = /(?:^|\s)\d?>>?\s*([^\s|&;]+)/g;
		let match: RegExpExecArray | null;
		while ((match = redirect.exec(segment)) !== null) targets.push(match[1]);

		const tokens = tokenize(segment);
		if (tokens.length === 0) continue;
		const index = commandIndex(tokens);
		const base = path.basename(tokens[index] ?? "");
		const args = tokens.slice(index + 1);
		if (base === "tee") {
			// tee takes the destination first: `tee [-a] file...`
			const destination = args.find((token) => !token.startsWith("-"));
			if (destination) targets.push(destination);
		}
		if (["cp", "mv", "install", "rsync", "truncate", "touch", "ln"].includes(base)) {
			// For cp/mv/install/rsync the destination is the last non-flag token.
			for (const token of [...args].reverse()) {
				if (!token.startsWith("-")) {
					targets.push(token);
					break;
				}
			}
		}
		if (base === "sed" && args.some((t) => t === "-i" || t.startsWith("-i"))) {
			for (const token of [...args].reverse()) {
				if (!token.startsWith("-") && !token.startsWith("s/") && !token.includes("/")) {
					targets.push(token);
					break;
				}
			}
		}
	}
	return targets.filter((target) => target !== "" && target !== "/dev/null");
}

export function bashWriteTargetDecision(command: string, ctx: BashPolicyContext): GuardDecision {
	for (const target of extractWriteTargets(command)) {
		if (target.startsWith("$") || target.startsWith("`")) continue; // cannot resolve variables
		const relative = toRepoRelative(target, ctx.cwd, ctx.root);
		if (relative.outsideRoot) continue;
		const decision = protectedPathDecision(relative.rel, {
			role: ctx.role,
			loopState: ctx.loopState,
		});
		if (!decision.allowed) {
			return deny(
				"PROTECTED_PATH",
				`command writes to protected path via shell redirection: ${relative.rel}`,
				`${decision.alternative} (guard detail: ${decision.reason})`,
			);
		}
	}
	return ALLOWED;
}

// ---------------------------------------------------------------------------
// Secret material in shell commands
// ---------------------------------------------------------------------------

const SECRET_ALTERNATIVE =
	"keep credentials outside the repository (an OS keyring or ~/.ssh); reference them through the environment and commit only examples with placeholders";

function secretPathProblem(candidate: string, ctx: BashPolicyContext, where: string): GuardDecision | null {
	const relative = toRepoRelative(candidate, ctx.cwd, ctx.root);
	const checked = relative.outsideRoot ? relative.abs : relative.rel;
	const decision = secretPathDecision(checked);
	if (!decision.blocked) return null;
	return deny(
		"SECRET",
		`${where} "${checked}": ${decision.reason ?? "matches a secret-bearing path pattern"}`,
		SECRET_ALTERNATIVE,
	);
}

function pathIsDirectory(candidate: string, ctx: BashPolicyContext): boolean {
	const { abs } = toRepoRelative(candidate, ctx.cwd, ctx.root);
	try {
		return fs.statSync(abs).isDirectory();
	} catch {
		return false;
	}
}

function pathIsFile(candidate: string, ctx: BashPolicyContext): boolean {
	const { abs } = toRepoRelative(candidate, ctx.cwd, ctx.root);
	try {
		return fs.statSync(abs).isFile();
	} catch {
		return false;
	}
}

/** Read-only git query; an empty list when the root is not a git repository. */
function gitLines(ctx: BashPolicyContext, args: string[]): string[] {
	try {
		const out = execFileSync("git", ["-C", ctx.root, ...args], {
			encoding: "utf8",
			stdio: ["ignore", "pipe", "ignore"],
		});
		return out
			.split("\n")
			.map((line) => line.trim())
			.filter((line) => line !== "");
	} catch {
		return [];
	}
}

/** Paths reported by `git status --porcelain --untracked-files=all` (renames resolved). */
function gitChangedPaths(ctx: BashPolicyContext): string[] {
	return gitLines(ctx, ["status", "--porcelain", "--untracked-files=all"]).map((line) => {
		const renamed = line.slice(3).split(" -> ");
		return renamed[renamed.length - 1];
	});
}

function gitAddCandidates(args: string[], ctx: BashPolicyContext): string[] {
	const explicit = args.filter((token) => !token.startsWith("-"));
	if (explicit.length === 0 || explicit.includes(".")) return gitChangedPaths(ctx);
	const out: string[] = [];
	for (const token of explicit) {
		if (pathIsDirectory(token, ctx)) out.push(...gitChangedPaths(ctx));
		else out.push(token);
	}
	return out;
}

function gitCommitCandidates(args: string[], ctx: BashPolicyContext): string[] {
	const out = gitLines(ctx, ["diff", "--cached", "--name-only"]);
	if (args.some((token) => token === "--all" || /^-[a-zA-Z]*a[a-zA-Z]*$/.test(token))) {
		out.push(...gitLines(ctx, ["diff", "--name-only"]));
	}
	// Explicit pathspecs exist as files; commit-message words do not.
	for (const token of args) {
		if (!token.startsWith("-") && pathIsFile(token, ctx)) out.push(token);
	}
	return out;
}

/**
 * D2: the shell is a second write path, so it gets the same secret rules as
 * `write`/`edit` — content scan, secret-bearing targets, and staging/committing.
 * Denials name the pattern and never echo the value.
 */
export function bashSecretDecision(command: string, ctx: BashPolicyContext): GuardDecision {
	// 1. Secret material anywhere in the command text (including heredoc bodies).
	const scan = scanForSecrets(command);
	if (scan.findings.length > 0) {
		return deny(
			"SECRET",
			`command text contains secret-like material (${describeFindings(scan.findings)}); the values are not echoed`,
			SECRET_ALTERNATIVE,
		);
	}

	// 2. Secret-bearing shell write targets.
	for (const target of extractWriteTargets(command)) {
		if (target.startsWith("$") || target.startsWith("`")) continue; // cannot resolve variables
		const problem = secretPathProblem(target, ctx, "command writes to a secret-bearing path");
		if (problem) return problem;
	}

	// 3. Staging/committing secret files (git is the commit path).
	for (const segment of splitSegments(command)) {
		const tokens = tokenize(segment);
		const index = commandIndex(tokens);
		if (tokens[index] !== "git") continue;
		const sub = tokens[index + 1];
		const args = tokens.slice(index + 2);
		const candidates =
			sub === "add" ? gitAddCandidates(args, ctx) : sub === "commit" ? gitCommitCandidates(args, ctx) : [];
		for (const candidate of candidates) {
			const problem = secretPathProblem(
				candidate,
				ctx,
				"refusing to stage or commit a secret-bearing path",
			);
			if (problem) return problem;
		}
	}

	return ALLOWED;
}
