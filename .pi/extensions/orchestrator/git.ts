/**
 * Git operations for the conductor (plan/02_ARCHITECTURE.md §2–3).
 *
 * Hard rules, encoded here and asserted by the tests:
 * - work happens in a worktree under `.pi/state/worktrees/<task_id>/` on branch
 *   `work/T-###-<slug>` branched from the run's base branch (project.json
 *   `baseBranch`; `@current` resolves to the checkout's branch at BOOT);
 * - the conductor never rebases and never force-pushes; `git push` does not exist
 *   in this module at all (a PR is a local ledger record unless GitHub is enabled);
 * - a merge is always `git merge --no-ff` with the review id in the message, run
 *   from a clean checkout of the base branch, and tagged `run/<run_id>/merge/<n>`;
 * - operating on a dirty checkout is refused, never "repaired".
 */

import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { appendRecord } from "../ledger/store.ts";
import { VangjaLoopError } from "../../lib/errors.ts";
import { statePath } from "../../lib/paths.ts";
import { sanitizeSegment } from "../../lib/agents.ts";

export interface GitResult {
	stdout: string;
	stderr: string;
	code: number;
}

export function git(root: string, args: string[], options: { allowFailure?: boolean } = {}): GitResult {
	try {
		const stdout = execFileSync("git", ["-C", root, ...args], {
			encoding: "utf8",
			stdio: ["ignore", "pipe", "pipe"],
		});
		return { stdout, stderr: "", code: 0 };
	} catch (error) {
		const failure = error as { status?: number; stdout?: string; stderr?: string; message?: string };
		if (!options.allowFailure) {
			throw new VangjaLoopError(
				"PATH_INVALID",
				`git ${args.join(" ")} failed in ${root}: (exit ${failure.status ?? "?"}) ` +
					`${(failure.stderr ?? failure.message ?? "").trim()}`,
			);
		}
		return {
			stdout: String(failure.stdout ?? ""),
			stderr: String(failure.stderr ?? failure.message ?? ""),
			code: failure.status ?? 1,
		};
	}
}

export function isGitRepo(root: string): boolean {
	return git(root, ["rev-parse", "--git-dir"], { allowFailure: true }).code === 0;
}

export function currentBranch(root: string): string | null {
	const result = git(root, ["rev-parse", "--abbrev-ref", "HEAD"], { allowFailure: true });
	return result.code === 0 ? result.stdout.trim() : null;
}

export function headSha(root: string, ref = "HEAD"): string {
	return git(root, ["rev-parse", ref]).stdout.trim();
}

export function branchExists(root: string, branch: string): boolean {
	return git(root, ["rev-parse", "--verify", `refs/heads/${branch}`], { allowFailure: true }).code === 0;
}

export function worktreeDirty(root: string, includeUntracked = false): boolean {
	const args = ["status", "--porcelain"];
	if (!includeUntracked) args.push("--untracked-files=no");
	return git(root, args).stdout.trim() !== "";
}

/** Refuse to operate on a dirty checkout (plan/05 §2: "git operations serialized"). */
export function assertCleanCheckout(root: string, why: string): void {
	if (!isGitRepo(root)) {
		throw new VangjaLoopError("PATH_INVALID", `${root} is not a git repository; cannot ${why}`);
	}
	if (worktreeDirty(root)) {
		throw new VangjaLoopError(
			"GUARD_DENIED",
			`refusing to ${why}: the checkout at ${root} has uncommitted changes. ` +
				`Commit or stash them first; the conductor never mutates or discards work in progress.`,
		);
	}
}

export function assertNotOnSharedHistoryRewrite(args: string[]): void {
	for (const arg of args) {
		if (["rebase", "filter-branch", "filter-repo"].includes(arg)) {
			throw new VangjaLoopError(
				"GUARD_DENIED",
				`refusing to run "git ${args.join(" ")}": the conductor never rewrites shared history (plan/02_ARCHITECTURE.md §5.1.6).`,
			);
		}
	}
}

export function worktreePath(root: string, taskId: string, worktreeDir = ".pi/state/worktrees"): string {
	return path.resolve(root, worktreeDir, sanitizeSegment(taskId));
}

export function branchForTask(taskId: string, slug: string): string {
	return `work/${sanitizeSegment(taskId)}-${sanitizeSegment(slug)}`;
}

export interface EnsureWorktreeOptions {
	taskId: string;
	slug: string;
	baseBranch: string;
	worktreeDir?: string;
}

export interface WorktreeInfo {
	path: string;
	branch: string;
	created: boolean;
}

/** Create (or reuse) the worktree for a task. Never touches the base branch. */
export function ensureWorktree(root: string, options: EnsureWorktreeOptions): WorktreeInfo {
	const dir = options.worktreeDir ?? ".pi/state/worktrees";
	const worktree = worktreePath(root, options.taskId, dir);
	const branch = branchForTask(options.taskId, options.slug);
	if (fs.existsSync(path.join(worktree, ".git"))) {
		const onBranch = currentBranch(worktree);
		if (onBranch !== branch) {
			throw new VangjaLoopError(
				"PATH_INVALID",
				`worktree ${worktree} exists on branch ${onBranch ?? "?"} but this task needs ${branch}; ` +
					`remove it deliberately (git worktree remove) or finish the other task first`,
			);
		}
		return { path: worktree, branch, created: false };
	}
	git(root, ["worktree", "prune"], { allowFailure: true });
	const args = ["worktree", "add"];
	if (branchExists(root, branch)) {
		args.push(worktree, branch);
	} else {
		args.push("-b", branch, worktree, options.baseBranch);
	}
	git(root, args);
	return { path: worktree, branch, created: true };
}

export function removeWorktree(root: string, worktree: string, force = false): void {
	const args = ["worktree", "remove"];
	if (force) args.push("--force");
	args.push(worktree);
	git(root, args, { allowFailure: true });
	git(root, ["worktree", "prune"], { allowFailure: true });
}

export interface CommitOptions {
	message: string;
	/** Explicit paths to stage; defaults to everything under `worktree/`. */
	files?: string[];
}

/** Stage and commit inside the worktree; returns the new HEAD SHA. */
export function commitAll(worktree: string, options: CommitOptions): string {
	if (options.files && options.files.length > 0) {
		git(worktree, ["add", "--", ...options.files]);
	} else {
		git(worktree, ["add", "-A"]);
	}
	const staged = git(worktree, ["diff", "--cached", "--name-only"]).stdout.trim();
	if (staged === "") {
		throw new VangjaLoopError("PATH_INVALID", `${worktree}: nothing staged to commit (${options.message})`);
	}
	git(worktree, ["commit", "-m", options.message]);
	return headSha(worktree);
}

export function changedFiles(root: string, base: string, head: string): string[] {
	return git(root, ["diff", "--name-only", base, head])
		.stdout.split("\n")
		.map((line) => line.trim())
		.filter((line) => line !== "");
}

export function diffText(root: string, base: string, head: string, paths: string[] = []): string {
	return git(root, ["diff", "-U3", base, head, ...(paths.length > 0 ? ["--", ...paths] : [])]).stdout;
}

export function changedTestFiles(root: string, base: string, head: string): string[] {
	return changedFiles(root, base, head).filter((file) => /(^|\/)(tests?|test_)/.test(file) || /_test\.(py|ts)$/.test(file) || /\.test\.ts$/.test(file));
}

export function isAncestor(root: string, ancestor: string, descendant: string): boolean {
	return git(root, ["merge-base", "--is-ancestor", ancestor, descendant], { allowFailure: true }).code === 0;
}

export interface MergeOptions {
	prId: string;
	reviewId: string;
	branch: string;
	baseBranch: string;
	summary: string;
	runId: string | null;
}

export interface MergeResult {
	sha: string;
	alreadyMerged: boolean;
	tag: string | null;
}

/** `merge --no-ff` from a clean checkout of the base branch, with the review id in the message. */
export function mergeNoFf(root: string, options: MergeOptions): MergeResult {
	assertCleanCheckout(root, `merge ${options.prId}`);
	assertNotOnSharedHistoryRewrite(["merge"]);
	if (!branchExists(root, options.branch)) {
		throw new VangjaLoopError(
			"PATH_INVALID",
			`refusing to merge ${options.prId}: branch ${options.branch} does not exist in ${root}`,
		);
	}
	const head = headSha(root, options.branch);
	if (isAncestor(root, head, options.baseBranch)) {
		return { sha: headSha(root, options.baseBranch), alreadyMerged: true, tag: null };
	}
	const previousBranch = currentBranch(root);
	if (previousBranch !== options.baseBranch) {
		git(root, ["checkout", options.baseBranch]);
	}
	try {
		const message = `merge ${options.prId} (review ${options.reviewId}): ${options.summary}`;
		git(root, ["merge", "--no-ff", options.branch, "-m", message]);
	} finally {
		if (previousBranch && previousBranch !== options.baseBranch) {
			git(root, ["checkout", previousBranch], { allowFailure: true });
		}
	}
	const sha = headSha(root, options.baseBranch);
	const tag = tagMerge(root, options.runId, options.prId, options.baseBranch);
	return { sha, alreadyMerged: false, tag };
}

export function tagPrefixes(): { merge: string; audit: string; submission: string } {
	return { merge: "run/", audit: "audit-cycle/", submission: "submission/" };
}

export function tagIndex(root: string, prefix: string): number {
	const result = git(root, ["tag", "--list", `${prefix}*`], { allowFailure: true });
	const names = result.stdout
		.split("\n")
		.map((line) => line.trim())
		.filter((line) => line !== "");
	return names.length;
}

/** Create an annotated tag; re-pointing an existing tag at a different SHA is refused. */
export function tagIfNew(root: string, name: string, message: string): boolean {
	const existing = git(root, ["rev-parse", "-q", "--verify", `refs/tags/${name}`], { allowFailure: true });
	const head = headSha(root);
	if (existing.code === 0) {
		if (existing.stdout.trim() === head) return false;
		throw new VangjaLoopError(
			"GUARD_DENIED",
			`refusing to move tag ${name} from ${existing.stdout.trim().slice(0, 12)} to ${head.slice(0, 12)}; ` +
				`tags are immutable evidence (use the next sequence number)`,
		);
	}
	git(root, ["tag", "-a", name, "-m", message]);
	return true;
}

export function tagMerge(root: string, runId: string | null, prId: string, baseBranch: string): string | null {
	const run = sanitizeSegment(runId ?? "no-run");
	const prefix = `run/${run}/merge/`;
	const index = tagIndex(root, prefix) + 1;
	const name = `${prefix}${index}`;
	tagIfNew(root, name, `merge ${prId} into ${baseBranch}`);
	return name;
}

export function tagAuditCycle(root: string, cycle: number): string {
	const name = `audit-cycle/${cycle}`;
	tagIfNew(root, name, `audit cycle ${cycle} snapshot`);
	return name;
}

export function tagSubmission(root: string): string {
	const index = tagIndex(root, "submission/") + 1;
	const name = `submission/${index}`;
	tagIfNew(root, name, `submission ${index}`);
	return name;
}

// ---------------------------------------------------------------------------
// PR records
// ---------------------------------------------------------------------------

export interface PrFooter {
	task_id: string;
	head_sha: string;
	base_sha: string;
	branch: string;
	summary: string;
	artifacts?: string[];
	tests?: string[];
	claims?: string[];
	hpc_jobs?: string[];
	risks?: string;
	kind?: string;
}

/** Record the PR in the ledger (the only PR mechanism unless GitHub is enabled). */
export function recordPr(root: string, footer: PrFooter, actor = "conductor"): Record<string, unknown> {
	for (const field of ["task_id", "head_sha", "base_sha", "branch", "summary"] as const) {
		if (typeof footer[field] !== "string" || footer[field].trim() === "") {
			throw new VangjaLoopError("LEDGER_INVALID", `PR footer is missing "${field}" (plan/03_AGENTS.md §3)`);
		}
	}
	return appendRecord(
		root,
		"pr",
		{
			...footer,
			artifacts: footer.artifacts ?? [],
			tests: footer.tests ?? [],
			claims: footer.claims ?? [],
			hpc_jobs: footer.hpc_jobs ?? [],
			status: "open",
		},
		{ root, role: "conductor", actor },
	);
}

/** The PR text file, so humans (and the supervisor) can read it without jq. */
export function writePrFile(root: string, prId: string, body: string): string {
	const file = statePath(root, path.join("prs", `${prId}.md`));
	fs.mkdirSync(path.dirname(file), { recursive: true });
	fs.writeFileSync(file, body.endsWith("\n") ? body : `${body}\n`, "utf8");
	return file;
}
