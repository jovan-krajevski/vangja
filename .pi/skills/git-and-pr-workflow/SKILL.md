---
name: git-and-pr-workflow
description: Branch, commit, open, or review a Vangja work PR. Use when creating a work branch or worktree, writing a commit or PR body, filling the machine-readable PR footer for pr_open, scanning a diff for protected paths or test weakening before review, or resolving a merge conflict. Encodes the no-force-push and supervisor-merges-only rules.
---

# Git and PR workflow

The conductor owns worktrees; the agent works in one, opens a PR record, and
**stops**. Merging into `update_pymc` is a supervisor action (`plan/03_AGENTS.md`
§1.3), and the conductor performs it. Read `templates/PR.md` and
`templates/review.md` before writing either artifact.

## Hard rules

1. **Never commit to `update_pymc` directly.** A unit of work happens on a
   branch `work/<task-id>-<slug>` in a conductor-managed worktree under the state directory
   (managed by the conductor). `git status` on the base branch must stay clean.
2. **Atomic commits.** One logical change per commit; the message names the
   task (`T-###`) and the evidence. Do not mix formatting-only churn with
   behaviour changes. `black .`/`ruff check .` belong in their own commit if
   they touch untouched files.
3. **No history rewrite.** `git push --force`, `filter-branch`, `reset --hard`
   on a shared branch, and rebases of a branch under review are blocked by the
   guard and by policy. Fix forward.
4. **Supervisor merges only.** `phd_candidate` opens the PR and stops; it may
   not merge, and may not approve its own work.
5. **Protected paths need a recorded reason.** `Vangja-project/**` is
   read-only, `research_agent/plan/**` and `fixed_case_studies/protocol/**` are
   frozen, `paper/**` only in the PAPER state, `AGENTS.md` and `.pi/state/**`
   are owned by the conductor/supervisor. Work around them, or record an
   amendment; the enforced list is `.pi/extensions/guards/protected.ts`.
6. **Every PR carries the machine footer.** `pr_open` validates
   `task_id`, `head_sha`, `base_sha`, `branch`, `summary`; artifacts, tests,
   claims and HPC jobs are lists (empty is allowed but must be present in the
   rendered body).

## The PR contract (`plan/03_AGENTS.md` §3, `plan/05_EXTENSIONS_AND_TOOLS.md`)

A PR is accepted only when the body answers, with evidence:

- what changed and why (summary ≤ 5 lines);
- the task and its acceptance criteria, each with the command that proves it;
- artifacts (paths) and claims (IDs) this PR creates or updates;
- tests run, with the exact commands and observed results;
- HPC jobs referenced (experiment IDs), if any;
- risks and the residual uncertainty;
- the reviewer checklist (`templates/review.md`).

Render it and the footer in one step:

```bash
python .pi/skills/git-and-pr-workflow/scripts/make_pr_record.py \
  --task-id T-012 --branch work/T-012-seeds \
  --base-sha "$(git merge-base HEAD update_pymc)" \
  --head-sha "$(git rev-parse HEAD)" \
  --summary "transfer regression: slope potential sign" \
  --artifact .pi/artifacts/E-0007/manifest.json \
  --test "uv run pytest tests/test_regularization.py -q -> 42 passed" \
  --claim C-003 --risks "ADVI source used for the smoke run only" \
  --out scratch/PR-003
```

Scan the diff **before** opening the PR:

```bash
git diff -U0 update_pymc...HEAD > scratch/change.patch
python .pi/skills/git-and-pr-workflow/scripts/diff_scan.py \
  --diff scratch/change.patch --repo . --out scratch/PR-003
```

`diff_scan.py` refuses (exit 1) on protected-path writes, deleted or weakened
tests, frozen-config edits without a justification, and anything the
`scientific-integrity-and-anti-gaming` scanner flags in the same patch. A hit
is a lead, not a verdict — triage each one in the PR body (`--allow
<rule>:<path>` with a reason writes an explicit waiver into the report).

## Merge-conflict policy

- Conflicts in generated or result files: regenerate, do not hand-merge.
- Conflicts in frozen protocol/config files: stop and record an amendment
  (the value must not change silently between dev and confirmation).
- Conflicts in tests: never resolve by deleting the stronger side; bring both
  expectations to the supervisor.
- After any conflict resolution, re-run the narrowest test and re-render the
  PR record with the new `head_sha`.

## Failure modes

- PR footer missing `base_sha` → `pr_open` refuses; use
  `git merge-base HEAD update_pymc`.
- Worktree on the wrong commit → the HPC record and artifact manifests will
  disagree with the PR; re-create the worktree rather than "fixing" the SHA.
- `diff_scan.py` says `scan_gaming: not run` → the scanner path moved; pass
  `--skills-dir .pi/skills` and fix the skill, do not skip the scan.
- Someone force-pushed the branch under review → the recorded `head_sha` is
  gone; open a new review, never rewrite the review history.
