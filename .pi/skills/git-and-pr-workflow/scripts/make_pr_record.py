#!/usr/bin/env python3
"""Render a PR body + machine-readable footer for the ``pr_open`` ledger tool.

The footer is the exact object ``pr_open`` validates (plan/03_AGENTS.md §3):
``task_id``, ``head_sha``, ``base_sha``, ``branch``, ``summary`` plus the
``artifacts``/``tests``/``claims``/``hpc_jobs`` lists.

Usage:
  make_pr_record.py --task-id T-012 --branch work/T-012-seeds \
    --base-sha $(git merge-base HEAD update_pymc) --head-sha $(git rev-parse HEAD) \
    --summary "..." --artifact path --test "command -> result" --claim C-003 \
    --acceptance "criterion::command::observed" --out scratch/PR-003

Exit codes: 0 rendered, 1 validation failure, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
BRANCH_RE = re.compile(r"^work/[A-Za-z0-9][A-Za-z0-9._/-]*$")

def render(template: str, values: dict[str, str]) -> str:
    out = template
    for key, value in values.items():
        out = out.replace("{{" + key + "}}", value)
    return out


def _bullets(items: list[str], empty: str = "- none") -> str:
    if not items:
        return empty
    return "\n".join(f"- {item}" for item in items)


def _rows(acceptance: list[str]) -> str:
    if not acceptance:
        return "| _(none recorded)_ | | |"
    rows = []
    for item in acceptance:
        parts = [p.strip() for p in item.split("::")]
        parts += [""] * (3 - len(parts))
        rows.append("| " + " | ".join(parts[:3]) + " |")
    return "\n".join(rows)


def build(args) -> tuple[dict | None, list[str]]:
    errors: list[str] = []
    for field in ("task_id", "branch", "base_sha", "head_sha", "summary"):
        if not getattr(args, field):
            errors.append(f"--{field.replace('_', '-')} is required")
    if args.base_sha and not SHA_RE.match(args.base_sha):
        errors.append(f"--base-sha does not look like a git SHA: {args.base_sha!r}")
    if args.head_sha and not SHA_RE.match(args.head_sha):
        errors.append(f"--head-sha does not look like a git SHA: {args.head_sha!r}")
    if args.branch and not BRANCH_RE.match(args.branch):
        errors.append(
            f"--branch must be 'work/<task-id>-<slug>', got {args.branch!r}"
        )
    if args.base_sha and args.base_sha == args.head_sha:
        errors.append("--base-sha and --head-sha are identical: no change to review")
    if errors:
        return None, errors
    footer = {
        "task_id": args.task_id,
        "head_sha": args.head_sha,
        "base_sha": args.base_sha,
        "branch": args.branch,
        "summary": args.summary,
        "artifacts": list(args.artifact),
        "tests": list(args.test),
        "claims": list(args.claim),
        "hpc_jobs": list(args.hpc_job),
        "risks": args.risks,
        "status": "open",
    }
    return footer, []


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a PR body and the pr_open footer.",
        epilog=(
            "Example:\n"
            "  make_pr_record.py --task-id T-012 --branch work/T-012-seeds \\\n"
            "    --base-sha $(git merge-base HEAD update_pymc) \\\n"
            "    --head-sha $(git rev-parse HEAD) --summary 'fix slope sign' \\\n"
            "    --test 'pytest tests/test_regularization.py -> 42 passed' \\\n"
            "    --out scratch/PR-003\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--task-title", default="")
    parser.add_argument("--branch", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--test", action="append", default=[])
    parser.add_argument("--claim", action="append", default=[])
    parser.add_argument("--hpc-job", action="append", default=[])
    parser.add_argument("--acceptance", action="append", default=[],
                        help="'criterion::command::observed' (repeatable)")
    parser.add_argument("--risks", default="none recorded")
    parser.add_argument("--diff-scan", default="not run")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true", help="print the footer JSON")
    args = parser.parse_args()

    footer, errors = build(args)
    if errors:
        for error in errors:
            print(f"REFUSED: {error}", file=sys.stderr)
        return 1
    assert footer is not None

    template_path = Path(__file__).resolve().parent.parent / "templates" / "PR.md"
    template = template_path.read_text()
    body = render(
        template,
        {
            "TITLE": args.task_title or args.summary[:72],
            "SUMMARY": args.summary,
            "TASK_ID": args.task_id,
            "TASK_TITLE": args.task_title or "(see brief)",
            "BRANCH": args.branch,
            "BASE_SHA": args.base_sha,
            "HEAD_SHA": args.head_sha,
            "STATUS": "open",
            "ACCEPTANCE_ROWS": _rows(args.acceptance),
            "ARTIFACTS": _bullets(args.artifact),
            "CLAIMS": _bullets(args.claim),
            "TESTS": _bullets(args.test),
            "HPC_JOBS": _bullets(args.hpc_job),
            "DIFF_SCAN": args.diff_scan,
            "RISKS": args.risks,
            "FOOTER_JSON": json.dumps(footer, indent=2),
        },
    )
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "PR.md").write_text(body)
        (args.out / "footer.json").write_text(json.dumps(footer, indent=2) + "\n")
        print(f"wrote {args.out / 'PR.md'} and {args.out / 'footer.json'}")
    if args.json or not args.out:
        print(json.dumps(footer, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
