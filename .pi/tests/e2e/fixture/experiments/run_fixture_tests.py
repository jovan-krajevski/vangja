#!/usr/bin/env python3
"""Run the synthetic fixture's tests from the worktree that carries them.

The pre-merge gate runs its configured test command in the base checkout, but
the task's test file lives on the PR branch inside a git worktree. This wrapper
(the smoke test's `testsCommand`) runs the newest worktree suite when one exists
and falls back to the checkout, so the gate really tests the PR head.

No network, no model calls.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def newest_worktree_tests() -> pathlib.Path | None:
    candidates = sorted(
        (path for path in ROOT.glob(".pi/state/worktrees/*/synthetic_fixture/tests") if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main() -> int:
    tests = newest_worktree_tests()
    if tests is not None:
        worktree = tests.parent.parent
        cwd = worktree
        target = "synthetic_fixture/tests"
    else:
        cwd = ROOT
        target = "synthetic_fixture/tests"
        if not (ROOT / target).is_dir():
            print(f"no fixture tests found in {ROOT / target}")
            return 1
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", target, "-t", "."],
        cwd=cwd,
        text=True,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
