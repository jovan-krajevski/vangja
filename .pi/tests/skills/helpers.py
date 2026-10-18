"""Shared helpers for the skill-script tests (stdlib unittest)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SKILLS = REPO_ROOT / ".pi" / "skills"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def script(skill_name: str, filename: str) -> Path:
    return SKILLS / skill_name / "scripts" / filename


def run(args, cwd: Path | None = None, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *[str(a) for a in args]],
        cwd=str(cwd or REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_script(path: Path, args=(), cwd: Path | None = None) -> subprocess.CompletedProcess:
    return run([path, *args], cwd=cwd)


def assert_help(case, path: Path) -> None:
    proc = run_script(path, ["--help"])
    case.assertEqual(proc.returncode, 0, proc.stderr)
    case.assertIn("usage", proc.stdout.lower(), proc.stdout)


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
