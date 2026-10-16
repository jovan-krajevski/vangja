#!/usr/bin/env python3
"""Scan a work diff for protected paths, test weakening, and gaming patterns.

Composes the other skills' scanner: the ``scan_gaming`` categories are
delegated to ``scientific-integrity-and-anti-gaming/scripts/scan_gaming.py
--patch`` and merged into one report. Its own rules:

  protected-path      write under Vangja-project/, research_agent/plan/,
                      fixed_case_studies/protocol/, paper/ (unless PAPER),
                      AGENTS.md, .git/, .pi/state/
  frozen-config       edit of fixed_case_studies/*/config.py
  test-file-deleted   deleted file under tests/
  assertion-removed   a `assert` line removed from tests/ and not re-added
  test-skip-added     new skip/xfail in tests/

Usage:
  git diff -U0 update_pymc...HEAD > change.patch
  diff_scan.py --diff change.patch --repo . --state UNINITIALIZED \
      --out scratch/PR-003 [--allow protected-path:paper/notes.md]

Exit codes: 0 no S0/S1 finding (after waivers), 1 findings, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = "vangja-diff-scan-1.0"
PROTECTED = [
    ("Vangja-project/", "read-only historical audit archive"),
    ("research_agent/plan/", "frozen build plan"),
    ("fixed_case_studies/protocol/", "frozen protocol files"),
    ("AGENTS.md", "shared project instructions (supervisor/conductor only)"),
    (".git/", "git object store"),
    (".pi/state/", "loop state, owned by the ledger tools"),
]
PAPER_PREFIX = "paper/"
ASSERT_RE = re.compile(r"^\s*assert\b")
SKIP_RE = re.compile(r"(pytest\.mark\.(skip|xfail)|unittest\.skip|@skip\b)")


def parse_diff(diff_text: str):
    """Yield per-file dicts: path, status, added lines, removed lines."""
    files: list[dict] = []
    current: dict | None = None
    for raw in diff_text.splitlines():
        if raw.startswith("diff --git "):
            if current:
                files.append(current)
            current = {"path": "", "old_path": "", "status": "modified",
                       "added": [], "removed": [], "is_test": False}
            continue
        if current is None:
            continue
        if raw.startswith("--- "):
            source = raw[4:].strip()
            if source.startswith("a/"):
                source = source[2:]
            current["old_path"] = source
            if source == "/dev/null":
                current["status"] = "added"
            continue
        if raw.startswith("+++ "):
            target = raw[4:].strip()
            if target.startswith("b/"):
                target = target[2:]
            if target == "/dev/null":
                current["status"] = "deleted"
                current["path"] = current["old_path"]
            else:
                current["path"] = target
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            current["added"].append(raw[1:])
        elif raw.startswith("-") and not raw.startswith("---"):
            current["removed"].append(raw[1:])
    if current:
        files.append(current)
    for file in files:
        path = file["path"]
        file["is_test"] = Path(path).name.startswith("test_") or "/tests/" in f"/{path}"
    return [f for f in files if f["path"]]


def own_findings(files, *, state: str, allow: list[tuple[str, str]]) -> list[dict]:
    findings: list[dict] = []
    for file in files:
        path = file["path"]
        if path == "/dev/null":
            continue
        protected_reason = None
        for prefix, reason in PROTECTED:
            if path == prefix or path.startswith(prefix):
                protected_reason = reason
                break
        if path.startswith(PAPER_PREFIX) and state != "PAPER":
            protected_reason = "paper/** is writable only in the PAPER state"
        if protected_reason:
            findings.append(_finding(path, "S1", "protected-path",
                                     f"writes a protected path: {protected_reason}"))
        if file["status"] == "deleted" and file["is_test"]:
            findings.append(_finding(path, "S1", "test-file-deleted",
                                     "a test file is deleted in this diff"))
        if file["is_test"]:
            removed_asserts = [
                line for line in file["removed"]
                if ASSERT_RE.match(line) and line not in file["added"]
            ]
            if removed_asserts:
                findings.append(_finding(
                    path, "S1", "assertion-removed",
                    f"{len(removed_asserts)} assertion(s) removed and not re-added"))
            added_skips = [line for line in file["added"] if SKIP_RE.search(line)]
            if added_skips:
                findings.append(_finding(
                    path, "S1", "test-skip-added",
                    f"{len(added_skips)} new skip/xfail marker(s) in tests"))
        if path.startswith("fixed_case_studies/") and path.endswith("/config.py"):
            findings.append(_finding(
                path, "S1", "frozen-config",
                "edits a frozen case-study config; requires a recorded amendment"))
    if allow:
        for finding in findings:
            for rule, prefix in allow:
                if finding["rule"] == rule and finding["location"].startswith(prefix):
                    finding["waived"] = True
                    finding["waiver"] = f"{rule}:{prefix}"
    return findings


def _finding(location: str, severity: str, rule: str, message: str) -> dict:
    return {"location": location, "severity": severity, "rule": rule,
            "message": message, "source": "diff_scan"}


def run_gaming_scan(diff_path: Path, repo: Path, skills_dir: Path) -> list[dict]:
    scanner = skills_dir / "scientific-integrity-and-anti-gaming" / "scripts" / "scan_gaming.py"
    if not scanner.exists():
        return [{
            "location": str(scanner), "severity": "S2", "rule": "scan-gaming-missing",
            "message": "scan_gaming.py not found; gaming scan skipped",
            "source": "diff_scan",
        }]
    proc = subprocess.run(
        [sys.executable, str(scanner), "--root", str(repo),
         "--patch", str(diff_path), "--json", "--fail-on", "none"],
        capture_output=True, text=True,
    )
    if proc.returncode not in (0, 1):
        return [{
            "location": str(scanner), "severity": "S1", "rule": "scan-gaming-error",
            "message": f"scan_gaming failed with exit {proc.returncode}: {proc.stderr[-300:]}",
            "source": "diff_scan",
        }]
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return [{
            "location": str(scanner), "severity": "S1", "rule": "scan-gaming-error",
            "message": "scan_gaming produced unparseable JSON", "source": "diff_scan",
        }]
    merged = []
    for finding in payload.get("findings", []):
        merged.append({
            "location": finding.get("location", ""),
            "severity": finding.get("severity", "S2"),
            "rule": f"gaming-{finding.get('category', 'unknown')}",
            "message": finding.get("message", ""),
            "source": "scan_gaming",
        })
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan a work diff for protected paths, test weakening, and gaming.",
        epilog=(
            "Example:\n"
            "  git diff -U0 update_pymc...HEAD > change.patch\n"
            "  diff_scan.py --diff change.patch --repo . --state UNINITIALIZED\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--diff", required=True, help="unified diff file, or '-' for stdin")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--skills-dir", type=Path, default=None)
    parser.add_argument("--state", default="UNINITIALIZED",
                        help="loop state; paper/** is allowed only in PAPER")
    parser.add_argument("--allow", action="append", default=[],
                        metavar="RULE:PATH", help="waive a rule for a path prefix")
    parser.add_argument("--skip-gaming-scan", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path, help="write report.json under this directory")
    args = parser.parse_args()

    repo = args.repo.resolve()
    skills_dir = (args.skills_dir or repo / ".pi" / "skills").resolve()
    diff_text = sys.stdin.read() if args.diff == "-" else Path(args.diff).read_text()
    allow = []
    for item in args.allow:
        if ":" not in item:
            parser.error(f"--allow expects RULE:PATH, got {item!r}")
        rule, prefix = item.split(":", 1)
        allow.append((rule, prefix))

    files = parse_diff(diff_text)
    findings = own_findings(files, state=args.state, allow=allow)
    if not args.skip_gaming_scan and args.diff != "-":
        findings += run_gaming_scan(Path(args.diff).resolve(), repo, skills_dir)

    failing = [
        f for f in findings
        if f["severity"] in ("S0", "S1") and not f.get("waived")
    ]
    report = {
        "schema": SCHEMA,
        "diff": args.diff,
        "repo": str(repo),
        "n_files": len(files),
        "n_findings": len(findings),
        "n_failing": len(failing),
        "findings": findings,
    }
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "diff_scan.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for finding in findings:
            mark = " (waived)" if finding.get("waived") else ""
            print(f"[{finding['severity']}] {finding['rule']} "
                  f"{finding['location']} {finding['message']}{mark}")
        print(f"FINDINGS: {len(findings)} (failing: {len(failing)})")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
