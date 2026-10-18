#!/usr/bin/env python3
"""Flag PyMC variable names that violate the Vangja naming rule.

Rule (AGENTS.md, tests/test_components.py:430): a PyMC variable name is
``{type}_{i} - {param}`` with a two-letter component prefix, an integer
model index, and a lowercase parameter name. It must never contain ``,``,
``.``, ``[``, ``]`` and must never encode priors or configuration
(period, order, bounds, hyperparameters).

Usage:
  lint_pymc_names.py --root src/vangja
  lint_pymc_names.py --root . --json --out scratch/names.json
  git diff HEAD~1 | lint_pymc_names.py --diff -
  lint_pymc_names.py --diff patch.diff --root .

Exit codes: 0 clean, 1 findings, 2 usage error.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

CANDIDATE = re.compile(r"^([a-z]{2})_([0-9]+) *- *(.*)$")
VALID_PREFIX = {"lt", "fs", "ft", "nc", "uc", "bc"}
# Parameter part: lowercase words/underscores only (slope, delta_sigma, c_shared).
VALID_PARAM = re.compile(r"^[a-z][a-z_]*$")

REPORT_SCHEMA = "vangja-pymc-name-lint-1.0"


def _finding(path: str, line: int, severity: str, rule: str, message: str) -> dict:
    return {
        "file": path,
        "line": line,
        "severity": severity,
        "rule": rule,
        "message": message,
    }


def _normalize(text: str) -> str:
    """Replace f-string placeholders and quoted keys with an index placeholder.

    ``lt_{self.model_idx} - slope`` -> ``lt_0 - slope``; ``prior_x`` stays.
    """
    return re.sub(r"\{[^{}]*\}", "0", text)


def _check_name(name: str) -> list[tuple[str, str]]:
    """Return (severity, message) tuples for one candidate name."""
    out: list[tuple[str, str]] = []
    normalized = _normalize(name)
    match = CANDIDATE.match(normalized)
    if not match:
        out.append(
            (
                "S2",
                f"name {name!r} is neither '{type}_{i} - {param}' nor a derived "
                "prior/column name",
            )
        )
        return out
    prefix, _index, param = match.groups()
    if prefix not in VALID_PREFIX:
        out.append(("S2", f"name {name!r} uses unknown component prefix {prefix!r}"))
    if not VALID_PARAM.match(param):
        out.append(
            (
                "S1",
                f"name {name!r} has parameter part {param!r}: use lowercase "
                "letters/underscores; never encode priors or configuration",
            )
        )
    for char in ",.[]":
        if char in normalized:
            out.append(
                (
                    "S1",
                    f"name {name!r} contains {char!r}, which PyMC/nutpie parse as "
                    "a dimension spec",
                )
            )
            break
    return out


def _literals_from_source(text: str, path: str, offset: int = 0) -> list[dict]:
    findings: list[dict] = []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        findings.append(
            _finding(path, offset + (exc.lineno or 1), "S1", "syntax-error", str(exc))
        )
        return findings
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
        elif isinstance(node, ast.JoinedStr):
            value = "".join(
                "{0}" if isinstance(part, ast.FormattedValue) else part.value
                for part in node.values
            )
        else:
            continue
        literal = _normalize(value)
        # Only string literals that follow the vangja naming convention can be
        # PyMC variable names; this keeps prose/docstrings out of the scan.
        if not CANDIDATE.match(literal):
            continue
        line = offset + getattr(node, "lineno", 1)
        for severity, message in _check_name(value):
            findings.append(_finding(path, line, severity, "pymc-name", message))
    return findings


STRING_LITERAL = re.compile(r"""(?:rf|fr|f|r)?(['"])(.*?)\1""")


def _literals_from_diff(diff_text: str) -> list[dict]:
    findings: list[dict] = []
    path = "<diff>"
    new_line = 0
    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            path = raw[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            continue
        if raw.startswith("@@"):
            match = re.search(r"\+(\d+)", raw)
            new_line = int(match.group(1)) if match else 0
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            for match in STRING_LITERAL.finditer(raw[1:]):
                literal = _normalize(match.group(2))
                if not CANDIDATE.match(literal):
                    continue
                for severity, message in _check_name(match.group(2)):
                    findings.append(
                        _finding(path, new_line, severity, "pymc-name", message)
                    )
            new_line += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            continue
        else:
            new_line += 1
    return findings


def _iter_python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        parts = set(path.parts)
        if parts & {"__pycache__", ".venv", "node_modules", "fixtures"}:
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flag PyMC variable names violating the Vangja naming rule.",
        epilog=(
            "Examples:\n"
            "  lint_pymc_names.py --root src/vangja\n"
            "  lint_pymc_names.py --root . --json --out scratch/names.json\n"
            "  git diff HEAD~1 | lint_pymc_names.py --diff -\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, help="scan every .py under this directory")
    parser.add_argument(
        "--diff",
        metavar="PATH",
        help="scan only added string literals in a unified diff ('-' for stdin)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument("--out", type=Path, help="also write the JSON report here")
    parser.add_argument(
        "--repo",
        type=Path,
        help="repository root used to relativise paths (default: cwd)",
    )
    args = parser.parse_args()

    if not args.root and not args.diff:
        parser.error("one of --root or --diff is required")

    findings: list[dict] = []
    if args.diff:
        if args.diff == "-":
            diff_text = sys.stdin.read()
        else:
            diff_text = Path(args.diff).read_text()
        findings += _literals_from_diff(diff_text)
    if args.root:
        root = args.root.resolve()
        if not root.is_dir():
            parser.error(f"--root is not a directory: {root}")
        for path in _iter_python_files(root):
            try:
                text = path.read_text()
            except (OSError, UnicodeDecodeError):
                continue
            findings += _literals_from_source(text, str(path))

    repo = (args.repo or Path.cwd()).resolve()
    for finding in findings:
        try:
            finding["file"] = str(Path(finding["file"]).resolve().relative_to(repo))
        except (ValueError, OSError):
            pass

    findings.sort(key=lambda f: (f["file"], f["line"], f["rule"]))
    payload = {
        "schema": REPORT_SCHEMA,
        "root": str(args.root) if args.root else None,
        "diff": args.diff,
        "n_findings": len(findings),
        "findings": findings,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for finding in findings:
            print(
                f"[{finding['severity']}] {finding['rule']} "
                f"{finding['file']}:{finding['line']} {finding['message']}"
            )
        print(f"FINDINGS: {len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
