#!/usr/bin/env python3
"""Scan a Vangja repository (or a diff/patch) for protocol violations.

Static rules:
  * configuration selection over test/confirmation metrics
    (``idxmin``/``idxmax``/``argmin``/``argmax`` or ``sort_values(...).head(1)``),
  * ``*.sha256`` sidecars that no longer match,
  * confirmation-result manifests without a hash-valid confirmation-origin file,
  * context series (``^GSPC``, ``source``, ``GC=F``) inside target aggregates.

Diff rules (``--diff REF`` or ``--patch FILE``):
  * edits to frozen split/metric/exclusion constants,
  * seed constants changed in a frozen config file.

Exit 1 when at least one S1 finding exists. Writes nothing unless ``--out``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

EXCLUDE_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".pytest_cache",
                ".ruff_cache", "docs/_build", ".ipynb_checkpoints", "fixtures"}
EXCLUDE_READONLY = {"Vangja-project"}
TEST_TOKENS = ("test", "confirm", "holdout", "hold_out", "out_of_sample", "oos")
METRIC_TOKENS = ("mae", "rmse", "mape", "elpd", "score", "metric", "error", "loss")
SELECTORS = {"idxmin", "idxmax", "argmin", "argmax"}
CONTEXT_VALUES = {"^GSPC", "source", "GC=F", "gspc"}
CONTEXT_SERIES_LITERAL = re.compile(r"(?:\^GSPC|\bGC=F\b)")
FROZEN_EDIT = re.compile(
    r"\b(REL_MAE_EPSILON|DEV_ORIGINS|HORIZON_DAYS|TARGET_WINDOW_DAYS|"
    r"CONTEXT_WINDOW_DAYS|MIN_TRAIN_OBS|CONFIRMATION_ORIGINS|CONTEXT_SERIES)\b"
    r"|^\s*(exclude[d]?(_series)?)\s*="
)
SEED_EDIT = re.compile(r"\b(BASE_SEED|FINALIST_SEEDS|random_seed)\b\s*=")
# Assertions counted as removed by the patch scan: bare `assert`, unittest
# camelCase (self.assertEqual/assertTrue/...), pytest.raises, JS expect().
ASSERT_RE = re.compile(r"\b(assert\b|(self\.)?assert[A-Z]\w*|pytest\.raises|expect\()")
HASH64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(repo: Path, include_readonly: bool, suffixes: tuple[str, ...]):
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        rel = path.relative_to(repo)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if not include_readonly and rel.parts and rel.parts[0] in EXCLUDE_READONLY:
            continue
        if "results" in rel.parts or "data" in rel.parts:
            continue
        yield path, rel


def finding(rule: str, severity: str, location: str, message: str, snippet: str = "") -> dict:
    return {
        "rule": rule,
        "severity": severity,
        "location": location,
        "message": message,
        "snippet": snippet.strip()[:200],
    }


# ---------------------------------------------------------------------------
# static rules
# ---------------------------------------------------------------------------


def scan_selection(repo: Path, include_readonly: bool) -> list[dict]:
    out: list[dict] = []
    for path, rel in iter_files(repo, include_readonly, (".py",)):
        try:
            source = path.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in SELECTORS:
                    segment = ast.get_source_segment(source, node) or ""
                    low = segment.lower()
                    if any(t in low for t in TEST_TOKENS) and any(m in low for m in METRIC_TOKENS):
                        out.append(finding(
                            "test-metric-selection", "S1",
                            f"{rel}:{node.lineno}",
                            f"selection by .{node.func.attr}() on an expression carrying "
                            "test/confirmation metric tokens",
                            segment,
                        ))
            if isinstance(node, (ast.Assign, ast.Expr)):
                value = node.value if isinstance(node, ast.Expr) else node.value
                if not any(isinstance(child, ast.Call) for child in ast.walk(value)):
                    continue
                segment = ast.get_source_segment(source, node) or ""
                low = segment.lower()
                if ("sort_values" in low and any(t in low for t in TEST_TOKENS)
                        and any(m in low for m in METRIC_TOKENS)
                        and re.search(r"\.(head\(1\)|tail\(1\)|iloc\[0\])", segment)):
                    out.append(finding(
                        "test-metric-selection", "S1",
                        f"{rel}:{node.lineno}",
                        "sort_values over a test metric followed by a single-row pick",
                        segment,
                    ))
    return out


def scan_sidecars(repo: Path, include_readonly: bool) -> list[dict]:
    out: list[dict] = []
    for path, rel in iter_files(repo, include_readonly, (".sha256",)):
        text = path.read_text().strip()
        if not text:
            out.append(finding("hashed-file-mismatch", "S1", f"{rel}:1", "empty .sha256 sidecar"))
            continue
        lines = [ln for ln in text.splitlines() if ln.strip()]
        for i, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) >= 2:
                digest, name = parts[0], " ".join(parts[1:]).lstrip("*")
                target = path.parent / name
            else:
                digest = parts[0]
                target = None
                for candidate in (path.parent / path.name[: -len(".sha256")],):
                    if candidate.exists():
                        target = candidate
                if target is None:
                    stem = path.name[: -len(".sha256")]
                    matches = sorted(
                        p for p in path.parent.glob(stem + ".*")
                        if p.suffix != ".sha256" and p.is_file()
                    )
                    target = matches[0] if matches else path.parent / stem
                name = target.name
            if not HASH64.match(digest):
                out.append(finding("hashed-file-mismatch", "S1", f"{rel}:{i}",
                                   f"sidecar line is not a sha256: {digest[:24]}…"))
                continue
            if not target.exists():
                out.append(finding("hashed-file-mismatch", "S1", f"{rel}:{i}",
                                   f"hashed file is missing: {target}"))
                continue
            if sha256_file(target) != digest:
                out.append(finding("hashed-file-mismatch", "S1", f"{rel}:{i}",
                                   f"hash mismatch for {target}"))
    return out


def origins_state(repo: Path) -> tuple[bool, str]:
    csv_path = repo / "fixed_case_studies/protocol/CONFIRMATION_ORIGINS.csv"
    sidecar = repo / "fixed_case_studies/protocol/CONFIRMATION_ORIGINS.sha256"
    if not csv_path.exists():
        return False, "CONFIRMATION_ORIGINS.csv is missing"
    if not sidecar.exists():
        return False, "CONFIRMATION_ORIGINS.sha256 is missing"
    stored = sidecar.read_text().split()
    expected = stored[0] if stored else ""
    actual = sha256_file(csv_path)
    if actual != expected:
        return False, f"CONFIRMATION_ORIGINS.csv hash mismatch ({actual[:12]}… != {expected[:12]}…)"
    return True, "frozen origins hash verified"


def scan_confirmation_results(repo: Path, include_readonly: bool) -> list[dict]:
    out: list[dict] = []
    manifests = []
    for path, rel in iter_files(repo, include_readonly, (".json",)):
        if "manifest" not in path.name:
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(payload, dict) and payload.get("stage") == "confirmation":
            manifests.append((path, rel))
    if not manifests:
        return out
    ok, detail = origins_state(repo)
    if not ok:
        for path, rel in manifests:
            out.append(finding(
                "confirmation-without-frozen-origins", "S1", str(rel),
                f"confirmation manifest present but {detail}",
            ))
    return out


def scan_context_aggregates(repo: Path, include_readonly: bool) -> list[dict]:
    out: list[dict] = []
    for path, rel in iter_files(repo, include_readonly, (".csv",)):
        name = path.name.lower()
        if "unit" not in name and "report" not in name and "aggregate" not in name:
            continue
        try:
            header = path.read_text().splitlines()[:1]
        except UnicodeDecodeError:
            continue
        if not header:
            continue
        columns = [c.strip() for c in header[0].split(",")]
        if "series" not in columns:
            continue
        idx = columns.index("series")
        for lineno, line in enumerate(path.read_text().splitlines()[1:], start=2):
            cells = line.split(",")
            if idx < len(cells) and cells[idx].strip().strip('"') in CONTEXT_VALUES:
                out.append(finding(
                    "context-series-in-aggregate", "S1", f"{rel}:{lineno}",
                    f"context series {cells[idx].strip()!r} appears in a target aggregate file",
                    line,
                ))
    return out


def scan_context_static(repo: Path, include_readonly: bool) -> list[dict]:
    out: list[dict] = []
    pattern = re.compile(
        r"(series|ticker).{0,40}(==|!=|isin|in).{0,40}['\"](\^GSPC|source|GC=F)['\"]",
        re.IGNORECASE,
    )
    for path, rel in iter_files(repo, include_readonly, (".py",)):
        name = path.name.lower()
        if not any(k in name for k in ("report", "aggregate", "table", "unit")):
            continue
        source = path.read_text()
        if "CONTEXT_SERIES" in source or "^GSPC" not in source:
            continue
        for i, line in enumerate(source.splitlines(), start=1):
            if pattern.search(line) and "rel_mae" in source:
                out.append(finding(
                    "context-series-in-aggregate", "S1", f"{rel}:{i}",
                    "context series compared/selected inside an aggregation module "
                    "that does not reference CONTEXT_SERIES",
                    line,
                ))
    return out


# ---------------------------------------------------------------------------
# diff rules
# ---------------------------------------------------------------------------


def parse_patch(text: str) -> list[tuple[str, int, str, str]]:
    """Return (path, line, kind, content) for +/- lines in a unified diff."""
    out: list[tuple[str, int, str, str]] = []
    path = "?"
    old_no = new_no = 0
    for raw in text.splitlines():
        if raw.startswith("+++ "):
            path = raw[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            continue
        if raw.startswith("@@"):
            m = re.search(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
            if m:
                old_no, new_no = int(m.group(1)), int(m.group(2))
            continue
        if raw.startswith("---") or raw.startswith("diff --git") or raw.startswith("index "):
            continue
        if raw.startswith("+") and not raw.startswith("+++"):
            out.append((path, new_no, "add", raw[1:]))
            new_no += 1
        elif raw.startswith("-") and not raw.startswith("---"):
            out.append((path, old_no, "del", raw[1:]))
            old_no += 1
        elif raw.startswith(" "):
            old_no += 1
            new_no += 1
    return out


def scan_patch(repo: Path, patch_text: str) -> list[dict]:
    out: list[dict] = []
    for path, line, kind, content in parse_patch(patch_text):
        location = f"{path}:{line}"
        verb = {"add": "added", "del": "deleted"}.get(kind, kind)
        if FROZEN_EDIT.search(content):
            out.append(finding(
                "frozen-split-metric-edit", "S1", location,
                f"{verb} line touches a frozen split/metric/exclusion constant",
                content,
            ))
        if SEED_EDIT.search(content) and re.search(r"config", path, re.IGNORECASE):
            out.append(finding(
                "seed-change-frozen-config", "S1", location,
                f"{verb} seed constant in a frozen config file",
                content,
            ))
        if ASSERT_RE.search(content) and kind == "del":
            out.append(finding(
                "assertion-removed", "S1", location,
                "an assertion was deleted",
                content,
            ))
        if re.search(r"(pytest\.mark\.skip|unittest\.skip|pytest\.skip|pytest\.mark\.xfail)", content) \
                and kind == "add":
            out.append(finding(
                "test-skip-added", "S1", location,
                "a skip/xfail was added",
                content,
            ))
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="check_protocol_compliance.py",
        description="Scan a Vangja repo/diff for study-protocol violations.",
        epilog="Example: check_protocol_compliance.py --repo . --diff HEAD~1",
    )
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--diff", metavar="REF", default=None,
                   help="scan `git diff -U0 REF` for frozen-constant edits")
    p.add_argument("--patch", type=Path, default=None, help="scan a unified diff file instead")
    p.add_argument("--include-readonly", action="store_true",
                   help="also scan the read-only Vangja-project/ archive")
    p.add_argument("--json", action="store_true", help="print JSON findings")
    p.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    args = p.parse_args(argv)

    repo = args.repo.resolve()
    if not repo.exists():
        print(f"repo not found: {repo}", file=sys.stderr)
        return 2

    findings: list[dict] = []
    findings += scan_selection(repo, args.include_readonly)
    findings += scan_sidecars(repo, args.include_readonly)
    findings += scan_confirmation_results(repo, args.include_readonly)
    findings += scan_context_aggregates(repo, args.include_readonly)
    findings += scan_context_static(repo, args.include_readonly)

    if args.diff:
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo), "diff", "-U0", args.diff],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"git diff failed: {exc.stderr.strip()}", file=sys.stderr)
            return 2
        findings += scan_patch(repo, proc.stdout)
    if args.patch:
        if not args.patch.exists():
            print(f"patch not found: {args.patch}", file=sys.stderr)
            return 2
        findings += scan_patch(repo, args.patch.read_text())

    findings.sort(key=lambda f: (f["location"], f["rule"]))
    ok, origins_detail = origins_state(repo)
    report = {
        "repo": str(repo),
        "origins_hash_ok": ok,
        "origins_detail": origins_detail,
        "n_findings": len(findings),
        "n_s1": sum(1 for f in findings if f["severity"] == "S1"),
        "findings": findings,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"confirmation origins: {origins_detail}")
        for f in findings:
            print(f"[{f['severity']}] {f['rule']} {f['location']}: {f['message']}")
        print(f"{'CLEAN' if not findings else 'FINDINGS'}: {len(findings)} "
              f"({report['n_s1']} S1)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
