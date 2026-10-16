#!/usr/bin/env python3
"""Scan for research-gaming patterns (static and diff modes).

Categories: test-metric selection, test weakening (skip/xfail, removed
assertions, relaxed tolerances), seed changes in frozen configs, failure
trimming, post-hoc exclusion changes, metric reassignment, denominator
clipping, hashed-file mismatches, horizon cropping.

Exit 1 when any finding at or above ``--fail-on`` (default S1) exists. Read-only
unless ``--out`` is given. See references/gaming-catalogue.md.
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
                ".ruff_cache", ".ipynb_checkpoints", "fixtures"}
SELECTORS = {"idxmin", "idxmax", "argmin", "argmax"}
TEST_TOKENS = ("test", "confirm", "holdout", "hold_out", "out_of_sample", "oos")
METRIC_TOKENS = ("mae", "rmse", "mape", "elpd", "score", "metric", "error", "loss")
SWALLOW_FUNCS = re.compile(
    r"(record|report|write|save|collect|aggregate|score|evaluate|export|publish|append|store)",
    re.IGNORECASE)
DROPNA_WRITE = re.compile(r"\.dropna\(\)\s*\.(to_csv|to_json|to_parquet|to_excel)\s*\(")
DROPNA_NEAR = re.compile(r"\b(to_csv|save_json|to_json|record_artifact)\b")
CLIP_DENOM = re.compile(
    r"\b(denom|persistence_mae|pers_mae\w*)\s*=\s*(np\.)?(clip|maximum|max|minimum)\s*\("
    r"|np\.clip\(\s*\w*(denom|pers_mae)",
    re.IGNORECASE)
EPSILON_DEF = re.compile(r"\bREL_MAE_EPSILON\s*=\s*([^\s#]+)")
EXPECTED_SEEDS = {
    "BASE_SEED": "42",
    "FINALIST_SEEDS": "(42, 20240816, 7)",
}
SEED_DEF = re.compile(r"^\s*(BASE_SEED|FINALIST_SEEDS)\s*=\s*(.+?)\s*(?:#.*)?$")
FROZEN_METRIC_EDIT = re.compile(
    r"\bREL_MAE_EPSILON\b|\bCONTEXT_SERIES\b"
    r"|\bdef\s+(relative_mae|unit_metrics|aggregate_rel_mae)\b")
FROZEN_SPLIT_EDIT = re.compile(r"\b(DEV_ORIGINS|CONFIRMATION_ORIGINS)\b")
SEED_DIFF = re.compile(r"^\s*(BASE_SEED|FINALIST_SEEDS)\s*=")
EXCLUSION_DIFF = re.compile(r"exclude[d]?(_series)?\s*=|\.isin\(\s*exclude|drop\(\s*exclude")
HORIZON_DIFF = re.compile(
    r"\b(HORIZON_DAYS|TARGET_WINDOW_DAYS|CONTEXT_WINDOW_DAYS|MIN_TRAIN_OBS)\b")
HASH64 = re.compile(r"^[0-9a-f]{64}$")
RESULT_CALLS = {"to_csv", "to_json", "to_parquet", "to_excel", "save_json",
                "record_artifact", "record_failure", "write_text", "json.dump"}
TOLERANCE_LIMITS = {"abs": 0.1, "atol": 0.1, "rel": 0.05, "rtol": 0.05}
SKIP_MARKER = re.compile(r"(pytest\.mark\.skip|unittest\.skip|pytest\.mark\.xfail|@skip\b)")


def severity_rank(severity: str) -> int:
    return {"S0": 0, "S1": 1, "S2": 2, "S3": 3}.get(severity.upper(), 9)


def finding(category: str, severity: str, location: str, message: str,
            snippet: str, consequence: str) -> dict:
    return {
        "category": category,
        "severity": severity,
        "location": location,
        "message": message,
        "snippet": snippet.strip()[:220],
        "consequence": consequence,
    }


def iter_files(root: Path, suffixes: tuple[str, ...]):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        rel = path.relative_to(root)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        yield path, rel


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# static scan
# ---------------------------------------------------------------------------


def scan_python_static(root: Path) -> list[dict]:
    out: list[dict] = []
    for path, rel in iter_files(root, (".py",)):
        try:
            source = path.read_text()
            tree = ast.parse(source)
        except (SyntaxError, UnicodeDecodeError):
            continue
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        is_test_file = "test" in rel.name.lower() or "tests" in rel.parts

        for node in ast.walk(tree):
            # --- test-metric selection ---
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in SELECTORS:
                    segment = ast.get_source_segment(source, node) or ""
                    low = segment.lower()
                    if any(t in low for t in TEST_TOKENS) and any(m in low for m in METRIC_TOKENS):
                        out.append(finding(
                            "test-metric-selection", "S0", f"{rel}:{node.lineno}",
                            f"selects by .{node.func.attr}() over test/metric tokens",
                            segment,
                            "configuration/metric chosen on the test horizon; the result is "
                            "selection-biased (FINDINGS F-1)",
                        ))
            if isinstance(node, (ast.Assign, ast.Expr)):
                value = node.value
                if not any(isinstance(child, ast.Call) for child in ast.walk(value)):
                    continue
                segment = ast.get_source_segment(source, node) or ""
                low = segment.lower()
                if ("sort_values" in low and any(t in low for t in TEST_TOKENS)
                        and any(m in low for m in METRIC_TOKENS)
                        and re.search(r"\.(head\(1\)|tail\(1\)|iloc\[0\])", segment)):
                    out.append(finding(
                        "test-metric-selection", "S0", f"{rel}:{node.lineno}",
                        "sort_values over a test metric followed by a single-row pick",
                        segment,
                        "best test row selected after the fact",
                    ))
            # --- test weakening: skip decorators ---
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                    and is_test_file:
                for decorator in node.decorator_list:
                    seg = ast.get_source_segment(source, decorator) or ""
                    if "skip" in seg.lower() or "xfail" in seg.lower():
                        out.append(finding(
                            "test-weakening-skip", "S1", f"{rel}:{node.lineno}",
                            f"test {node.name!r} is skipped/xfailed unconditionally",
                            seg,
                            "a failing behaviour can no longer be detected",
                        ))
            # --- test weakening: relaxed tolerance ---
            if isinstance(node, ast.Call):
                seg = ast.get_source_segment(source, node) or ""
                func_name = ""
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id
                if func_name in ("approx", "allclose", "isclose", "assert_allclose"):
                    for keyword in node.keywords or []:
                        if keyword.arg not in TOLERANCE_LIMITS:
                            continue
                        try:
                            tol = float(ast.literal_eval(keyword.value))
                        except (ValueError, TypeError):
                            continue
                        if tol > TOLERANCE_LIMITS[keyword.arg]:
                            out.append(finding(
                                "test-weakening-tolerance", "S1", f"{rel}:{node.lineno}",
                                f"{func_name}(..., {keyword.arg}={tol}) exceeds the "
                                f"{TOLERANCE_LIMITS[keyword.arg]} guideline",
                                seg,
                                "assertions pass for materially wrong values",
                            ))
            # --- failure trimming: swallowed exceptions around result recording ---
            if isinstance(node, ast.Try):
                if node.handlers and all(
                    h.body and all(isinstance(stmt, (ast.Pass, ast.Continue)) for stmt in h.body)
                    for h in node.handlers
                ):
                    func = enclosing_function_name(parents, node)
                    bare = any(h.type is None for h in node.handlers)
                    body = "\n".join(
                        ast.get_source_segment(source, stmt) or ""
                        for stmt in node.body
                    )
                    records = any(
                        (isinstance(call.func, ast.Attribute) and call.func.attr in RESULT_CALLS)
                        or (isinstance(call.func, ast.Name) and call.func.id in RESULT_CALLS)
                        for call in ast.walk(node)
                        if isinstance(call, ast.Call)
                    ) or any(name in body for name in ("to_csv", "save_json", "record_artifact"))
                    if bare or (records and func and SWALLOW_FUNCS.search(func)):
                        out.append(finding(
                            "failure-trimming-swallow", "S1", f"{rel}:{node.lineno}",
                            "exception swallowed with `pass`"
                            + (f" inside {func}()" if func else "")
                            + (" around result recording" if records else ""),
                            first_line(source, node.lineno),
                            "failed units/results disappear instead of being reported (F-11)",
                        ))
    return out


def enclosing_function_name(parents: dict, node: ast.AST) -> str | None:
    current: ast.AST | None = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current.name
    return None


def first_line(source: str, lineno: int) -> str:
    lines = source.splitlines()
    return lines[lineno - 1] if 0 < lineno <= len(lines) else ""


def scan_line_static(root: Path) -> list[dict]:
    out: list[dict] = []
    for path, rel in iter_files(root, (".py",)):
        source = path.read_text()
        lines = source.splitlines()
        for i, line in enumerate(lines, start=1):
            if DROPNA_WRITE.search(line):
                out.append(finding(
                    "failure-trimming-dropna", "S1", f"{rel}:{i}",
                    "dropna() chained straight into a file write",
                    line,
                    "missing/failed units are removed from the reported table",
                ))
            if ".dropna()" in line:
                window = "\n".join(lines[i:i + 3])
                if DROPNA_NEAR.search(window) and not DROPNA_WRITE.search(line):
                    out.append(finding(
                        "failure-trimming-dropna", "S1", f"{rel}:{i}",
                        "dropna() result written/reported within the next 3 lines",
                        line,
                        "rows removed before reporting",
                    ))
            if CLIP_DENOM.search(line):
                out.append(finding(
                    "denominator-clipping", "S0", f"{rel}:{i}",
                    "persistence denominator is clipped/bounded",
                    line,
                    "near-zero denominators are turned into large relative errors "
                    "instead of being excluded (PROTOCOL §4)",
                ))
            match = EPSILON_DEF.search(line)
            if match and match.group(1) not in ("1e-3", "0.001", "1.0e-3"):
                out.append(finding(
                    "metric-reassignment", "S0", f"{rel}:{i}",
                    f"REL_MAE_EPSILON changed to {match.group(1)}",
                    line,
                    "the exclusion rule changed; aggregates are not comparable",
                ))
            seed = SEED_DEF.match(line)
            if seed and "config" in rel.name.lower():
                name, value = seed.group(1), seed.group(2).strip().rstrip(",")
                expected = EXPECTED_SEEDS[name]
                if value.replace(" ", "") != expected.replace(" ", ""):
                    out.append(finding(
                        "seed-change-frozen-config", "S1", f"{rel}:{i}",
                        f"{name} = {value} in a frozen config file (expected {expected})",
                        line,
                        "seeds changed after results; not reproducible",
                    ))
    return out


def scan_sidecars(root: Path) -> list[dict]:
    out: list[dict] = []
    for path, rel in iter_files(root, (".sha256",)):
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                digest, name = parts[0], " ".join(parts[1:]).lstrip("*")
                target = path.parent / name
            else:
                digest = parts[0]
                stem = path.name[: -len(".sha256")] if path.name.endswith(".sha256") else path.stem
                matches = sorted(p for p in path.parent.glob(stem + ".*")
                                 if p.suffix != ".sha256" and p.is_file())
                target = matches[0] if matches else path.parent / stem
            if not HASH64.match(digest):
                out.append(finding("hashed-file-mismatch", "S1", f"{rel}:{i}",
                                   "sidecar line is not a sha256", line,
                                   "hashes cannot be trusted"))
            elif not target.exists():
                out.append(finding("hashed-file-mismatch", "S1", f"{rel}:{i}",
                                   f"hashed file missing: {target}", line,
                                   "the recorded artifact no longer exists"))
            elif sha256_file(target) != digest:
                out.append(finding("hashed-file-mismatch", "S1", f"{rel}:{i}",
                                   f"hash mismatch for {target.name}", line,
                                   "the file changed after being hashed"))
    return out


# ---------------------------------------------------------------------------
# diff scan
# ---------------------------------------------------------------------------


def parse_patch(text: str) -> list[tuple[str, int, str, str]]:
    out: list[tuple[str, int, str, str]] = []
    path, old_no, new_no = "?", 0, 0
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
        if raw.startswith(("---", "diff --git", "index ", "new file", "deleted file")):
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


def scan_patch(patch_text: str) -> list[dict]:
    out: list[dict] = []
    for path, line, kind, content in parse_patch(patch_text):
        location = f"{path}:{line}"
        verb = {"add": "added", "del": "deleted"}.get(kind, kind)
        if kind == "del" and re.search(r"\bassert\b", content):
            out.append(finding("test-weakening-assertion-removed", "S1", location,
                               "assertion deleted", content,
                               "a regression can no longer be caught"))
        if kind == "add" and SKIP_MARKER.search(content):
            out.append(finding("test-weakening-skip", "S1", location,
                               "skip/xfail added", content,
                               "a failing behaviour is now bypassed"))
        if kind == "add" and re.search(r"except[^:]*:\s*$", content):
            out.append(finding("failure-trimming-swallow", "S1", location,
                               "new exception handler added", content,
                               "failures may be swallowed silently"))
        if kind == "add" and DROPNA_WRITE.search(content):
            out.append(finding("failure-trimming-dropna", "S1", location,
                               "dropna().to_csv() added", content,
                               "rows removed before reporting"))
        if SEED_DIFF.search(content) and re.search(r"config", path, re.IGNORECASE):
            out.append(finding("seed-change-frozen-config", "S1", location,
                               f"seed constant {verb} in a frozen config", content,
                               "results produced with a different seed than recorded"))
        if EXCLUSION_DIFF.search(content):
            out.append(finding("exclusion-change", "S1", location,
                               f"exclusion logic {verb}", content,
                               "units excluded after the fact (post-hoc exclusion)"))
        if FROZEN_SPLIT_EDIT.search(content):
            out.append(finding("frozen-split-edit", "S1", location,
                               f"frozen split/origin constant {verb}", content,
                               "development/confirmation splits changed without an amendment"))
        if FROZEN_METRIC_EDIT.search(content):
            out.append(finding("metric-reassignment", "S0", location,
                               f"frozen metric definition {verb}", content,
                               "the protocol definition changed without an amendment"))
        if HORIZON_DIFF.search(content) and kind in ("add", "del"):
            out.append(finding("horizon-cropping", "S1", location,
                               f"horizon/window expression {verb}", content,
                               "arms may be evaluated over different horizons"))
        if kind == "add" and CLIP_DENOM.search(content):
            out.append(finding("denominator-clipping", "S0", location,
                               "denominator clipping added", content,
                               "relative MAE no longer excludes near-zero denominators"))
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="scan_gaming.py",
        description="Scan for research-gaming patterns (static and diff).",
        epilog="Example: scan_gaming.py --root . --patch change.patch --fail-on S1",
    )
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--patch", type=Path, default=None, help="unified diff file to scan")
    p.add_argument("--diff", metavar="REF", default=None, help="scan `git diff -U0 REF`")
    p.add_argument("--fail-on", choices=("S0", "S1", "S2", "S3", "none"), default="S1")
    p.add_argument("--out", type=Path, default=None, help="write gaming_scan.json/.md here")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        print(f"--root is not a directory: {root}", file=sys.stderr)
        return 2

    findings = scan_python_static(root) + scan_line_static(root) + scan_sidecars(root)
    if args.diff:
        try:
            proc = subprocess.run(["git", "-C", str(root), "diff", "-U0", args.diff],
                                  capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"git diff failed: {exc.stderr.strip()}", file=sys.stderr)
            return 2
        findings += scan_patch(proc.stdout)
    if args.patch:
        if not args.patch.exists():
            print(f"patch not found: {args.patch}", file=sys.stderr)
            return 2
        findings += scan_patch(args.patch.read_text())

    findings.sort(key=lambda f: (severity_rank(f["severity"]), f["location"], f["category"]))
    report = {
        "root": str(root),
        "fail_on": args.fail_on,
        "n_findings": len(findings),
        "by_severity": {s: sum(1 for f in findings if f["severity"] == s)
                        for s in ("S0", "S1", "S2", "S3")},
        "findings": findings,
    }

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "gaming_scan.json").write_text(json.dumps(report, indent=2) + "\n")
        lines = [f"# Gaming scan — {root}", "",
                 f"{len(findings)} finding(s); fail-on={args.fail_on}", ""]
        for f in findings:
            lines.append(f"- **{f['severity']} {f['category']}** {f['location']}: "
                         f"{f['message']}")
        (args.out / "gaming_scan.md").write_text("\n".join(lines) + "\n")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for f in findings:
            print(f"[{f['severity']}] {f['category']} {f['location']}: {f['message']}")
        print(f"{'CLEAN' if not findings else 'FINDINGS'}: {len(findings)} "
              f"(S0={report['by_severity']['S0']}, S1={report['by_severity']['S1']}, "
              f"S2={report['by_severity']['S2']})")

    if args.fail_on == "none":
        return 0
    threshold = severity_rank(args.fail_on)
    failing = [f for f in findings if severity_rank(f["severity"]) <= threshold]
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
