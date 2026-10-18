#!/usr/bin/env python3
"""Flag per-group shape-contract violations in Vangja components.

Contract (AGENTS.md): non-complete pooling always has a per-group dimension,
even when ``n_groups == 1``. Therefore, for ``pool_type`` in
{"partial", "individual"}:

  * every per-group initval is an array shaped ``(n_groups,)`` — a scalar
    initval for a vector RV breaks ADVI's start handling;
  * every ``_predict_columns`` value is 1-D, ``(horizon,)`` — a 2-D column
    makes the folded composition broadcast wrongly.

Two modes:

  --report FILE   check a JSON shape report (pure, no pymc import):
      {"schema": "vangja-shape-report-1.0", "component": "LinearTrend",
       "pool_type": "partial", "n_groups": 1,
       "initvals":       {"lt_0 - slope": {"shape": []}},
       "predict_columns": {"lt_0_0": {"ndim": 2}}}
  --root DIR      static scan of component sources for the two known
                  anti-patterns (heuristic; triage every hit).

Exit codes: 0 clean, 1 findings, 2 usage error.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPORT_SCHEMA = "vangja-shape-report-1.0"
NON_COMPLETE = {"partial", "individual"}
SCALAR_CALLS = {"float", "int", "mean", "max", "min", "sum", "round", "len"}
RESULT_SCHEMA = "vangja-pooling-shape-check-1.0"


def _finding(location: str, severity: str, rule: str, message: str) -> dict:
    return {
        "location": location,
        "severity": severity,
        "rule": rule,
        "message": message,
    }


def check_report(report: dict) -> list[dict]:
    findings: list[dict] = []
    component = report.get("component", "<report>")
    pool_type = report.get("pool_type")
    n_groups = report.get("n_groups", 1)
    if report.get("schema") != REPORT_SCHEMA:
        findings.append(
            _finding(
                component,
                "S2",
                "report-schema",
                f"expected schema {REPORT_SCHEMA!r}, got {report.get('schema')!r}",
            )
        )
    if pool_type not in NON_COMPLETE and pool_type != "complete":
        findings.append(
            _finding(component, "S2", "report-schema", f"unknown pool_type {pool_type!r}")
        )

    if pool_type in NON_COMPLETE:
        for name, spec in (report.get("initvals") or {}).items():
            shape = spec.get("shape") if isinstance(spec, dict) else spec
            if not isinstance(shape, (list, tuple)):
                shape = []
            if len(shape) == 0:
                findings.append(
                    _finding(
                        f"{component}:{name}",
                        "S1",
                        "scalar-initval-for-vector",
                        f"pool_type={pool_type!r} with n_groups={n_groups}: initval "
                        "is scalar but the RV is shaped (n_groups,); return an array",
                    )
                )
            elif len(shape) >= 1 and n_groups and shape[0] != n_groups:
                findings.append(
                    _finding(
                        f"{component}:{name}",
                        "S1",
                        "initval-group-dim",
                        f"initval shape {list(shape)} does not start with "
                        f"n_groups={n_groups}",
                    )
                )
        for name, spec in (report.get("predict_columns") or {}).items():
            ndim = spec.get("ndim") if isinstance(spec, dict) else None
            if ndim is None:
                findings.append(
                    _finding(
                        f"{component}:{name}",
                        "S2",
                        "report-schema",
                        "predict column spec needs an integer 'ndim'",
                    )
                )
            elif ndim != 1:
                findings.append(
                    _finding(
                        f"{component}:{name}",
                        "S1",
                        "2d-predict-column",
                        f"predict column has ndim={ndim}; expected 1-D (horizon,) "
                        "for non-complete pooling",
                    )
                )
    return findings


# ---------------------------------------------------------------------------
# Static scan
# ---------------------------------------------------------------------------


class _Parents:
    def __init__(self, tree: ast.AST):
        self.parents: dict[int, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self.parents[id(child)] = node

    def ancestors(self, node: ast.AST):
        child = node
        current = self.parents.get(id(node))
        while current is not None:
            yield current, child
            child = current
            current = self.parents.get(id(current))


def _scalarish(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp):
        return _scalarish(node.operand)
    if isinstance(node, ast.Call):
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None) or ""
        return name in SCALAR_CALLS
    if isinstance(node, ast.Subscript):
        return True  # e.g. slopes[0] — a single element of a per-group list
    if isinstance(node, ast.Name):
        return True  # unknown; triage
    return False


def _mentions_pool_type(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and child.attr == "pool_type":
            return True
        if isinstance(child, ast.Name) and child.id == "pool_type":
            return True
    return False


def _is_predict_columns_target(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "_predict_columns"
    )


def _has_2d_slice(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript):
            continue
        if not isinstance(child.slice, ast.Tuple) or not child.slice.elts:
            continue
        first = child.slice.elts[0]
        if isinstance(first, ast.Slice) and first.lower is None and first.upper is None:
            return True
    return False


def _scan_initval(path: str, func: ast.FunctionDef, parents: _Parents) -> list[dict]:
    findings: list[dict] = []
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not node.value or not _scalarish(node.value):
            continue
        guarded = any(
            isinstance(anc, ast.If)
            and _mentions_pool_type(anc.test)
            and any(branch is child for branch in anc.body)
            for anc, child in parents.ancestors(node)
        )
        if guarded:
            continue
        for target in targets:
            if isinstance(target, ast.Subscript) and not _is_predict_columns_target(target):
                findings.append(
                    _finding(
                        f"{path}:{node.lineno}",
                        "S1",
                        "scalar-initval-for-vector",
                        "scalar value assigned to a per-group initval without a "
                        "pool_type guard; use an array for non-complete pooling",
                    )
                )
    return findings


def scan_source(display: str, text: str) -> list[dict]:
    findings: list[dict] = []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [
            _finding(
                f"{display}:{exc.lineno or 1}",
                "S1",
                "syntax-error",
                str(exc),
            )
        ]
    parents = _Parents(tree)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_get_initval":
            findings += _scan_initval(display, node, parents)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is None or not _has_2d_slice(value):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if _is_predict_columns_target(target):
                    findings.append(
                        _finding(
                            f"{display}:{node.lineno}",
                            "S1",
                            "2d-predict-column",
                            "2-D slice assigned into _predict_columns; index "
                            "[group_code] and store a 1-D column",
                        )
                    )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Flag per-group shape-contract violations in Vangja components.",
        epilog=(
            "Examples:\n"
            "  check_pooling_shapes.py --root src/vangja/components\n"
            "  check_pooling_shapes.py --report shape_report.json --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, help="directory to scan statically")
    parser.add_argument("--report", type=Path, help="JSON shape report to check")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--out", type=Path, help="also write the JSON report here")
    args = parser.parse_args()

    if not args.root and not args.report:
        parser.error("one of --root or --report is required")

    findings: list[dict] = []
    if args.report:
        report = json.loads(args.report.read_text())
        findings += check_report(report)
    if args.root:
        root = args.root.resolve()
        if not root.is_dir():
            parser.error(f"--root is not a directory: {root}")
        for path in sorted(root.rglob("*.py")):
            if set(path.parts) & {"__pycache__", ".venv", "node_modules"}:
                continue
            try:
                findings += scan_source(str(path), path.read_text())
            except (OSError, UnicodeDecodeError):
                continue

    findings.sort(key=lambda f: (f["location"], f["rule"]))
    payload = {
        "schema": RESULT_SCHEMA,
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
                f"{finding['location']} {finding['message']}"
            )
        print(f"FINDINGS: {len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
