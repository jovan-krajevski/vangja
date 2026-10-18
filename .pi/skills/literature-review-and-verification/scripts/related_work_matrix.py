#!/usr/bin/env python3
"""Render a related-work extraction matrix and flag incomplete/unsafe rows.

Input: a JSON list of sources (see the literature skill and
``references/source-hierarchy.md``). Each row needs key, level, claim, method,
dataset, limitation, relevance, doi_or_url and checked. A row that is not
``checked`` is a finding (S1): it must not appear in the paper.

Usage:
  related_work_matrix.py --sources sources.json --out scratch/related-work.md
  related_work_matrix.py --sources sources.json --json

Exit codes: 0 clean, 1 findings, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "vangja-related-work-1.0"
LEVELS = {"primary", "secondary", "tertiary", "preprint", "docs"}
REQUIRED = ("key", "level", "claim", "method", "dataset", "limitation",
            "relevance", "doi_or_url", "checked")
COLUMNS = ["Key", "Level", "Claim it supports", "Method", "Dataset",
           "Limitation", "Relevance to Vangja", "DOI/URL", "Checked"]


def check(sources: list[dict]) -> list[dict]:
    findings: list[dict] = []
    seen: set[str] = set()
    for index, source in enumerate(sources, start=1):
        key = str(source.get("key", f"#{index}"))
        if key in seen:
            findings.append(_finding(key, "S2", "duplicate-key",
                                     f"source {key!r} appears more than once"))
        seen.add(key)
        for field in REQUIRED:
            if source.get(field) in (None, "", []):
                findings.append(_finding(key, "S2", "missing-field",
                                         f"missing {field!r}"))
        if source.get("level") not in LEVELS:
            findings.append(_finding(
                key, "S2", "invalid-level",
                f"level {source.get('level')!r} not in {sorted(LEVELS)}"))
        if source.get("level") == "tertiary" and not source.get("primary_for"):
            findings.append(_finding(
                key, "S2", "tertiary-without-primary",
                "a tertiary source cannot carry a claim on its own; add the "
                "primary source or drop the row"))
        if source.get("checked") is not True:
            findings.append(_finding(key, "S1", "unchecked-source",
                                     "checked is not true: do not cite this row"))
    return findings


def _finding(key: str, severity: str, rule: str, message: str) -> dict:
    return {"location": key, "severity": severity, "rule": rule,
            "message": message}


def render(sources: list[dict]) -> str:
    lines = ["# Related-work extraction matrix", "", "| " + " | ".join(COLUMNS) + " |",
             "|" + "---|" * len(COLUMNS)]
    for source in sources:
        row = [
            source.get("key", ""),
            source.get("level", ""),
            source.get("claim", ""),
            source.get("method", ""),
            source.get("dataset", ""),
            source.get("limitation", ""),
            source.get("relevance", ""),
            source.get("doi_or_url", ""),
            "yes" if source.get("checked") is True else "NO",
        ]
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|")
                                       for cell in row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render and validate a related-work extraction matrix.",
        epilog=(
            "Example:\n"
            "  related_work_matrix.py --sources sources.json \\\n"
            "    --out scratch/related-work.md --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="write the Markdown matrix here")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not args.sources.is_file():
        parser.error(f"--sources does not exist: {args.sources}")
    payload = json.loads(args.sources.read_text())
    sources = payload if isinstance(payload, list) else payload.get("sources", [])
    findings = check(sources)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render(sources))
    report = {
        "schema": SCHEMA,
        "sources": str(args.sources),
        "n_sources": len(sources),
        "n_findings": len(findings),
        "findings": findings,
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for item in findings:
            print(f"[{item['severity']}] {item['rule']} {item['location']} "
                  f"{item['message']}")
        print(f"SOURCES: {len(sources)} FINDINGS: {len(findings)}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
