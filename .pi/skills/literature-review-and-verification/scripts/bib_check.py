#!/usr/bin/env python3
"""Check a BibTeX file for fabricated or unverified citations.

A citation is usable only when it has a DOI or URL **and** a
``verified.json`` sidecar record with ``checked: true`` for its key. The
sidecar is the evidence that a human/agent opened the source (see
``references/source-hierarchy.md``); the ``.bib`` is only a rendering.

Usage:
  bib_check.py --bib refs.bib --verified verified.json --json
  bib_check.py --bib refs.bib --out scratch/bib-report.json

Exit codes: 0 clean, 1 findings, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHEMA = "vangja-bib-check-1.0"
ENTRY_START = re.compile(r"@(\w+)\s*[{(]\s*([^,\s]+)\s*,")
LEVELS = {"primary", "secondary", "tertiary", "preprint", "docs"}


def parse_bib(text: str) -> list[dict]:
    entries: list[dict] = []
    index = 0
    while True:
        match = ENTRY_START.search(text, index)
        if not match:
            break
        entry_type, key = match.group(1).lower(), match.group(2)
        opener = "{" if "{" in match.group(0) else "("
        closer = "}" if opener == "{" else ")"
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth > 0:
            if text[cursor] == opener:
                depth += 1
            elif text[cursor] == closer:
                depth -= 1
            cursor += 1
        body = text[match.end():cursor - 1]
        fields: dict[str, str] = {}
        for field_match in re.finditer(
            r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[^,\n]+)",
            body,
        ):
            value = field_match.group(2).strip()
            if value[:1] in "{\"" and value[-1:] in "}\"":
                value = value[1:-1]
            fields[field_match.group(1).lower()] = " ".join(value.split())
        entries.append({"type": entry_type, "key": key, "fields": fields})
        index = cursor
    return entries


def load_verified(path: Path) -> tuple[dict, list[dict]]:
    if not path.exists():
        return {}, [finding(str(path), 1, "S1", "no-verification-sidecar",
                            f"sidecar {path.name} does not exist; nothing is verified")]
    payload = json.loads(path.read_text())
    records = payload.get("records", payload) if isinstance(payload, dict) else {}
    findings: list[dict] = []
    if not isinstance(records, dict):
        findings.append(finding(str(path), 1, "S2", "invalid-sidecar",
                                "verified.json must map key -> record"))
        records = {}
    return records, findings


def finding(location: str, line: int, severity: str, rule: str, message: str) -> dict:
    return {"location": f"{location}:{line}", "severity": severity, "rule": rule,
            "message": message}


def check(entries: list[dict], records: dict) -> list[dict]:
    findings: list[dict] = []
    seen: dict[str, int] = {}
    for entry in entries:
        key = entry["key"]
        fields = entry["fields"]
        location = f"@{entry['type']}{{{key}}}"
        if key in seen:
            findings.append(finding(location, 1, "S2", "duplicate-key",
                                    f"key {key!r} appears more than once"))
        seen[key] = 1
        for required in ("title", "author", "year"):
            if not fields.get(required):
                findings.append(finding(location, 1, "S2", "incomplete-entry",
                                        f"missing field {required!r}"))
        doi = fields.get("doi", "")
        url = fields.get("url", "")
        if not doi and not url:
            findings.append(finding(location, 1, "S2", "no-identifier",
                                    "no DOI and no URL: cannot be verified"))
        record = records.get(key)
        if not record:
            findings.append(finding(location, 1, "S1", "unverified-citation",
                                    f"{key!r} has no verified.json record"))
            continue
        if record.get("checked") is not True:
            findings.append(finding(location, 1, "S1", "unverified-citation",
                                    f"{key!r} record has checked="
                                    f"{record.get('checked')!r}"))
        sidecar_doi = record.get("doi", "")
        sidecar_url = record.get("url", "")
        if doi and sidecar_doi and doi != sidecar_doi:
            findings.append(finding(location, 1, "S1", "verification-mismatch",
                                    f"{key!r}: bib doi {doi!r} != sidecar "
                                    f"{sidecar_doi!r}"))
        if url and sidecar_url and url != sidecar_url:
            findings.append(finding(location, 1, "S1", "verification-mismatch",
                                    f"{key!r}: bib url {url!r} != sidecar "
                                    f"{sidecar_url!r}"))
        if not record.get("quote"):
            findings.append(finding(location, 1, "S2", "missing-quote",
                                    f"{key!r} record has no supporting quote"))
        if record.get("level") and record.get("level") not in LEVELS:
            findings.append(finding(location, 1, "S2", "invalid-level",
                                    f"{key!r} level {record.get('level')!r} not in "
                                    f"{sorted(LEVELS)}"))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check BibTeX entries against a verification sidecar.",
        epilog=(
            "Example:\n"
            "  bib_check.py --bib refs.bib --verified verified.json --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bib", type=Path, required=True)
    parser.add_argument("--verified", type=Path, default=None,
                        help="verified.json sidecar (default: next to the .bib)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    args = parser.parse_args()

    if not args.bib.is_file():
        parser.error(f"--bib does not exist: {args.bib}")
    verified_path = args.verified or args.bib.parent / "verified.json"
    entries = parse_bib(args.bib.read_text())
    records, findings = load_verified(verified_path)
    findings += check(entries, records)

    report = {
        "schema": SCHEMA,
        "bib": str(args.bib),
        "verified": str(verified_path),
        "n_entries": len(entries),
        "n_verified": len(records),
        "n_findings": len(findings),
        "n_unverified_s1": sum(1 for f in findings if f["rule"] == "unverified-citation"),
        "findings": findings,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for item in findings:
            print(f"[{item['severity']}] {item['rule']} {item['location']} "
                  f"{item['message']}")
        print(f"ENTRIES: {len(entries)} FINDINGS: {len(findings)}")
    return 1 if any(f["severity"] in ("S0", "S1") for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main())
