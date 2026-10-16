#!/usr/bin/env python3
"""Validate an audit findings register (plan/07 §4).

Checks: required fields, id format and uniqueness, severity/confidence/status
enums, the "no command → not a finding" rule, the ``reproducible`` field, and a
dedupe hash over ``title + location``. Emits meta-findings about the register
itself; it never edits the register.

Usage:
  validate_findings.py --findings audit/cycle-3/ws-s/findings.json
  validate_findings.py --findings findings.json --workstream WS-S --json --out report.json

Exit codes: 0 valid, 1 meta-findings, 2 usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA = "vangja-findings-validation-1.0"
REQUIRED = (
    "id", "title", "location", "snapshot", "command", "expected", "observed",
    "consequence", "severity", "confidence", "status", "reproducible",
)
SEVERITIES = {"S0", "S1", "S2", "S3"}
CONFIDENCES = {"high", "medium", "low"}
STATUSES = {"verified", "to-verify", "refuted", "unresolved"}
ID_RE = re.compile(r"^([A-Z]{1,3}(?:-[A-Z]{1,3})?)-(\d{3})$")
COMMAND_TOKENS = (
    "pytest", "python", "uv run", "git ", "grep", "find ", "sha256sum",
    "diff ", "curl", "sqlite3", "jq ", "awk ", "sed ", "rg ", "test:",
)
STATIC_READING = re.compile(r"^static (reading|read|inspection)", re.IGNORECASE)


def validate(findings: list[dict], workstream: str | None) -> list[dict]:
    meta: list[dict] = []
    seen_ids: dict[str, int] = {}
    seen_hashes: dict[str, str] = {}
    expected_prefix = workstream if workstream else None

    for index, finding in enumerate(findings, start=1):
        label = str(finding.get("id", f"#{index}"))
        for field in REQUIRED:
            if finding.get(field) in (None, "", []):
                meta.append(_mf(label, "missing-field", f"missing field {field!r}"))
        finding_id = str(finding.get("id", ""))
        match = ID_RE.match(finding_id)
        if not match:
            meta.append(_mf(label, "bad-id", f"id {finding_id!r} is not <PREFIX>-NNN"))
        elif expected_prefix and match.group(1) != expected_prefix:
            meta.append(_mf(label, "wrong-workstream",
                            f"id prefix {match.group(1)!r} != {expected_prefix!r}"))
        if finding_id in seen_ids:
            meta.append(_mf(label, "duplicate-id",
                            f"id {finding_id!r} already used"))
        seen_ids[finding_id] = 1

        severity = finding.get("severity")
        if severity not in SEVERITIES:
            meta.append(_mf(label, "bad-severity", f"severity {severity!r}"))
        confidence = finding.get("confidence")
        if confidence not in CONFIDENCES:
            meta.append(_mf(label, "bad-confidence", f"confidence {confidence!r}"))
        status = finding.get("status")
        if status not in STATUSES:
            meta.append(_mf(label, "bad-status", f"status {status!r}"))
        if status == "verified" and confidence == "low":
            meta.append(_mf(label, "verified-low-confidence",
                            "status=verified with confidence=low is a contradiction"))

        command = str(finding.get("command", ""))
        command_like = (
            any(token in command for token in COMMAND_TOKENS)
            or bool(STATIC_READING.match(command))
        )
        if not command_like:
            meta.append(_mf(label, "no-command",
                            f"command {command!r} is not a command or a static "
                            "reading; without one this is not a finding"))

        reproducible = str(finding.get("reproducible", ""))
        if reproducible != "yes" and not reproducible.startswith("no ("):
            meta.append(_mf(label, "bad-reproducible",
                            f"reproducible {reproducible!r} must be 'yes' or "
                            "'no (<why>)'"))
        if reproducible.startswith("no") and severity in ("S0", "S1") and status == "verified":
            meta.append(_mf(label, "unreproducible-high-severity",
                            "a verified S0/S1 must be reproducible; downgrade or "
                            "defer it"))

        dedupe = hashlib.sha256(
            (" ".join(str(finding.get("title", "")).lower().split())
             + "|" + " ".join(str(finding.get("location", "")).lower().split())
             ).encode()
        ).hexdigest()[:16]
        if dedupe in seen_hashes:
            meta.append(_mf(label, "duplicate-finding",
                            f"same title+location as {seen_hashes[dedupe]}"))
        seen_hashes[dedupe] = label

    return meta


def _mf(location: str, rule: str, message: str) -> dict:
    return {"location": location, "rule": rule, "message": message}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate an audit findings register.",
        epilog=(
            "Example:\n"
            "  validate_findings.py --findings findings.json --workstream WS-S --json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--findings", type=Path, required=True)
    parser.add_argument("--workstream", default=None,
                        help="expected id prefix, e.g. WS-S")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    args = parser.parse_args()

    if not args.findings.is_file():
        parser.error(f"--findings does not exist: {args.findings}")
    payload = json.loads(args.findings.read_text())
    findings = payload if isinstance(payload, list) else payload.get("findings", [])
    if not isinstance(findings, list):
        parser.error("findings.json must be a list or {'findings': [...]}")

    meta = validate(findings, args.workstream)
    report = {
        "schema": SCHEMA,
        "findings": str(args.findings),
        "workstream": args.workstream,
        "n_findings": len(findings),
        "n_meta_findings": len(meta),
        "severity_histogram": {
            sev: sum(1 for f in findings if f.get("severity") == sev)
            for sev in ("S0", "S1", "S2", "S3")
        },
        "meta_findings": meta,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for item in meta:
            print(f"[META] {item['rule']} {item['location']} {item['message']}")
        print(f"FINDINGS: {len(findings)} META: {len(meta)}")
    return 1 if meta else 0


if __name__ == "__main__":
    sys.exit(main())
