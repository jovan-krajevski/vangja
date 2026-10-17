#!/usr/bin/env python3
"""Validate an audit findings register (plan/07 §4).

Checks: required fields, id format and uniqueness, severity/confidence/status
enums, the "no command → not a finding" rule (a static reading is not a command),
the ``reproducible`` field, and a dedupe hash over normalized ``title + location``.
Duplicates are folded into a canonical finding with a cross-ref; the report
carries the deduped register and the duplicate map. Emits meta-findings about the
register itself; it never edits the input file.

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
    "evidence",
)
SEVERITIES = {"S0", "S1", "S2", "S3"}
CONFIDENCES = {"high", "medium", "low"}
STATUSES = {"verified", "to-verify", "refuted", "unresolved"}
ID_RE = re.compile(r"^([A-Z]{1,3}(?:-[A-Z]{1,3})?)-(\d{3})$")
SHELL_FIRST_TOKENS = {
    "uv", "pytest", "python", "python3", "node", "bash", "sh", "git", "grep",
    "rg", "find", "ls", "cat", "head", "tail", "sed", "awk", "jq", "diff",
    "cmp", "sha256sum", "shasum", "md5sum", "make", "npm", "npx", "pip",
    "uvx", "sqlite3", "test", "true", "false", "wc", "sort", "uniq", "xargs",
    "cp", "mv", "rm", "mkdir", "touch", "curl", "wget", "tar", "unzip", "zip",
}
STATIC_READING = re.compile(
    r"^(static (reading|read|inspection)|reading|manual( inspection)?|"
    r"inspection|none|n/?a|tbd|todo|see above|as discussed)\b",
    re.IGNORECASE,
)
OWNERS = {"code", "paper", "protocol", "docs", "env"}
PHASES = {"finder", "verify", "synth", "adjudicate", "canary"}
REPRODUCIBLE = re.compile(r"^(yes\b.*|no\b[: (].+)$", re.IGNORECASE)


def _executable(command: str) -> bool:
    trimmed = command.strip()
    if len(trimmed) < 4 or STATIC_READING.match(trimmed):
        return False
    first = trimmed.split()[0]
    base = first.rsplit("/", 1)[-1]
    if first.startswith(("./", "/", "../")):
        return True
    if base in SHELL_FIRST_TOKENS:
        return True
    return bool(re.match(r"^[\w.-]+\.(py|sh|js|ts|mjs|pl|rb)$", base))


def _normalize_location(location: str) -> str:
    text = location.strip().lower().replace("\\", "/")
    text = re.sub(r"^./", "", text)
    text = re.sub(r":\d+(?:-\d+)?(?::\d+)?$", "", text)
    return re.sub(r"\s+", " ", text).replace("`", "").replace("'", "").replace('"', "")


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", title.lower())).strip()


def _dedupe_key(finding: dict) -> str:
    payload = f"{_normalize_location(str(finding.get('location', '')))}|{_normalize_title(str(finding.get('title', '')))}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def validate(findings: list[dict], workstream: str | None) -> tuple[list[dict], dict]:
    meta: list[dict] = []
    seen_ids: dict[str, int] = {}
    seen_hashes: dict[str, dict] = {}
    duplicates: list[dict] = []
    register = [dict(finding) for finding in findings]
    expected_prefix = workstream if workstream else None
    expected_plain = expected_prefix[3:] if expected_prefix and expected_prefix.startswith("WS-") else expected_prefix

    for index, finding in enumerate(findings, start=1):
        label = str(finding.get("id", f"#{index}"))
        for field in REQUIRED:
            if finding.get(field) in (None, "", []):
                meta.append(_mf(label, "missing-field", f"missing field {field!r}"))
        finding_id = str(finding.get("id", ""))
        match = ID_RE.match(finding_id)
        if not match:
            meta.append(_mf(label, "bad-id", f"id {finding_id!r} is not <PREFIX>-NNN"))
        elif expected_prefix and match.group(1) not in {expected_prefix, expected_plain}:
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

        evidence = finding.get("evidence")
        if evidence is not None and (not isinstance(evidence, list) or not evidence):
            meta.append(_mf(label, "missing-field", "evidence must be a non-empty list"))

        command = str(finding.get("command", ""))
        if not _executable(command):
            meta.append(_mf(label, "no-command",
                            f"command {command!r} is not an executable command "
                            "(a static reading is not a command); without one this "
                            "is not a finding"))

        reproducible = str(finding.get("reproducible", ""))
        if not REPRODUCIBLE.match(reproducible) or reproducible.strip().lower() == "no":
            meta.append(_mf(label, "bad-reproducible",
                            f"reproducible {reproducible!r} must be 'yes' (optionally "
                            "explained) or 'no: <why>'"))
        if status == "verified" and not reproducible.lower().startswith("yes"):
            meta.append(_mf(label, "unreproducible-verified",
                            "a verified finding must be reproducible; downgrade it "
                            "to to-verify or drop it"))

        owner = finding.get("owner")
        if owner is not None and owner not in OWNERS:
            meta.append(_mf(label, "bad-owner", f"owner {owner!r} is not one of {sorted(OWNERS)}"))
        phase = finding.get("phase")
        if phase is not None and phase not in PHASES:
            meta.append(_mf(label, "bad-phase", f"phase {phase!r} is not one of {sorted(PHASES)}"))
        if status == "unresolved" and finding.get("question_tag") not in ("blocking", "non-blocking"):
            meta.append(_mf(label, "unresolved-no-question-tag",
                            "an unresolved finding needs question_tag blocking|non-blocking"))

        dedupe = _dedupe_key(finding)
        if dedupe in seen_hashes:
            canonical = seen_hashes[dedupe]
            canonical_id = str(canonical.get("id", "?"))
            meta.append(_mf(label, "duplicate-finding",
                            f"same normalized title+location as {canonical_id}"))
            duplicates.append({
                "id": finding_id or label,
                "duplicate_of": canonical_id,
                "key": dedupe,
            })
            refs = canonical.setdefault("cross_refs", [])
            if isinstance(refs, list) and finding_id and finding_id not in refs:
                refs.append(finding_id)
            register[index - 1]["duplicate_of"] = canonical_id
        else:
            seen_hashes[dedupe] = finding

    return meta, {
        "register": [entry for entry in register if "duplicate_of" not in entry],
        "duplicates": duplicates,
        "dedupe_keys": {str(entry.get("id", f"#{index}")): _dedupe_key(entry)
                        for index, entry in enumerate(findings, start=1)},
    }


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

    meta, deduped = validate(findings, args.workstream)
    report = {
        "schema": SCHEMA,
        "findings": str(args.findings),
        "workstream": args.workstream,
        "n_findings": len(findings),
        "n_deduped": len(deduped["register"]),
        "n_meta_findings": len(meta),
        "register": deduped["register"],
        "duplicates": deduped["duplicates"],
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
