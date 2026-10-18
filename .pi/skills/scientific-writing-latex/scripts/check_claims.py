#!/usr/bin/env python3
"""Check that every number in the manuscript has a claims-registry entry.

Rules:

  hand-typed-number   a number lives outside the generated include directory
  dangling-number     a number has no CLAIMS.json entry with the same value
  invalid-claim       a claim is missing a required field
  unverified-claim    a claim's status is not ``verified``
  orphan-claim        a claim's value appears nowhere in the scanned sources
  sha-not-ancestor    a claim's commit is not an ancestor of the base ref
  ancestry-unchecked  git or the base ref is unavailable (informational)

Mode: by default the checker **reports** (exit 0) so the legacy paper can be
scanned while drafting; ``--gate`` exits 1 when any finding exists.

Usage:
  check_claims.py --paper paper/paper.tex --claims CLAIMS.json
  check_claims.py --paper paper/paper.tex --claims CLAIMS.json --gate --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

SCHEMA = "vangja-claims-check-1.0"
CLAIMS_SCHEMA = "vangja-claims-1.0"
GENERATED_DIR = "generated"

SKIP_ENVIRONMENTS = {
    "document", "abstract", "equation", "align", "amsmath", "enumerate",
    "itemize", "table", "tabular", "figure", "center", "theorem", "proof",
    "split", "gather", "multline", "cases", "array", "verbatim", "tikzpicture",
}
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.\\{])(\d+(?:\.\d+)?)(?![A-Za-z0-9_}])")
COMMAND_ARG_RE = re.compile(
    r"\\(?:ref|cite|label|input|include|includegraphics|bibliography|"
    r"usepackage|documentclass|begin|end|pageref|eqref)\s*(?:\[[^\]]*\])?\{[^{}]*\}"
)
NUM_MACRO_RE = re.compile(r"\\num\{([^{}]*)\}")
COMMENT_RE = re.compile(r"(?<!\\)%.*$")


def strip_latex(text: str) -> str:
    text = COMMENT_RE.sub("", text)
    text = COMMAND_ARG_RE.sub(" ", text)
    return text


def numbers_in(text: str) -> list[tuple[str, int]]:
    """Return (normalised value, line number) for every number-like token."""
    out: list[tuple[str, int]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for match in NUM_MACRO_RE.finditer(line):
            out.append((normalise(match.group(1)), line_no))
        cleaned = strip_latex(line)
        for match in NUMBER_RE.finditer(cleaned):
            out.append((normalise(match.group(1)), line_no))
    return out


def normalise(value: str) -> str:
    value = value.strip().strip("$").rstrip("%").rstrip(".")
    return value


def expand_inputs(paper: Path) -> list[Path]:
    files = [paper]
    text = paper.read_text()
    for match in re.finditer(r"\\input\{([^{}]+)\}", text):
        target = (paper.parent / match.group(1)).resolve()
        if target.suffix != ".tex":
            target = target.with_suffix(".tex")
        if target.exists() and target not in files:
            files.append(target)
    return files


def load_claims(path: Path) -> tuple[list[dict], list[dict]]:
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text())
    findings: list[dict] = []
    if payload.get("schema") != CLAIMS_SCHEMA:
        findings.append(finding(str(path), 1, "S2", "invalid-claim",
                                f"schema is {payload.get('schema')!r}, "
                                f"expected {CLAIMS_SCHEMA!r}"))
    claims = payload.get("claims", [])
    required = ("id", "value", "tex", "artifact", "artifact_sha256", "commit",
                "status")
    for index, claim in enumerate(claims, start=1):
        missing = [field for field in required if not claim.get(field)]
        line = claim.get("line", index)
        if missing:
            findings.append(finding(str(path), line, "S2", "invalid-claim",
                                    f"{claim.get('id', f'#{index}')} missing {missing}"))
        if claim.get("status") not in ("verified", "pending", "stale"):
            findings.append(finding(str(path), line, "S2", "invalid-claim",
                                    f"{claim.get('id')} has status "
                                    f"{claim.get('status')!r}"))
    return claims, findings


def finding(location: str, line: int, severity: str, rule: str, message: str) -> dict:
    return {"location": f"{location}:{line}", "severity": severity,
            "rule": rule, "message": message}


def artifact_findings(claims: list[dict], repo: Path) -> list[dict]:
    findings: list[dict] = []
    for claim in claims:
        artifact = claim.get("artifact")
        expected = claim.get("artifact_sha256")
        if not artifact:
            continue
        path = repo / artifact
        if not path.is_file():
            findings.append(finding(claim.get("tex", str(repo)), 1, "S2",
                                    "artifact-missing",
                                    f"{claim.get('id')} cites a missing artifact: "
                                    f"{artifact}"))
            continue
        if expected:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != str(expected).lower():
                findings.append(finding(claim.get("tex", str(repo)), 1, "S1",
                                        "artifact-hash-mismatch",
                                        f"{claim.get('id')}: {artifact} hashes to "
                                        f"{actual[:12]}..., claim says {str(expected)[:12]}..."))
    return findings


def ancestry_failures(claims: list[dict], repo: Path, base_ref: str) -> list[dict]:
    if base_ref.lower() in ("none", ""):
        return []
    try:
        head = subprocess.run(["git", "rev-parse", "--verify", base_ref],
                              cwd=repo, capture_output=True, text=True)
    except FileNotFoundError:
        return [finding(str(repo), 1, "S3", "ancestry-unchecked",
                        "git is not available; ancestry not checked")]
    if head.returncode != 0:
        return [finding(str(repo), 1, "S3", "ancestry-unchecked",
                        f"base ref {base_ref!r} not found; ancestry not checked")]
    findings = []
    for claim in claims:
        commit = claim.get("commit", "")
        proc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, base_ref],
            cwd=repo, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            findings.append(finding(claim.get("tex", str(repo)),
                                    claim.get("line", 1), "S1", "sha-not-ancestor",
                                    f"{claim.get('id')} cites {commit!r}, not an "
                                    f"ancestor of {base_ref}"))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the manuscript's numbers against the claims registry.",
        epilog=(
            "Examples:\n"
            "  check_claims.py --paper paper/paper.tex --claims CLAIMS.json\n"
            "  check_claims.py --paper paper/paper.tex --claims CLAIMS.json --gate\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--paper", type=Path, default=Path("paper/paper.tex"))
    parser.add_argument("--claims", type=Path, default=Path("CLAIMS.json"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--base-ref", default="update_pymc",
                        help="ancestry base (use 'none' to skip)")
    parser.add_argument("--generated-dir", default=None,
                        help="directory allowed to hold generated numbers "
                             "(default: a 'generated' directory under the paper's parent)")
    parser.add_argument("--gate", action="store_true",
                        help="exit 1 when any finding exists (default: report only)")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    args = parser.parse_args()

    paper = args.paper.resolve()
    if not paper.is_file():
        parser.error(f"--paper does not exist: {paper}")
    generated_dir = (args.generated_dir or str(paper.parent / GENERATED_DIR)).rstrip("/")

    sources = expand_inputs(paper)
    tokens: list[tuple[str, Path, int]] = []
    findings: list[dict] = []
    for source in sources:
        text = source.read_text()
        for value, line in numbers_in(text):
            tokens.append((value, source, line))
            if not str(source).startswith(str(Path(generated_dir).resolve())):
                findings.append(finding(str(source), line, "S1",
                                        "hand-typed-number",
                                        f"{value!r} is not in a generated include"))

    claims, claim_findings = load_claims(args.claims.resolve())
    findings += claim_findings
    findings += artifact_findings(claims, args.repo.resolve())
    claim_values = {}
    for claim in claims:
        if claim.get("status") == "verified" and claim.get("value"):
            claim_values.setdefault(normalise(str(claim["value"])), []).append(claim)
        elif claim.get("status") and claim.get("status") != "verified":
            findings.append(finding(claim.get("tex", str(args.claims)), 1, "S2",
                                    "unverified-claim",
                                    f"{claim.get('id')} status={claim.get('status')}"))

    used_values = set()
    for value, source, line in tokens:
        if value in claim_values:
            used_values.add(value)
        else:
            findings.append(finding(str(source), line, "S1", "dangling-number",
                                    f"{value!r} has no verified claims entry"))
    for value, entries in claim_values.items():
        if value not in used_values:
            for claim in entries:
                findings.append(finding(claim.get("tex", str(args.claims)), 1, "S2",
                                        "orphan-claim",
                                        f"{claim.get('id')} value {value!r} not "
                                        "found in the paper"))

    findings += ancestry_failures(claims, args.repo.resolve(), args.base_ref)

    report = {
        "schema": SCHEMA,
        "paper": str(paper),
        "sources": [str(s) for s in sources],
        "n_numbers": len(tokens),
        "n_claims": len(claims),
        "n_findings": len(findings),
        "by_severity": {
            sev: sum(1 for f in findings if f["severity"] == sev)
            for sev in ("S0", "S1", "S2", "S3")
        },
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
        print(f"FINDINGS: {len(findings)} (S1={report['by_severity']['S1']})")
        print(f"numbers={len(tokens)} claims={len(claims)}")
    if args.gate and findings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
