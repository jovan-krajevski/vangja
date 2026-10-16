#!/usr/bin/env python3
"""Validate a `vangja-run-manifest-1.0` manifest and its hashed files.

Checks the schema, status/exit-code consistency, hash formats, seed types, and
(with --root) recomputes every data/config hash. Exit 0 only when every check
passes. Reads the manifest and the files it names; writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA = "vangja-run-manifest-1.0"
HEX40 = re.compile(r"^[0-9a-f]{7,40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED = (
    "manifest_schema", "run_id", "created_at", "started_at", "ended_at",
    "status", "exit_code", "command", "git", "environment", "seeds",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(manifest: dict, root: Path | None, allow_missing_data: bool,
             allow_unknown_commit: bool) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if manifest.get("manifest_schema") != SCHEMA:
        errors.append(f"manifest_schema must be {SCHEMA!r}, got {manifest.get('manifest_schema')!r}")
    for key in REQUIRED:
        if key not in manifest:
            errors.append(f"missing required field: {key}")
    if errors:
        return errors, warnings

    if manifest["status"] not in ("ok", "failed", "running"):
        errors.append(f"invalid status: {manifest['status']!r}")
    if manifest["status"] == "ok" and manifest["exit_code"] != 0:
        errors.append("status ok requires exit_code 0")
    if manifest["status"] == "failed":
        if not manifest.get("error"):
            errors.append("status failed requires a non-empty 'error'")
        if manifest["exit_code"] == 0:
            errors.append("status failed must not have exit_code 0")

    if not isinstance(manifest["command"], list) or not manifest["command"]:
        errors.append("command must be a non-empty list")
    if "argv" in manifest and not isinstance(manifest["argv"], list):
        errors.append("argv must be a list")

    git = manifest["git"]
    commit = git.get("commit")
    if not isinstance(commit, str):
        errors.append("git.commit must be a string")
    elif commit == "unknown":
        if not allow_unknown_commit:
            errors.append("git.commit is 'unknown'; rerun the manifest step inside a git worktree")
        else:
            warnings.append("git.commit is 'unknown' (allowed by --allow-unknown-commit)")
    elif not HEX40.match(commit):
        errors.append(f"git.commit is not a hex SHA: {commit!r}")

    env = manifest["environment"]
    if not isinstance(env, dict):
        errors.append("environment must be an object")
    else:
        if "python" not in env:
            warnings.append("environment.python missing")
        if not isinstance(env.get("packages", {}), dict):
            errors.append("environment.packages must be an object")

    seeds = manifest["seeds"]
    if not isinstance(seeds, dict) or not isinstance(seeds.get("values"), dict):
        errors.append("seeds.values must be an object")
    else:
        for label, value in seeds["values"].items():
            if not label or not isinstance(value, int) or value < 0:
                errors.append(f"bad seed entry {label!r}: {value!r}")

    data = manifest.get("data", {})
    if not isinstance(data, dict):
        errors.append("data must be an object")
        data = {}
    for dirname, entry in data.items():
        # Schema v1.0: {"path": <repo-relative>, "files": {rel: sha256}}.
        if isinstance(entry, dict) and "files" in entry:
            base_text = entry.get("path", "")
            hashmap = entry.get("files", {})
        else:  # tolerate a bare {rel: sha256} map (no path information)
            base_text, hashmap = "", entry
        if not isinstance(hashmap, dict):
            errors.append(f"data[{dirname!r}] files must be an object")
            continue
        base = Path(base_text) if base_text else None
        if base is not None and not base.is_absolute() and root is not None:
            base = root / base
        for rel, digest in hashmap.items():
            if not isinstance(digest, str) or not HEX64.match(digest):
                errors.append(f"bad sha256 for {dirname}/{rel}")
                continue
            if root is None:
                continue
            candidate = (base / rel) if base is not None else (root / rel)
            if not candidate.exists():
                message = f"hashed data file missing: {candidate}"
                (warnings if allow_missing_data else errors).append(message)
                continue
            if sha256_file(candidate) != digest:
                errors.append(f"data hash mismatch: {candidate}")

    configs = manifest.get("configs", {})
    if isinstance(configs, dict):
        for rel, digest in configs.items():
            if not isinstance(digest, str) or not HEX64.match(digest):
                errors.append(f"bad config sha256 for {rel}")
                continue
            if root is None:
                continue
            candidate = root / rel
            if not candidate.exists():
                (warnings if allow_missing_data else errors).append(
                    f"hashed config file missing: {candidate}")
            elif sha256_file(candidate) != digest:
                errors.append(f"config hash mismatch: {candidate}")
    elif configs:
        errors.append("configs must be an object")

    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="check_manifest.py",
        description="Validate a vangja-run-manifest-1.0 and recompute its hashes.",
        epilog="Example: check_manifest.py results/run42/manifest.json --root .",
    )
    p.add_argument("manifest", type=Path)
    p.add_argument("--root", type=Path, default=None,
                   help="repo/data root used to resolve hashed relative paths")
    p.add_argument("--allow-missing-data", action="store_true",
                   help="downgrade missing hashed files to warnings")
    p.add_argument("--allow-unknown-commit", action="store_true",
                   help="accept git.commit == 'unknown'")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = p.parse_args(argv)

    if not args.manifest.exists():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(args.manifest.read_text())
    except json.JSONDecodeError as exc:
        print(f"manifest is not valid JSON: {exc}", file=sys.stderr)
        return 2

    errors, warnings = validate(
        manifest, args.root, args.allow_missing_data, args.allow_unknown_commit
    )
    ok = not errors
    if args.json:
        print(json.dumps({"ok": ok, "errors": errors, "warnings": warnings}, indent=2))
    else:
        for message in errors:
            print(f"FAIL  {message}")
        for message in warnings:
            print(f"WARN  {message}")
        print(f"{'PASS' if ok else 'FAIL'}  {args.manifest}"
              f"  ({len(errors)} error(s), {len(warnings)} warning(s))")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
