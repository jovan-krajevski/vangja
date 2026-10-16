#!/usr/bin/env python3
"""Verify SHA-256 sidecars for a data directory.

Accepts both the project `data_sha256.json` format
(``{"files": {...}}``) and sha256sum text (``<hash>  <path>``, or a bare hash
in a ``<name>.sha256`` sidecar). Missing files, mismatches, and (unless
``--allow-extra``) unrecorded files fail. Exit 1 on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXCLUDES = ("__pycache__", "*.pyc", ".git", ".DS_Store")


def excluded(rel: Path) -> bool:
    if any(part in rel.parts for part in EXCLUDES) or rel.name.endswith(".pyc"):
        return True
    if rel.name.endswith(".sha256") or rel.name == "data_sha256.json":
        return True
    return False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_json_sidecar(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    files = payload.get("files", payload)
    if not isinstance(files, dict):
        raise ValueError(f"{path}: no 'files' mapping")
    return {str(k): str(v) for k, v in files.items()}


def parse_text_sidecar(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            entries[" ".join(parts[1:]).lstrip("*")] = parts[0]
        else:
            base = path.name[: -len(".sha256")] if path.name.endswith(".sha256") else path.stem
            matches = sorted(
                p for p in path.parent.glob(base + ".*") if p.suffix != ".sha256" and p.is_file()
            )
            name = matches[0].name if matches else base
            entries[name] = parts[0]
    return entries


def load_sidecar(path: Path) -> dict[str, str]:
    if path.suffix == ".json" or path.name.endswith("_sha256.json"):
        return parse_json_sidecar(path)
    return parse_text_sidecar(path)


def discover(root: Path) -> list[Path]:
    found: list[Path] = []
    for base in (root, root.parent):
        for candidate in ("data_sha256.json",):
            path = base / candidate
            if path.exists() and path not in found:
                found.append(path)
        for path in sorted(base.glob("*.sha256")):
            if path not in found:
                found.append(path)
    return found


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="check_sidecars.py",
        description="Verify SHA-256 sidecars for a data directory.",
        epilog="Example: check_sidecars.py --root data/tickers --sidecar data_sha256.json",
    )
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--sidecar", action="append", default=[], type=Path,
                   help="sidecar file (repeatable; default: auto-discover)")
    p.add_argument("--allow-extra", action="store_true",
                   help="downgrade unrecorded files to warnings")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    if not args.root.is_dir():
        print(f"--root is not a directory: {args.root}", file=sys.stderr)
        return 2

    sidecars = list(args.sidecar) or discover(args.root)
    if not sidecars:
        print(f"no sidecar found for {args.root} (looked for data_sha256.json, *.sha256)",
              file=sys.stderr)
        return 2

    recorded: dict[str, str] = {}
    origins: dict[str, str] = {}
    for path in sidecars:
        if not path.exists():
            print(f"sidecar not found: {path}", file=sys.stderr)
            return 2
        try:
            entries = load_sidecar(path)
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            print(f"cannot parse sidecar {path}: {exc}", file=sys.stderr)
            return 2
        for name, digest in entries.items():
            recorded[name] = digest
            origins[name] = str(path)

    errors: list[str] = []
    warnings: list[str] = []
    checked = 0
    seen: set[str] = set()
    for name, digest in sorted(recorded.items()):
        candidate = (args.root / name)
        seen.add(name)
        if not candidate.exists():
            errors.append(f"recorded file is missing: {name} (in {origins[name]})")
            continue
        checked += 1
        if sha256_file(candidate) != digest:
            errors.append(f"hash mismatch: {name} (in {origins[name]})")

    for path in sorted(args.root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(args.root)
        if excluded(rel) or rel.as_posix() in seen:
            continue
        message = f"unrecorded file: {rel.as_posix()}"
        (warnings if args.allow_extra else errors).append(message)

    result = {
        "root": str(args.root),
        "sidecars": [str(s) for s in sidecars],
        "n_recorded": len(recorded),
        "n_checked": checked,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for message in errors:
            print(f"FAIL  {message}")
        for message in warnings:
            print(f"WARN  {message}")
        print(f"{'PASS' if not errors else 'FAIL'}  {args.root}: "
              f"{checked}/{len(recorded)} verified, {len(errors)} error(s)")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
