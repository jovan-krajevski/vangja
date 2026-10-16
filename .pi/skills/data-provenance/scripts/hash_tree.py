#!/usr/bin/env python3
"""Hash a file tree with SHA-256 and write a hash manifest.

Writes either the project `data_sha256.json` shape
(``{"created_at": ..., "files": {relpath: sha256}}``) or sha256sum text.
Only writes to ``--out`` / ``--sidecar``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

DEFAULT_EXCLUDES = ("__pycache__", "*.pyc", ".git", ".DS_Store")


def excluded(rel: Path, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if pattern in rel.parts:
            return True
        if pattern.startswith("*") and rel.name.endswith(pattern[1:]):
            return True
    return False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_tree(root: Path, patterns: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if excluded(rel, patterns):
            continue
        out[rel.as_posix()] = sha256_file(path)
    return out


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".hash-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hash_tree.py",
        description="SHA-256 hash a directory and write a manifest.",
        epilog="Example: hash_tree.py --root data/tickers --out data/tickers_sha256.json",
    )
    p.add_argument("--root", required=True, type=Path, help="directory to hash")
    p.add_argument("--out", type=Path, default=None,
                   help="write data_sha256.json-format output here")
    p.add_argument("--sidecar", type=Path, default=None,
                   help="write sha256sum-format output here")
    p.add_argument("--exclude", action="append", default=[],
                   help=f"extra exclude pattern (default: {', '.join(DEFAULT_EXCLUDES)})")
    args = p.parse_args(argv)

    if not args.root.is_dir():
        print(f"--root is not a directory: {args.root}", file=sys.stderr)
        return 2
    if not args.out and not args.sidecar:
        print("nothing to write: pass --out and/or --sidecar", file=sys.stderr)
        return 2

    patterns = tuple(DEFAULT_EXCLUDES) + tuple(args.exclude)
    try:
        hashes = hash_tree(args.root, patterns)
    except OSError as exc:
        print(f"hashing failed: {exc}", file=sys.stderr)
        return 2

    if args.out:
        payload = {
            "created_at": datetime.now().isoformat(),
            "algorithm": "sha256",
            "root": args.root.name,
            "files": hashes,
        }
        atomic_write(args.out, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.sidecar:
        lines = [f"{digest}  {rel}" for rel, digest in sorted(hashes.items())]
        atomic_write(args.sidecar, "\n".join(lines) + "\n")

    print(json.dumps({
        "root": str(args.root),
        "n_files": len(hashes),
        "out": str(args.out) if args.out else None,
        "sidecar": str(args.sidecar) if args.sidecar else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
