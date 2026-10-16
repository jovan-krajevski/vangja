#!/usr/bin/env python3
"""Write a `vangja-run-manifest-1.0` manifest for a run directory.

Records: commit/branch/dirty, package versions, uv.lock hash, command + argv,
seeds, data and config hashes, start/end timestamps, status and exit code.
Writes atomically into --out only; refuses to overwrite an existing manifest
unless --force is given.

See references/manifest-schema.md for the schema.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "vangja-run-manifest-1.0"
PACKAGES = ("vangja", "pymc", "pymc-extras", "arviz", "numpy", "pandas", "pytensor", "nutpie", "numba")
DEFAULT_EXCLUDES = ("__pycache__", "*.pyc", ".git", ".ipynb_checkpoints")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def excluded(rel: Path, globs: tuple[str, ...]) -> bool:
    parts = rel.parts
    for pattern in globs:
        if pattern in parts:
            return True
        if pattern.startswith("*") and rel.name.endswith(pattern[1:]):
            return True
    return False


def hash_tree(root: Path, globs: tuple[str, ...] = DEFAULT_EXCLUDES) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if excluded(rel, globs):
            continue
        out[rel.as_posix()] = sha256_file(path)
    return out


def repo_root(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def git_info(root: Path | None) -> dict:
    if root is None:
        return {"commit": "unknown", "branch": "unknown", "dirty": None}
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            return "unknown"
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": branch,
        "dirty": (status not in ("unknown", "")),
    }


def environment_info(root: Path | None) -> dict:
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = None
    lock = None
    if root is not None and (root / "uv.lock").exists():
        lock = sha256_file(root / "uv.lock")
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "packages": packages,
        "uv_lock_sha256": lock,
    }


def parse_seed(text: str, index: int) -> tuple[str, int]:
    if "=" in text:
        label, value = text.split("=", 1)
        label = label.strip()
        if not label:
            raise argparse.ArgumentTypeError(f"empty seed label in {text!r}")
    else:
        label, value = f"seed_{index}", text
    try:
        return label, int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"seed value must be an int: {text!r}") from exc


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".manifest-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_manifest.py",
        description="Write a vangja-run-manifest-1.0 for a run directory.",
        epilog="Example: make_manifest.py --out results/run42 --data data/tickers "
               "--base-seed 42 --seed origin=0 --seed cell=42 --status ok",
    )
    p.add_argument("--out", required=True, type=Path, help="run directory (only write target)")
    p.add_argument("--name", default="manifest.json", help="manifest file name (default: manifest.json)")
    p.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    p.add_argument("--command", nargs=argparse.REMAINDER, default=[],
                   help="logical command line (everything after the flag; must be last)")
    p.add_argument("--data", action="append", default=[], type=Path,
                   help="data directory to hash (repeatable)")
    p.add_argument("--config", action="append", default=[], type=Path,
                   help="config file to hash (repeatable)")
    p.add_argument("--base-seed", type=int, default=42, help="frozen base seed (default 42)")
    p.add_argument("--seed", action="append", default=[], metavar="[LABEL=]INT",
                   help="recorded seed (repeatable)")
    p.add_argument("--status", choices=("ok", "failed", "running"), default="ok")
    p.add_argument("--exit-code", type=int, default=None,
                   help="exit code (default 0 for ok, 1 for failed)")
    p.add_argument("--error", default=None, help="error message when --status failed")
    p.add_argument("--started-at", default=None, help="ISO-8601 start time (default: now)")
    p.add_argument("--ended-at", default=None, help="ISO-8601 end time (default: now)")
    p.add_argument("--fit-info", type=Path, default=None, help="JSON file merged into fit_info")
    p.add_argument("--aggregate", type=Path, default=None, help="JSON file merged into aggregate")
    p.add_argument("--study", default=None)
    p.add_argument("--stage", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    manifest_path = args.out / args.name
    if manifest_path.exists() and not args.force:
        print(f"refusing to overwrite {manifest_path}; pass --force", file=sys.stderr)
        return 2
    if not args.command:
        print("--command is required: record the logical command line", file=sys.stderr)
        return 2
    if args.status == "failed" and not args.error:
        print("--status failed requires --error", file=sys.stderr)
        return 2
    if args.status == "ok" and args.error:
        print("--error is only valid with --status failed", file=sys.stderr)
        return 2

    exit_code = args.exit_code
    if exit_code is None:
        exit_code = {"ok": 0, "failed": 1, "running": None}[args.status]

    root = repo_root(Path.cwd().resolve())
    now = utc_now()

    data_hashes: dict[str, dict] = {}
    for i, data_dir in enumerate(args.data):
        if not data_dir.exists():
            print(f"--data path does not exist: {data_dir}", file=sys.stderr)
            return 2
        key = data_dir.name if data_dir.name not in data_hashes else f"{data_dir.name}_{i}"
        try:
            rel_dir = data_dir.resolve().relative_to(root).as_posix() if root else str(data_dir)
        except ValueError:
            rel_dir = str(data_dir)
        data_hashes[key] = {"path": rel_dir, "files": hash_tree(data_dir)}

    config_hashes: dict[str, str] = {}
    for cfg in args.config:
        if not cfg.exists():
            print(f"--config path does not exist: {cfg}", file=sys.stderr)
            return 2
        try:
            rel = cfg.resolve().relative_to(root) if root else cfg
            config_hashes[rel.as_posix()] = sha256_file(cfg)
        except ValueError:
            config_hashes[str(cfg)] = sha256_file(cfg)

    seeds: dict[str, int] = {}
    for i, text in enumerate(args.seed):
        label, value = parse_seed(text, i)
        if label in seeds:
            print(f"duplicate seed label: {label}", file=sys.stderr)
            return 2
        seeds[label] = value

    payload: dict = {
        "manifest_schema": SCHEMA,
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + os.urandom(4).hex(),
        "created_at": now,
        "started_at": args.started_at or now,
        "ended_at": args.ended_at or now,
        "status": args.status,
        "exit_code": exit_code,
        "command": list(args.command),
        "argv": list(sys.argv),
        "git": git_info(root),
        "environment": environment_info(root),
        "seeds": {"base_seed": args.base_seed, "values": seeds},
        "data": data_hashes,
        "configs": config_hashes,
    }
    if args.error:
        payload["error"] = args.error
    if args.study:
        payload["study"] = args.study
    if args.stage:
        payload["stage"] = args.stage
    if args.fit_info:
        payload["fit_info"] = json.loads(args.fit_info.read_text())
    if args.aggregate:
        payload["aggregate"] = json.loads(args.aggregate.read_text())

    atomic_write_json(manifest_path, payload)
    print(json.dumps({
        "manifest": str(manifest_path),
        "run_id": payload["run_id"],
        "status": payload["status"],
        "n_data_hashes": sum(len(v["files"]) for v in data_hashes.values()),
        "n_config_hashes": len(config_hashes),
        "commit": payload["git"]["commit"][:12],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
