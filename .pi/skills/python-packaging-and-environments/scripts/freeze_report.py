#!/usr/bin/env python3
"""Emit the environment fingerprint block used in run manifests.

Records the Python implementation/version, the versions of the packages that
decide numerical behaviour (pymc, pytensor, arviz, numpy, numba, jax,
pymc-extras, blackjax, pandas, scikit-learn, vangja), plus SHA-256 of
uv.lock and of a container image when one is given. It reads
``importlib.metadata`` only (no package imports, no network), so it is safe to
run inside a job and from a bare interpreter.

Usage:
  freeze_report.py                        # print JSON to stdout
  freeze_report.py --json --out env.json  # write JSON under --out
  freeze_report.py --container case_studies/vangja.sif

Exit codes: 0 emitted, 2 usage error (missing file).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

SCHEMA = "vangja-freeze-report-1.0"
PACKAGES = [
    "vangja",
    "pymc",
    "pytensor",
    "arviz",
    "numpy",
    "numba",
    "jax",
    "jaxlib",
    "pymc-extras",
    "blackjax",
    "pandas",
    "scikit-learn",
    "scipy",
    "nutpie",
    "preliz",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build(repo: Path, container: Path | None) -> dict:
    uv_lock = repo / "uv.lock"
    report = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": package_versions(),
        "uv_lock": None,
        "container": None,
    }
    if uv_lock.is_file():
        report["uv_lock"] = {"path": str(uv_lock), "sha256": sha256_file(uv_lock)}
    if container is not None:
        report["container"] = {
            "path": str(container),
            "sha256": sha256_file(container),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Emit the environment fingerprint block used in manifests.",
        epilog=(
            "Example:\n"
            "  freeze_report.py --json --out scratch/run-env.json \\\n"
            "    --container case_studies/vangja.sif\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--repo", type=Path, default=Path("."),
                        help="repository root holding uv.lock")
    parser.add_argument("--container", type=Path,
                        help="container image whose SHA-256 to record")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON (always true; kept for symmetry)")
    parser.add_argument("--out", type=Path,
                        help="write the JSON report to this file")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not repo.is_dir():
        parser.error(f"--repo is not a directory: {repo}")
    if args.container is not None and not args.container.is_file():
        parser.error(f"--container does not exist: {args.container}")

    report = build(repo, args.container)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized)
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
