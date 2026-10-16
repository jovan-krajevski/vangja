#!/usr/bin/env bash
# Validate the project skills under .pi/skills (or a directory passed as $1).
#
# Checks per skill directory with SKILL.md:
#   * frontmatter parses and declares name + description
#   * name matches the directory, is a valid skill name, description <= 1024
#   * SKILL.md <= 120 lines
#   * every path mentioned in SKILL.md resolves (skill dir, repo root, or basename)
#   * no absolute paths outside the repo and no secret material in any file
#   * .pi/skills/README.md mentions every skill
#
# Exit 0 only when every check passes.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SKILLS_DIR="${1:-$ROOT/.pi/skills}"
REPO_ROOT="${2:-$ROOT}"

exec python3 - "$SKILLS_DIR" "$REPO_ROOT" <<'PY'
import re
import sys
from pathlib import Path

skills_dir = Path(sys.argv[1]).resolve()
repo_root = Path(sys.argv[2]).resolve()

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
PATH_TOKEN = re.compile(r"`([^`\n]+)`")
MD_LINK = re.compile(r"\]\(([^)\s]+)\)")
EXTENSIONS = (".py", ".md", ".sh", ".json", ".csv", ".slurm", ".tex", ".ipynb",
              ".toml", ".lock", ".txt", ".yaml", ".yml", ".cfg", ".sbatch")
ABS_PATH = re.compile(r"(?:^|[\s`\"'(=])/(home|Users|root|tmp|var|etc|opt|mnt|srv)/")
TILDE_PATH = re.compile(r"(?:^|[\s`\"'(=])~/")
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
]
failures = 0


def fail(skill: str, message: str) -> None:
    global failures
    failures += 1
    print(f"FAIL  {skill}: {message}")


def parse_frontmatter(text: str):
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    data = {}
    for line in text[4:end].splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            return None
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def path_candidates(text: str):
    for match in PATH_TOKEN.finditer(text):
        yield match.group(1)
    for match in MD_LINK.finditer(text):
        yield match.group(1)


def looks_like_path(token: str) -> bool:
    token = token.strip().rstrip(".,;:)")
    if not token or "://" in token or token.startswith("-"):
        return False
    if any(ch in token for ch in " \t\"'()[]{}*$<>|="):
        return False
    # Only check path-like tokens: a directory or a repo-relative path.
    return "/" in token


RUNTIME_PARTS = {"data", "results", "results_hyperparams", ".venv", "node_modules"}


basename_index = {p.name for p in repo_root.rglob("*") if p.is_file()}

skill_dirs = sorted(
    d for d in skills_dir.iterdir()
    if d.is_dir() and (d / "SKILL.md").exists()
)
if not skill_dirs:
    print(f"FAIL  no skill directories with SKILL.md under {skills_dir}")
    sys.exit(1)

readme = skills_dir / "README.md"
readme_text = readme.read_text() if readme.exists() else ""

for skill_dir in skill_dirs:
    name = skill_dir.name
    skill_md = skill_dir / "SKILL.md"
    text = skill_md.read_text()
    ok = True

    data = parse_frontmatter(text)
    if data is None:
        fail(name, "SKILL.md has no parseable frontmatter block")
        continue
    if "name" not in data or not data["name"]:
        fail(name, "frontmatter is missing 'name'")
        ok = False
    if "description" not in data or not data["description"]:
        fail(name, "frontmatter is missing 'description'")
        ok = False
    if data.get("name") and data["name"] != name:
        fail(name, f"frontmatter name {data['name']!r} != directory name")
        ok = False
    if data.get("name") and not NAME_RE.match(data["name"]):
        fail(name, f"name {data['name']!r} is not a valid skill name")
        ok = False
    if len(data.get("name", "")) > 64:
        fail(name, "name longer than 64 characters")
        ok = False
    if len(data.get("description", "")) > 1024:
        fail(name, "description longer than 1024 characters")
        ok = False

    n_lines = len(text.splitlines())
    if n_lines > 120:
        fail(name, f"SKILL.md is {n_lines} lines (limit 120)")
        ok = False

    for token in path_candidates(text):
        candidate = token.strip().strip("`").rstrip(".,;:)")
        candidate = re.sub(r":\d+(?:-\d+)?$", "", candidate)
        candidate = re.sub(r":[A-Za-z_][\w.]*$", "", candidate)
        if not looks_like_path(candidate):
            continue
        candidate = candidate.strip("/") if candidate.endswith("/") else candidate
        if set(Path(candidate).parts) & RUNTIME_PARTS:
            continue
        if candidate.startswith((".pi/",)):
            target = repo_root / candidate
        else:
            target = skill_dir / candidate
        if target.exists() or (repo_root / candidate).exists() \
                or (repo_root / ".pi" / "skills" / candidate).exists():
            continue
        if Path(candidate).name in basename_index:
            continue
        fail(name, f"referenced path does not exist: {candidate} (SKILL.md)")
        ok = False

    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            content = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if ABS_PATH.search(content):
            fail(name, f"absolute path outside the repo in {path.relative_to(skill_dir)}")
            ok = False
        if TILDE_PATH.search(content):
            fail(name, f"tilde-home path in {path.relative_to(skill_dir)}")
            ok = False
        for pattern in SECRET_PATTERNS:
            if pattern.search(content):
                fail(name, f"possible secret material in {path.relative_to(skill_dir)}")
                ok = False

    scripts = skill_dir / "scripts"
    if scripts.is_dir():
        for script in sorted(scripts.iterdir()):
            if script.is_file() and not script.stat().st_mode & 0o111:
                print(f"WARN  {name}: {script.name} is not executable")

    if readme_text and name not in readme_text:
        fail(name, "not listed in .pi/skills/README.md")
        ok = False

    if ok:
        print(f"PASS  {name}: description {len(data.get('description', ''))} chars, "
              f"{n_lines} lines")

if failures:
    print(f"\nvalidate_skills: {failures} failure(s)")
    sys.exit(1)
print(f"\nvalidate_skills: all {len(skill_dirs)} skills valid")
PY
