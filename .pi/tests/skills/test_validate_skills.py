import subprocess
import tempfile
import unittest
from pathlib import Path

from helpers import REPO_ROOT, SKILLS, run_script

VALIDATOR = SKILLS / "validate_skills.sh"

try:
    import yaml  # PyYAML: the same real parser pi uses for SKILL.md frontmatter

except ImportError:  # pragma: no cover - environment without PyYAML
    yaml = None


def require_yaml(case):
    """Fail (never skip) when the parser the validator relies on is absent.

    A skip decorator here would be a test-weakening marker; instead the test
    fails with the missing dependency named, which is the honest state of the
    environment for a YAML frontmatter check.
    """
    if yaml is None:
        case.fail(
            "PyYAML is required to validate SKILL.md frontmatter; the validator "
            "degrades to a warning without it, so this check cannot be skipped"
        )

BAD_YAML_FRONTMATTER = """---
name: {name}
description: A skill about colons: this value is not valid YAML unquoted. Use when testing.
---

# Test skill
"""

GOOD_FRONTMATTER = """---
name: {name}
description: A valid test skill used by the validator tests. Use when validating skills.
---

# Test skill

No project facts; test only.
"""


def run_validator(skills_dir: Path, repo_root: Path = REPO_ROOT):
    return subprocess.run(
        ["bash", str(VALIDATOR), str(skills_dir), str(repo_root)],
        capture_output=True, text=True, timeout=120,
    )


class TestValidateSkills(unittest.TestCase):
    def test_real_skills_are_valid(self):
        proc = run_validator(SKILLS)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("all", proc.stdout)

    def test_bad_frontmatter_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = root / "wrong-dir"
            skill.mkdir()
            (skill / "SKILL.md").write_text(GOOD_FRONTMATTER.format(name="other-name"))
            (root / "README.md").write_text("wrong-dir other-name")
            proc = run_validator(root)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("!= directory name", proc.stdout)

    def test_missing_referenced_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = root / "bad-path"
            skill.mkdir()
            content = GOOD_FRONTMATTER.format(name="bad-path") + "\nSee `scripts/missing.py`.\n"
            (skill / "SKILL.md").write_text(content)
            (root / "README.md").write_text("bad-path")
            proc = run_validator(root)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("does not exist", proc.stdout)

    def test_too_many_lines_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = root / "long-skill"
            skill.mkdir()
            content = GOOD_FRONTMATTER.format(name="long-skill") + "\n" + "filler\n" * 130
            (skill / "SKILL.md").write_text(content)
            (root / "README.md").write_text("long-skill")
            proc = run_validator(root)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("limit 120", proc.stdout)

    def test_absolute_path_outside_repo_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = root / "abs-path"
            skill.mkdir()
            content = GOOD_FRONTMATTER.format(name="abs-path") + "\nWrite to /tmp/out now.\n"
            (skill / "SKILL.md").write_text(content)
            (root / "README.md").write_text("abs-path")
            proc = run_validator(root)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("absolute path", proc.stdout)

    def test_unquoted_colon_in_description_is_rejected(self):
        """Pi parses SKILL.md frontmatter as YAML; the validator must too."""
        require_yaml(self)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill = root / "bad-yaml"
            skill.mkdir()
            (skill / "SKILL.md").write_text(BAD_YAML_FRONTMATTER.format(name="bad-yaml"))
            (root / "README.md").write_text("bad-yaml")
            proc = run_validator(root)
            self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
            self.assertIn("not valid YAML", proc.stdout)

    def test_every_real_skill_frontmatter_is_valid_yaml(self):
        require_yaml(self)
        for skill_dir in sorted(p for p in SKILLS.iterdir() if (p / "SKILL.md").exists()):
            with self.subTest(skill=skill_dir.name):
                text = (skill_dir / "SKILL.md").read_text()
                end = text.find("\n---", 3)
                self.assertNotEqual(end, -1, "frontmatter fence is never closed")
                data = yaml.safe_load(text[4:end])
                self.assertIsInstance(data, dict, "frontmatter must be a YAML mapping")
                self.assertEqual(data.get("name"), skill_dir.name)
                self.assertIsInstance(data.get("description"), str)


if __name__ == "__main__":
    unittest.main()
