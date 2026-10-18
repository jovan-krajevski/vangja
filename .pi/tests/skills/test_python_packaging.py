import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from helpers import FIXTURES, REPO_ROOT, assert_help, run_script, script

FREEZE = script("python-packaging-and-environments", "freeze_report.py")
FAKE_CONTAINER = FIXTURES / "packaging" / "fake_container.img"


class TestFreezeReport(unittest.TestCase):
    def test_help(self):
        assert_help(self, FREEZE)

    def test_report_has_versions_and_lock_hash(self):
        proc = run_script(FREEZE, ["--repo", REPO_ROOT, "--json"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["schema"], "vangja-freeze-report-1.0")
        self.assertTrue(report["python"]["version"])
        expected = hashlib.sha256((REPO_ROOT / "uv.lock").read_bytes()).hexdigest()
        self.assertEqual(report["uv_lock"]["sha256"], expected)
        for package in ("pymc", "pytensor", "arviz", "numpy", "numba", "jax"):
            self.assertIn(package, report["packages"])

    def test_container_hash(self):
        proc = run_script(FREEZE, ["--repo", REPO_ROOT, "--container", FAKE_CONTAINER])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        expected = hashlib.sha256(FAKE_CONTAINER.read_bytes()).hexdigest()
        self.assertEqual(report["container"]["sha256"], expected)

    def test_missing_container_is_usage_error(self):
        proc = run_script(FREEZE, ["--container", FAKE_CONTAINER.parent / "nope.img"])
        self.assertEqual(proc.returncode, 2)

    def test_out_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "env.json"
            proc = run_script(FREEZE, ["--repo", REPO_ROOT, "--out", out])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(out.read_text())
            self.assertEqual(report["schema"], "vangja-freeze-report-1.0")
            self.assertEqual(len(list(Path(tmp).rglob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
