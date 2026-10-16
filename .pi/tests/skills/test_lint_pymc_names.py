import json
import unittest
from pathlib import Path

from helpers import FIXTURES, REPO_ROOT, assert_help, run_script, script

LINT = script("vangja-codebase", "lint_pymc_names.py")
FIXTURE_DIR = FIXTURES / "vangja_codebase"


class TestLintPymcNames(unittest.TestCase):
    def test_help(self):
        assert_help(self, LINT)

    def test_diff_fixture_is_flagged(self):
        proc = run_script(LINT, ["--diff", FIXTURE_DIR / "bad_names.diff", "--json"])
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        report = json.loads(proc.stdout)
        messages = " ".join(f["message"] for f in report["findings"])
        self.assertIn("365.25", messages)
        self.assertIn("beta_sd1.0", messages)
        # The two well-formed names in the same hunk must not be flagged.
        self.assertNotIn("'lt_0 - slope'", messages)
        self.assertNotIn("nc_0 - mu", messages)

    def test_valid_names_only(self):
        diff = (
            "--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n"
            '+    a = pm.Normal(f"lt_{self.model_idx} - slope", 0, 1)\n'
            '+    b = pm.Normal(f"fs_{self.model_idx} - beta", 0, 1)\n'
        )
        path = FIXTURE_DIR / "_tmp_valid.diff"
        path.write_text(diff)
        try:
            proc = run_script(LINT, ["--diff", path])
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        finally:
            path.unlink()

    def test_src_vangja_is_clean(self):
        proc = run_script(LINT, ["--root", REPO_ROOT / "src" / "vangja"])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("FINDINGS: 0", proc.stdout)

    def test_out_written_only_under_out(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "report.json"
            proc = run_script(
                LINT,
                ["--diff", FIXTURE_DIR / "bad_names.diff", "--json", "--out", out],
            )
            self.assertEqual(proc.returncode, 1)
            self.assertTrue(out.exists())
            payload = json.loads(out.read_text())
            self.assertGreaterEqual(payload["n_findings"], 2)


if __name__ == "__main__":
    unittest.main()
