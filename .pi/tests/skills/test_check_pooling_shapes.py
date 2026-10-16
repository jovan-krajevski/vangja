import json
import unittest

from helpers import FIXTURES, REPO_ROOT, assert_help, run_script, script

CHECK = script("vangja-codebase", "check_pooling_shapes.py")
FIXTURE_DIR = FIXTURES / "vangja_codebase"


class TestCheckPoolingShapes(unittest.TestCase):
    def test_help(self):
        assert_help(self, CHECK)

    def test_bad_report_flagged(self):
        proc = run_script(
            CHECK, ["--report", FIXTURE_DIR / "shape_report_bad.json", "--json"]
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        report = json.loads(proc.stdout)
        rules = [f["rule"] for f in report["findings"]]
        self.assertIn("scalar-initval-for-vector", rules)
        self.assertIn("2d-predict-column", rules)
        self.assertEqual(report["n_findings"], 2)

    def test_clean_report_passes(self):
        proc = run_script(
            CHECK, ["--report", FIXTURE_DIR / "shape_report_clean.json"]
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("FINDINGS: 0", proc.stdout)

    def test_static_fixture_flagged(self):
        proc = run_script(CHECK, ["--root", FIXTURE_DIR, "--json"])
        self.assertEqual(proc.returncode, 1)
        report = json.loads(proc.stdout)
        locations = {f["rule"] for f in report["findings"]}
        self.assertIn("scalar-initval-for-vector", locations)
        self.assertIn("2d-predict-column", locations)
        # The guarded complete-pooling initval must not be flagged.
        guarded = [
            f
            for f in report["findings"]
            if "bad_component.py" in f["location"] and f["rule"] == "scalar-initval-for-vector"
        ]
        self.assertEqual(len(guarded), 2)

    def test_real_components_clean(self):
        proc = run_script(CHECK, ["--root", REPO_ROOT / "src" / "vangja" / "components"])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("FINDINGS: 0", proc.stdout)


if __name__ == "__main__":
    unittest.main()
