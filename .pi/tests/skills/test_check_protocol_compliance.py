import json
import unittest

from helpers import FIXTURES, REPO_ROOT, assert_help, run_script, script

CHECK = script("experiment-protocol-design", "check_protocol_compliance.py")
PROTOCOL_FIXTURES = FIXTURES / "protocol"


class TestProtocolCompliance(unittest.TestCase):
    def test_help(self):
        assert_help(self, CHECK)

    def test_real_repo_is_clean(self):
        proc = run_script(CHECK, ["--repo", REPO_ROOT])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("CLEAN", proc.stdout)

    def test_test_set_selection_is_flagged_with_file_line(self):
        proc = run_script(CHECK, ["--repo", PROTOCOL_FIXTURES / "dirty"])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("test-metric-selection", proc.stdout)
        self.assertIn("evaluation.py:6", proc.stdout)

    def test_development_selection_is_not_flagged(self):
        proc = run_script(CHECK, ["--repo", PROTOCOL_FIXTURES / "clean"])
        self.assertEqual(proc.returncode, 0, proc.stdout)

    def test_hashed_file_mismatch_is_flagged(self):
        proc = run_script(CHECK, ["--repo", PROTOCOL_FIXTURES / "sidecar"])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("hash mismatch", proc.stdout)

    def test_context_series_in_units_is_flagged(self):
        proc = run_script(CHECK, ["--repo", PROTOCOL_FIXTURES / "context"])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("context-series-in-aggregate", proc.stdout)
        self.assertIn("units_main.csv:2", proc.stdout)

    def test_frozen_edit_patch_is_flagged(self):
        proc = run_script(CHECK, [
            "--repo", REPO_ROOT, "--patch", PROTOCOL_FIXTURES / "frozen_edit.patch",
        ])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("frozen-split-metric-edit", proc.stdout)
        self.assertNotIn("deled", proc.stdout)
        self.assertIn("deleted", proc.stdout)

    def test_unittest_assertion_removal_patch_is_flagged(self):
        proc = run_script(CHECK, [
            "--repo", REPO_ROOT, "--patch", PROTOCOL_FIXTURES / "assert_removed.patch",
        ])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("assertion-removed", proc.stdout)
        self.assertIn("deleted", proc.stdout)
        self.assertNotIn("deled", proc.stdout)

    def test_json_report_written_to_out(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            proc = run_script(CHECK, ["--repo", PROTOCOL_FIXTURES / "dirty",
                                      "--json", "--out", out])
            self.assertEqual(proc.returncode, 1)
            report = json.loads(out.read_text())
            self.assertEqual(report["n_s1"], report["n_findings"])
            self.assertTrue(report["origins_hash_ok"] is False)


if __name__ == "__main__":
    unittest.main()
