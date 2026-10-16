import json
import tempfile
import unittest
from pathlib import Path

from helpers import FIXTURES, assert_help, run_script, script

VALIDATE = script("scientific-audit-and-adversarial-review", "validate_findings.py")
AUDIT_FIXTURES = FIXTURES / "audit"


class TestValidateFindings(unittest.TestCase):
    def test_help(self):
        assert_help(self, VALIDATE)

    def test_valid_register_passes(self):
        proc = run_script(
            VALIDATE,
            ["--findings", AUDIT_FIXTURES / "findings_ok.json",
             "--workstream", "WS-S", "--json"],
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["n_meta_findings"], 0, report)
        self.assertEqual(report["severity_histogram"]["S0"], 1)

    def test_bad_register_flags_every_rule(self):
        proc = run_script(
            VALIDATE,
            ["--findings", AUDIT_FIXTURES / "findings_bad.json",
             "--workstream", "WS-S", "--json"],
        )
        self.assertEqual(proc.returncode, 1)
        report = json.loads(proc.stdout)
        rules = {f["rule"] for f in report["meta_findings"]}
        for expected in ("no-command", "bad-severity", "bad-confidence",
                         "bad-status", "bad-reproducible", "duplicate-id",
                         "duplicate-finding", "verified-low-confidence"):
            self.assertIn(expected, rules, report)

    def test_wrong_workstream_flagged(self):
        proc = run_script(
            VALIDATE,
            ["--findings", AUDIT_FIXTURES / "findings_ok.json",
             "--workstream", "WS-M", "--json"],
        )
        self.assertEqual(proc.returncode, 1)
        report = json.loads(proc.stdout)
        self.assertTrue(any(f["rule"] == "wrong-workstream"
                            for f in report["meta_findings"]))

    def test_out_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit-validation.json"
            proc = run_script(
                VALIDATE,
                ["--findings", AUDIT_FIXTURES / "findings_ok.json", "--out", out],
            )
            self.assertEqual(proc.returncode, 0)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
