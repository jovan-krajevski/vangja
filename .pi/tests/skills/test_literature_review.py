import json
import tempfile
import unittest
from pathlib import Path

from helpers import FIXTURES, assert_help, run_script, script

BIB_CHECK = script("literature-review-and-verification", "bib_check.py")
MATRIX = script("literature-review-and-verification", "related_work_matrix.py")
BIB_FIXTURES = FIXTURES / "bib"


class TestBibCheck(unittest.TestCase):
    def test_help(self):
        assert_help(self, BIB_CHECK)

    def test_fabricated_citation_flagged(self):
        proc = run_script(
            BIB_CHECK,
            ["--bib", BIB_FIXTURES / "refs.bib",
             "--verified", BIB_FIXTURES / "verified.json", "--json"],
        )
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        report = json.loads(proc.stdout)
        rules_by_location = {
            (f["rule"], f["location"].split(":")[0]) for f in report["findings"]
        }
        self.assertIn(("unverified-citation", "@article{fake2024}"),
                      rules_by_location)
        self.assertFalse(any("good2023" in f["location"]
                             for f in report["findings"]), report)
        self.assertIn(("no-identifier", "@article{fake2024}"), rules_by_location)

    def test_missing_sidecar_flagged(self):
        proc = run_script(
            BIB_CHECK,
            ["--bib", BIB_FIXTURES / "refs.bib",
             "--verified", BIB_FIXTURES / "does-not-exist.json", "--json"],
        )
        self.assertEqual(proc.returncode, 1)
        report = json.loads(proc.stdout)
        self.assertTrue(any(f["rule"] == "no-verification-sidecar"
                            for f in report["findings"]))

    def test_doi_mismatch_flagged(self):
        proc = run_script(
            BIB_CHECK,
            ["--bib", BIB_FIXTURES / "refs.bib",
             "--verified", BIB_FIXTURES / "verified_mismatch.json", "--json"],
        )
        self.assertEqual(proc.returncode, 1)
        report = json.loads(proc.stdout)
        self.assertTrue(any(f["rule"] == "verification-mismatch"
                            for f in report["findings"]), report)

    def test_out_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bib-report.json"
            proc = run_script(
                BIB_CHECK,
                ["--bib", BIB_FIXTURES / "refs.bib",
                 "--verified", BIB_FIXTURES / "verified.json",
                 "--out", out],
            )
            self.assertEqual(proc.returncode, 1)
            self.assertEqual(len(list(Path(tmp).rglob("*.json"))), 1)


class TestRelatedWorkMatrix(unittest.TestCase):
    def test_help(self):
        assert_help(self, MATRIX)

    def test_unchecked_source_flagged(self):
        proc = run_script(
            MATRIX, ["--sources", BIB_FIXTURES / "sources.json", "--json"])
        self.assertEqual(proc.returncode, 1)
        report = json.loads(proc.stdout)
        self.assertTrue(any(f["rule"] == "unchecked-source"
                            for f in report["findings"]), report)

    def test_clean_sources_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "matrix.md"
            proc = run_script(
                MATRIX,
                ["--sources", BIB_FIXTURES / "sources_clean.json", "--out", out],
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            text = out.read_text()
            self.assertIn("Related-work extraction matrix", text)
            self.assertIn("T_prophet", text)
            self.assertIn("| yes |", text)


if __name__ == "__main__":
    unittest.main()
