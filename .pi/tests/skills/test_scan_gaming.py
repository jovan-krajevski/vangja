import json
import tempfile
import unittest
from pathlib import Path

from helpers import REPO_ROOT, SKILLS, assert_help, run_script, script

SCAN = script("scientific-integrity-and-anti-gaming", "scan_gaming.py")
FIXTURES = SKILLS / "scientific-integrity-and-anti-gaming" / "fixtures"
CLEAN = FIXTURES / "clean"
SEEDED = FIXTURES / "seeded"

STATIC_CATEGORIES = {
    "test-metric-selection",
    "metric-reassignment",
    "denominator-clipping",
    "seed-change-frozen-config",
    "hashed-file-mismatch",
    "failure-trimming-swallow",
    "failure-trimming-dropna",
    "test-weakening-tolerance",
    "test-weakening-skip",
}
PATCH_CATEGORIES = {
    "removed_assert.patch": "test-weakening-assertion-removed",
    "seed_change.patch": "seed-change-frozen-config",
    "exclusion_change.patch": "exclusion-change",
    "horizon_crop.patch": "horizon-cropping",
    "swallow.patch": "failure-trimming-swallow",
}


def findings_for(root: Path, patch: Path | None = None) -> list[dict]:
    args = ["--root", root, "--json", "--fail-on", "none"]
    if patch is not None:
        args += ["--patch", patch]
    proc = run_script(SCAN, args)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout)["findings"]


class TestScanGaming(unittest.TestCase):
    def test_help(self):
        assert_help(self, SCAN)

    def test_clean_fixture_has_no_false_positives(self):
        proc = run_script(SCAN, ["--root", CLEAN])
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("CLEAN", proc.stdout)
        self.assertEqual(findings_for(CLEAN), [])

    def test_seeded_fixture_detects_every_static_category(self):
        proc = run_script(SCAN, ["--root", SEEDED])
        self.assertEqual(proc.returncode, 1)
        categories = {f["category"] for f in findings_for(SEEDED)}
        missing = STATIC_CATEGORIES - categories
        self.assertFalse(missing, f"undetected seeded categories: {missing}")
        report = json.loads(run_script(SCAN, ["--root", SEEDED, "--json"]).stdout)
        self.assertGreater(report["by_severity"]["S0"], 0)

    def test_patch_categories_is_detected(self):
        for name, expected in PATCH_CATEGORIES.items():
            with self.subTest(patch=name):
                categories = {
                    f["category"]
                    for f in findings_for(CLEAN, SEEDED / "patches" / name)
                }
                self.assertIn(expected, categories)

    def test_out_writes_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "scan"
            proc = run_script(SCAN, ["--root", SEEDED, "--out", out])
            self.assertEqual(proc.returncode, 1)
            report = json.loads((out / "gaming_scan.json").read_text())
            self.assertEqual(report["n_findings"], len(report["findings"]))
            self.assertIn("gaming scan", (out / "gaming_scan.md").read_text().lower())

    def test_real_repo_is_clean_statically(self):
        proc = run_script(SCAN, ["--root", REPO_ROOT])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
