import json
import tempfile
import unittest
from pathlib import Path

from helpers import REPO_ROOT, SKILLS, assert_help, run_script, script, write_text

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
    "removed_unittest_assert.patch": "test-weakening-assertion-removed",
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
                findings = findings_for(CLEAN, SEEDED / "patches" / name)
                categories = {f["category"] for f in findings}
                self.assertIn(expected, categories)
                messages = " ".join(f["message"] for f in findings)
                self.assertNotIn("deled", messages)

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

    def test_runtime_state_is_not_scanned(self):
        # `.pi/state/**` holds gitignored runtime state (test sandboxes, worktrees,
        # calibration snapshots) — a planted defect there is not repository source.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_text(
                root / ".pi" / "state" / "sandbox" / "runners" / "run_grid.py",
                'best = min(rows, key=lambda row: row["test_mape"])\n',
            )
            write_text(
                root / ".pi" / "artifacts" / "E-0001" / "scripts" / "pick.py",
                'best = min(rows, key=lambda row: row["holdout_rmse"])\n',
            )
            write_text(root / "runners" / "run_grid.py", 'best = min(rows, key=lambda row: row["test_mape"])\n')
            locations = [finding["location"] for finding in findings_for(root)]
            self.assertTrue(any("runners/run_grid.py" in location for location in locations), locations)
            self.assertFalse(any(".pi/state" in location or ".pi/artifacts" in location for location in locations), locations)

    def test_builtin_and_pandas_selection_is_detected(self):
        cases = {
            "min": 'best = min(rows, key=lambda r: r["test_mape"])\n',
            "max": 'best = max(runs, key=lambda r: r["holdout_rmse"])\n',
            "sorted": 'best = sorted(runs, key=lambda r: r["oos_loss"])[0]\n',
            "nsmallest": 'best = frame.nsmallest(1, "test_mape")\n',
            "largest": 'best = frame.nlargest(1, "test_score")\n',
            "argmax": 'best = frame["test_mae"].argmax()\n',
        }
        for selector, source in cases.items():
            with self.subTest(selector=selector):
                with tempfile.TemporaryDirectory() as tmp:
                    Path(tmp, "runner.py").write_text(source)
                    findings = findings_for(Path(tmp))
                self.assertEqual(
                    [f["category"] for f in findings], ["test-metric-selection"], findings
                )
                self.assertEqual(findings[0]["severity"], "S0")

    def test_builtin_selection_negative_controls(self):
        source = (
            "a = min(rows, key=lambda r: r['dev_mape'])\n"
            "b = max(elpds)\n"
            "c = min(0.0, value)\n"
            "d = sorted(names)\n"
            "e = max(rows, key=lambda r: r['validation_mae'])\n"
            "f = sorted(table, key=lambda r: r['series'])[0]\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "runner.py").write_text(source)
            findings = findings_for(Path(tmp))
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
