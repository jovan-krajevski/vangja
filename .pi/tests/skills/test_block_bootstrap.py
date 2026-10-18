import json
import tempfile
import unittest
from pathlib import Path

from helpers import assert_help, run_script, script

BOOT = script("forecasting-evaluation", "block_bootstrap.py")

PAIRED = """series,origin_block,diff_rel_mae
A,2013-H1,-0.20
B,2013-H1,-0.10
C,2013-H1,0.05
A,2013-H2,-0.15
B,2013-H2,0.02
C,2013-H2,0.30
D,2014-H1,-0.05
E,2014-H1,0.10
"""


class TestBlockBootstrap(unittest.TestCase):
    def test_help(self):
        assert_help(self, BOOT)

    def test_shape_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired = root / "paired.csv"
            paired.write_text(PAIRED)
            args = ["--paired", paired, "--n-iter", "500", "--seed", "7"]
            first = run_script(BOOT, args)
            second = run_script(BOOT, args)
            self.assertEqual(first.returncode, 0, first.stderr)
            report = json.loads(first.stdout)
            for key in ("n_paired_units", "n_iter", "median_diff", "ci_2p5",
                        "ci_97p5", "prop_diff_lt_0", "negative_transfer_rate"):
                self.assertIn(key, report)
            self.assertEqual(report["n_paired_units"], 8)
            self.assertLessEqual(report["ci_2p5"], report["median_diff"])
            self.assertLessEqual(report["median_diff"], report["ci_97p5"])
            self.assertEqual(report, json.loads(second.stdout))

    def test_out_writes_bootstrap_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired = root / "paired.csv"
            paired.write_text(PAIRED)
            out = root / "out"
            proc = run_script(BOOT, ["--paired", paired, "--n-iter", "100",
                                     "--out", out])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((out / "bootstrap.json").exists())

    def test_missing_column_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("series,block\nA,x\n")
            proc = run_script(BOOT, ["--paired", path])
            self.assertEqual(proc.returncode, 2)
            self.assertIn("missing column", proc.stderr)

    def test_bad_n_iter_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paired.csv"
            path.write_text(PAIRED)
            proc = run_script(BOOT, ["--paired", path, "--n-iter", "0"])
            self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
