import json
import tempfile
import unittest
from pathlib import Path

from helpers import assert_help, run_script, script

RELMAE = script("forecasting-evaluation", "relative_mae.py")

TRAIN = """ds,series,y
d1,A,0.0
d2,A,1.0
d3,A,2.0
d1,B,0.0
d2,B,0.0
d3,B,0.0
"""

TEST = """ds,series,y
d4,A,3.0
d5,A,3.0
d6,A,3.0
d4,B,0.0
d5,B,0.0
d6,B,0.0
"""

PRED = """ds,series,yhat
d4,A,3.0
d5,A,3.0
d6,A,3.0
d4,B,0.5
d5,B,0.5
d6,B,0.5
"""


def write_inputs(root: Path) -> tuple[Path, Path, Path]:
    paths = []
    for name, content in (("train.csv", TRAIN), ("test.csv", TEST), ("pred.csv", PRED)):
        path = root / name
        path.write_text(content)
        paths.append(path)
    return tuple(paths)


class TestRelativeMae(unittest.TestCase):
    def test_help(self):
        assert_help(self, RELMAE)

    def test_perfect_forecast_and_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test, pred = write_inputs(root)
            proc = run_script(RELMAE, ["--train", train, "--test", test, "--pred", pred])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(proc.stdout)
            self.assertEqual(report["metric"], "relative_mae_vs_persistence")
            self.assertEqual(report["n_units_total"], 2)
            self.assertEqual(report["n_excluded"], 1)
            self.assertEqual(report["n_units_scored"], 1)
            self.assertAlmostEqual(report["aggregate"]["median"], 0.0)
            self.assertEqual(report["units"][0]["series"], "A")
            self.assertFalse(report["units"][0]["excluded"])
            self.assertTrue(report["units"][1]["excluded"])
            self.assertIsNone(report["units"][1]["rel_mae"])

    def test_epsilon_threshold_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test, pred = write_inputs(root)
            # B's persistence MAE is 0 so excluded regardless; A is scored.
            proc = run_script(RELMAE, ["--train", train, "--test", test, "--pred", pred,
                                       "--epsilon", "0.0"])
            report = json.loads(proc.stdout)
            self.assertEqual(report["n_excluded"], 1)

    def test_out_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test, pred = write_inputs(root)
            out = root / "out"
            proc = run_script(RELMAE, ["--train", train, "--test", test, "--pred", pred,
                                       "--out", out])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            units = (out / "units.csv").read_text()
            self.assertIn("persistence_mae", units)
            aggregate = json.loads((out / "aggregate.json").read_text())
            self.assertEqual(aggregate["n_excluded"], 1)

    def test_missing_column_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "bad.csv"
            path.write_text("ds,series\n")
            proc = run_script(RELMAE, ["--train", path, "--test", path, "--pred", path])
            self.assertEqual(proc.returncode, 2)
            self.assertIn("missing column", proc.stderr)

    def test_no_units_is_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.csv"
            test = root / "test.csv"
            pred = root / "pred.csv"
            train.write_text("ds,series,y\n")
            test.write_text("ds,series,y\n")
            pred.write_text("ds,series,yhat\n")
            proc = run_script(RELMAE, ["--train", train, "--test", test, "--pred", pred])
            self.assertEqual(proc.returncode, 1, proc.stderr)


if __name__ == "__main__":
    unittest.main()
