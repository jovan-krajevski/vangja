import json
import tempfile
import unittest
from pathlib import Path

from helpers import assert_help, run_script, script

DIAGNOSE = script("bayesian-workflow", "diagnose_idata.py")

GOOD = {
    "label": "stocks-main-nuts",
    "method": "nuts",
    "target_accept": 0.9,
    "r_hat": {"lt_0 - slope": 1.003, "fs_0 - beta": 1.001},
    "ess_bulk": {"lt_0 - slope": 620, "fs_0 - beta": 510},
    "ess_tail": {"lt_0 - slope": 510, "fs_0 - beta": 460},
    "divergences": 0,
    "max_tree_depth_hits": 0,
    "bfmi": [0.61, 0.58],
}

BAD = dict(GOOD, r_hat={"lt_0 - slope": 1.05}, ess_bulk={"lt_0 - slope": 12},
           ess_tail={"lt_0 - slope": 9}, divergences=3, max_tree_depth_hits=2,
           target_accept=0.8)


def write_json(tmp: Path, payload: dict) -> Path:
    path = tmp / "diag.json"
    path.write_text(json.dumps(payload))
    return path


class TestDiagnoseIdata(unittest.TestCase):
    def test_help(self):
        assert_help(self, DIAGNOSE)

    def test_good_fit_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json(Path(tmp), GOOD)
            proc = run_script(DIAGNOSE, ["--json", path, "--json-stdout"])
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            report = json.loads(proc.stdout)
            self.assertEqual(report["verdict"], "pass")
            self.assertEqual(report["n_fail"], 0)

    def test_bad_fit_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json(Path(tmp), BAD)
            proc = run_script(DIAGNOSE, ["--json", path, "--json-stdout"])
            self.assertEqual(proc.returncode, 1)
            report = json.loads(proc.stdout)
            self.assertEqual(report["verdict"], "fail")
            self.assertGreaterEqual(report["n_fail"], 5)

    def test_writes_reports_to_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            path = write_json(Path(tmp), GOOD)
            proc = run_script(DIAGNOSE, ["--json", path, "--out", out])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((out / "diagnostics_stocks-main-nuts.json").exists())
            markdown = (out / "diagnostics_stocks-main-nuts.md").read_text()
            self.assertIn("verdict", markdown.lower())

    def test_map_fit_with_diagnostics_passes(self):
        payload = {"label": "map", "method": "mapx",
                   "map_diagnostics": {"status": "converged", "gradient_norm": 1e-6}}
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json(Path(tmp), payload)
            proc = run_script(DIAGNOSE, ["--json", path, "--json-stdout"])
            self.assertEqual(proc.returncode, 0, proc.stdout)
            report = json.loads(proc.stdout)
            self.assertEqual(report["verdict"], "pass")

    def test_map_fit_without_diagnostics_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_json(Path(tmp), {"label": "map", "method": "mapx"})
            proc = run_script(DIAGNOSE, ["--json", path, "--json-stdout"])
            self.assertEqual(proc.returncode, 0, proc.stdout)
            self.assertEqual(json.loads(proc.stdout)["verdict"], "warn")

    def test_threshold_override(self):
        payload = dict(GOOD, r_hat={"x": 1.02})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_json(root, payload)
            thresholds = root / "thresholds.json"
            thresholds.write_text(json.dumps({"r_hat_max": 1.05}))
            proc = run_script(DIAGNOSE, ["--json", path, "--thresholds", thresholds])
            self.assertEqual(proc.returncode, 0, proc.stdout)

    def test_missing_input_is_error(self):
        proc = run_script(DIAGNOSE, ["--json", "/nonexistent/diag.json"])
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
