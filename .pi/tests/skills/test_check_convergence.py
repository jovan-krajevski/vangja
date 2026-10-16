import json
import tempfile
import unittest
from pathlib import Path

from helpers import assert_help, run_script, script

CHECK = script("bayesian-workflow", "check_convergence.py")

GOOD = {
    "label": "nuts-good", "method": "nuts", "target_accept": 0.9,
    "r_hat": {"a": 1.004}, "ess_bulk": {"a": 600}, "ess_tail": {"a": 500},
    "divergences": 0, "max_tree_depth_hits": 0, "bfmi": [0.6],
}
BAD = dict(GOOD, divergences=4, r_hat={"a": 1.2})
WARN = {"label": "map", "method": "mapx"}


def write(tmp: Path, payload: dict) -> Path:
    path = tmp / "diag.json"
    path.write_text(json.dumps(payload))
    return path


class TestCheckConvergence(unittest.TestCase):
    def test_help(self):
        assert_help(self, CHECK)

    def test_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_script(CHECK, ["--json", write(Path(tmp), GOOD)])
            self.assertEqual(proc.returncode, 0, proc.stdout)
            self.assertIn("CONVERGENCE PASS", proc.stdout)

    def test_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_script(CHECK, ["--json", write(Path(tmp), BAD)])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("CONVERGENCE FAIL", proc.stdout)

    def test_strict_turns_warning_into_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), WARN)
            self.assertEqual(run_script(CHECK, ["--json", path]).returncode, 0)
            self.assertEqual(run_script(CHECK, ["--json", path, "--strict"]).returncode, 1)

    def test_waiver_records_reason_and_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp), BAD)
            out = Path(tmp) / "report.json"
            proc = run_script(CHECK, ["--json", path, "--waiver-reason",
                                      "supervisor waiver WV-001", "--out", out])
            self.assertEqual(proc.returncode, 0, proc.stdout)
            self.assertIn("WAIVED", proc.stdout)
            report = json.loads(out.read_text())
            self.assertEqual(report["waiver"]["reason"], "supervisor waiver WV-001")

    def test_missing_input_is_error(self):
        proc = run_script(CHECK, ["--json", "/nonexistent/diag.json"])
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
