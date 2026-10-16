import json
import tempfile
import unittest
from pathlib import Path

from helpers import FIXTURES, assert_help, run_script, script

PR = script("git-and-pr-workflow", "make_pr_record.py")
DIFF = script("git-and-pr-workflow", "diff_scan.py")
GIT_FIXTURES = FIXTURES / "git"

HEAD = "a" * 40
BASE = "b" * 40


class TestMakePrRecord(unittest.TestCase):
    def test_help(self):
        assert_help(self, PR)

    def _args(self, out: Path, *extra):
        return [
            "--task-id", "T-012", "--branch", "work/T-012-seeds",
            "--base-sha", BASE, "--head-sha", HEAD,
            "--summary", "transfer regression: slope potential sign",
            "--artifact", ".pi/artifacts/E-0007/manifest.json",
            "--test", "uv run pytest tests/test_regularization.py -q -> 42 passed",
            "--claim", "C-003", "--hpc-job", "E-0007",
            "--acceptance", "slope sign fixed::pytest -k trend::42 passed",
            "--out", str(out), *extra,
        ]

    def test_renders_body_and_footer(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "PR-003"
            proc = run_script(PR, self._args(out))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            body = (out / "PR.md").read_text()
            footer = json.loads((out / "footer.json").read_text())
            self.assertIn("PR_FOOTER_JSON", body)
            self.assertIn("slope sign fixed", body)
            for field in ("task_id", "head_sha", "base_sha", "branch", "summary",
                          "artifacts", "tests", "claims", "hpc_jobs"):
                self.assertIn(field, footer)
            self.assertEqual(footer["status"], "open")
            self.assertEqual(footer["head_sha"], HEAD)

    def test_refuses_bad_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp) / "PR-003")
            args[args.index("work/T-012-seeds")] = "update_pymc"
            proc = run_script(PR, args)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("work/<task-id>-<slug>", proc.stderr)

    def test_refuses_identical_shas(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp) / "PR-003")
            args[args.index(BASE)] = HEAD
            proc = run_script(PR, args)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("identical", proc.stderr)


class TestDiffScan(unittest.TestCase):
    def _run(self, diff: str, *extra):
        proc = run_script(DIFF, ["--diff", GIT_FIXTURES / diff, "--json", *extra])
        if proc.returncode not in (0, 1):
            raise AssertionError(proc.stderr)
        return proc.returncode, json.loads(proc.stdout)

    def test_help(self):
        assert_help(self, DIFF)

    def test_clean_diff_passes(self):
        code, report = self._run("clean.diff")
        self.assertEqual(code, 0, report)
        self.assertEqual(report["n_failing"], 0)

    def test_weakened_diff_flags_all_categories(self):
        code, report = self._run("weakened.diff")
        self.assertEqual(code, 1)
        rules = {f["rule"] for f in report["findings"]}
        for expected in ("protected-path", "frozen-config", "assertion-removed",
                         "test-skip-added", "test-file-deleted"):
            self.assertIn(expected, rules, report)

    def test_gaming_scan_composed(self):
        code, report = self._run("gaming.diff")
        self.assertEqual(code, 1)
        rules = {f["rule"] for f in report["findings"]}
        self.assertTrue(any(r.startswith("gaming-") for r in rules), report)
        self.assertIn("protected-path", rules)  # paper/ outside PAPER state

    def test_paper_state_and_waiver(self):
        code, report = self._run("gaming.diff", "--state", "PAPER")
        paper = [f for f in report["findings"]
                 if f["location"].startswith("paper/")
                 and f["rule"] == "protected-path"]
        self.assertEqual(paper, [])
        code, report = self._run(
            "gaming.diff", "--allow", "protected-path:paper/")
        waived = [f for f in report["findings"] if f.get("waived")]
        self.assertTrue(waived)

    def test_out_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "PR-003"
            proc = run_script(
                DIFF,
                ["--diff", GIT_FIXTURES / "clean.diff", "--out", out,
                 "--skip-gaming-scan"],
            )
            self.assertEqual(proc.returncode, 0)
            self.assertTrue((out / "diff_scan.json").exists())


if __name__ == "__main__":
    unittest.main()
