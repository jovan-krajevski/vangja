import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from helpers import assert_help, run_script, script

MAKE = script("reproducible-experiments", "make_manifest.py")
CHECK = script("reproducible-experiments", "check_manifest.py")


def build_run(root: Path, cwd: Path | None = None) -> Path:
    data = root / "tickers"
    data.mkdir()
    (data / "AAPL.csv").write_text("ds,y\n2020-01-01,1.0\n")
    cfg = root / "config.py"
    cfg.write_text("BASE_SEED = 42\n")
    out = root / "run"
    proc = run_script(MAKE, ["--out", out, "--data", data, "--config", cfg,
                             "--base-seed", "42", "--status", "ok",
                             "--command", "python", "run.py"], cwd=cwd)
    assert proc.returncode == 0, proc.stderr
    return out


class TestCheckManifest(unittest.TestCase):
    def test_help(self):
        assert_help(self, CHECK)

    def test_valid_manifest_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = build_run(root)
            proc = run_script(CHECK, [out / "manifest.json", "--root", root, "--json"])
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            report = json.loads(proc.stdout)
            self.assertTrue(report["ok"])

    def test_tampered_data_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = build_run(root)
            (root / "tickers" / "AAPL.csv").write_text("ds,y\n2020-01-01,9.9\n")
            proc = run_script(CHECK, [out / "manifest.json", "--root", root,
                                      "--allow-unknown-commit"])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("hash mismatch", proc.stdout)

    def test_missing_field_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps({"manifest_schema": "vangja-run-manifest-1.0"}))
            proc = run_script(CHECK, [path, "--allow-unknown-commit"])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("missing required field", proc.stdout)

    def test_status_exit_code_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = build_run(root)
            manifest = json.loads((out / "manifest.json").read_text())
            manifest["exit_code"] = 1
            (out / "manifest.json").write_text(json.dumps(manifest))
            proc = run_script(CHECK, [out / "manifest.json", "--root", root,
                                      "--allow-unknown-commit"])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("status ok requires exit_code 0", proc.stdout)

    def test_allow_missing_data_downgrades_to_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = build_run(root)
            (root / "tickers" / "AAPL.csv").unlink()
            proc = run_script(CHECK, [out / "manifest.json", "--root", root,
                                      "--allow-unknown-commit", "--allow-missing-data"])
            self.assertEqual(proc.returncode, 0, proc.stdout)
            self.assertIn("WARN", proc.stdout)

    def test_hash_format_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = build_run(root)
            manifest = json.loads((out / "manifest.json").read_text())
            manifest["data"]["tickers"]["files"]["AAPL.csv"] = "not-a-hash"
            (out / "manifest.json").write_text(json.dumps(manifest))
            proc = run_script(CHECK, [out / "manifest.json", "--root", root,
                                      "--allow-unknown-commit"])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("bad sha256", proc.stdout)

    def test_unknown_commit_can_be_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = build_run(root, cwd=root)
            proc = run_script(CHECK, [out / "manifest.json", "--root", root])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("unknown", proc.stdout)
            proc = run_script(CHECK, [out / "manifest.json", "--root", root,
                                      "--allow-unknown-commit"])
            self.assertEqual(proc.returncode, 0, proc.stdout)


if __name__ == "__main__":
    unittest.main()
