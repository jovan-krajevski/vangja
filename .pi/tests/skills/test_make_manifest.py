import json
import tempfile
import unittest
from pathlib import Path

from helpers import assert_help, run_script, script

MAKE = script("reproducible-experiments", "make_manifest.py")


class TestMakeManifest(unittest.TestCase):
    def test_help(self):
        assert_help(self, MAKE)

    def test_writes_schema_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "tickers"
            data.mkdir()
            (data / "AAPL.csv").write_text("ds,y\n2020-01-01,1.0\n")
            (data / "MSFT.csv").write_text("ds,y\n2020-01-01,2.0\n")
            cfg = root / "config.py"
            cfg.write_text("BASE_SEED = 42\n")
            out = root / "run"
            proc = run_script(MAKE, [
                "--out", out, "--data", data, "--config", cfg,
                "--base-seed", "42", "--seed", "cell=42", "--seed", "7",
                "--command", "python", "run.py",
            ])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["manifest_schema"], "vangja-run-manifest-1.0")
            self.assertEqual(manifest["status"], "ok")
            self.assertEqual(manifest["exit_code"], 0)
            self.assertEqual(manifest["seeds"]["base_seed"], 42)
            self.assertEqual(manifest["seeds"]["values"]["cell"], 42)
            entry = manifest["data"]["tickers"]
            self.assertEqual(sorted(entry["files"]), ["AAPL.csv", "MSFT.csv"])
            self.assertEqual(len(entry["files"]["AAPL.csv"]), 64)
            config_keys = list(manifest["configs"])
            self.assertEqual(len(config_keys), 1)
            self.assertTrue(config_keys[0].endswith("config.py"))
            self.assertEqual(len(manifest["configs"][config_keys[0]]), 64)

    def test_refuses_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            base = ["--out", out, "--status", "ok"]
            command = ["--command", "python", "run.py"]
            self.assertEqual(run_script(MAKE, base + command).returncode, 0)
            self.assertEqual(run_script(MAKE, base + command).returncode, 2)
            self.assertEqual(run_script(MAKE, base + ["--force"] + command).returncode, 0)

    def test_command_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_script(MAKE, ["--out", Path(tmp) / "run"])
            self.assertEqual(proc.returncode, 2)
            self.assertIn("--command is required", proc.stderr)

    def test_failed_requires_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            command = ["--command", "python", "run.py"]
            proc = run_script(MAKE, ["--out", out, "--status", "failed"] + command)
            self.assertEqual(proc.returncode, 2)
            proc = run_script(MAKE, ["--out", out, "--status", "failed",
                                     "--error", "OOM"] + command)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            manifest = json.loads((out / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["exit_code"], 1)
            self.assertEqual(manifest["error"], "OOM")

    def test_missing_data_path_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_script(MAKE, ["--out", Path(tmp) / "run",
                                     "--data", Path(tmp) / "nope",
                                     "--command", "python", "run.py"])
            self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
