import json
import tempfile
import unittest
from pathlib import Path

from helpers import assert_help, run_script, script

CHECK = script("data-provenance", "check_sidecars.py")
HASH = script("data-provenance", "hash_tree.py")


def build_fixture(root: Path) -> tuple[Path, Path]:
    data = root / "data"
    data.mkdir()
    (data / "a.csv").write_text("x\n1\n")
    (data / "b.csv").write_text("y\n2\n")
    sidecar = root / "data_sha256.json"
    proc = run_script(HASH, ["--root", data, "--out", sidecar])
    assert proc.returncode == 0, proc.stderr
    return data, sidecar


class TestCheckSidecars(unittest.TestCase):
    def test_help(self):
        assert_help(self, CHECK)

    def test_valid_sidecar_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, sidecar = build_fixture(root)
            proc = run_script(CHECK, ["--root", data, "--sidecar", sidecar])
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("PASS", proc.stdout)

    def test_auto_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, _ = build_fixture(root)
            proc = run_script(CHECK, ["--root", data])
            self.assertEqual(proc.returncode, 0, proc.stdout)

    def test_tampered_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, sidecar = build_fixture(root)
            (data / "a.csv").write_text("x\n9\n")
            proc = run_script(CHECK, ["--root", data, "--sidecar", sidecar])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("hash mismatch", proc.stdout)

    def test_missing_recorded_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, sidecar = build_fixture(root)
            (data / "b.csv").unlink()
            proc = run_script(CHECK, ["--root", data, "--sidecar", sidecar])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("missing", proc.stdout)

    def test_extra_file_fails_unless_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, sidecar = build_fixture(root)
            (data / "c.csv").write_text("z\n3\n")
            strict = run_script(CHECK, ["--root", data, "--sidecar", sidecar])
            self.assertEqual(strict.returncode, 1)
            self.assertIn("unrecorded", strict.stdout)
            relaxed = run_script(CHECK, ["--root", data, "--sidecar", sidecar,
                                         "--allow-extra"])
            self.assertEqual(relaxed.returncode, 0, relaxed.stdout)
            self.assertIn("WARN", relaxed.stdout)

    def test_bare_sha256_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "conf.csv").write_text("o,h\n2023-01-01,2024-01-01\n")
            digest = __import__("hashlib").sha256((root / "conf.csv").read_bytes()).hexdigest()
            (root / "conf.csv.sha256").write_text(digest + "\n")
            proc = run_script(CHECK, ["--root", root, "--sidecar",
                                      root / "conf.csv.sha256"])
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("PASS", proc.stdout)

    def test_json_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, sidecar = build_fixture(root)
            proc = run_script(CHECK, ["--root", data, "--sidecar", sidecar, "--json"])
            report = json.loads(proc.stdout)
            self.assertTrue(report["ok"])
            self.assertEqual(report["n_recorded"], 2)


if __name__ == "__main__":
    unittest.main()
