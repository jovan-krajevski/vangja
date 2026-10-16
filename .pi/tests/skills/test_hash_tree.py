import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from helpers import assert_help, run_script, script

HASH = script("data-provenance", "hash_tree.py")


class TestHashTree(unittest.TestCase):
    def test_help(self):
        assert_help(self, HASH)

    def test_writes_json_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            (root / "sub").mkdir(parents=True)
            (root / "a.csv").write_text("x\n1\n")
            (root / "sub" / "b.csv").write_text("y\n2\n")
            out = Path(tmp) / "hash.json"
            proc = run_script(HASH, ["--root", root, "--out", out])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(out.read_text())
            self.assertEqual(sorted(payload["files"]), ["a.csv", "sub/b.csv"])
            expected = hashlib.sha256((root / "a.csv").read_bytes()).hexdigest()
            self.assertEqual(payload["files"]["a.csv"], expected)

    def test_sidecar_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            root.mkdir()
            (root / "a.csv").write_text("x\n1\n")
            sidecar = Path(tmp) / "data.sha256"
            proc = run_script(HASH, ["--root", root, "--sidecar", sidecar])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            line = sidecar.read_text().strip()
            digest, name = line.split("  ")
            self.assertEqual(name, "a.csv")
            self.assertEqual(len(digest), 64)

    def test_excludes_pycache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            (root / "__pycache__").mkdir(parents=True)
            (root / "__pycache__" / "x.pyc").write_text("junk")
            (root / "real.csv").write_text("x\n1\n")
            out = Path(tmp) / "hash.json"
            run_script(HASH, ["--root", root, "--out", out])
            payload = json.loads(out.read_text())
            self.assertEqual(list(payload["files"]), ["real.csv"])

    def test_requires_an_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            root.mkdir()
            proc = run_script(HASH, ["--root", root])
            self.assertEqual(proc.returncode, 2)

    def test_missing_root_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = run_script(HASH, ["--root", Path(tmp) / "nope",
                                     "--out", Path(tmp) / "out.json"])
            self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
