import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from helpers import FIXTURES, REPO_ROOT, assert_help, run_script, script

CHECK = script("scientific-writing-latex", "check_claims.py")
PAPER_FIXTURES = FIXTURES / "manuscript"
CLEAN = PAPER_FIXTURES / "clean"
DANGLING = PAPER_FIXTURES / "dangling"


def head_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def clean_claims(commit: str, *, value: str = "0.31") -> dict:
    artifact = CLEAN / "units.csv"
    return {
        "schema": "vangja-claims-1.0",
        "claims": [{
            "id": "C-001",
            "value": value,
            "tex": "fixtures/manuscript/clean/generated/table.tex",
            "artifact": str(artifact.relative_to(REPO_ROOT)),
            "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "commit": commit,
            "status": "verified",
        }],
    }


class TestCheckClaims(unittest.TestCase):
    def test_help(self):
        assert_help(self, CHECK)

    def _run(self, paper: Path, claims: Path, *extra):
        proc = run_script(
            CHECK,
            ["--paper", paper, "--claims", claims, "--repo", REPO_ROOT,
             "--base-ref", "HEAD", "--json", *extra],
        )
        return proc, (json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else None)

    def test_clean_fixture_passes_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            claims = Path(tmp) / "CLAIMS.json"
            claims.write_text(json.dumps(clean_claims(head_sha())))
            proc, report = self._run(CLEAN / "paper.tex", claims, "--gate")
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(report["n_findings"], 0, report)
            self.assertEqual(report["n_numbers"], 1)

    def test_dangling_number_reported_and_gated(self):
        missing = DANGLING / "no-claims-here.json"
        proc, report = self._run(DANGLING / "paper.tex", missing)
        self.assertEqual(proc.returncode, 0, "report mode must not fail")
        self.assertTrue(any(f["rule"] == "dangling-number" for f in report["findings"]))
        self.assertTrue(any(f["rule"] == "hand-typed-number" for f in report["findings"]))
        gated, _ = self._run(DANGLING / "paper.tex", missing, "--gate")
        self.assertEqual(gated.returncode, 1)

    def test_sha_not_ancestor_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            claims = Path(tmp) / "CLAIMS.json"
            claims.write_text(json.dumps(clean_claims("0" * 40)))
            proc, report = self._run(CLEAN / "paper.tex", claims)
            self.assertEqual(proc.returncode, 0)
            self.assertTrue(any(f["rule"] == "sha-not-ancestor"
                                for f in report["findings"]), report)

    def test_artifact_hash_mismatch_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            claims = clean_claims(head_sha())
            claims["claims"][0]["artifact_sha256"] = "f" * 64
            path = Path(tmp) / "CLAIMS.json"
            path.write_text(json.dumps(claims))
            proc, report = self._run(CLEAN / "paper.tex", path)
            self.assertTrue(any(f["rule"] == "artifact-hash-mismatch"
                                for f in report["findings"]), report)

    def test_legacy_paper_is_a_report_not_an_error(self):
        proc = run_script(
            CHECK,
            ["--paper", REPO_ROOT / "paper" / "paper.tex",
             "--claims", REPO_ROOT / "CLAIMS.json",
             "--repo", REPO_ROOT, "--base-ref", "none", "--json"],
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(proc.stdout)
        self.assertGreater(report["n_numbers"], 50)
        self.assertGreater(report["by_severity"]["S1"], 50)
        gated = run_script(
            CHECK,
            ["--paper", REPO_ROOT / "paper" / "paper.tex",
             "--claims", REPO_ROOT / "CLAIMS.json",
             "--repo", REPO_ROOT, "--base-ref", "none", "--gate"],
        )
        self.assertEqual(gated.returncode, 1)


if __name__ == "__main__":
    unittest.main()
