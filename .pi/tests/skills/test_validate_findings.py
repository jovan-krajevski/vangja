import json
import tempfile
import unittest
from pathlib import Path

from helpers import FIXTURES, assert_help, run_script, script

VALIDATE = script("scientific-audit-and-adversarial-review", "validate_findings.py")
AUDIT_FIXTURES = FIXTURES / "audit"


def base_finding(**overrides):
    finding = {
        "id": "WS-S-001",
        "title": "Trend transfer potential has the wrong sign",
        "location": "src/vangja/components/linear_trend.py:397",
        "snapshot": "audit-cycle/3@abc1234",
        "command": "grep -n loss_factor src/vangja/components/linear_trend.py",
        "expected": "the potential is -loss*(slope-target)^2",
        "observed": "the potential increases with distance",
        "consequence": "transfer moves away from the context posterior",
        "severity": "S1",
        "confidence": "high",
        "status": "verified",
        "reproducible": "yes",
        "evidence": ["src/vangja/components/linear_trend.py:397"],
        "cross_refs": [],
        "owner": "code",
        "phase": "finder",
        "found_in_cycle": 3,
    }
    finding.update(overrides)
    return finding


class TestValidateFindings(unittest.TestCase):
    def test_help(self):
        assert_help(self, VALIDATE)

    def test_valid_register_passes(self):
        proc = run_script(
            VALIDATE,
            ["--findings", AUDIT_FIXTURES / "findings_ok.json",
             "--workstream", "WS-S", "--json"],
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        report = json.loads(proc.stdout)
        self.assertEqual(report["n_meta_findings"], 0, report)
        self.assertEqual(report["severity_histogram"]["S0"], 1)

    def test_bad_register_flags_every_rule(self):
        proc = run_script(
            VALIDATE,
            ["--findings", AUDIT_FIXTURES / "findings_bad.json",
             "--workstream", "WS-S", "--json"],
        )
        self.assertEqual(proc.returncode, 1)
        report = json.loads(proc.stdout)
        rules = {f["rule"] for f in report["meta_findings"]}
        for expected in ("no-command", "bad-severity", "bad-confidence",
                         "bad-status", "bad-reproducible", "duplicate-id",
                         "duplicate-finding", "verified-low-confidence"):
            self.assertIn(expected, rules, report)

    def test_wrong_workstream_flagged(self):
        proc = run_script(
            VALIDATE,
            ["--findings", AUDIT_FIXTURES / "findings_ok.json",
             "--workstream", "WS-M", "--json"],
        )
        self.assertEqual(proc.returncode, 1)
        report = json.loads(proc.stdout)
        self.assertTrue(any(f["rule"] == "wrong-workstream"
                            for f in report["meta_findings"]))

    def _report(self, findings):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "findings.json"
            path.write_text(json.dumps({"findings": findings}))
            proc = run_script(
                VALIDATE,
                ["--findings", path, "--workstream", "WS-S", "--json"],
            )
        self.assertIn(proc.returncode, (0, 1))
        return json.loads(proc.stdout)

    def test_static_reading_is_not_a_command(self):
        report = self._report([base_finding(command="static reading of lines 397-410")])
        rules = {f["rule"] for f in report["meta_findings"]}
        self.assertIn("no-command", rules, report)

    def test_duplicate_is_folded_with_a_cross_ref(self):
        first = base_finding()
        second = base_finding(id="WS-S-002", location="src/vangja/components/linear_trend.py:410")
        report = self._report([first, second])
        self.assertEqual(report["n_deduped"], 1, report)
        self.assertEqual(len(report["duplicates"]), 1, report)
        self.assertEqual(report["duplicates"][0]["duplicate_of"], "WS-S-001")
        canonical = report["register"][0]
        self.assertIn("WS-S-002", canonical["cross_refs"], canonical)
        rules = {f["rule"] for f in report["meta_findings"]}
        self.assertIn("duplicate-finding", rules)

    def test_reproducible_requires_a_reason_for_no(self):
        bare = self._report([base_finding(status="to-verify", confidence="medium", reproducible="no")])
        self.assertIn("bad-reproducible", {f["rule"] for f in bare["meta_findings"]})
        reasoned = self._report(
            [base_finding(status="to-verify", confidence="medium", reproducible="no: data snapshot unavailable")]
        )
        self.assertNotIn("bad-reproducible", {f["rule"] for f in reasoned["meta_findings"]})
        explained_yes = self._report([base_finding(reproducible="yes — reproduced with the cited grep")])
        self.assertNotIn("bad-reproducible", {f["rule"] for f in explained_yes["meta_findings"]})

    def test_verified_requires_reproducible_yes(self):
        report = self._report([base_finding(reproducible="no: not reproducible")])
        self.assertIn("unreproducible-verified", {f["rule"] for f in report["meta_findings"]})

    def test_evidence_is_required(self):
        finding = base_finding()
        finding.pop("evidence")
        report = self._report([finding])
        self.assertIn("missing-field", {f["rule"] for f in report["meta_findings"]})

    def test_unresolved_needs_a_question_tag(self):
        report = self._report(
            [base_finding(status="unresolved", confidence="medium", reproducible="no: needs author input")]
        )
        self.assertIn("unresolved-no-question-tag", {f["rule"] for f in report["meta_findings"]})
        tagged = self._report(
            [base_finding(status="unresolved", confidence="medium",
                          reproducible="no: needs author input", question_tag="blocking")]
        )
        self.assertNotIn("unresolved-no-question-tag", {f["rule"] for f in tagged["meta_findings"]})

    def test_out_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit-validation.json"
            proc = run_script(
                VALIDATE,
                ["--findings", AUDIT_FIXTURES / "findings_ok.json", "--out", out],
            )
            self.assertEqual(proc.returncode, 0)
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
