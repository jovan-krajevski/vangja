import json
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import FIXTURES, assert_help, run_script, script

PARSE = script("hpc-slurm-and-checkpointing", "parse_sacct.py")
RECORD = script("hpc-slurm-and-checkpointing", "make_job_record.py")
SMOKE = script("hpc-slurm-and-checkpointing", "checkpoint_smoke.py")
HPC_FIXTURES = FIXTURES / "hpc"
SACCT = HPC_FIXTURES / "sacct_states.json"


def classify(job_id: int, *extra) -> dict:
    proc = run_script(PARSE, [SACCT, "--json", *extra])
    if proc.returncode not in (0, 1):
        raise AssertionError(proc.stderr)
    report = json.loads(proc.stdout)
    for job in report["jobs"]:
        if job["job_id"] == str(job_id):
            return job
    raise AssertionError(f"job {job_id} not classified")


class TestParseSacct(unittest.TestCase):
    def test_help(self):
        assert_help(self, PARSE)

    def test_completed_collects(self):
        self.assertEqual(classify(4711)["action"], "collect")
        self.assertEqual(classify(4719)["state"], "completed")

    def test_timeout_requeue_depends_on_progress(self):
        with_progress = classify(4712, "--progress-advanced")
        self.assertEqual(with_progress["state"], "timed_out")
        self.assertEqual(with_progress["action"], "requeue")
        self.assertEqual(with_progress["exit_code"], 99)
        without = classify(4712)
        self.assertEqual(without["action"], "requeue-no-progress")

    def test_oom_memory_classification(self):
        self.assertEqual(classify(4713)["state"], "oom")
        self.assertEqual(classify(4713)["action"], "failed-final")
        retry = classify(4713, "--memory-can-double")
        self.assertEqual(retry["action"], "retry-once-doubled-memory")
        self.assertEqual(classify(4713)["elapsed_seconds"], 720)

    def test_node_fail_requeues_with_progress(self):
        self.assertEqual(classify(4714, "--progress-advanced")["action"], "requeue")
        self.assertEqual(classify(4714)["state"], "preempted")
        self.assertEqual(classify(4714)["node"], "")

    def test_active_states_wait(self):
        self.assertEqual(classify(4715)["action"], "wait")
        self.assertEqual(classify(4715)["state"], "running")
        self.assertEqual(classify(4716)["state"], "queued")

    def test_failed_and_cancelled(self):
        self.assertEqual(classify(4717)["action"], "failed-final")
        self.assertEqual(classify(4718)["action"], "stop")

    def test_max_attempts_caps_requeue(self):
        capped = classify(4712, "--progress-advanced", "--attempt", "6")
        self.assertEqual(capped["action"], "failed-final")
        self.assertIn("maxAttempts", capped["reason"])

    def test_out_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "status.json"
            proc = run_script(PARSE, [SACCT, "--json", "--out", out])
            self.assertEqual(proc.returncode, 0)
            self.assertTrue(out.exists())


class TestMakeJobRecord(unittest.TestCase):
    def _args(self, out: Path, *extra):
        return [
            "--id", "E-0007", "--task-id", "T-012",
            "--branch", "work/T-012-seeds",
            "--commit", "abc123", "--command", "python -m runner",
            "--data-hash", "DATA", "--uv-lock-sha", "LOCK",
            "--checkpoint-dir", "$SCRATCH/vangja/E-0007/checkpoints",
            "--resume-command", "python -m runner --resume $CKPT",
            "--expected-output", "results/units.csv",
            "--seeds", "42,20240816", "--smoke-ok",
            "--out", str(out), *extra,
        ]

    def test_help(self):
        assert_help(self, RECORD)

    def test_valid_record_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "E-0007"
            proc = run_script(RECORD, self._args(out))
            self.assertEqual(proc.returncode, 0, proc.stderr)
            record = json.loads((out / "record.json").read_text())
            self.assertEqual(record["id"], "E-0007")
            self.assertEqual(record["status"], "draft")
            self.assertTrue(record["checkpoints"]["flushOnSignal"])
            first = record["idempotencyKey"]
            self.assertEqual(len(first), 64)
            run_script(RECORD, self._args(out))
            again = json.loads((out / "record.json").read_text())
            self.assertEqual(again["idempotencyKey"], first)

    def test_seed_change_changes_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "E-0007"
            run_script(RECORD, self._args(out))
            key_a = json.loads((out / "record.json").read_text())["idempotencyKey"]
            run_script(RECORD, self._args(out, "--seeds", "7"))
            key_b = json.loads((out / "record.json").read_text())["idempotencyKey"]
            self.assertNotEqual(key_a, key_b)

    def test_refuses_without_smoke_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = [a for a in self._args(Path(tmp) / "E-0007") if a != "--smoke-ok"]
            proc = run_script(RECORD, args)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("--smoke-ok", proc.stderr)

    def test_refuses_resume_command_without_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._args(Path(tmp) / "E-0007")
            index = args.index("python -m runner --resume $CKPT")
            args[index] = "python -m runner"
            proc = run_script(RECORD, args)
            self.assertEqual(proc.returncode, 1)
            self.assertIn("--resume", proc.stderr)


class TestCheckpointSmoke(unittest.TestCase):
    def _command(self, out: Path, runner: Path, units="a,b,c,d"):
        return [
            sys.executable, str(runner), "--out", str(out),
            "--units", units, "--sleep", "0.15", "--linger", "30",
            "{limit}", "{resume}",
        ]

    def test_smoke_proves_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "smoke"
            proc = run_script(
                SMOKE,
                ["--out", out, "--limit", "1", "--", *self._command(
                    out, HPC_FIXTURES / "fixture_runner.py")],
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            evidence = json.loads((out / "smoke.json").read_text())
            self.assertTrue(evidence["ok"], evidence)
            self.assertTrue(evidence["first_completed"])
            self.assertEqual(evidence["missing_skips"], [])
            self.assertEqual(evidence["refitted_after_resume"], [])
            self.assertEqual(evidence["resume_returncode"], 0)
            # the resume run finished all units
            self.assertEqual(len(evidence["completed_after_resume"]), 4)

    def test_bad_runner_fails_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "smoke-bad"
            proc = run_script(
                SMOKE,
                ["--out", out, "--limit", "1", "--", *self._command(
                    out, HPC_FIXTURES / "fixture_runner_bad.py")],
            )
            self.assertEqual(proc.returncode, 1)
            evidence = json.loads((out / "smoke.json").read_text())
            self.assertFalse(evidence["ok"])

    def test_refuses_dirty_out_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "smoke"
            out.mkdir()
            (out / "progress.json").write_text('{"version": 1, "completed": ["a"]}')
            proc = run_script(
                SMOKE,
                ["--out", out, "--limit", "1", "--", *self._command(
                    out, HPC_FIXTURES / "fixture_runner.py")],
            )
            self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
