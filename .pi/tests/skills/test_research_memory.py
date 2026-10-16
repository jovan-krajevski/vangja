import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from helpers import REPO_ROOT, SKILLS

CHECK = SKILLS / "research-memory-and-handoff" / "scripts" / "check_state_files.sh"


def run(args, timeout: int = 60):
    return subprocess.run(
        ["bash", str(CHECK), *[str(a) for a in args]],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=timeout,
    )


def make_root(tmp: Path, *, loop_state="default", context_lines=10,
              ledger=None) -> Path:
    state = tmp / ".pi" / "state"
    state.mkdir(parents=True)
    (state / "CONTEXT.md").write_text("\n".join(f"line {i}" for i in range(context_lines)))
    if loop_state == "default":
        payload = {
            "schema": "pi-loop-state/1", "run_id": None, "state": "UNINITIALIZED",
            "paused_from": None, "task_id": None, "pr_id": None,
            "experiment_id": None, "attempt": 0, "next_gate": None,
            "reason": None, "updated_at": "", "updated_by": "",
        }
        (state / "LOOP_STATE.json").write_text(json.dumps(payload))
    elif loop_state is not None:
        (state / "LOOP_STATE.json").write_text(json.dumps(loop_state))
    if ledger is not None:
        (state / "EVENTS.jsonl").write_text(ledger)
    return tmp


class TestCheckStateFiles(unittest.TestCase):
    def test_help(self):
        proc = run(["--help"])
        self.assertEqual(proc.returncode, 0)
        self.assertIn("usage", proc.stdout.lower())

    def test_repo_state_passes(self):
        proc = run([])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("PASS  state-dir", proc.stdout)

    def test_bad_schema_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_root(Path(tmp), loop_state={
                "schema": "wrong/1", "state": "WORK"})
            proc = run(["--root", tmp])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("FAIL", proc.stdout)

    def test_illegal_state_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "schema": "pi-loop-state/1", "run_id": None, "state": "SLEEPING",
                "paused_from": None, "task_id": None, "pr_id": None,
                "experiment_id": None, "attempt": 0, "next_gate": None,
                "reason": None, "updated_at": "", "updated_by": "",
            }
            make_root(Path(tmp), loop_state=payload)
            proc = run(["--root", tmp])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("unknown state", proc.stdout)

    def test_long_context_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_root(Path(tmp), context_lines=250)
            proc = run(["--root", tmp])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("limit 200", proc.stdout)

    def test_missing_loop_state_warns_then_strict_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_root(Path(tmp), loop_state=None)
            proc = run(["--root", tmp])
            self.assertEqual(proc.returncode, 0, proc.stdout)
            self.assertIn("WARN", proc.stdout)
            strict = run(["--root", tmp, "--strict"])
            self.assertEqual(strict.returncode, 1)

    def test_bad_ledger_line_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_root(Path(tmp), ledger='{"schema": "x", "kind": "event"}\n')
            proc = run(["--root", tmp])
            self.assertEqual(proc.returncode, 1)
            self.assertIn("missing", proc.stdout)


if __name__ == "__main__":
    unittest.main()
