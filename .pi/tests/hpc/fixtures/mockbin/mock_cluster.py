#!/usr/bin/env python3
"""A SLURM cluster that lives in a directory — the S5 mock-cluster shims.

Every executable in this directory (`ssh`, `scp`, `rsync`, `sbatch`, `squeue`,
`sacct`, `scancel`, `ssh-agent`, `ssh-add`) is a two-line wrapper around this
file. Nothing here touches a network or a real cluster: the "remote" root is
`$MOCK_CLUSTER_DIR/remote`, and every invocation is appended to
`$MOCK_CLUSTER_DIR/calls.jsonl` so a test can assert exactly what ran.

Driving file: `$MOCK_CLUSTER_DIR/scenario.json`

    {
      "host": "mock-cluster",
      "user": "mockuser",
      "startJobId": 900001,
      "states": ["PENDING", "RUNNING", "RUNNING", "COMPLETED"],
      "exitCode": 0,
      "elapsed": 754,
      "maxRss": "1024M",
      "node": "mocknode01",
      "progress": {"version": 1, "completed": ["2023-01-01:main"]},
      "checkpoints": {"fit_a.pkl": "checkpoint-a"},
      "outputs": {"results/units_a.csv": "unit,value\\na,1\\n"},
      "partialOutputs": {},
      "logLines": {"out": ["started"], "err": []},
      "failSsh": false,
      "sbatchExit": 0,
      "squeueRows": null
    }

`states` is consumed one entry per `squeue` call (the collector's poll), so a
test drives the timeline precisely. `sacct` never advances the counter: it
reports the state the last poll observed, which is how the real accounting lags
the queue.

Safety: refuses to run without `MOCK_CLUSTER_DIR`, and only "connects" to the
host named in the scenario.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import time

MOCK_DIR = os.environ.get("MOCK_CLUSTER_DIR", "")
TOOL = sys.argv[1] if len(sys.argv) > 1 else os.path.basename(sys.argv[0])
ARGS = sys.argv[2:] if len(sys.argv) > 1 else sys.argv[1:]
RUN_ID = f"{os.getpid()}-{time.time_ns()}"

TERMINAL_STATES = {"COMPLETED", "TIMEOUT", "TIME_LIMIT", "DEADLINE", "OUT_OF_MEMORY", "FAILED", "CANCELLED", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "NODE_DOWN"}


def die(message: str, code: int = 90) -> None:
    sys.stderr.write(f"mock-cluster {TOOL}: {message}\n")
    sys.exit(code)


def require_dir() -> str:
    if MOCK_DIR == "" or not os.path.isdir(MOCK_DIR):
        die("MOCK_CLUSTER_DIR is not set or does not exist; refusing to run")
    return MOCK_DIR


def scenario() -> dict:
    path = os.path.join(require_dir(), "scenario.json")
    if not os.path.isfile(path):
        die(f"missing scenario file {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def state_path() -> str:
    return os.path.join(require_dir(), "state.json")


def load_state() -> dict:
    if not os.path.isfile(state_path()):
        return {"next_job_id": int(scenario().get("startJobId", 900001))}
    with open(state_path(), encoding="utf-8") as handle:
        return json.load(handle)


def save_state(state: dict) -> None:
    tmp = f"{state_path()}.tmp-{RUN_ID}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    os.replace(tmp, state_path())


def job_dir(job_id: str) -> str:
    return os.path.join(require_dir(), "jobs", str(job_id))


def load_job(job_id: str) -> dict:
    path = os.path.join(job_dir(job_id), "job.json")
    if not os.path.isfile(path):
        die(f"unknown job {job_id}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_job(job_id: str, job: dict) -> None:
    path = os.path.join(job_dir(job_id), "job.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(job, handle, indent=2)


def record(extra: dict | None = None) -> None:
    entry = {
        "ts": time.time(),
        "tool": TOOL,
        "argv": ARGS,
        "pid": os.getpid(),
    }
    if extra:
        entry.update(extra)
    with open(os.path.join(require_dir(), "calls.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def remote_root() -> str:
    return os.path.join(require_dir(), "remote")


def remote_path(path: str) -> str:
    """Map an absolute remote path to the fake remote root."""
    if path.startswith(remote_root()):
        return path
    return os.path.join(remote_root(), path.lstrip("/"))


def split_host(spec: str) -> tuple[str | None, str]:
    if ":" not in spec:
        return None, spec
    host_part, path = spec.split(":", 1)
    if "/" in host_part or host_part == "":
        return None, spec
    if "@" in host_part:
        host_part = host_part.split("@", 1)[1]
    return host_part, path


def expect_mock_host(host: str) -> None:
    expected = scenario().get("host", "mock-cluster")
    if host != expected:
        die(f"refusing to connect to host {host!r}: this is a mock cluster ({expected!r} only)", 96)


# ---------------------------------------------------------------------------
# ssh
# ---------------------------------------------------------------------------

FLAGS_WITH_VALUE = {"-o", "-p", "-i", "-F", "-l", "-c", "-m", "-b", "-e", "-J", "-L", "-R", "-D", "-W", "-Q", "-S", "-w", "-E", "-I", "-O"}


def parse_ssh_args(args: list[str]) -> tuple[str, str]:
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            index += 1
            break
        if arg in FLAGS_WITH_VALUE:
            index += 2
            continue
        if arg.startswith("-") and len(arg) > 2 and arg[:2] in FLAGS_WITH_VALUE:
            index += 1
            continue
        if arg.startswith("-") and len(arg) == 2:
            index += 1
            continue
        break
    if index >= len(args):
        die("no user@host given")
    target = args[index]
    command = " ".join(args[index + 1 :])
    if "@" not in target:
        die(f"expected user@host, got {target!r}")
    host = target.split("@", 1)[1]
    return host, command


def cmd_ssh() -> int:
    host, command = parse_ssh_args(ARGS)
    if os.environ.get("MOCK_FAIL_IF_SSH") == "1":
        record({"violation": "ssh-called-while-disabled"})
        sys.stderr.write("mock-cluster ssh: SSH MUST NOT BE CALLED in this test\n")
        return 97
    sc = scenario()
    expect_mock_host(host)
    record({"remote": command, "host": host})
    if sc.get("failSsh"):
        sys.stderr.write("mock-cluster ssh: simulated transport failure\n")
        return 255
    env = dict(os.environ)
    env["USER"] = sc.get("user", "mockuser")
    env["HOME"] = os.path.join(require_dir(), "remote", "home")
    os.makedirs(env["HOME"], exist_ok=True)
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=remote_root(),
        env=env,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


# ---------------------------------------------------------------------------
# scp / rsync
# ---------------------------------------------------------------------------

SCP_VALUE_FLAGS = {"-P", "-o", "-i", "-c", "-F", "-J", "-S", "-l"}
SCP_FLAGS = {"-q", "-p", "-r", "-C", "-B", "-3", "-4", "-6", "-T", "-v"}


def parse_scp_args(args: list[str]) -> tuple[str, str]:
    positional: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in SCP_VALUE_FLAGS:
            index += 2
            continue
        if arg.startswith("-") and len(arg) > 2 and arg[:2] in SCP_VALUE_FLAGS:
            index += 1
            continue
        if arg in SCP_FLAGS or (arg.startswith("-") and arg not in {"-"}):
            index += 1
            continue
        positional.append(arg)
        index += 1
    if len(positional) != 2:
        die(f"expected exactly two paths, got {positional!r}")
    return positional[0], positional[1]


def copy_tree(source: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.isdir(source):
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
    else:
        os.makedirs(dest if dest.endswith("/") else os.path.dirname(dest) or ".", exist_ok=True)
        target = dest
        if os.path.isdir(dest):
            target = os.path.join(dest, os.path.basename(source))
        shutil.copy2(source, target)


def cmd_scp() -> int:
    source_spec, dest_spec = parse_scp_args(ARGS)
    source_host, source_path = split_host(source_spec)
    dest_host, dest_path = split_host(dest_spec)
    if source_host and dest_host:
        die("remote-to-remote scp is not supported by the mock")
    if source_host:
        expect_mock_host(source_host)
        record({"from": source_spec, "to": dest_spec, "direction": "download"})
        copy_tree(remote_path(source_path), dest_path)
        return 0
    if dest_host:
        expect_mock_host(dest_host)
        record({"from": source_spec, "to": dest_spec, "direction": "upload"})
        copy_tree(source_spec, remote_path(dest_path))
        return 0
    record({"from": source_spec, "to": dest_spec, "direction": "local"})
    copy_tree(source_spec, dest_path)
    return 0


def cmd_rsync() -> int:
    relative = False
    positional: list[str] = []
    index = 0
    while index < len(ARGS):
        arg = ARGS[index]
        if arg in {"--relative", "-R"}:
            relative = True
            index += 1
            continue
        if arg == "-e":
            index += 2
            continue
        if arg.startswith("--") or (arg.startswith("-") and arg != "-"):
            index += 1
            continue
        positional.append(arg)
        index += 1
    if len(positional) != 2:
        die(f"expected exactly two paths, got {positional!r}")
    source_spec, dest_spec = positional
    source_host, source_path = split_host(source_spec)
    dest_host, dest_path = split_host(dest_spec)
    if source_host and dest_host:
        die("remote-to-remote rsync is not supported by the mock")
    if source_host:
        expect_mock_host(source_host)
        record({"from": source_spec, "to": dest_spec, "direction": "download", "relative": relative})
        copied: list[str] = []
        if "/./" in source_path:
            base, rel = source_path.split("/./", 1)
            for match in sorted(glob.glob(os.path.join(remote_path(base), rel))):
                rel_out = os.path.relpath(match, remote_path(base))
                target = os.path.join(dest_path, rel_out)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                copy_tree(match, target)
                copied.append(rel_out)
        elif source_path.endswith("/"):
            # Directory contents into the destination directory.
            os.makedirs(dest_path, exist_ok=True)
            for entry in sorted(os.listdir(remote_path(source_path))):
                copy_tree(os.path.join(remote_path(source_path), entry), os.path.join(dest_path, entry))
                copied.append(entry)
        else:
            for match in sorted(glob.glob(remote_path(source_path))):
                copy_tree(match, dest_path)
                copied.append(os.path.basename(match))
        if not copied:
            sys.stderr.write(f"mock-cluster rsync: no files matched {source_spec}\n")
            return 94
        return 0
    if dest_host:
        expect_mock_host(dest_host)
        record({"from": source_spec, "to": dest_spec, "direction": "upload", "relative": relative})
        copy_tree(source_spec, remote_path(dest_path))
        return 0
    record({"from": source_spec, "to": dest_spec, "direction": "local", "relative": relative})
    copy_tree(source_spec, dest_path)
    return 0


# ---------------------------------------------------------------------------
# sbatch
# ---------------------------------------------------------------------------

def parse_sbatch(args: list[str]) -> dict:
    options: dict[str, str] = {}
    script = ""
    for arg in args:
        if arg.startswith("--") and "=" in arg:
            key, value = arg.split("=", 1)
            options[key] = value
        elif arg in {"--parsable", "-p"}:
            options["--parsable"] = "true"
        elif not arg.startswith("-"):
            script = arg
    return {"options": options, "script": script}


def materialize(job_id: str, state: str) -> list[str]:
    """Write the files a real job would have produced by this state."""
    sc = scenario()
    job = load_job(job_id)
    scratch = os.path.join(remote_root(), "scratch", job["experiment_id"])
    checkpoints = os.path.join(scratch, "checkpoints")
    # The job runs in the checkout and writes declared outputs relative to it.
    run_dir = os.path.join(remote_root(), "repo", "runs", job["experiment_id"])
    written: list[str] = []
    if state in {"RUNNING", "COMPLETING", "SUSPENDED"} or state in TERMINAL_STATES:
        os.makedirs(checkpoints, exist_ok=True)
        progress = sc.get("progress")
        if progress is not None and "checkpoints/progress.json" not in job.get("written", []):
            with open(os.path.join(checkpoints, "progress.json"), "w", encoding="utf-8") as handle:
                json.dump(progress, handle, indent=2)
            written.append("checkpoints/progress.json")
        for name, content in (sc.get("checkpoints") or {}).items():
            target = os.path.join(checkpoints, name)
            if not os.path.exists(target):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write(content)
                written.append(f"checkpoints/{name}")
    if state == "COMPLETED":
        for name, content in (sc.get("outputs") or {}).items():
            target = os.path.join(run_dir, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as handle:
                handle.write(content)
            written.append(name)
    if state in {"TIMEOUT", "TIME_LIMIT", "DEADLINE", "OUT_OF_MEMORY", "FAILED", "NODE_FAIL", "PREEMPTED"}:
        for name, content in (sc.get("partialOutputs") or {}).items():
            target = os.path.join(run_dir, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if not os.path.exists(target):
                with open(target, "w", encoding="utf-8") as handle:
                    handle.write(content)
                written.append(name)
    job["written"] = sorted(set(job.get("written", [])) | set(written))
    save_job(job_id, job)
    return written


def cmd_sbatch() -> int:
    sc = scenario()
    parsed = parse_sbatch(ARGS)
    options = parsed["options"]
    job_name = options.get("--job-name", "mock-job")
    if sc.get("sbatchExit", 0) != 0:
        sys.stderr.write("sbatch: error: simulated submission failure (sbatchExit)\n")
        record({"job_name": job_name, "ok": False})
        return int(sc.get("sbatchExit", 1))
    state = load_state()
    job_id = str(state["next_job_id"])
    state["next_job_id"] = int(job_id) + 1
    save_state(state)
    experiment_id = job_name.split("-a")[0] if "-a" in job_name else job_name
    scratch = os.path.join(remote_root(), "scratch", experiment_id)
    os.makedirs(os.path.join(scratch, "checkpoints"), exist_ok=True)
    job = {
        "job_id": job_id,
        "job_name": job_name,
        "experiment_id": experiment_id,
        "script": parsed["script"],
        "options": options,
        "polls": 0,
        "states": sc.get("states", ["PENDING", "RUNNING", "COMPLETED"]),
        "created": time.time(),
        "written": [],
    }
    save_job(job_id, job)
    log_lines = sc.get("logLines") or {}
    with open(os.path.join(scratch, f"slurm-{job_id}.out"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(log_lines.get("out", ["mock job started"])) + "\n")
    with open(os.path.join(scratch, f"slurm-{job_id}.err"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(log_lines.get("err", [])) + "\n")
    record({"job_id": job_id, "job_name": job_name, "script": parsed["script"], "options": options})
    print(job_id)
    return 0


# ---------------------------------------------------------------------------
# squeue
# ---------------------------------------------------------------------------

def parse_squeue(args: list[str]) -> dict:
    filters: dict[str, str] = {}
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in {"-u", "-o", "-t", "-p", "-w", "-j", "--name"}:
            filters[arg] = args[index + 1] if index + 1 < len(args) else ""
            index += 2
            continue
        index += 1
    return filters


def cmd_squeue() -> int:
    filters = parse_squeue(ARGS)
    requested = [job for job in filters.get("-j", "").split(",") if job]
    want_name = filters.get("--name", "")
    jobs = []
    for job_id in sorted(os.listdir(os.path.join(require_dir(), "jobs"))):
        job = load_job(job_id)
        if requested and job_id not in requested and job["job_name"] not in requested:
            continue
        if want_name and job["job_name"] != want_name:
            continue
        states = job["states"]
        observed = states[min(job["polls"], len(states) - 1)]
        job["polls"] += 1
        save_job(job_id, job)
        materialized = materialize(job_id, observed)
        jobs.append({"job": job, "state": observed, "materialized": materialized})
    record({"jobs": [{"job_id": job["job"]["job_id"], "state": job["state"], "materialized": job["materialized"]} for job in jobs], "filters": filters})
    rows = []
    for entry in jobs:
        if entry["state"] in TERMINAL_STATES:
            continue
        job = entry["job"]
        elapsed = int(float(scenario().get("elapsed", 0)))
        rows.append(f"{job['job_id']}|{entry['state']}|{elapsed // 60:02d}:{elapsed % 60:02d}|{scenario().get('node', 'mocknode01')}")
    sys.stdout.write("\n".join(rows) + ("\n" if rows else ""))
    return 0


# ---------------------------------------------------------------------------
# sacct
# ---------------------------------------------------------------------------

def observed_state(job: dict) -> str:
    states = job["states"]
    index = min(max(job["polls"] - 1, 0), len(states) - 1)
    return states[index]


def cmd_sacct() -> int:
    filters = parse_squeue(ARGS)
    requested = [job for job in filters.get("-j", "").split(",") if job]
    sc = scenario()
    state = {"limited": [], "jobs": []}
    for job_id in sorted(os.listdir(os.path.join(require_dir(), "jobs"))):
        job = load_job(job_id)
        if requested and job_id not in requested:
            continue
        current = observed_state(job)
        exit_code = int(sc.get("exitCode", 0))
        if current == "COMPLETED":
            exit_code = int(sc.get("exitCode", 0))
        payload = {
            "job_id": int(job_id),
            "name": job["job_name"],
            "state": {"current": [current], "reason": "mock"},
            "exit_code": {"return_code": exit_code, "signal": 0},
            "derived_exit_code": {"return_code": exit_code, "signal": 0},
            "time": {
                "elapsed": int(sc.get("elapsed", 0)),
                "start": int(job["created"]),
                "end": int(time.time()) if current in TERMINAL_STATES else None,
            },
            "nodes": sc.get("node", "mocknode01"),
            "alloc_cpus": int(sc.get("cpus", 8)),
            "max_rss": sc.get("maxRss", "1024M"),
        }
        state["jobs"].append(payload)
    record({"jobs": [job["job_id"] for job in state["jobs"]], "filters": filters})
    print(json.dumps(state, indent=2))
    return 0 if state["jobs"] else 1


def cmd_scancel() -> int:
    job_ids = [arg for arg in ARGS if not arg.startswith("-")]
    for job_id in job_ids:
        try:
            job = load_job(job_id)
        except SystemExit:
            continue
        job["states"] = ["CANCELLED"]
        job["polls"] = 1
        save_job(job_id, job)
    record({"cancelled": job_ids})
    return 0


# ---------------------------------------------------------------------------
# ssh-agent / ssh-add (hermetic stand-ins for the real binaries)
# ---------------------------------------------------------------------------

def agent_state_path() -> str:
    return os.path.join(require_dir(), "agent.json")


def cmd_ssh_agent() -> int:
    if "-k" in ARGS:
        path = agent_state_path()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                agent = json.load(handle)
            if os.path.exists(agent.get("socket", "")):
                os.remove(agent["socket"])
            os.remove(path)
        record({"agent": "kill"})
        print("unset SSH_AUTH_SOCK;")
        return 0
    socket = os.path.join(require_dir(), "agent.sock")
    with open(socket, "w", encoding="utf-8") as handle:
        handle.write("mock ssh-agent socket\n")
    agent = {"socket": socket, "pid": os.getpid(), "identities": []}
    with open(agent_state_path(), "w", encoding="utf-8") as handle:
        json.dump(agent, handle)
    record({"agent": "start", "socket": socket})
    print(f"SSH_AUTH_SOCK={socket}; export SSH_AUTH_SOCK;")
    print(f"SSH_AGENT_PID={os.getpid()}; export SSH_AGENT_PID;")
    print("Agent pid %d" % os.getpid())
    return 0


def cmd_ssh_add() -> int:
    path = agent_state_path()
    if not os.path.exists(path):
        sys.stderr.write("Could not open a connection to your authentication agent.\n")
        return 2
    with open(path, encoding="utf-8") as handle:
        agent = json.load(handle)
    if "-l" in ARGS:
        for identity in agent.get("identities", []):
            print(f"256 SHA256:mockfingerprint {identity} (ED25519)")
        record({"agent": "list", "count": len(agent.get("identities", []))})
        return 0 if agent.get("identities") else 1
    if "-D" in ARGS:
        agent["identities"] = []
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(agent, handle)
        record({"agent": "delete-all"})
        print("All identities removed.")
        return 0
    keys = [arg for arg in ARGS if not arg.startswith("-")]
    if not keys:
        sys.stderr.write("ssh-add: no key given\n")
        return 2
    key = keys[0]
    if not os.path.isfile(key):
        sys.stderr.write(f"ssh-add: cannot stat {key}: No such file or directory\n")
        record({"agent": "add", "key": key, "ok": False})
        return 1
    agent.setdefault("identities", []).append(key)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(agent, handle)
    record({"agent": "add", "key": key, "ok": True})
    print(f"Identity added: {key} ({key})")
    return 0


COMMANDS = {
    "ssh": cmd_ssh,
    "scp": cmd_scp,
    "rsync": cmd_rsync,
    "sbatch": cmd_sbatch,
    "squeue": cmd_squeue,
    "sacct": cmd_sacct,
    "scancel": cmd_scancel,
    "ssh-agent": cmd_ssh_agent,
    "ssh-add": cmd_ssh_add,
}


def main() -> int:
    command = COMMANDS.get(TOOL)
    if command is None:
        die(f"unknown mock tool {TOOL!r}")
    return command()


if __name__ == "__main__":
    sys.exit(main())
