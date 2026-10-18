# specs.md — everything the loop needs before it can run

This is the fill-in-the-blanks sheet for running the Vangja research loop end to
end. Two columns matter for every item: **where the information goes** (a file
and field) and **how to prove it is right** (a command whose output you can
check). Nothing here is a secret: real keys and tokens stay outside the
repository.

The loop has two modes:

- **Local mode (no HPC).** Fill sections 1, 2, 4, 6 and 7. Leave
  `.pi/config/hpc.json` absent or with `"host": ""`; every `hpc_*` tool then
  reports `{enabled:false}` and `WAITING_HPC` is skipped.
- **Full mode (with HPC).** Everything above plus section 3.

Run the one-screen checklist at the end (§10) after filling anything in.

---

## 0. The information map

| # | Information | File and field | Required? | Verified by |
|---|---|---|---|---|
| 1 | pi binary and pinned provider credential | `pi` on `PATH`; credential in `~/.pi/agent/auth.json` or a provider env var | yes | `pi --print "say ok"`, `make -C .pi verify-env` |
| 2 | Pinned model + thinking level | `.pi/config/model.json` (`provider`, `model`, `thinking`) and `.pi/settings.json` and `.pi/config/agents.json` (`pinnedModel`, `pinnedThinking`) | yes (already set) | `/loop:verify-env` (model-pin line) |
| 3 | Branch the loop merges into | `.pi/config/project.json` → `baseBranch` (`"@current"` = branch checked out at BOOT) | yes (already set) | `/loop:verify-env` → `base=…` |
| 4 | Human-merge policy | `.pi/config/project.json` → `requireHumanMerge` | yes (default `true`) | `/loop:verify-env` |
| 5 | Git author identity | your global `user.name` / `user.email` | yes | `git config --global user.name` |
| 6 | HPC login host / IP | `.pi/config/hpc.json` → `host` | only for HPC | `hpc_config` tool, `/loop:verify-env` |
| 7 | HPC username and port | `user`, `port` | only for HPC | `ssh` preflight below |
| 8 | HPC private key path | `hpcKey` | only for HPC | `resolveHpc` check via `/loop:verify-env` |
| 9 | Remote repo clone path | `remoteRepoDir` | only for HPC | remote `git -C … remote -v` |
| 10 | Remote scratch path | `remoteScratchDir` | only for HPC | remote `mkdir -p` check |
| 11 | Git key for remote fetch | `gitKey`, `forwardGitAgent` | only for HPC | `ssh-add -l` inside the private agent; submit evidence `git_forwarding` |
| 12 | Container image path | `container` (remote absolute path) | only for HPC with a container | submit gate `container` check |
| 13 | SLURM partition | `partition` | only for HPC | `sinfo -p <partition>` |
| 14 | Resource defaults | `cpus`, `memGb`, `gpus`, `wallClockHours` | only for HPC | submit gate `wall-clock` check |
| 15 | Poll/requeue caps | `pollSeconds`, `maxPolls`, `maxAttemptsPerExperiment`, `maxParallelJobs` | only for HPC | `hpc_config` |
| 16 | Budget caps | `.pi/config/budgets.json` | yes (defaults) | `make -C .pi env-check` |
| 17 | Kaggle token (smart-home temperature) | `~/.kaggle/kaggle.json` (mode 600) or `KAGGLE_USERNAME`/`KAGGLE_KEY` | only for the Kaggle dataset | `python fixed_case_studies/smart_home/01_fetch_data.py` |
| 18 | Local data | `fixed_case_studies/*/data/` (gitignored) + `data_sha256.json` sidecars | yes | `01_fetch_data.py`; `.pi/skills/data-provenance/scripts/check_sidecars.py` |
| 19 | Toolchain | `uv`, Python per `uv.lock`, Node 24.x, `make` | yes | `make -C .pi verify-env` |

---

## 1. pi and the model provider

The loop runs `pi` as the conductor and spawns `pi` children. It does not
contain a model client of its own.

### 1.1 Where the credential goes

The pinned provider is `opencode-go` (`provider/model` =
`opencode-go/deepseek-v4.1-flash`, thinking `max`). Pick one authentication
route:

- **Interactive (recommended):** run `pi`, then `/login opencode-go` and follow
  the prompt. The credential is stored in `~/.pi/agent/auth.json` (outside this
  repository; never copy it into the repo).
- **Headless / cluster:** export the provider's API-key environment variable in
  the shell that starts the loop and in the SLURM job environment. Children
  inherit the parent environment (`dispatch/lib.ts` copies `process.env`).

Verify the credential before starting the loop:

```bash
pi --print "reply with the single word: ready"     # must print an answer
pi --version                                       # 1.3.x or newer
```

Do not change the model pin to make authentication work. `model.json`,
`settings.json` and `agents.json` must agree; the guard blocks a session whose
active model or thinking level differs from the pin. To change the pin
deliberately, edit all three files and re-run `make -C .pi verify-env`.

If `pi` is not on the children's `PATH`, set `PI_LOOP_COMMAND` to its absolute
path in the environment that launches the conductor.

### 1.2 What is already set (do not fill)

| File | Field | Current value |
|---|---|---|
| `.pi/config/model.json` | `provider` / `model` / `thinking` / `contextWindow` / `enforce` | `opencode-go` / `deepseek-v4.1-flash` / `max` / `1000000` / `true` |
| `.pi/settings.json` | `defaultProvider`, `defaultModel`, `enabledModels`, `defaultThinkingLevel` | must match `model.json` |
| `.pi/config/agents.json` | `pinnedModel`, `pinnedThinking`, every `agents[].model` | must match `model.json` |

---

## 2. Git: the branch the loop merges into

The loop creates `work/T-###-<slug>` branches and merges them into **the branch
that was checked out when the run booted**. That is what
`.pi/config/project.json` → `"baseBranch": "@current"` means. BOOT resolves the
name once and pins it in `.pi/state/LOOP_STATE.json` → `base_branch`, so a
branch switch mid-run cannot retarget the merge.

To use a fixed branch instead, set a literal name in `project.json`:

```json
{ "baseBranch": "update_pymc" }
```

Rules that go with this:

- **Check out the intended target before `/loop:start`.** On this checkout the
  current branch is `vangja_loop_20261018`; the loop will merge there, not into
  `update_pymc`.
- **The checkout must be clean** at BOOT (`git status --porcelain` with no
  modified tracked files). The conductor refuses a dirty checkout rather than
  stashing or discarding your work.
- **The conductor never pushes, rebases, or force-pushes.** GitHub setup is not
  required. If you want the merged branch on a remote, push it yourself.
- **Git identity is required** because children commit in worktrees:

```bash
git config --global user.name  "Your Name"
git config --global user.email "you@example.com"
```

- Detached HEAD is refused when `baseBranch` is `@current`: check out a branch
  first.

Verify:

```bash
git branch --show-current
git status --porcelain          # empty output
pi --print "/loop:verify-env"   # last lines include: human merge gate: … base=<branch>
```

---

## 3. HPC (only for the full mode)

### 3.1 Create the runtime file safely

`.pi/config/hpc.json` is gitignored and matches the guard's secret-path
patterns, so the agent's file tools refuse to create it. Create it from a shell:

```bash
cp .pi/config/hpc.example.json .pi/config/hpc.json
chmod 600 .pi/config/hpc.json
```

`host: ""` disables the HPC path. The moment `host` is non-empty, the loader
requires `user`, `remoteRepoDir`, `remoteScratchDir` and `hpcKey`, and the SSH
key must be readable.

### 3.2 Field-by-field

All paths here are **on the cluster** unless the field says otherwise.

| Field | Put this in | Example | Notes |
|---|---|---|---|
| `host` | login-node hostname or IP | `hpc.example.edu` | Empty = HPC disabled. This is the "HPC IP address". |
| `user` | SSH username | `jdoe` | |
| `port` | SSH port | `22` | |
| `remoteRepoDir` | absolute path to a **git clone of this repo** on the cluster | `/home/jdoe/vangja` | Run worktrees live under `<remoteRepoDir>/runs/<experiment>`. Must have an `origin` the remote host can `git fetch` (see §3.4). |
| `remoteScratchDir` | absolute path on fast/high-quota storage | `/scratch/jdoe/vangja` | Checkpoints, logs and the rendered `job.slurm` live under `<remoteScratchDir>/<experiment>`. |
| `hpcKey` | **local** path to the private key for the login node | `~/.ssh/id_ed25519_hpc` | `~` is expanded locally. Must be `chmod 600` and readable by the loop process. |
| `gitKey` | **local** path to the private key used for remote `git fetch` | `~/.ssh/vangja_github` | Loaded into a private `ssh-agent`; your own agent is never used. |
| `forwardGitAgent` | `true` to forward that private agent | `true` | `false` = rsync-only; remote `git fetch` then needs its own credentials. |
| `container` | absolute remote path to the `.sif` | `/scratch/jdoe/vangja.sif` | Empty = run the command directly. The submit gate hashes it remotely and compares with the record's `container_sha`. |
| `partition` | SLURM partition | `openlab-queue` | Empty omits `--partition` (cluster default). |
| `cpus` / `memGb` / `gpus` | defaults, overridable per experiment record | `8` / `32` / `0` | `gpus > 0` renders `--gres=gpu:<n>`. |
| `wallClockHours` | maximum the conductor will accept | `24` | The job template requests `--time=24:00:00`; longer work must be split into `stages`. |
| `pollSeconds` / `maxPolls` | collector cadence and poll budget | `60` / `40` | |
| `maxAttemptsPerExperiment` | requeue cap | `6` | |
| `maxParallelJobs` | **keep `1`** | `1` | `> 1` is refused without a recorded amendment (hard single-slot rule). |

### 3.3 SSH keys

The HPC client runs non-interactively (`BatchMode=yes`,
`IdentitiesOnly=yes`), so password prompts and agent guessing will not work.

```bash
# 1. Choose (or create) a dedicated key for the cluster.
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_hpc -C "vangja-loop-hpc"
chmod 600 ~/.ssh/id_ed25519_hpc
ssh-copy-id -i ~/.ssh/id_ed25519_hpc.pub jdoe@hpc.example.edu

# 2. Prove the same options the loop uses.
ssh -o IdentitiesOnly=yes -o BatchMode=yes -i ~/.ssh/id_ed25519_hpc \
    -p 22 jdoe@hpc.example.edu \
    'hostname; which sbatch; which uv; command -v apptainer || command -v singularity'
```

Every command in that check must print a path (except `apptainer`/`singularity`,
which is optional when `container` is empty). `uv` must be available on the
**compute node** because the job template runs `uv sync --frozen --no-dev`
before the container command.

The second key (`gitKey`) is only needed when `forwardGitAgent: true`:

```bash
chmod 600 ~/.ssh/vangja_github
eval "$(ssh-agent -s)" && ssh-add ~/.ssh/vangja_github && ssh-add -l   # local sanity check only
```

The loop starts its own agent and adds only `gitKey`; it never touches your
shell's agent.

### 3.4 The remote repository and commit reachability

This is the step most often missed. `hpc_submit` runs a remote preflight that
does, inside `remoteRepoDir`:

```text
git fetch --quiet origin
git worktree add --force --detach <runDir> <commit_sha>
git checkout --quiet --detach <commit_sha>
git status --porcelain --untracked-files=no     # must be empty
sha256sum uv.lock                               # must equal the record's uv_lock_sha
```

So, before submitting:

1. `remoteRepoDir` exists on the cluster and is a clone of the same repository
   whose `origin` can be fetched (GitHub over SSH, an institutional mirror, or a
   path reachable from the login node).
2. **The experiment's `commit_sha` is reachable from that remote.** The
   conductor never pushes, so this is a human step: push the branch or commit
   that the experiment record names (`git push origin <branch>`) before
   `hpc_submit`. If you deliberately do not want to push work branches, push the
   single commit to a dedicated ref, e.g.
   `git push origin <sha>:refs/heads/hpc/<experiment-id>`.
3. The remote checkout's `uv.lock` is identical to the local one (the gate
   compares SHA-256).
4. `forwardGitAgent: true` + a readable `gitKey` that can read `origin`, or a
   credential already configured on the cluster (deploy key, `insteadOf` rule).

Preflight check you can run yourself (replace the placeholders):

```bash
ssh -i ~/.ssh/id_ed25519_hpc jdoe@hpc.example.edu \
  'cd /home/jdoe/vangja && git fetch --quiet origin && git cat-file -e <commit_sha>^{commit} && echo COMMIT-REACHABLE'
```

### 3.5 The container (optional but recommended)

Build the image locally (root needed), then copy it to the path you set in
`container`:

```bash
sudo singularity build case_studies/vangja.sif case_studies/singularity.def
scp case_studies/vangja.sif jdoe@hpc.example.edu:/scratch/jdoe/vangja.sif
ssh -i ~/.ssh/id_ed25519_hpc jdoe@hpc.example.edu 'sha256sum /scratch/jdoe/vangja.sif'
```

Put that SHA-256 in the experiment record's `container_sha`; the submit gate
recomputes it remotely and refuses a mismatch. The rendered job wraps the resume
command as `apptainer exec <container> <resumeCommand>`; the `uv sync` step still
runs on the host, outside the container.

### 3.6 SLURM specifics

- The job template is versioned at
  `.pi/skills/hpc-slurm-and-checkpointing/templates/job.slurm` and is rendered
  per attempt. Do not hand-edit a rendered copy on the cluster.
- The template requests `--time=24:00:00`, `--signal=B:USR1@300`, `--cpus-per-task`,
  `--mem`, `--partition`, optional `--gres`, and the log paths. Checkpoints must
  flush on `USR1` and exit `99`.
- **SLURM account:** the config schema has no `account` field. If your cluster
  requires `#SBATCH --account=…`, add it to the template deliberately (and run
  the HPC tests) or set a site default account.
- The submit gate caps wall clock at `min(config.wallClockHours, 24)` per
  attempt, unless the record is split with `"stages": N` and a stage index is
  requested.

### 3.7 What an experiment record must contain before `hpc_submit`

The agent creates this with `experiment_queue` (capability: `phd_candidate`,
`supervisor`, or `conductor`). Prepare these values for it:

```jsonc
{
  "task_id": "T-001",
  "branch": "work/T-001-slug",
  "commit_sha": "<40-hex, reachable from remoteRepoDir's origin>",
  "data_hash": "sha256:<hash of the frozen data sidecar>",
  "command": "uv run python fixed_case_studies/stocks/run_one.py --config …",
  "resources": { "kind": "hpc", "cpus": 8, "memGb": 32, "gpus": 0, "wallClockHours": 24 },
  "checkpoints": {
    "dir": "/scratch/jdoe/vangja/E-0007/checkpoints",   // must match remoteScratchDir/E-0007/checkpoints
    "progressFile": "progress.json",
    "resumeCommand": "uv run python … --resume \"$CKPT\"", // $CKPT is exported by the job template
    "flushOnSignal": true
  },
  "expectedOutputs": ["metrics.csv"],
  "seeds": [42],
  "idempotencyKey": "<stable key>",
  "resumeSmokeTest": "<command + observed evidence that --resume skips completed units>",
  "uv_lock_sha": "<sha256 of local uv.lock>",
  "container_sha": "<sha256 on the cluster, or \"-\" when container is empty>",
  "stages": 1
}
```

`hpc_submit` refuses while another experiment is active, while the record is not
`queued`, when the checkpoint block is incomplete, when `resumeSmokeTest`
evidence is missing, when `uv.lock` drifted, when the container hash differs,
when the remote checkout is not a clean detached checkout at `commit_sha`, or
when the wall clock is too long. Every failure names the check.

Verify the HPC path:

```bash
pi --print "/loop:verify-env"          # hpc gate line: not-configured or enabled
# In a pi session: ask for the `hpc_config` tool, which reports the resolved
# host (redacted), keys, container and limits without connecting.
```

---

## 4. Datasets

Data is fetched once, hashed, and then read from the local cache.

```bash
uv sync --extra test --extra datasets

# Stocks: yfinance + the frozen Wikipedia revision; writes per-origin
# universes, availability.csv, cached OHLCV CSVs and data_sha256.json.
python fixed_case_studies/stocks/01_fetch_data.py

# Smart home: Kaggle temperature data + the household series.
python fixed_case_studies/smart_home/01_fetch_data.py
```

Kaggle needs a token for `load_kaggle_temperature` / the smart-home fetch:
put `kaggle.json` at `~/.kaggle/kaggle.json` with mode `600`, or export
`KAGGLE_USERNAME` and `KAGGLE_KEY`. Never put the token in the repository.

Where the data goes (all gitignored; do not commit raw dumps):

| Study | Path |
|---|---|
| Stocks tickers / constituents / universes | `fixed_case_studies/stocks/data/` |
| Smart home | `fixed_case_studies/smart_home/data/` |
| Generic loader cache | `data/` |

Verify the hashes before trusting any run:

```bash
python .pi/skills/data-provenance/scripts/check_sidecars.py \
  --root fixed_case_studies/stocks/data/tickers --json
```

---

## 5. Secrets: where they must **not** go

- Keys stay in `~/.ssh` (mode 600), tokens in `~/.pi/agent/auth.json`,
  `~/.kaggle/kaggle.json`, or environment variables.
- The guard blocks writes to `.env*`, `hpc.json`, `auth.json`, `credentials*`,
  `secrets.*`, `*.pem`, `*.p12`, `*.pfx`, `id_rsa*`, `id_ed25519*`,
  `id_ecdsa*`.
- Command output and file writes are scanned for PEM blocks, provider tokens,
  JWTs and bearer tokens; matches are blocked or redacted.
- `.pi/config/hpc.json` and `.pi/state/**` are gitignored. Commit manifests and
  hashes, not credentials or data dumps.

---

## 6. Local toolchain

| Tool | Version | Why |
|---|---|---|
| `uv` | current | `uv sync --frozen`, `uv run pytest`, the HPC job template |
| Python | per `uv.lock` (3.13 in the container) | package and experiments |
| Node.js | 24.x | `.pi` tests use `node --test` with built-in TypeScript type stripping |
| `git` | 2.30+ | worktrees, `merge --no-ff`, tags |
| `ssh`, `scp`, `rsync`, `ssh-agent`, `ssh-add` | any recent OpenSSH | HPC mode |
| `singularity` / `apptainer` | optional, local build only | container image |

Prepare and verify:

```bash
uv sync --extra test --extra datasets      # uv.lock is authoritative
uv run pytest                              # Python suite
make -C .pi verify-env                     # node suite + env checks + orchestrator + model pin
```

---

## 7. Runtime directories, budgets and stop controls

| Path | Contents | Tracked? |
|---|---|---|
| `.pi/state/` | `LOOP_STATE.json`, ledgers, worktrees, sessions, locks | no (gitignored) |
| `.pi/artifacts/` | collected experiment outputs + manifests | no (manifests are committed per protocol) |
| `.pi/state/UNDERSTAND_LOOP.json` | `/understand_loop` tutorial progress | no |
| `.pi/config/hpc.json` | your HPC connection details | no (gitignored) |

Budgets live in `.pi/config/budgets.json` (nine caps, including a hard
single HPC job slot and a cost cap). Exceeding a cap pauses the loop with a
reason. The kill switches are `/loop:pause "<reason>"` (recorded, resumable) and
the presence of `.pi/state/STOP` (visible in the status line as `STOP`).

---

## 8. Useful local checks before you start

```bash
git branch --show-current                  # the branch the loop will merge into
git status --porcelain                     # must be empty
pi --version && node --version && uv --version
pi --print "reply with the single word: ready"
make -C .pi verify-env
pi --print "/loop:verify-env"              # 4/4 orchestrator checks
```

Expected in `/loop:verify-env`:

```text
PASS  human merge gate: requireHumanMerge=true base=<your current branch>
PASS  hpc gate: not-configured: …            # without hpc.json
# or: PASS  hpc gate: enabled: host <redacted> …   # with a valid hpc.json
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `LOOP UNINITIALIZED` footer | no run has booted | expected; `/loop:start` writes `LOOP_STATE.json` |
| BOOT refuses: "checkout has uncommitted changes" | dirty working tree | commit or `git stash push -m wip`, then retry |
| `baseBranch "@current" … detached HEAD` | no current branch | `git switch <branch>` or set a literal `baseBranch` |
| merge lands on the wrong branch | the run booted on another branch, or `baseBranch` is literal | check `LOOP_STATE.base_branch`; restart the run from the intended branch |
| model-pin drift / input dropped | provider, model or thinking changed in the session | restore the pin in `model.json` + `settings.json` + `agents.json`; reload pi |
| `hpcKey … is not readable` | wrong path or permissions | fix `hpcKey`, `chmod 600`, or set `host: ""` |
| submit gate: `cannot hash <container> remotely` | bad `container` path | correct the remote path; `sha256sum` it manually |
| submit gate: remote HEAD ≠ `commit_sha` / fetch fails | commit not pushed to `origin`, or no git credential on the cluster | push the commit to the remote; fix `forwardGitAgent`/`gitKey` |
| submit gate: `uv-lock` mismatch | local `uv.lock` changed after the record was written | rebuild the record with the new `uv_lock_sha` |
| submit gate: `record has no container_sha` | record predates the container | add the remote `sha256sum` to the record |
| `checkpoint metadata is missing …` | incomplete checkpoint block | supply `dir`, `progressFile`, `resumeCommand`, `flushOnSignal: true` |
| `local resume smoke-test evidence is missing` | no proof `--resume` works | add `resumeSmokeTest` (command + observed skip) |
| Kaggle 401/403 | missing/incorrect token | `~/.kaggle/kaggle.json` mode 600, or `KAGGLE_USERNAME`/`KAGGLE_KEY` |
| `verify-env` node failures | Node older than 24 or deps missing | install Node 24.x; re-run `make -C .pi verify-env` |

---

## 10. Final readiness checklist

Local mode:

- [ ] `pi --print "reply with the single word: ready"` answers.
- [ ] `git branch --show-current` is the branch you want the loop to merge into.
- [ ] `git status --porcelain` is empty.
- [ ] `git config --global user.name` and `user.email` are set.
- [ ] `make -C .pi verify-env` is green.
- [ ] `pi --print "/loop:verify-env"` prints `4/4 orchestrator checks passed`
      and `base=<intended branch>`.
- [ ] `uv run pytest` is green.
- [ ] Data is fetched and `check_sidecars.py` passes.
- [ ] A first `/loop:start` reaches `SELECT_TASK`; `/loop:status` shows the
      budget and the next action.

Full mode, additionally:

- [ ] `.pi/config/hpc.json` exists (mode 600) and `hpc_config` reports it enabled.
- [ ] The SSH preflight in §3.3 prints a hostname and paths.
- [ ] `remoteRepoDir` is a clone with `origin`; the experiment commit is
      reachable there (§3.4).
- [ ] The container (if used) exists at `container` and its SHA-256 is recorded.
- [ ] `partition` exists (`sinfo -p <partition>`), and any required
      `--account` is handled in the job template.
- [ ] An experiment record passes `hpc_submit`'s gate; the job reaches
      `running`; the collector can read `squeue`/`sacct`.
- [ ] `hpc_fetch` returns a manifest whose SHA-256 matches every file.

If every box is ticked, the loop has no external unknown left: the remaining
gates are the intentional human ones (merge, confirmation, audit sign-off,
submission).
