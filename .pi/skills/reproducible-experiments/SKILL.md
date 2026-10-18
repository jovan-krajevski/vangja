---
name: reproducible-experiments
description: Produce or verify a reproducible Vangja experiment run. Use when fitting a model, running a case-study cell, collecting results, writing or checking a run manifest, resuming checkpointed work, or anytime a number will be reported. Covers the frozen seed policy, uv-frozen environments, data hashes, the manifest schema, and checkpoints.
---

# Reproducible experiments

Every reported number must be recomputable from committed artifacts + data
hashes. This skill is the contract for a single run; the study-wide rules are in
`experiment-protocol-design`.

House rules: `AGENTS.md` § Non-negotiables and § Working style. Read
`references/manifest-schema.md` before writing a manifest. The canonical
producer is `fixed_case_studies/common.py:record_artifact`
(`manifest_schema: vangja-fixed-case-studies-1.0`) and the reference CLI here
is `scripts/make_manifest.py`.

## When to use

- Any `model.fit(...)` that feeds a table, figure, or claim.
- Any HPC submission, checkpoint, resume, or collection step.
- Before reporting a number: the run manifest must exist and validate.

## Run contract (do all seven)

1. **Environment is frozen.** `uv sync --frozen --extra test --extra datasets`;
   `uv.lock` is authoritative. Never relax the pins
   (`pymc[nutpie]~=6.2`, `numpy>=2.0,<2.5`, `numba>=0.66`).
2. **Seed every fit.** `random_seed` always; NUTS also `target_accept`.
   Never pass an explicit `step` together with `nuts_sampler`.
3. **Hash the data.** `record_sidecars` writes `data_sha256.json` next to the
   data dir (`fixed_case_studies/common.py:121`); carry the hash map into the
   manifest. Never re-download at run time.
4. **Record provenance.** Commit SHA (full), branch, dirty flag, package
   versions, `fit_info`, command + argv, start/end, exit state.
5. **Name deterministically.** `{config}__{origin}__seed{seed}` with `/` → `_`;
   horizon overrides append `__h{horizon}` (`fixed_case_studies/stocks/runner.py:306`).
6. **Checkpoint.** A completed cell is `manifest_<tag>.json` + `units_<tag>.csv`;
   `run_cell` returns the CSV instead of refitting (`runner.py:307`). Source
   fits cache as `source__{origin}__{method}__seed{seed}.pkl` (`runner.py:452`).
7. **Never silently drop.** Failures go to `failures.csv` via `record_failure`
   (`common.py:558`); availability goes into the manifest `extra.availability`.

## Seed policy (frozen)

| What | Value / rule | Source |
|---|---|---|
| Global base | `BASE_SEED = 42` | `fixed_case_studies/common.py:48` |
| Per cell | `seed_for(origin_index, config_index) = base + origin_index*100 + config_index` | `fixed_case_studies/stocks/runner.py:405`, `fixed_case_studies/smart_home/runner.py:249` |
| Context fit | `source_seed_for(origin) = base + 1000*(year-2013)*12 + (month-1)` — **one seed per origin**, so all configs transfer the identical source posterior | `runner.py:397` |
| Finalists | `FINALIST_SEEDS = (42, 20240816, 7)`; applied to `main` and `no_transfer` (other configs get `seed_for`) | `fixed_case_studies/common.py:49`, `fixed_case_studies/smart_home/03_run_main.py:60-64` |
| Subsets | subset seed recorded in `availability.subset_seed` | `runner.py:101` |

Seeds are recorded in every artifact. A seed chosen *after* seeing results is
seed shopping and is a finding (`scientific-integrity-and-anti-gaming`).

## Manifest in one command

```bash
python .pi/skills/reproducible-experiments/scripts/make_manifest.py \
  --out fixed_case_studies/stocks/results/<run> \
  --data fixed_case_studies/stocks/data/tickers \
  --config fixed_case_studies/stocks/config.py \
  --base-seed 42 --seed origin=0 --seed cell=42 \
  --status ok --exit-code 0 \
  --command python fixed_case_studies/stocks/03_run_main.py --origins 2013-01-01
```

`--seed` takes `[LABEL=]INT` (here `origin` is the origin *index* and `cell`
the config index, matching `seed_for`); the origin date belongs in the run
command, not the seed.

Validate before using a number:

```bash
python .pi/skills/reproducible-experiments/scripts/check_manifest.py \
  fixed_case_studies/stocks/results/<run>/manifest.json --root .
```

See `templates/run_dir.md` for the expected layout of a results directory.

## Non-negotiables

- No network at run time; fetching belongs to `01_fetch_data.py`.
- Never commit data dumps or result directories (`.gitignore` covers
  `fixed_case_studies/**/data/` and `results*/`); commit manifests and hashes.
- `commit_sha()` returns `"unknown"` outside git. That is not acceptable for a
  reported run: fix the checkout and re-run the manifest step.
- Retries on HPC must reuse the same run dir; the checkpoint is the resume
  contract (see `hpc-slurm-and-checkpointing` later). One active job.
- PPC coverage, R̂/ESS and fit diagnostics live in `fit_info`; the Bayesian
  thresholds are in `bayesian-workflow/references/thresholds.md`.

## Failure modes seen in this repo

- Fit succeeded but `fit_info` empty → the artifact cannot be checked; fix
  `fit()` call parameters, not the manifest.
- Manifest says `commit: unknown` → run outside a git worktree (HPC tarball);
  forward the git dir or record the archive SHA instead.
- Re-running overwrote a checkpoint with a different seed → per-cell seed must
  come from `seed_for`, never from the clock.
