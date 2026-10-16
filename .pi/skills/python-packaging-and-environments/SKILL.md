---
name: python-packaging-and-environments
description: Change or verify the Vangja Python environment: dependencies, lockfile, extras, containers, or CI. Use when adding or relaxing a pin, running uv sync/lock, building the Singularity image, preparing an experiment environment fingerprint, or when an import fails on a cluster. Contains the pin rationale and the freeze_report fingerprint block used in manifests.
---

# Python packaging and environments

`uv.lock` is authoritative; `pyproject.toml` declares intent. Read
`references/pins.md` before touching either. An experiment run without a
recorded fingerprint is not reproducible, however good the code is.

## Hard rules

1. **`uv sync --frozen --extra test --extra datasets`** for experiments and
   tests. `--frozen` fails instead of re-resolving; that is the point.
2. **Never relax a pin without checking why it exists.** The pymc/numba/numpy
   triangle is load-bearing (`references/pins.md`); the failure mode is an
   import-time crash deep in `pytensor.link.numba.dispatch`, not a test
   failure.
3. **No global installs in a run.** Containers and lockfiles, not
   `pip install` at job time; `uv sync --frozen --no-dev` inside the job.
4. **Record the fingerprint.** Every manifest should carry the
   `freeze_report.py` block: python, pymc, pytensor, arviz, numpy, numba,
   jax (plus pandas/scikit-learn/pymc-extras/blackjax when present),
   `uv.lock` SHA-256, container SHA-256.
5. **Docs stay buildable.** `cd docs && make html` uses the `docs` extra; a
   dependency change must keep the doc build green locally before it lands.
6. **No secrets in packaging metadata** (credential files, tokens in URLs, `.pypirc`
   in the sdist); the guards scan for secret material.

## Commands

```bash
uv sync --frozen --extra test --extra datasets
uv lock --check                 # lockfile matches pyproject
uv run pytest                   # full suite
uv run black . && uv run ruff check .

# Environment fingerprint for a manifest
python .pi/skills/python-packaging-and-environments/scripts/freeze_report.py \
  --container case_studies/vangja.sif --out scratch/run-env.json

# Container sanity (build locally, then verify the %test)
singularity build case_studies/vangja.sif case_studies/singularity.def
sha256sum case_studies/vangja.sif
```

## Changing a dependency (checklist)

- [ ] The failing import/test is reproduced first; the change is the *fix*,
      not a guess.
- [ ] `pyproject.toml` comment explains the constraint if it is non-obvious.
- [ ] `uv lock` and commit the lockfile diff; `uv lock --check` passes.
- [ ] `uv sync --frozen` in a clean environment installs and `uv run pytest`
      is green locally.
- [ ] `pytensor.link.numba.dispatch` imports (pymc/numba/numpy interaction).
- [ ] ArviZ usage still matches 1.x (DataTree; no `az.waic`, `az.compare` has
      no `ic`).
- [ ] Container build still succeeds and `%test` prints the three import
      lines; record the new `.sif` SHA-256.
- [ ] The PR body names the old and new versions and the evidence command.

## Failure modes

- `uv sync` without `--frozen` silently upgrades a transitive package; the
  run is then unreproducible even though tests pass.
- `numpy>=2.5` → `np.row_stack` removal crashes numba ≤ 0.66; `numba` 0.63.0b1
  → `np.trapz` crash on numpy ≥ 2.0. Both are pre-existing, documented pins.
- Old container (base tag pinned, but `pymc>=5.27.1` in `singularity.def`) vs
  the `pymc[nutpie]~=6.2` pin: the container satisfies both only because
  PyMC 6 ≥ 5.27.1 — verify the installed version with `%test`-style imports
  and record it, and do not assume the container matches the lockfile.
- `docs` build failing on a missing pandoc → install the `docs` extra, not
  pandoc-by-hand on the CI runner.
- `freeze_report.py` prints `null` for a package → it is not installed in the
  *current* interpreter; run it with `uv run` inside the experiment env.
