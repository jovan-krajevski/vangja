# Dependency pins — why each one exists

Source of truth: `pyproject.toml` (with its inline comments) and `uv.lock`.
`AGENTS.md` § Vangja/PyMC gotchas lists the three load-bearing pins. Never
relax a pin to make a local install succeed; find the interaction that breaks.

## Runtime pins

| Requirement | Pin | Why |
|---|---|---|
| Python | `>=3.12` | uses modern typing syntax; the container builds 3.13 |
| PyMC | `pymc[nutpie]~=6.2` | PyMC 6 line; nutpie is the NUTS backend used by the fixed case studies. Every reported NUTS fit must state this version (`fit_info` records it) |
| PyMC extras | `pymc-extras~=0.14.0` | `pmx.find_MAP` (JAX backend) is the default `mapx` method; the transfer MVN prior uses its DataTree helpers |
| BlackJAX | `blackjax~=1.6.2` | alternative NUTS backend; keep aligned with pymc-extras |
| scikit-learn | `scikit-learn~=1.9.0` | preprocessing/metrics used by the package and case studies |
| numba | `>=0.66.0` | transitive via preliz. `0.63.0b1` crashes on import with `numpy>=2.0` (`np.trapz` removed); `<=0.66` crashes with `numpy>=2.5` (`np.row_stack` removed); pytensor caps numba at `<=0.66` |
| numpy | `>=2.0,<2.5` | the other half of the numba triangle. The `windows`/`numba` combination is the reason the upper bound exists; do not widen it without re-running the numba dispatch import |

## Extras

| Extra | Contents | Used by |
|---|---|---|
| `datasets` | `pyreadr`, `yfinance`, `lxml`, `kagglehub`, `requests` | `01_fetch_data.py`, loaders that download (never at run time) |
| `test` | `pytest`, `pytest-cov` | `uv run pytest` |
| `docs` | `sphinx`, `sphinx-book-theme`, `myst-parser`, `nbsphinx`, `pandoc`, `ipykernel` | `cd docs && make html` |
| `dev` | `vangja[test,docs]`, `ruff` | local formatting/lint |
| `reproducibility` | `vangja[datasets]`, `statsmodels`, `matplotlib`, `seaborn` | classical baselines and report scripts |

`uv sync --frozen --extra test --extra datasets` is the experiment environment;
add `--extra reproducibility` for baseline scripts.

## Environment/API interactions to keep in mind

- **ArviZ 1.x is DataTree-based.** Fixtures are
  `az.InferenceData(xr.Dataset(), children={group: xr.DataTree(ds)})`; there is
  no `az.waic`, and `az.compare` has no `ic`.
- **pytensor 3** removed `pt.constant`/`pt.as_tensor_variable`; use
  `pt.tensor.as_tensor`.
- **`fit_info`** records `{vangja,pymc,arviz,numpy,pandas,pytensor}_version`;
  use `freeze_report.py` for `jax`, `numba`, `pymc-extras`, `blackjax` and the
  lock/container hashes that the manifest needs.
- **Container** (`case_studies/singularity.def`): base
  `continuumio/miniconda3:v25.11.1` (pinned by tag), conda env `pymc_env`
  (Python 3.13) with `pymc>=5.27.1` and `pip install '/vangja[reproducibility]'`.
  The lower bound is satisfied by PyMC 6, but the image is built from a
  *timestamp* of the repo, so its `.sif` SHA-256 is part of the provenance —
  record it, and prefer the lockfile for exact versions.
- **CI** has docs and publish workflows only. There is no test workflow:
  `uv run pytest` locally (plus `make -C .pi verify-env` for the loop) is the
  gate; do not assume CI catches a broken test.

## Checks

```bash
uv lock --check                       # pyproject <-> uv.lock
uv run python -c "from pytensor.link.numba import dispatch"   # numba triangle
uv run python -c "import arviz, pymc, vangja; print(arviz.__version__, pymc.__version__)"
python .pi/skills/python-packaging-and-environments/scripts/freeze_report.py --json
```

`freeze_report.py` reads installed metadata with `importlib.metadata`; it never
imports the packages, so it also works from a bare interpreter (values are
`null` there, which is itself a useful signal that you are in the wrong env).
