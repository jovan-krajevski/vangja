# Run directory template

Use one directory per run stage. Paths below are relative to the run root
(e.g. `fixed_case_studies/stocks/results/<run>/`).

```text
<run>/
├── manifest.json                 # the run-level manifest (this skill)
├── manifest_<tag>.json           # per-cell artifact manifest (record_artifact)
├── units_<tag>.csv               # per target x origin metrics (units)
├── forecasts_<tag>.csv           # aligned observations + predictions
├── failures.csv                  # append-only; one row per failed cell
├── source_cache/                 # source__<origin>__<method>__seed<seed>.pkl/.json
├── data_sha256.json              # hash map of the data dir used (if not at data root)
└── notes.md                      # free text; must not contain unverifiable numbers
```

`<tag>` = `{config}__{origin}__seed{seed}` with `/` replaced by `_`; fast
iteration runs append `__h{horizon}` (`fixed_case_studies/stocks/runner.py:306`).

## Resume checklist

1. `manifest_<tag>.json` **and** `units_<tag>.csv` both exist → cell is done;
   do not refit.
2. A source cache hit is keyed by origin × method × seed
   (`runner.py:452`); never reuse a cache built with a different seed.
3. On restart, re-run `make_manifest.py --force` only for the run-level
   manifest; per-cell manifests are immutable.

## What must never appear here

- Raw data dumps (`.gitignore` excludes `fixed_case_studies/**/data/`).
- Notebooks or exploratory figures that no table cites.
- A number in `notes.md` without a matching artifact ID.
