# Run manifest schema — `vangja-run-manifest-1.0`

One manifest per run directory, written atomically (temp file + rename) as
`manifest.json`. The study runner writes per-artifact manifests with
`artifact_schema: vangja-fixed-case-studies-1.0`
(`fixed_case_studies/common.py:507`); this general schema is a superset used by
`scripts/make_manifest.py` and validated by `scripts/check_manifest.py`.

## Required fields

| Field | Type | Rule |
|---|---|---|
| `manifest_schema` | str | exactly `vangja-run-manifest-1.0` |
| `run_id` | str | UTC `YYYYMMDDTHHMMSSZ` + 8 hex chars; unique per run |
| `created_at` | str | ISO-8601 UTC |
| `started_at` / `ended_at` | str | ISO-8601; `ended_at >= started_at` |
| `status` | str | `ok` \| `failed` \| `running` |
| `exit_code` | int | 0 iff `status == ok` |
| `error` | str | required non-empty when `status == failed` |
| `command` | list[str] | non-empty; the logical command line (pass it last: `--command` is an `argparse.REMAINDER`) |
| `argv` | list[str] | the exact `sys.argv` of the manifest step |
| `git.commit` | str | 40-hex (or 7–40 hex) SHA; `unknown` only with `--allow-unknown-commit` |
| `git.branch` | str | branch name or `detached` |
| `git.dirty` | bool | `git status --porcelain` non-empty |
| `environment.python` | str | e.g. `3.14.4` |
| `environment.platform` | str | `sys.platform` |
| `environment.packages` | object | name → version (may be `null`) |
| `environment.uv_lock_sha256` | str \| null | SHA-256 of `uv.lock`, `null` if absent |
| `seeds.base_seed` | int | frozen policy default 42 |
| `seeds.values` | object | label → int (per-origin, per-cell, subset, …) |
| `data` | object | one entry per hashed directory: `{"<key>": {"path": <repo-relative dir>, "files": {<relpath>: <sha256>}}}` |
| `configs` | object | path (repo-relative when inside the repo, otherwise absolute) → 64-hex SHA-256 |

## Optional fields

| Field | Purpose |
|---|---|
| `study`, `stage`, `origin`, `config`, `method`, `target_accept` | case-study identity (mirrors `record_artifact`) |
| `fit_info` | the model's `fit_info` dict (diagnostics, versions) |
| `aggregate` | median/mean/IQR/`prop_below_1`/`n_units`/`n_excluded` |
| `extra.availability` | requested/available/missing series — **never drop silently** |
| `artifacts` | list of produced files with SHA-256 |
| `checkpoint` | `{cell, completed: bool, resumed: bool, source_cache: str}` |
| `notes` | free text; no unverifiable claims |

## Validation rules (`check_manifest.py`)

1. Schema string, required types, `exit_code`/`status` consistency.
2. Every hash is 64 lowercase hex.
3. If `--root DIR` is given, every `data.files` and `configs` entry is
   recomputed and must match (missing file = failure; use
   `--allow-missing-data` only for a manifest transferred between machines).
4. `git.commit` must be hex unless `--allow-unknown-commit`.
5. `seeds.values` keys are non-empty strings, values are ints ≥ 0.
6. `status == failed` requires `error`.

Exit code is 0 only when every check passes. `--json` prints
`{"ok": bool, "errors": [...], "warnings": [...]}`.

## Provenance rules of thumb

- One manifest per *cell attempt*; retries append `attempt` in the file name,
  they never overwrite the first manifest.
- If a run produces tables, the table generator records the manifest paths it
  consumed.
- A number is reportable only when its manifest validates **and** the commit is
  an ancestor of the audited tag.
