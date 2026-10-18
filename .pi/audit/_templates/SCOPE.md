# Audit scope — cycle {{cycle}}

> Filled by the conductor at audit start, before any finder runs. Every path is
> relative to the repository root. The snapshot is immutable: findings cite it,
> never a moving branch.

## Snapshot

- Audited commit: `{{sha}}` on `{{branch}}`
- Audit tag: `audit-cycle/{{cycle}}` (annotated)
- Paper commit: `{{paper_sha}}`; built PDF SHA-256: `{{pdf_sha256}}`
- Snapshot date (UTC): `{{started_at}}`
- Base: `{{base_sha}}`

## Frozen protocol and environment

| Item | Path | SHA-256 |
|---|---|---|
| Confirmation origins | `fixed_case_studies/protocol/CONFIRMATION_ORIGINS.csv` | `{{origins_sha256}}` |
| Development origins | `fixed_case_studies/protocol/DEV_ORIGINS.csv` | `{{dev_origins_sha256}}` |
| Frozen configs | `{{config_paths}}` | `{{config_hashes}}` |
| Lockfile | `uv.lock` | `{{uv_lock_sha256}}` |
| Container | `{{container_path}}` | `{{container_sha256}}` |
| Python / PyMC / NumPy / Numba | recorded by `run_gates` `env-pinned` | see `gates.json` |

## Artifact index

Every artifact a paper claim depends on, with its manifest and manifest SHA
(the registry itself is `paper/CLAIMS.json`):

| Artifact | Manifest | Manifest SHA-256 | Commit SHA |
|---|---|---|---|
| `{{artifact_id}}` | `{{manifest_path}}` | `{{manifest_sha256}}` | `{{artifact_commit}}` |

## Loop ledgers (read-only evidence for R and E)

- `.pi/state/EVENTS.jsonl`, `EXPERIMENTS.jsonl`, `REVIEWS.jsonl`, `PRS.jsonl`
- Gate runs: `.pi/state/gate-runs/`
- Experiment manifests: `.pi/artifacts/*/manifest.json`

## What is explicitly out of scope

- `Vangja-project/**` (historical audit archive; calibration scorer only)
- anything merged after the audit tag
- data that cannot be identified by a recorded hash (reported as unavailable, never guessed)

## Reproduction of the audit itself

```bash
run_gates(cycle={{cycle}}, snapshot="{{sha}}")
# then the finders in parallel, the verifier, the synthesizer, the adjudicator
```
