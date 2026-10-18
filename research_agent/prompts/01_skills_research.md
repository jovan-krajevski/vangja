# Prompt 01 — Build stage S1: research & methodology skills

Prerequisite: S0 complete (`make -C .pi verify-env` green). Read
`research_agent/plan/04_SKILLS.md` (rows 1–6) and the source material each skill
must be built from. Skills live at the **repo root**: `.pi/skills/<name>/`.

Create these six skills, each with `SKILL.md` (frontmatter: `name`,
`description`; ≤120 lines) plus `references/` and `scripts/` as listed:

1. `.pi/skills/reproducible-experiments/`
   Source material: `fixed_case_studies/README.md` §3.11, `common.py`,
   `fixed_case_studies/stocks/runner.py`, `Vangja-project/REVIEW/PROTOCOL.md` §10–11.
   Scripts: `make_manifest.py` (write a run manifest: SHA, branch, seeds,
   versions, data hashes, command, argv, start/end, exit state),
   `check_manifest.py` (validate a manifest and its files).
2. `.pi/skills/experiment-protocol-design/`
   Source: `Vangja-project/REVIEW/{PROTOCOL,FINDINGS,PLAN}.md`,
   `fixed_case_studies/{README,HYPERPARAMETER_SELECTION}.md`,
   `fixed_case_studies/protocol/CONFIRMATION_ORIGINS.csv(.sha256)`.
   Script: `check_protocol_compliance.py` — scans a diff/repo for: config
   selection on test metrics, changes to frozen splits/metrics/exclusions, edits
   to hashed files, confirmation results present without the hash-committed
   origins, context series in target aggregates. Fixture-driven tests.
3. `.pi/skills/bayesian-workflow/`
   Source: `src/vangja/`, `tests/test_regularization.py`,
   `fixed_case_studies/hyperparams/*.py`, `docs/source/notebooks/10_bayesian_workflow.ipynb`,
   `Vangja-project/audit/stage-2/ws-s-stats-simulation/FINDINGS.md`.
   Scripts: `diagnose_idata.py` (R̂/ESS/divergences/tree depth/energy + verdict
   against stated thresholds), `check_convergence.py` (fail a run that does not
   meet them).
4. `.pi/skills/forecasting-evaluation/`
   Source: `src/vangja/utils.py` (`relative_mae`, `persistence_forecast`),
   `fixed_case_studies/common.py` (`two_way_block_bootstrap`, `REL_MAE_EPSILON`),
   `fixed_case_studies/stocks/09_report.py`.
   Scripts: `relative_mae.py` (reference implementation with the ε rule),
   `block_bootstrap.py` (two-way resampling with percentile CIs).
5. `.pi/skills/scientific-integrity-and-anti-gaming/`
   Source: `research_agent/plan/07_AUDIT_HARNESS.md` §6, the historical audit's
   E-adjacent findings.
   Script: `scan_gaming.py` — detects test weakening (removed assertions, `skip`,
   relaxed tolerances), test-metric selection (`idxmin`/`argmin` over a frame
   containing test metrics), seed changes in frozen configs, failure trimming
   (`except: pass`, `dropna` before reporting), excluded-unit changes, metric
   reassignment, denominator clipping, edits to hashed files. Ship a fixture repo
   with one seeded instance of each and assert each is detected; assert no
   false positives on the clean fixture.
6. `.pi/skills/data-provenance/`
   Source: `src/vangja/datasets/stocks.py` (frozen Wikipedia `oldid`),
   `fixed_case_studies/stocks/01_fetch_data.py`, `data_sha256.json` patterns.
   Scripts: `hash_tree.py`, `check_sidecars.py`.

## Cross-cutting requirements

- `.pi/skills/validate_skills.sh` — checks frontmatter parses, `name` matches the
  directory, description length ≤1024, every path mentioned in `SKILL.md` exists,
  no absolute paths outside the repo, no secrets, and no skill >120 lines.
- `.pi/skills/README.md` — one table row per skill (name, trigger, owner stage).
- Each `scripts/*` entry: `--help`, non-zero exit on failure, no network, no
  hidden state outside a `--out` directory.
- Tests: `.pi/tests/skills/` with one test file per skill script; run them from
  `make -C .pi verify-env` (extend the Makefile).
- Skills must teach the *project's* rules, not generic advice: quote the real
  thresholds (e.g. `REL_MAE_EPSILON = 1e-3`, PPC coverage 30–60%), name the real
  files, and link to `AGENTS.md` sections.

## Acceptance tests (paste output into `REPORT_BUILD.md`)

1. `bash .pi/skills/validate_skills.sh` → all six valid.
2. `bash .pi/tests/run.sh skills` → all green, including the `scan_gaming.py`
   fixture matrix (each seeded defect detected, clean fixture clean).
3. `python .pi/skills/forecasting-evaluation/scripts/relative_mae.py --help` and
   `block_bootstrap.py --help` run; a demo invocation on synthetic data prints the
   documented output shape.
4. `check_protocol_compliance.py` on the real repo → clean; on a fixture with a
   test-set `idxmin` → flagged with file:line.
5. `pi --print --model opencode-go/deepseek-v4.1-flash --thinking max
   "/skill:reproducible-experiments summarize the seed policy"` returns content
   from the skill (proves routing works).

## Do not

- Do not touch `src/`, `tests/`, `fixed_case_studies/`, `paper/`,
  `Vangja-project/` (read-only sources).
- Do not create agents, orchestrator, or HPC code (later stages).

## Final report

Append an `## S1 — research & methodology skills` section to
`research_agent/REPORT_BUILD.md`: skills created, scripts + tests, acceptance
results, deviations.
