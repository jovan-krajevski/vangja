# 04 — Skills inventory

> Skills and their supporting files live at the **repository root**, under
> `.pi/skills/` (project resources are auto-discovered there).

A skill is a directory `.pi/skills/<name>/SKILL.md` with frontmatter
(`name`, `description`) plus `references/` and `scripts/`. All are **local** to
this repo. The `description` must say *what it does and when it applies* (it is
the routing surface). Skills are loaded on demand; the loop's briefs name the
skills each task needs.

Rule: a skill that only repeats general knowledge is not worth writing. Every
skill below must contain either project-specific facts, a checkable procedure, or
a script that removes judgement from a step.

## Research & methodology

| # | Skill | Triggers when | Must contain | Bundled artifacts |
|---|---|---|---|---|
| 1 | `reproducible-experiments` | any experiment/run/result is produced | seed policy (`BASE_SEED`, per-cell `seed_for`), env pinning with `uv --frozen`, data hashes, manifest schema, checkpoint format, deterministic naming, "no network at run time" | `scripts/make_manifest.py`, `scripts/check_manifest.py`, `references/manifest-schema.md`, `templates/run_dir.md` |
| 2 | `experiment-protocol-design` | designing or amending a study | dev/confirmation separation, frozen-config rule, hash-committed origins, primary/secondary metrics, baseline list, negative controls, exclusion rules, what counts as a protocol amendment and how to record one | `references/vangja-protocol.md` (distilled from `Vangja-project/REVIEW/PROTOCOL.md`), `scripts/check_protocol_compliance.py` |
| 3 | `bayesian-workflow` | priors, samplers, transfer, calibration | PPC workflow and coverage targets, R̂/ESS/divergence/tree-depth thresholds, PSIS-LOO and LFO-CV, SBC, `prior_from_idata` vs moment matching, ADVI ≈ diagonal warning, MAP multi-start diagnostics, `target_accept`, seed threading | `scripts/diagnose_idata.py`, `scripts/check_convergence.py`, `references/thresholds.md` |
| 4 | `forecasting-evaluation` | computing/reporting metrics | Relative MAE vs persistence (unit = target×origin, aggregate = median), ε-exclusion rule, CRPS/coverage/sharpness, MASE/RMSSE, two-way block bootstrap, horizon overlap and multiplicity, negative-transfer rate, failure-rate reporting | `scripts/relative_mae.py` (reference impl.), `scripts/block_bootstrap.py`, `references/metrics.md` |
| 5 | `scientific-integrity-and-anti-gaming` | before claiming a result, and as a review checklist | the gaming catalogue: test weakening, seed shopping, test-set selection, metric/split edits, failure trimming, post-hoc exclusions, horizon cropping, denominator clipping, artifact-less numbers; required countermeasures; how to write an amendment | `scripts/scan_gaming.py` (AST/grep over runners: `idxmin` on test metrics, `except: pass`, `skip`, metric reassignment, split edits), `references/gaming-catalogue.md` |
| 6 | `data-provenance` | any dataset is fetched/used | source URL + revision, SHA-256 sidecars, licence/redistribution limits, survivorship and availability accounting, no silent drops, cache layout, frozen-revision rules (e.g. Wikipedia `oldid`) | `scripts/hash_tree.py`, `scripts/check_sidecars.py`, `references/datasets.md` |

## Engineering

| # | Skill | Triggers when | Must contain | Bundled artifacts |
|---|---|---|---|---|
| 7 | `vangja-codebase` | touching `src/`, tests, or `fixed_case_studies/` | the AGENTS.md facts (expanded), component contract, pooling rules, naming rules, operator semantics, transfer-potential conventions, `fit()`/`fit_info`, dataset helpers, fixed-case-study entry points, common failure modes with fixes | `scripts/lint_pymc_names.py`, `scripts/check_pooling_shapes.py`, `references/architecture.md` |
| 8 | `hpc-slurm-and-checkpointing` | submitting/monitoring/collecting HPC work | **single active job**, 24h wall limit, mandatory checkpoint + resume contract, idempotency key, submission template, `sbatch --parsable`, `sacct` parsing, timeout/OOM/node-fail → requeue rules, agent-forwarded git, container usage | `templates/job.slurm`, `scripts/make_job_record.py`, `scripts/parse_sacct.py`, `references/slurm-fields.md` |
| 9 | `git-and-pr-workflow` | branching, committing, opening/reviewing PRs | branch naming, atomic commits, worktrees, PR body + machine footer, review checklist, merge policy (no force push, no history rewrite, supervisor merges only), protected paths, resolving conflicts | `scripts/make_pr_record.py`, `scripts/diff_scan.py`, `templates/PR.md`, `templates/review.md` |
| 10 | `python-packaging-and-environments` | dependency, lockfile, container, CI changes | `uv` conventions, pin rationale (pymc/numba/numpy/arviz/pytensor), extras layout, singularity/apptainer builds, CI-equivalent local checks, "frozen env" rule for experiments | `scripts/freeze_report.py`, `references/pins.md` |

## Writing, review, and process

| # | Skill | Triggers when | Must contain | Bundled artifacts |
|---|---|---|---|---|
| 11 | `scientific-writing-latex` | writing/revising the paper or a report | claim discipline (every number → artifact ID), LaTeX house style for `paper/paper.tex`, table/figure generation rules (never hand-typed numbers), limitations + AI-use + reproducibility statements, notation consistency, abstract honesty | `templates/claim.tex`, `templates/limitations.md`, `scripts/check_claims.py`, `references/house-style.md` |
| 12 | `literature-review-and-verification` | related work, positioning, new ideas | search strategy, source hierarchy (primary papers > surveys > blogs), **verify every citation** with `source_check`/`fetch_content` before use, extraction matrix (claim / method / dataset / limitation / relevance), how to write a related-work paragraph that cannot be accused of strawmanning | `scripts/bib_check.py`, `templates/related_work_matrix.md` |
| 13 | `venue-norms-and-reviewing` | venue choice, review criteria, grades | target-venue norms (Journal of Forecasting / IJF / Machine Learning / JSS), grade scale (strong reject → strong accept), reviewer criteria (novelty, correctness, evidence, reproducibility, clarity, significance), desk-reject triggers, response-to-reviewers structure | `references/grades.md`, `references/venues.md`, `templates/review_form.md` |
| 14 | `scientific-audit-and-adversarial-review` | auditing or reviewing any work | the nine-field finding schema, severity S0–S3 with examples, confidence rules, refutation protocol (state what would falsify, then try), evidence-first writing, anti-anchoring (never copy another workstream's finding), coverage checklist | `templates/finding.md`, `scripts/validate_findings.py`, `references/severity-examples.md` |
| 15 | `research-memory-and-handoff` | starting, ending, or pausing any session | what to read first, brief-writing, ADR format, state-file discipline, "fresh context" rules, how to hand off mid-task, what never goes in prose (numbers without artifacts) | `templates/handoff.md`, `templates/adr.md`, `scripts/check_state_files.sh` |

## Build and validation

- Each skill's `SKILL.md` is ≤120 lines; long material lives in `references/`.
- Every `scripts/` entry takes `--help`, exits non-zero on failure, and is
  covered by at least one test in `.pi/tests/` (a small shell/pytest suite the
  conductor runs with `make verify-env`).
- `scripts/validate_skills.sh` checks: frontmatter parses, `name` matches the
  directory, description length ≤1024, all referenced files exist, no absolute
  paths outside the repo, no secrets.
- Skills are reviewed like code: they are diffed in PRs under
  `.pi/**` and require supervisor approval.
- Optional third-party skills (e.g. from public pi-skill collections) may be
  used as *references* for authoring; if any text is vendored it is copied into
  the repo with its licence and pinned source, and it still passes the same
  validation.
