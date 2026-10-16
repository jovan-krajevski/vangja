# Prompt 02 — Build stage S2: engineering, writing, review, and process skills

Prerequisite: S0 and S1 complete. Read `research_agent/plan/04_SKILLS.md`
(rows 7–15). Skills live at `.pi/skills/<name>/`, same contract as S1
(frontmatter, ≤120 lines, references/, scripts with tests, no network).

7. `.pi/skills/vangja-codebase/`
   Source: `AGENTS.md` (expand, don't duplicate), `src/vangja/**`,
   `tests/test_regularization.py`, `tests/test_time_series.py`,
   `fixed_case_studies/**`. References: `architecture.md` (component contract,
   operator semantics, pooling, transfer potentials, `fit_info`, naming rules),
   `datasets.md`, `failure-modes.md` (each with symptom → cause → fix).
   Scripts: `lint_pymc_names.py` (flags `,`, `[`, `]`, `.`, config suffixes in
   PyMC variable names in a diff), `check_pooling_shapes.py` (flags scalar
   initvals / 2-D prediction columns for non-complete pooling with `n_groups==1`).
8. `.pi/skills/hpc-slurm-and-checkpointing/`
   Source: `case_studies/{singularity.def,singularity_gpu.def,submit_ablation.slurm}`,
   `fixed_case_studies/README.md` §5.4, `research_agent/plan/06_HPC_AND_COLLECTOR.md`.
   Must encode: **single active job**, 24h wall limit, mandatory checkpoint +
   resume + local smoke test, `sbatch --parsable`, exit-code 99 convention,
   `sacct --json` parsing, requeue rules, agent-forwarded git.
   Templates: `templates/job.slurm`. Scripts: `make_job_record.py`,
   `parse_sacct.py` (fixture-driven), `checkpoint_smoke.py` (runs a command with
   `--limit 1`, kills it, resumes, asserts completed units are skipped).
9. `.pi/skills/git-and-pr-workflow/`
   Source: `case_studies/README.md` §3 (SSH agent forwarding), repo history.
   Scripts: `make_pr_record.py` (renders the PR body + machine-readable footer),
   `diff_scan.py` (protected paths, test-diff, gaming scan — composes the other
   skills' scanners). Templates: `PR.md`, `review.md`.
10. `.pi/skills/python-packaging-and-environments/`
    Source: `pyproject.toml`, `uv.lock`, `AGENTS.md` pins section.
    Script: `freeze_report.py` (emit the environment fingerprint block used in
    manifests: python, pymc, pytensor, arviz, numpy, numba, jax, uv-lock-hash,
    container hash). References: `pins.md` (why each pin exists).
11. `.pi/skills/scientific-writing-latex/`
    Source: `paper/paper.tex` (structure + notation), `paper/case_studies/tables/*.tex`,
    `Vangja-project/REVIEW/FINDINGS.md` (what must not be claimed).
    Must define: claim discipline (every number → `paper/CLAIMS.json` entry →
    artifact ID → manifest SHA), no hand-typed numbers in `.tex` (generated
    `paper/generated/**` includes only), limitations + AI-use + reproducibility
    statements, abstract honesty rules, house style.
    Templates: `claim.tex`, `limitations.md`. Script: `check_claims.py`
    (every `\num`/number-like token in the paper has a claims entry; no claims
    entry cites an artifact whose SHA is not an ancestor of `update_pymc`).
12. `.pi/skills/literature-review-and-verification/`
    Source: `paper/paper.tex` (related work), `Vangja-project/audit/stage-2/ws-h-docs-hygiene/FINDINGS.md`
    (citation problems found historically).
    Must define the source hierarchy; **mandatory verification of every citation**
    with `source_check`/`fetch_content` before use; the extraction matrix; how to
    state a competing method fairly.
    Scripts: `bib_check.py` (flags entries with no verified URL/DOI or whose
    claimed content was not checked; keeps a `verified.json` sidecar),
    `related_work_matrix.py`.
13. `.pi/skills/venue-norms-and-reviewing/`
    Must encode the grade scale (strong reject → strong accept), blocking rules,
    target venues and their criteria, desk-reject triggers, and the review form
    used by the supervisor and the audit. References: `grades.md`, `venues.md`.
    Template: `review_form.md`.
14. `.pi/skills/scientific-audit-and-adversarial-review/`
    Source: `Vangja-project/audit/stage-2/_PROTOCOL.md`, `DEFINITIONS.md`,
    `ws-v-verification/REPORT.md`, and `research_agent/plan/07_AUDIT_HARNESS.md`.
    Templates: `finding.md` (nine fields + severity/confidence/status +
    reproducible), `workstream_brief.md`, `verify.md`, `synthesis.md`.
    Script: `validate_findings.py` (schema + dedupe hash + severity/confidence
    enums + "no command → not a finding" rule). References:
    `severity-examples.md` (real examples from the historical audit).
15. `.pi/skills/research-memory-and-handoff/`
    Must define: what to read first in a fresh session (`state/CONTEXT.md`), the
    brief template, ADR format, handoff note, what never goes into prose
    numbers, how to leave a task resumable mid-way. Templates: `handoff.md`,
    `adr.md`, `brief.md`. Script: `check_state_files.sh` (validates the state
    file schemas and that `CONTEXT.md` is ≤200 lines).

## Cross-cutting requirements

- Extend `.pi/skills/README.md` and `.pi/skills/validate_skills.sh`.
- Add tests under `.pi/tests/skills/` and wire them into `make -C .pi verify-env`.
- Where a skill restates something in `AGENTS.md`, link rather than copy.
- Scripts that take a repo/paths argument must default to the repo root and
  refuse to write outside `--out`.

## Acceptance tests

1. `validate_skills.sh` → 15 skills valid.
2. `make -C .pi verify-env` green with the new skill tests.
3. `lint_pymc_names.py` on a fixture with `fs_0 - beta(p=365.25,n=5)` → flagged;
   on `src/vangja` → clean.
4. `checkpoint_smoke.py` on a fixture runner → proves skip-completed resume.
5. `parse_sacct.py` on fixtures including TIMEOUT / OOM / NODE_FAIL → correct
   state mapping and requeue classification.
6. `check_claims.py` on `paper/paper.tex` → expected large dangling list (the
   old paper has no claims registry) **reported as a finding, not an error**;
   on a small fixture paper + claims file → clean.
7. `bib_check.py` on a fixture with a fabricated citation → flagged.
8. `/skill:` routing check for three of the skills.

## Do not

- Do not modify `paper/`, `src/`, `tests/`, `fixed_case_studies/`,
  `Vangja-project/` except by reading.
- Do not implement the loop, agents, dispatch, or HPC tools (later stages).

## Final report

Append `## S2 — engineering, writing, review, process skills` to
`research_agent/REPORT_BUILD.md` with the same evidence format as S1.
