# Project skills

Skills are loaded on demand by pi (`.pi/skills/<name>/SKILL.md`, discovered as
project resources) and named in agent briefs. Validate them with:

```bash
bash .pi/skills/validate_skills.sh          # all skills under .pi/skills
bash .pi/tests/run.sh skills                # validate + script tests
make -C .pi verify-env                      # full environment check
```

Each skill is a directory: `SKILL.md` (≤120 lines), `references/`,
`scripts/` and `templates/` (where useful). Scripts take `--help`, exit
non-zero on failure, use no network, and write only under an explicit
`--out`/`--sidecar` (or not at all).

## S1 — research & methodology

| Skill | Triggers when | Owner stage |
|---|---|---|
| `reproducible-experiments` | any experiment/run/result is produced; manifests, seeds, environments, checkpoints | S1 |
| `experiment-protocol-design` | designing or amending a study; frozen configs, splits, origins, metrics, exclusions | S1 |
| `bayesian-workflow` | priors, samplers, transfer, MAP/ADVI/NUTS diagnostics, calibration | S1 |
| `forecasting-evaluation` | computing/reporting metrics; Relative MAE, bootstrap CIs, CRPS, failure rates | S1 |
| `scientific-integrity-and-anti-gaming` | before claiming a result; reviewing test/metric/seed/split changes | S1 |
| `data-provenance` | fetching/using datasets; hashes, sidecars, universes, availability | S1 |

## S2 — engineering, writing, review, process

| Skill | Triggers when | Owner stage |
|---|---|---|
| `vangja-codebase` | touching `src/`, tests, or `fixed_case_studies/`; components, pooling, transfer potentials, loaders, PyMC names | S2 |
| `hpc-slurm-and-checkpointing` | submitting/monitoring/collecting SLURM work; checkpoints, requeues, `sacct`, container jobs | S2 |
| `git-and-pr-workflow` | branching, committing, opening/reviewing a PR; protected paths, test-diff and gaming scans | S2 |
| `python-packaging-and-environments` | dependency, lockfile, container, or CI changes; environment fingerprints | S2 |
| `scientific-writing-latex` | writing/revising the paper or a report; claims, tables, limitations, AI-use statements | S2 |
| `literature-review-and-verification` | related work, positioning, new citations; verifying a source before use | S2 |
| `venue-norms-and-reviewing` | venue choice, review criteria, grades, desk-reject triggers, review forms | S2 |
| `scientific-audit-and-adversarial-review` | auditing or verifying any work; finding schema, severity, refutation, synthesis | S2 |
| `research-memory-and-handoff` | starting, ending, or pausing any session; briefs, ADRs, handoffs, state-file checks | S2 |

## Conventions

- Project facts beat generic advice: quote real thresholds (`REL_MAE_EPSILON =
  1e-3`, PPC coverage 30–60 %, R̂ < 1.01) and real file:line references.
- Long material lives in `references/`; procedures that remove judgement live
  in `scripts/` with tests in `.pi/tests/skills/`.
- Where a skill restates `AGENTS.md`, it links to the section instead of
  copying it; the skill adds the exact file:line evidence.
- Skills are reviewed like code; they must pass `validate_skills.sh`.
- Scripts that take repository/paths arguments default to the repo root and
  refuse to write outside an explicit output path.
