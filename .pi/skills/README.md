# Project skills

Skills are loaded on demand by pi (`.pi/skills/<name>/SKILL.md`, discovered as
project resources) and named in agent briefs. Validate them with:

```bash
bash .pi/skills/validate_skills.sh          # all skills under .pi/skills
bash .pi/tests/run.sh skills                # validate + script tests
make -C .pi verify-env                      # full S1 environment check
```

Each skill is a directory: `SKILL.md` (≤120 lines), `references/`, `scripts/`
(and `templates/`/`fixtures/` where useful). Scripts take `--help`, exit
non-zero on failure, use no network, and write only under an explicit `--out`.

## S1 — research & methodology

| Skill | Triggers when | Owner stage |
|---|---|---|
| `reproducible-experiments` | any experiment/run/result is produced; manifests, seeds, environments, checkpoints | S1 |
| `experiment-protocol-design` | designing or amending a study; frozen configs, splits, origins, metrics, exclusions | S1 |
| `bayesian-workflow` | priors, samplers, transfer, MAP/ADVI/NUTS diagnostics, calibration | S1 |
| `forecasting-evaluation` | computing/reporting metrics; Relative MAE, bootstrap CIs, CRPS, failure rates | S1 |
| `scientific-integrity-and-anti-gaming` | before claiming a result; reviewing test/metric/seed/split changes | S1 |
| `data-provenance` | fetching/using datasets; hashes, sidecars, universes, availability | S1 |

## Planned (later stages)

Engineering skills `vangja-codebase`, `hpc-slurm-and-checkpointing`,
`git-and-pr-workflow`, `python-packaging-and-environments` arrive in S2; the
writing/review skills (11–15) arrive in S3. They are already named in
`.pi/config/agents.json` and will be added to this table as they land.

## Conventions

- Project facts beat generic advice: quote real thresholds (`REL_MAE_EPSILON =
  1e-3`, PPC coverage 30–60 %, R̂ < 1.01) and real file:line references.
- Long material lives in `references/`; procedures that remove judgement live
  in `scripts/` with tests in `.pi/tests/skills/`.
- Skills are reviewed like code; they must pass `validate_skills.sh`.
