# 00 — Goal, constraints, and open decisions

## 1. Objective

Produce the best defensible version of the Vangja paper — a rigorous,
reproducible, honestly-scoped empirical study of Bayesian transfer learning for
short time series — using a self-driving multi-agent loop, with an audit process
strictly stronger than the historical `Vangja-project/audit`.

"Best paper" is defined operationally, not aspirationally:

1. Every quantitative claim is recomputable from committed artifacts by a single
   documented command, and traceable to a commit SHA, a frozen config, a seed,
   and a data hash.
2. The experimental design is pre-registered before confirmation results are
   seen: one frozen config per study, development/confirmation separation,
   frozen confirmation origins with hashes, no test-horizon selection.
3. Baselines are honest (persistence/drift/seasonal-naïve/ARIMA/Holt-Winters and
   *clearly named* Vangja ablations; no unnamed stand-ins called "Prophet").
4. Uncertainty is either calibrated (posterior predictive with coverage/CRPS
   reported) or explicitly labelled as a heuristic.
5. Known limitations (S&P mechanical overlap, survivorship, retrospective
   smart-home data, MAP point forecasts) are disclosed in the paper, not buried.
6. An adversarial audit with independent verification grades the work, and the
   grade gates whether the loop continues.

Venue target (default, configurable): *Journal of Forecasting*; *Machine
Learning* is conditional on the mechanism study and calibration results; no
AISTATS/TMLR claims in the current empirical state.

## 2. Starting point (what already exists)

- `update_pymc` contains the code fixes from the historical audit (`a562629` and
  later): regularisation sign/form, dispatch, interpolation, seeds,
  `target_accept`, NUTS dispatch, `relative_mae`, `get_sp500_tickers_at_date`,
  PyMC pinning.
- `fixed_case_studies/` is the audit-compliant rerun suite: frozen configs, no
  grid, Relative-MAE primary metric, hash-frozen confirmation origins, block
  bootstrap, covariance-transfer script, calibration script, report script.
  Its outputs do **not** exist yet — the runs have never been executed to
  completion.
- `paper/paper.tex` is the *old* manuscript (759 lines) containing claims the
  audit refuted. It must be rewritten against new results, not patched quietly.
- `Vangja-project/` is the read-only historical audit: Stage-1 plan, six
  workstreams, verification (WS-V), synthesis (WS-Y), adjudication, 89 confirmed
  findings, 4 rejected leads, 7 open questions. It is the template — and the
  calibration set — for the new audit.
- There is no compiled paper build, no claims↔artifact traceability, no
  experiment ledger, no agent environment.

## 3. Constraints

1. **Model:** every agent, every session = `opencode-go/deepseek-v4.1-flash`,
   thinking `max`, 1M context. No exceptions, no downgrades, no cheaper models for
   "simple" subtasks.
2. **Locality:** every skill, extension, tool, prompt, and agent definition is a
   file in this repository. No global installs, no runtime downloads of agent
   resources. Third-party resources may be *read* as references, but what runs is
   ours and pinned.
3. **Read-only inputs:** `Vangja-project/**` is never modified; the published
   PyPI package and remote branches are inspection-only; the historical
   manuscript is not edited in place — it is superseded by a new build.
4. **Branches:** all work on `work/T-###-<slug>` branches; only the supervisor/
   conductor merges into `update_pymc`; no force-pushes or history rewrites.
5. **Claims:** a claim may only cite an artifact produced on `update_pymc` whose
   manifest records the producing SHA. No number enters the paper from an agent's
   memory or from an unlogged local run.
6. **Human gates:** pause before merging (default on), before confirmation runs,
   before the audit, before anything called "submission".
7. **No fabrication:** missing data, failed fits, and unavailable jobs are
   recorded as such. Agents may not interpolate, impute, or "reconstruct" a
   missing result.
8. **Cost discipline:** budgets are enforced by the conductor; a run that exceeds
   its budget pauses and asks, it does not silently continue.

## 4. Definition of done (for the loop)

The loop may declare `DONE` only when all hold:

- Audit grade ∈ {accept, strong accept}, or weak-accept-grade fixes explicitly
  waived by the human.
- No open S0/S1 findings; all S2 findings either fixed or explicitly accepted
  with a documented reason in `paper/LIMITATIONS.md`.
- `make verify` in the paper build reproduces every table/figure/number from
  artifacts, and the traceability matrix covers 100% of claims.
- Confirmation runs complete with all protocol artifacts (`PROTOCOL.md` §11).
- The paper states the honest scope (retrospective smart-home, confirmation
  stocks, calibration disclosure, baseline naming, AI-use disclosure).
- Human sign-off recorded in `state/HUMAN_GATES.json`.

## 5. Open decisions for the human (confirm before running prompt 04)

1. **Human gate before merge:** default is yes (supervisor prepares, human
   approves each merge into `update_pymc`). Set `require_human_merge=false` for a
   fully autonomous merge policy, at the cost of a weaker audit trail.
2. **HPC:** is there a cluster? If yes, provide host/user/keys/partition and the
   remote work dir. If no, the loop runs locally with reduced parallelism and the
   audit notes the reduced scale.
3. **Confirmation data:** the stocks confirmation origins (2023–2024, in
   `fixed_case_studies/protocol/CONFIRMATION_ORIGINS.csv`) are frozen and ready.
   Are they approved for use, or should new origins be chosen? Smart-home has no
   independent confirmation data: confirm the retrospective-only framing.
4. **Venue and deadline:** affects the paper rubric and the "done" bar.
5. **Budget:** max sessions/day, max parallel workers, max $ (or token) spend,
   max wall-clock per task. Defaults are in `plan/02_ARCHITECTURE.md` §6.
6. **Scope of "new ideas":** may the loop propose and run *new* experiments
   beyond the audit fix-list (e.g. a synthetic mechanism study), or is it confined
   to closing findings and producing the paper? Default: new experiments allowed
   only if they address an audit finding or a paper claim, and always on
   development data first.
7. **Autonomy of the audit:** may the audit look at the paper's git history and
   the loop's own state (useful for provenance findings, but risks anchoring)?
   Default: yes for provenance/reproducibility workstreams only.
