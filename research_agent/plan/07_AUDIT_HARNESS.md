# 07 — The audit harness (stronger than `Vangja-project/audit`)

> Audit cycles are written to `.pi/audit/cycle-<n>/` at the repository root; all
> paths below are relative to it unless stated otherwise.

The historical audit was good: six independent workstreams, a nine-field evidence
schema, a refutation pass, a synthesis pass, an adjudication, and a verified
89-finding register. This harness keeps all of that and adds the things it was
missing: deterministic reproduction gates, a claims↔artifact traceability
workstream, an anti-gaming workstream, a coverage meta-review, machine-readable
findings, rubric-derived grades, and **calibration against the historical audit
itself**.

## 1. Principles

1. **Finders never verify; verifiers never synthesise; synthesis adds no new
   primary claims.** Four phases, four fresh contexts, files as the only channel.
2. **Every finding is reproducible**: a `file:line` (or an artifact + config) plus
   an exact command plus the observed output. No command → not a finding.
3. **Deterministic gates dominate**: if `pytest`, a hash check, or a recomputation
   disagrees with a claimed result, that is a finding regardless of prose.
4. **Try to refute.** The verification phase's default hypothesis is that the
   finding is wrong.
5. **Calibration before trust.** The audit is measured against the historical
   findings on the historical snapshot; a run that misses known S1s or confirms
   known-rejected leads is not allowed to grade the paper.
6. **Assume gaming.** The evaluated entity (the loop) also writes the code under
   audit; the E workstream and the canary checks exist for exactly that.

## 2. Snapshot and scope

At audit start the conductor freezes:

- `audit-cycle/<n>` git tag on `update_pymc` (the code under audit), with SHA.
- The paper commit SHA and the built PDF hash.
- The artifact index: every manifest + SHA used by any paper claim.
- Protocol hashes: `CONFIRMATION_ORIGINS.csv.sha256`, frozen config hashes,
  `uv.lock`, container hashes.
- The loop's own ledgers (read-only) for the R/ E workstreams.

`audit/cycle-<n>/SCOPE.md` records all of the above. Audit output is
append-only within a cycle; corrections are new files (`VERIFY_CORRECTIONS.md`)
rather than edits to findings.

## 3. Workstreams

| WS | Name | Scope (hard boundaries) | Typical evidence |
|---|---|---|---|
| M | Methodology & experimental design | design validity, selection, baselines, horizon framing, attribution, negative transfer, generalisation limits | protocol files, runner code, result tables |
| S | Statistics & math | formulas vs code, priors, pooling parameterisation, transfer moments, gradients/potentials, estimator correctness | component code, tests, numeric probes |
| P | Probabilistic UQ & calibration | posterior predictive validity, coverage/sharpness/CRPS, MAP-heuristic disclosure, sampler diagnostics | idata, calibration script output, diagnostics CSVs |
| D | Data handling & leakage | splits, scaling fit-on-train, interpolations, context cutoffs, joins, availability/exclusions | loaders, harness code, data hashes |
| I | Implementation & software quality | API correctness, dispatch, error handling, types, dead paths, dependency behaviour, test quality | code, tests, runtime probes |
| R | Reproducibility & provenance | env pinning, seeds, SHAs, manifests, ledger integrity, CI, containers, one-command reproduction | manifests, locks, remote records |
| H | Documentation & hygiene | README/docs/claims vs behaviour, naming honesty, licence, packaging, notebook staleness | docs, docstrings, notebooks |
| E | Evaluation integrity & anti-gaming | selection on test metrics, seed shopping, test weakening, failure trimming, post-hoc exclusions, horizon cropping, metric edits, denominator tricks | diffs, scan reports, configs |
| C | Claims ↔ artifacts traceability | every number/claim in the paper maps to an artifact whose SHA is an ancestor of the audited tag; statement-conformance (the code computes what the paper says) | `paper/CLAIMS.json`, manifests, reproduction runs |
| L | Literature & positioning | related-work accuracy, no strawmen, citations exist and say what is claimed, novelty claims defensible | bibliography, `source_check` results |

Each WS writes `FINDINGS.md` + `findings.json`. A WS may not report outside its
boundary; cross-boundary leads are recorded as `cross-ref: <WS>-NNN` and left to
the owner. Flooding is penalised: the brief caps each WS at 25 findings, ordered
by severity; everything else goes to an appendix list.

## 4. Finding schema (machine-checkable)

```json
{
  "id": "S-002",
  "title": "one line",
  "location": "src/vangja/components/linear_trend.py:358",
  "snapshot": "audit-cycle/3@<sha>",
  "command": "uv run pytest tests/test_regularization.py -k trend -q",
  "expected": "what should hold, per the paper/spec/sound practice",
  "observed": "what actually happens (with numbers)",
  "consequence": "scientific or practical impact",
  "severity": "S0|S1|S2|S3",
  "confidence": "high|medium|low",
  "status": "verified|to-verify|refuted|unresolved",
  "reproducible": "yes|no (why)",
  "evidence": ["artifact id", "file:line", "command output hash"],
  "cross_refs": [], "owner": "code|paper|protocol|docs|env",
  "found_in_cycle": 3
}
```

Severity: **S0** = correctness-defining (a result is wrong or the method cannot
support any claim) · **S1** = claim-affecting (a published/plausible claim is
unsupported or overstated) · **S2** = documentation/usability/reproducibility
degradation · **S3** = cosmetic. Confidence high = reproduced directly; medium =
inferred from code; low = hypothesis needing data/run. `unresolved` requires a
question in `QUESTIONS_FOR_AUTHORS.md` tagged `[blocking]`/`[non-blocking]`.

## 5. Deterministic gates (run before any rubric scoring)

Every gate emits a machine result (`audit/cycle-<n>/gates.json`) and can itself
produce findings:

| Gate | What it proves |
|---|---|
| `tests` | `uv run pytest` at the audited tag |
| `protocol-hashes` | frozen files match their committed hashes |
| `confirmation-frozen` | confirmation origins were hash-committed before results existed (ledger `ts` vs file `ts`) |
| `no-test-selection` | static scan: no test-metric-based selection in runners |
| `test-diff` | tests were only strengthened across the cycle |
| `no-failures-hidden` | failure/availability files exist and counts reconcile with the reported N |
| `context-excluded` | context series absent from target aggregates |
| `artifacts-complete` | every required artifact type from `PROTOCOL.md` §11 exists and validates |
| `reproduce-headline` | the top-N paper numbers are recomputed from artifacts (tolerance stated) |
| `traceability` | `CLAIMS.json` covers every number in the paper; every artifact SHA is an ancestor of the audited tag |
| `env-pinned` | `uv.lock`, versions, container hashes recorded and consistent |
| `gaming-scan` | the anti-gaming catalogue scan is clean (or each hit is explained) |
| `canary` | seeded-defect canaries injected into a scratch copy are detected by the corresponding workstream (sensitivity) |

A failed gate is an automatic S0/S1 finding assigned to the relevant workstream;
the audit cannot award above `weak reject` with a failed gate unless the
adjudicator records why the gate is inapplicable.

## 6. Anti-gaming catalogue (workstream E + `scan_gaming.py`)

Detect, and report with evidence, any of: selection of configurations/thresholds
on the test horizon; seed shopping (a seed chosen after seeing results, or seeds
not recorded); test weakening (assertions removed, tolerances relaxed, `skip`
added, seeds changed in frozen tests); failure trimming (failed units dropped
rather than reported); post-hoc exclusions (units excluded after seeing results);
horizon cropping; metric redefinition or denominator clipping; split or scaling
changes; comparing against ill-named baselines; reporting the best arm without
multiple-comparison disclosure; claiming covariance preservation from mean-field
ADVI; reporting numbers whose artifacts are missing or whose SHA is not an
ancestor of the audited tag. Each hit: `location`, `command`/scan output,
`consequence`, severity.

## 7. Rubrics, grades, and calibration

`rubrics/audit.json` is a tree; leaves are binary/scale with weights, some
`auto` (filled from gates) and the rest scored by the synthesizer **with an
evidence pointer per score**. `auditkit.audit_grade(rubric, findings)` derives the
grade:

- Scale: `strong reject · reject · weak reject · borderline · weak accept ·
  accept · strong accept` (+ `major/minor revision` mapping for journals).
- Blocking rule: any confirmed S0, or any confirmed S1 in M/S/P/C, caps the grade
  at `weak reject`; an unfixed confirmed S1 elsewhere caps at `borderline`.
- A failed deterministic gate caps at `weak reject` unless the adjudicator
  records an inapplicability reason.
- `accept`/`strong accept` require: no confirmed S0/S1, all gates green,
  traceability complete, and the limitation disclosures present.
- The grade comes with a **fix list**, each item tagged feasible/needs-human with
  an effort estimate and an owner.

**Calibration (mandatory, once per audit cycle, cheap subset).** Run workstreams
S, M, R against the frozen historical snapshot (`main@8028c51`, the pre-fix
commit, extracted read-only into a scratch worktree) in `calibration` mode:
expected to rediscover a defined list of historical S1 findings (e.g.
WS-S-001/002/003, WS-M-001/002, WS-R-001/004) and **not** to confirm the four
rejected Phase-A leads (F-012, F-014, F-015/F-016 as framed). Report recall over
the expected list and false-positive count; a recall < 0.6 or any confirmed
rejected lead marks the audit run `uncalibrated` and the conductor narrows briefs
and re-runs those workstreams. The historical `Vangja-project/audit` is available
to the *calibration scorer* (not to the finders).

## 8. Phases and files

```
audit/cycle-<n>/
  SCOPE.md                     # snapshot, SHAs, hashes, environment
  gates.json                   # deterministic gate results
  _PROTOCOL.md                 # schema + rules (adapted from the historical one)
  <ws>/BRIEF.md                # narrow task + coverage checklist + effort cap
  <ws>/FINDINGS.md             # human-readable, schema-ordered
  <ws>/findings.json           # machine-readable
  VERIFY.md                    # confirmed / weakened / refuted, with corrections
  verify/VERIFY_<ws>.md        # per-workstream verification detail
  FINAL_REPORT.md              # author-facing (verified findings only)
  rubric.json                  # rubric with auto + scored leaves and evidence
  GRADE.md                     # grade, blocking rule trace, fix list
  ADJUDICATED_REPORT.md        # disputes resolved, fix mapping, residual risk
  ROOT_ISSUE_REGISTER.csv      # root issue → findings → owner → fix → status
  QUESTIONS_FOR_AUTHORS.md     # blocking / non-blocking
  sessions/                    # persisted sessions for every phase
```

Phase separation is enforced by the conductor (different agent definitions,
different session dirs, and a rule that synthesis may only cite verified IDs).

## 9. Improvements over the historical audit (explicit)

1. Deterministic gates + reproduction of headline numbers (the historical audit
   deferred reproduction in several places).
2. Claims↔artifact traceability (statement conformance) as a first-class
   workstream — the analogue of the human premise/conclusion check in the
   prover/verifier protocol.
3. Anti-gaming workstream + seeded canaries (DeepMind's 14% failure mode).
4. Machine-readable findings + rubrics + computed grade, so the supervisor's
   decision is a rule application, not a judgement call.
5. Calibration against the historical 89 findings and 4 rejected leads.
6. Split UQ workstream (P) so calibration cannot hide inside methodology.
7. Coverage meta-review (what was *not* checked) and per-WS caps to prevent
   flooding.
8. Runs on a frozen snapshot tag every cycle, with loop-ledger access for
   provenance/reproducibility, so findings are about a specific artifact set.
9. Fix list with owners and effort, feeding directly into `SELECT_TASK`.
10. No staging gate that stalls the loop: the audit is triggered by the readiness
    rubric, not by an author's approval — the human sits at merge/submission.
