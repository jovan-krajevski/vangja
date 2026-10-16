# 01 — Research notes: how other agentic research systems work

Findings only; each entry ends with the mechanism we adopt and where it lands.
(Researched 2026; links are the primary sources read.)

## 1. OpenAI's Navier–Stokes run (Sept 8, 2026) — the swarm pattern and its risks

Sources: OpenAI announcement `openai.com/index/navier-stokes-solution/`; Quanta
`quantamagazine.org/ai-has-solved-one-of-maths-1-million-millennium-prize-problems-20260908/`;
BBC `bbc.com/news/articles/cy7zygy3rl2o`; coverage at `the-agent-report.com`,
`shattered.io`; Wikipedia `Navier–Stokes_priority_controversy`.

What happened, factually:

- ~10,000 concurrent agents on an internal model, ~88 hours; ~3–5M messages,
  130B output tokens (single problem); a **separate** model spent ~17h producing
  the Lean 4 formalisation. Estimated cost: several million dollars.
- Decomposition: the swarm was pointed at a portfolio of open problems and the
  winning one was pursued by many specialised agents exchanging messages —
  i.e. massive parallel decomposition with a shared, accumulating proof state.
- Verification was **mechanised** where possible (Lean) and human where not: a
  human must confirm the formalised statement matches the informal claim, and the
  community must still review the paper, the priority, and the framing.
- The release was criticised for announcing before independent verification, and
  for a priority/credit dispute with a concurrent human+AI result; an early
  LLM-written draft was described as needing substantial cleanup.
- DeepMind had previously found that **~14% of AI agents "cheated"** on formal
  math tasks by exploiting scoring loopholes rather than proving the statement.
- Both AI efforts stood on a human framework (Córdoba/Martínez-Zoroa cascades);
  the humans were the intellectual origin, the agents the search engine.

**Adopted:** (i) many small parallel workers over a shared artifact, but with our
scale (10² sessions, not 10⁴); (ii) a *separate* formalisation/verification pass —
here that is the deterministic reproduction gate and the claims↔artifact matrix;
(iii) a statement-conformance check (does the code compute what the paper
claims?); (iv) provenance and "don't announce before verification" as hard rules;
(v) anti-gaming checks as first-class guardrails.
→ `plan/02_ARCHITECTURE.md` §5, `plan/07_AUDIT_HARNESS.md` §5–6, `plan/05_EXTENSIONS_AND_TOOLS.md` §4.

## 2. OpenAI prover/verifier protocol and math artifacts

Sources: arXiv 2510.12829 *Mathematics with large language models as provers and
verifiers* (different prover and verifier *instances* working collaboratively;
final proof Lean-verified; human checks premise/conclusion conformance; 5/6 IMO
2025 problems, ~1/3 of 66 number-theory conjectures); `github.com/openai/math`
(manuscripts published "at different stages of verification"); *Early science
acceleration experiments with GPT-5* (case studies with domain experts, explicit
"where it fell short").

**Adopted:** separate **prover** and **verifier** sessions with different prompts
and, where cheap, different provenance (verifier reopens source files and re-runs
commands); explicit verification-status per artifact; honest documentation of
where the AI fell short.
→ `plan/03_AGENTS.md` (audit_verifier), `plan/07_AUDIT_HARNESS.md` §3–4.

## 3. PaperBench — rubrics as the evaluation substrate

Source: OpenAI PaperBench (`openai.com/index/paperbench/`, arXiv 2504.01848).
Agents replicate 20 ICML-2024 papers; evaluation uses **hierarchical rubrics**
decomposing each task into **8,316 individually graded, mostly binary nodes**
with explicit grading criteria. A follow-up **JudgeEval** calibrates LLM judges
against human ground truth with accuracy/precision/recall/F1.

**Adopted:** the audit produces a machine-readable rubric tree with binary leaves,
not just prose findings; the readiness and PR gates use rubrics too; and the audit
judge is **calibrated** on our golden set (89 historical findings + 4 rejected
leads) to measure sensitivity and specificity before it is trusted.
→ `plan/07_AUDIT_HARNESS.md` §2, §7; `rubrics/`.

## 4. Anthropic's multi-agent research system — orchestrator/worker discipline

Source: `anthropic.com/engineering/multi-agent-research-system`.

Concrete, transferable lessons:

- Orchestrator–worker: a lead agent plans, delegates to specialised subagents
  with **isolated context windows**, then synthesises. Multi-agent (Opus lead +
  Sonnet workers) beat single-agent by 90.2% on their internal eval; token usage
  alone explained ~80% of BrowseComp variance; multi-agent uses ~15× chat tokens.
- Subagent briefs must specify **objective, output format, tools/sources, and
  task boundaries** — vague briefs cause duplicated work and gaps.
- **Effort scaling rules** are written into prompts (simple → 1 agent/3–10 tool
  calls; complex → 10+ subagents with divided responsibility).
- The lead agent **saves its plan to memory** because context gets truncated;
  long-horizon state belongs in files.
- Parallelisation multiplies speed: 3–5 subagents in parallel, 3+ tools per
  subagent; synchronous waiting is the bottleneck.
- Evaluation: start with ~20 representative cases; a **single LLM judge with a
  rubric** (accuracy, citation accuracy, completeness, source quality, tool
  efficiency) scored 0–1 was more consistent than multiple judges; human review
  still caught what automation missed (source-quality bias).
- Production: agents are stateful; errors compound; build **resumability and
  checkpoints**, observability/tracing, and rainbow deployments.

**Adopted:** our conductor is the orchestrator; every worker brief is a file with
objective/format/tools/boundaries; effort budget per task type; plan-in-a-file;
fresh context per work unit; human gates; traceability of every run.
→ `plan/03_AGENTS.md` (brief template), `plan/02_ARCHITECTURE.md` §3–4.

## 5. DeepMind AlphaEvolve / FunSearch — verify-then-evolve

Sources: `deepmind.google/blog/alphaevolve-...`, arXiv 2506.13131, FunSearch
(Nature 2023). An LLM ensemble proposes code mutations; **automated evaluators**
score them; a program database keeps the best; evolution iterates. Requirement:
a programmatic target — no human judgment inside the loop. Known failure: agents
exploit evaluator loopholes.

**Adopted:** for engineering choices (fixes, configs, analysis variants) the loop
always pairs a generator with a programmatic evaluator (tests, metrics, hash
checks) and keeps a candidate registry; and it treats "the evaluator can be
gamed" as an explicit audit workstream item (E).
→ `plan/02_ARCHITECTURE.md` §5; `plan/07_AUDIT_HARNESS.md` §6.

## 6. Google AI co-scientist — specialist coalition + tournaments

Source: `research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/`.
A **Supervisor** agent assigns specialised agents — Generation, Reflection,
Ranking, Evolution, Proximity, Meta-review — to a worker queue; self-play debate
generates hypotheses; **Elo tournaments** rank them; an evolution step improves
the best; test-time compute scales quality; expert humans rated novelty/impact and
lab experiments validated a subset.

**Adopted:** add a `meta_review` step to the audit (an agent whose only job is to
find holes in the other workstreams' coverage), and a ranking/tournament step when
several candidate fixes exist; keep the supervisor as an agent distinct from the
generators.
→ `plan/03_AGENTS.md` (`audit_synthesizer`, `critic`), `plan/07_AUDIT_HARNESS.md`.

## 7. Sakana AI Scientist v1/v2 — the pipeline we are *not* copying

Sources: `sakana.ai/ai-scientist/`, arXiv 2408.06292, v2 (agentic tree search;
first fully AI-written workshop paper accepted through peer review); independent
evaluation of v1 found bold claims and mixed results.

**Adopted:** the phase structure (idea → experiment → analysis → paper → review)
and tree search over experiment variants; **rejected:** unconstrained idea
generation and self-review without a verification gate. Our audit is external,
adversarial, and calibrated; the paper may only cite artifacts.
→ `plan/02_ARCHITECTURE.md` §3.

## 8. Other multi-agent research pipelines

- *AI-Researcher* (arXiv 2505.18705) and *Agent Laboratory* / *cmbagent*
  (arXiv 2507.07257, ~30 agents with a Planning & Control orchestrator) confirm
  the same phase decomposition (literature → experiments → report) and the value
  of a central controller; they also show the failure mode of no verification
  gate: plausible but unchecked outputs.

**Adopted:** the phase machine and the "no phase without an artifact + a verifier"
rule.

## 9. Net design decisions

| Mechanism | Source pattern | In our plan |
|---|---|---|
| Orchestrator + fresh-context workers | Anthropic / OpenAI swarm | conductor + `phd_candidate` + workers |
| Generator ≠ verifier | FunSearch, prover/verifier | `audit_*` sessions, `audit_verifier`, deterministic gate |
| Mechanised verification first | Lean / evaluators | pytest + artifact recomputation + hashes + frozen protocol |
| Statement-conformance check | prover/verifier human step | claims↔artifact traceability matrix |
| Hierarchical rubrics + judge calibration | PaperBench / JudgeEval | `rubrics/` + golden-set calibration of the audit |
| Effort scaling rules | Anthropic | per-task-type budgets in `.pi/config/budgets.json` |
| Plan/state in files, resumable | Anthropic + AI co-scientist | `state/*.jsonl` + `LOOP_STATE.json` + worktrees |
| Human gates on irreversible/public steps | Navier–Stokes episode | merge/confirmation/audit/submission gates |
| Anti-gaming threat model | DeepMind 14% | guard + integrity workstream (E) |
| Honest labelling + AI-use disclosure | OpenAI early science | paper rubric + `LIMITATIONS.md` |
