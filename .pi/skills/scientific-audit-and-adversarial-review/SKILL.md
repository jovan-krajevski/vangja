---
name: scientific-audit-and-adversarial-review
description: Run or review an audit workstream over Vangja code, results, or the paper. Use when writing a finding, verifying someone else's finding, drafting a workstream brief, synthesising a final report, or checking that a findings register is schema-valid and deduplicated. Encodes the nine-field evidence schema and the no-command-then-not-a-finding rule.
---

# Scientific audit and adversarial review

Four phases, four fresh contexts, files as the only channel:
**find → verify → synthesise → adjudicate**. Finders never verify; verifiers
never synthesise; synthesis adds no new primary claims. The authority is
`research_agent/plan/07_AUDIT_HARNESS.md`; the historical execution model is
`Vangja-project/audit/stage-2/_PROTOCOL.md`.

## The finding schema (all nine fields, no exceptions)

```
### <WS>-<NNN> · <one-line title>
- **Location:** <file:line, or artifact path + config/seed>
- **Command / test:** <exact command, or "static reading of lines X–Y">
- **Expected:** <what should hold, per the paper/spec/sound practice>
- **Observed:** <what actually happens, with numbers>
- **Consequence:** <scientific or practical impact>
- **Severity:** S0 | S1 | S2 | S3
- **Confidence:** high | medium | low
- **Status:** verified | to-verify | refuted | unresolved
```

Machine form adds `id`, `snapshot`, `reproducible`, `evidence`, `cross_refs`,
`owner`, `found_in_cycle` (plan/07 §4). **No command (or exact file:line plus
observed output) → not a finding**; it goes to a question, not the register.

Severity: **S0** a result is wrong or the method cannot support any claim ·
**S1** a claim is unsupported/overstated · **S2** documentation, usability or
reproducibility degradation · **S3** cosmetic. Confidence: high = reproduced
directly · medium = inferred from code · low = hypothesis needing data.
`unresolved` requires a tagged question to the authors.

## Operating rules

1. **Refutation first.** The verifier's default hypothesis is that the finding
   is wrong: reopen the cited lines, re-run the cited command, and try to
   falsify. Record `confirmed` / `weakened` / `refuted` with your own evidence.
2. **Anti-anchoring.** Do not read another workstream's findings before
   deriving your own; never copy a finding body. Cross-boundary leads are
   `cross-ref: <WS>-NNN`, not duplicates.
3. **Coverage is part of the result.** A workstream that checked less than its
   brief must say so; the meta-review lists what was *not* checked.
4. **Flooding is penalised.** Cap 25 findings per workstream, ordered by
   severity; everything else goes to an appendix list.
5. **Deterministic gates dominate prose.** A failing test/hash/recomputation
   is a finding regardless of how good the argument reads.
6. **The historical audit is calibration, not content.** Its findings may be
   used by the calibration scorer, not by the finder.

## Commands

```bash
python .pi/skills/scientific-audit-and-adversarial-review/scripts/validate_findings.py \
  --findings audit/cycle-3/ws-s/findings.json --workstream WS-S --json
```

The validator checks the schema, enums, id format, the no-command rule,
`reproducible`, and a dedupe hash over `title + location`; it exits non-zero
on any meta-finding. A workstream is not complete until `FINDINGS.md` and
`findings.json` agree and the validator is clean.

## Coverage checklist (per workstream)

- [ ] Scope and hard boundaries stated in the brief; cross-boundary leads
      recorded as cross-references only.
- [ ] Every finding has all nine fields and a command/test.
- [ ] Every S0/S1 was attempted to be refuted by the verifier.
- [ ] Counts (n findings, n units, n parameters) are reproducible by one
      command; approximations are labelled.
- [ ] What was NOT checked is written down.
- [ ] Severity examples consulted (`references/severity-examples.md`) so the
      calibration matches the historical scale.
- [ ] No finding depends on data that is known to be deferred (label
      `[deferred]` and provide a question instead).

## Writing style (evidence-first)

- Lead with `location` + `command`; the prose is a caption for the evidence.
- Quote the observed output (trimmed), not a paraphrase.
- Numbers in the consequence must be checkable by the same command that
  produced them.
- No rhetorical escalation: "wrong" means a gate failed, not "concerning".
- State what would change the finding ("if X were true, this is not a
  finding") — that is the refutation protocol in one line.
