---
name: scientific-writing-latex
description: Write or revise the Vangja manuscript and its tables. Use when editing paper/paper.tex, adding a figure or table, reporting a number in prose, preparing limitations/AI-use/reproducibility statements, or checking whether every number in the paper is backed by a claims entry and an artifact. Enforces generated includes over hand-typed numbers.
---

# Scientific writing (LaTeX)

Claim discipline is the rule of this skill: **every number in the manuscript
traces to a `CLAIMS.json` entry, which traces to an artifact path + a
manifest SHA that is an ancestor of the frozen results**. Prose that cannot
cite an artifact does not go in the paper. Read
`references/house-style.md` for the notation/structure conventions and
`Vangja-project/REVIEW/FINDINGS.md` (§Decisions already settled) plus
`Vangja-project/audit/stage-2/ws-h-docs-hygiene/FINDINGS.md` for the list of
things the current drafts must not claim.

## Hard rules

1. **No hand-typed numbers.** Results live in generated includes; the only
   source is a report script and the artifact it read. `check_claims.py`
   flags any number outside the generated directory, and any number with no
   claims entry.
2. **One claim, one artifact.** `CLAIMS.json` entries carry
   `{id, value, tex, artifact, artifact_sha256, commit, status}`. The
   artifact SHA must be an ancestor of the frozen results branch; a claim
   whose artifact is missing or superseded is `stale`, not silently updated.
3. **Abstract honesty.** The abstract may only state the sign and rough
   magnitude of what the results support, with the dataset framing attached
   (stocks 4× horizon; smart home ≈2.9×, retrospective/illustrative). No
   best-of-grid number may be presented as a single frozen configuration.
4. **Every limitation has a home.** `templates/limitations.md` is the
   checklist; a missing disclosure is a finding, not a style choice.
5. **AI-use and reproducibility statements are mandatory** in any submitted
   version: model/tool use, prompt-to-artifact traceability, seeds, and the
   data hashes.
6. **The paper is written in the PAPER state only** (guard). Until then,
   record results in artifacts and task branches.

## Commands

```bash
python .pi/skills/scientific-writing-latex/scripts/check_claims.py \
  --paper paper/paper.tex --claims CLAIMS.json          # report (exit 0)
python .pi/skills/scientific-writing-latex/scripts/check_claims.py \
  --paper paper/paper.tex --claims CLAIMS.json --gate   # gate (exit 1)
```

Today the report is expected to be long: the legacy `paper/paper.tex` has no
claims registry and hand-typed tables. That is a finding to fix during the
rewrite, not an error in the checker.

## Writing a results paragraph (procedure)

1. Open the artifact (`manifest_*.json` + `units_*.csv`) and read the number
   from the file, not from a previous draft.
2. Put the number in the generated include (or the claims entry), never in
   the sentence.
3. In the sentence, cite the claim inline:
   `Vangja reaches a median relative MAE of 0.31~[C-007]`.
4. State the framing next to the number: horizon multiple, arm names,
   retrospective/confirmatory, n units after exclusions.
5. Re-run `check_claims.py`; then `cd docs && make html` if the change
   touched anything the docs embed.

## Failure modes

- Table numbers copied from an older draft: invisible until the audit
  recomputes them; always regenerate the include.
- Two claims with the same value (or the same claim cited twice) → check
  which artifact each belongs to; values alone do not identify a quantity.
- `\input{generated/...}` missing at build time → the report script did not
  run; run it, do not paste the numbers.
- A claim's `commit` is not an ancestor → the number came from an
  unfrozen run; re-run under the frozen protocol or mark `stale`.
- Prose says "MAP" for `parametric` transfer → wrong (`WS-H-002`): it is
  posterior moment matching.
