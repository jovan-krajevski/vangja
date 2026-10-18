---
name: literature-review-and-verification
description: Find, verify, and position related work for the Vangja paper. Use when adding a citation, writing a related-work paragraph, comparing against Prophet/TimeSeers or another method, or checking a bibliography for fabricated or unverified entries. Requires source_check/fetch_content verification of every citation before use and keeps a verified.json sidecar.
---

# Literature review and verification

**No citation enters the paper unverified.** `paper/paper.tex` currently has
no `.bib` file at all (`\bibliography{my_paper}` points at a file that is not
in the repository) — restoring the bibliography is a literature task, and
every restored entry must be checked against the source, not reconstructed
from memory.

## Source hierarchy (prefer top-down; record the level)

1. **Primary**: peer-reviewed paper, official documentation, standard,
   archived dataset page. Cite by DOI or a stable URL.
2. **Secondary**: survey or a well-cited review.
3. **Tertiary**: blog post, tutorial, vendor page.
4. **Not citable**: model-generated summaries, forum answers, search-result
   snippets, anything you cannot open.

State the level in the extraction matrix (`templates/related_work_matrix.md`)
so the reader can weigh the claim. A secondary source never overrides the
primary it summarises.

## Verification procedure (mandatory per citation)

1. Find the canonical record (DOI page / arXiv / official docs).
2. `source_check` (or `fetch_content`) the claim you intend to attribute;
   save the exact passage, not a paraphrase.
3. Write the entry into the `.bib` with DOI/URL, then add a
   `verified.json` record: `{key: {checked: true, url, doi, date, quote}}`.
4. `bib_check.py` must pass for the changed entries.
5. If the source cannot be opened, either drop the citation or mark it
   `checked: false` and remove it from the paper.

```bash
python .pi/skills/literature-review-and-verification/scripts/bib_check.py \
  --bib references.bib --verified verified.json --json
python .pi/skills/literature-review-and-verification/scripts/related_work_matrix.py \
  --sources sources.json --out scratch/related-work.md
```

## Stating a competing method fairly

- Quote the method's own formulation before contrasting; never attribute a
  limitation it does not claim.
- Compare like with like: same data, same horizon, same metric. Prophet and
  TimeSeers are **prophet-class baselines**; the historical audit found the
  old "Prophet/TimeSeers" numbers were actually Vangja configurations
  (F-14) — never resurrect that comparison.
- Say what the other method does *better* if it does; a related-work
  paragraph that only lists weaknesses is a strawman and will be flagged by
  review.
- For TimeSeers state the partial-pooling comparison truthfully: Vangja
  adds the missing independent-pooling mode
  (`fixed_case_studies/README.md` §3); "Vangja is hierarchical" is not a
  novelty claim (the paper itself says so).
- Novelty claims must be defensible against the closest prior work. If the
  closest work already does posterior-to-prior transfer for seasonality,
  say so and narrow the contribution to trend transfer + the potentials.

## Failure modes

- Citation with no DOI/URL and no verification record → `bib_check.py`
  flags it; that is a fabricated/unverifiable citation until proven
  otherwise.
- `verified.json` says DOI X, the `.bib` says DOI Y → one of them is wrong;
  check the source again, do not "fix" the sidecar to match.
- A claim attributed to a survey that actually reports it from a primary
  source → cite the primary.
- Numbers quoted from related work without checking the current version of
  the paper → verify at the cited revision; record the date.
