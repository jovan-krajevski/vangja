# Source hierarchy and the verification log

## Levels

| Level | Examples | Use for | Never use for |
|---|---|---|---|
| `primary` | peer-reviewed paper (DOI), official package documentation, standard, archived dataset page | method claims, benchmark numbers, API behaviour | — |
| `secondary` | survey, review, well-cited tutorial | orientation, framing, finding the primary | overriding the primary it summarises |
| `tertiary` | blog post, release notes, forum thread | implementation hints, historical context | any number or priority claim |
| `preprint` | arXiv/bioRxiv without peer review | recent work, with the status stated | "published" claims |
| `docs` | README of the package being compared | installed-version behaviour | scientific claims |

Record `level` in the extraction matrix. A `tertiary` source alone cannot
support a related-work claim; find the primary or drop the claim.

## What each citation must record

```json
{
  "key": "T_prophet",
  "level": "primary",
  "claim": "Prophet models trend + seasonality as a GAM with changepoints",
  "method": "piecewise linear trend, Fourier seasonality, MAP/L-BFGS",
  "dataset": "internal benchmarks + case studies",
  "limitation": "no hierarchical pooling; transfer not supported",
  "relevance": "baseline and formal starting point",
  "doi": "10.1080/00031305.2017.1380080",
  "url": "https://doi.org/10.1080/00031305.2017.1380080",
  "checked": true,
  "checked_date": "2026-01-01",
  "quote": "exact passage supporting the claim",
  "quoted_from": "page/§/paragraph"
}
```

The `checked` flag lives in the sidecar (`verified.json`), never in the
bibliography file itself: a `.bib` is a rendering artifact, the sidecar is
the evidence.

## Verification etiquette

- Verify at the version you cite; if the paper has a v2, record which.
- Quote, do not paraphrase, in the sidecar; the quote is what a reviewer
  checks.
- Numbers taken from a paper record the table/figure number and the page.
- If the source is paywalled, use the DOI landing page + abstract and state
  in the matrix that only the abstract was checked.
- Re-verify anything older than the audit cycle it was recorded in.
