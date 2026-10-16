# Related-work extraction matrix

One row per source. Columns are fixed; `related_work_matrix.py` renders this
from `sources.json` and flags missing cells. A row with `checked: false` must
not appear in the paper.

| Key | Level | Claim it supports | Method | Dataset | Limitation | Relevance to Vangja | DOI/URL | Checked |
|---|---|---|---|---|---|---|---|---|
| {{key}} | primary/secondary/tertiary/preprint/docs | {{claim}} | {{method}} | {{dataset}} | {{limitation}} | {{relevance}} | {{doi_or_url}} | {{checked}} |

## Fair-comparison rules

1. State the competing method's contribution in its own terms first.
2. Compare on identical data, horizon, and metric; if that is impossible,
   say why and label the comparison as indicative.
3. Attribute limitations to the source (with the quote in the sidecar), not
   to your reading of it.
4. Never cite a number you have not read in the source at the cited version.
5. Prophet/TimeSeers comparisons: only as *prophet-class baselines*, and only
   with the official packages actually run; the historical audit (F-14)
   found the old baselines were Vangja configurations.

## Position paragraph skeleton

> Prior work A shows `<claim, cited>` but assumes `<assumption>`. Work B
> relaxes `<assumption>` yet does not `<gap>`. Vangja differs in `<specific
> mechanism>`, which matters because `<consequence for the forecast>`. The
> closest prior work is `<key>`, which already does `<overlap>`; the
> contribution here is `<narrow, defensible delta>`.
