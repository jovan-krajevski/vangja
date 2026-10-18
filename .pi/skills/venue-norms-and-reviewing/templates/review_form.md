# Review form

```yaml
pr_id: PR-###
head_sha: <40-hex>            # must match the PR record; a different SHA invalidates the review
reviewer: <role>             # author != reviewer
decision: accept | changes_requested | reject
grade: strong reject | reject | weak reject | borderline | weak accept | accept | strong accept
```

## Evidence actually run

```
<exact commands + observed output summary; one per acceptance criterion>
```

## Acceptance criteria

| # | Criterion | Verdict (pass/fail/na) | Evidence pointer |
|---|---|---|---|
| 1 | | | |

## Rubric (score 0–4 each, with an evidence pointer)

| Criterion | Score | Pointer |
|---|---|---|
| Novelty/contribution | | |
| Correctness | | |
| Evidence/experiments | | |
| Reproducibility | | |
| Clarity | | |
| Significance | | |

## Blocking-rule trace

```
confirmed S0: none | <IDs>
confirmed S1 (M/S/P/C): none | <IDs>
unfixed S1 elsewhere: none | <IDs>
failed gates: none | <names + inapplicability reason>
=> grade cap: <grade>
```

## Required changes (fix list)

| Finding | Owner | Fix | Effort | Feasibility |
|---|---|---|---|---|

## Residual risk

<what remains uncertain after this review, and what would resolve it>

## Decision rationale

<one paragraph: why this grade, what would raise it>
