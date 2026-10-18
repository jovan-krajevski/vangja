# House style — structure, notation, tables, claims

`paper/paper.tex` is the manuscript (`amsart`, `runningheads, reqno`). The
six result tables under `paper/case_studies/tables/` are `\input` by
`paper/paper.tex:673-719`. This file fixes the conventions so revisions do
not drift; it does not restate the mathematics, which lives in the paper.

## Structure

| Part | Section / label convention |
|---|---|
| Abstract | no citations, no config names, framing + sign + rough magnitude only |
| Formal definition | labels `ch:formal_def`, `timeseries_def` |
| Extending Prophet | `ch_main`, `subch_hierarchical`, `subch_prior_from_posterior` |
| Methodology | `ch_methodology` (datasets, task/evaluation, specifications, baselines) |
| Results | `ch_results`, `subch_best` (accuracy), `subch_ablation` |
| Closing | `ch_conclusion`, Limitations, Future work |

Labels: `tab:…`, `fig:…`, `ch…`/`subch…`/`subsubch…`. Refer with `\ref`.
Use `\begin{table}` immediately around `\input` of a generated table; do not
edit generated table bodies by hand.

## Notation (keep consistent)

- Sets: `T` equally spaced, spacing `\Delta t`; `T_C` context span,
  `T_X` target span (`\max(T_X) = \max(T_C)`), `T_F` forecast horizon.
- Indices: `i` time series (`i = 1` is the context), `j` period.
- Trend: `g`, changepoint indicator matrix `A`, slope `w` (or `\mathbf{w}`),
  changepoint deltas `\boldsymbol{\delta}`, intercept `b`.
- Seasonality: `s^{(j)}` with Fourier coefficients `\boldsymbol{\beta}`.
- Model form: `y_i = g_i (1 + \sum_j s_i^{(j)}) + \varepsilon` (Prophet-style;
  the `**` operator). Say "multiplicative" only for this form.
- Pooling: **complete** (shared), **partial** (hierarchical, shrinkage `\lambda`),
  **individual** (free). Never call partial "random effects" without saying
  the parameterisation (non-centred, `HalfCauchy(\sigma/\lambda)`).
- Transfer: `parametric` = posterior moment matching (say *mean and standard
  deviation*, never MAP/mode — the code uses `.mean()`; audit WS-H-002);
  `prior_from_idata` = MVN approximation of the joint posterior (says
  nothing about a mean-field ADVI run).
- Changepoint side: `delta_side="right"` means the slope is the
  end-of-history slope; `"left"` is the Prophet default. State which one the
  experiment uses.

## Tables and numbers

- Cells are generated; the include is committed, the generator is committed,
  the artifact is committed (or hashed). Formatting: 4 decimals for MAPE,
  2 for RMSE/MAE in the legacy tables, bold only the best cell in a column,
  no trailing zeros invented for alignment.
- Percentages: relative differences, one decimal, and name the reference
  (`+1168.8% vs the frozen reference`) — "absolute reduction of 78.2%" was a
  real error (WS-H-016): it is a *relative* reduction.
- Ranges must be checked against the table before writing ("8.0% to 25.9%",
  not "8.5% to 25.9%"; WS-H-010).
- Every table caption states the arm, the metric, the unit of aggregation
  (median over target×origin), and the data split (dev or confirmation).
- Smart-home numbers are labelled **retrospective/illustrative** wherever
  they appear until independent confirmation data exists.

## Claims and generated includes

- Numbers appear only inside generated includes; the generator writes
  `CLAIMS.json` alongside them (or is checked against it).
- Each claims entry: `{id, value, tex, artifact, artifact_sha256, commit,
  status}`; `status` is `verified` / `pending` / `stale`.
- Prose cites `[C-###]` immediately after the number; a paragraph with no
  claim ID must contain no numbers.
- `check_claims.py` is the gate: run it in report mode while drafting, in
  `--gate` mode before handing the PR to review.

## Statements that must be present before submission

- **Reproducibility:** the exact commands, environment fingerprint
  (`freeze_report.py`), data hashes, and where artifacts live.
- **AI use:** which model/agent wrote or checked what, and that every number
  was recomputed from artifacts by a deterministic script.
- **Limitations:** use `templates/limitations.md`. In particular: no
  calibration claim for MAP intervals, no covariance claim from ADVI,
  no "bit-for-bit reproduction" of provider data (sidecars freeze the *used*
  data), the S&P-overlap/survivorship disclosure, the horizon framing, and
  the retrospective status of smart home.
- **Data/availability:** failure counts and excluded units reconcile with
  the reported N; context series never enter target aggregates.

## Things the current drafts get wrong (do not repeat)

From `Vangja-project/audit/stage-2/ws-h-docs-hygiene/FINDINGS.md`:
parametric ≠ MAP; the two smart-home tables come from different config
selections (per-series test-best vs global reference) and must be labelled or
replaced; stocks `series_order` is 6 in the old code, the paper said 5; the
`load_nyc_temperature` docstring is wrong (mean/Kelvin, not max/Fahrenheit);
the baseline improvement range was misattributed; there is no bibliography
file in `paper/` at this revision (`\bibliography{my_paper}` refers to a
missing `.bib`) — a literature task must restore it, not invent citations.
