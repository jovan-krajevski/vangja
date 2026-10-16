# Target venues — norms and what each one rewards

The project default is **Journal of Forecasting**
(`.pi/config/project.json` `venue`). Choose per contribution type; the
framing changes more than the formatting.

## Journal of Forecasting

- **Wants:** an applied forecasting problem, an honest evaluation protocol,
  practical takeaway; incremental-but-solid method papers are welcome.
- **Framing:** lead with the forecasting task and the metric (relative MAE
  vs persistence, horizon multiples), not with the package API.
- **Watch:** baselines must be real packages; uncertainty should be
  addressed; long-horizon claims need the horizon ratio stated per study.
- **Fit for Vangja:** yes for the transfer-learning + pooling evaluation.

## International Journal of Forecasting

- **Wants:** rigorous comparison, dependence-aware inference, multiple
  comparison control, reproducible protocols; strong statistics reviewing.
- **Framing:** state the protocol (frozen configs, dev/confirmation split)
  up front; report failure/exclusion accounting and the paired analysis.
- **Watch:** the block-bootstrap caveat (few independent origin blocks) must
  be disclosed; no "significance" language without it.
- **Fit:** yes, if the protocol is foregrounded and the claims are narrow.

## Machine Learning

- **Wants:** methodological novelty, theory (identifiability, priors,
  estimator properties), and experiments that isolate the mechanism.
- **Framing:** contribution = the transfer parameterisation and the
  regularisation potentials; the package is incidental.
- **Watch:** "we released a Python package" is not a contribution; weak
  baselines and no theory will be desk-rejected.
- **Fit:** only after the methods story is separated from the software story.

## Journal of Statistical Software

- **Wants:** a stable, documented API; reproducible installation; worked
  examples; comparison to alternatives; attention to numerical correctness.
- **Framing:** software paper — architecture, API, component catalogue,
  examples, computational details, version provenance.
- **Watch:** code must be public and versioned; vignettes/examples must run;
  "bit-for-bit reproducibility" claims are scrutinised (and were found
  unsupported historically).
- **Fit:** as a second, companion paper (the historical `jstat.tex` draft),
  not as the venue for the methods result.

## Desk-reject triggers (any venue)

- test-horizon selection presented as a frozen configuration;
- baselines that are not the packages they are named after (F-14);
- numbers with no artifact, or artifacts whose SHA is not an ancestor;
- missing bibliography entries or unverifiable citations;
- no limitations / AI-use / data-availability statement;
- a headline horizon/ratio that holds for one study only and is generalised.

## Cover-letter checklist

- [ ] Contribution type matches the venue's list.
- [ ] Protocol (frozen config, dev/confirmation, exclusions) stated in one
      paragraph with artifact locations.
- [ ] Baselines named with versions and packages actually run.
- [ ] Limitations paragraph written (`templates/limitations.md`).
- [ ] AI-use and reproducibility statements attached.
- [ ] Companion papers disclosed (avoid duplicate-submission confusion).
