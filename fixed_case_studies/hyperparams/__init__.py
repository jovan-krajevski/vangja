"""Shared machinery for the Bayesian hyperparameter-selection workflow.

The workflow itself is documented in ``fixed_case_studies/HYPERPARAMETER_SELECTION.md``.
Every procedure here runs on **training windows only**; no test-horizon or
confirmation data is ever used (review FINDINGS F-1 / PROTOCOL.md §5).
"""

from fixed_case_studies.hyperparams import bayesian, candidates, full_bayes, robustness, scoring, stacking, ts_cv

__all__ = ["bayesian", "candidates", "full_bayes", "robustness", "scoring", "stacking", "ts_cv"]
