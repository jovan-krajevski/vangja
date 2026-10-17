"""Negative control: a real Prophet baseline must NOT be reported as mislabelled."""

from prophet import Prophet


def run_baseline(series):
    return {"label": "Prophet", "model": Prophet()}
