"""Baseline runner (seeded defect D6)."""

from vangja import LinearTrend


def run_baseline(series):
    model = LinearTrend()
    # BUG (seeded): the label says Prophet but no Prophet is imported or run.
    return {"label": "Prophet", "model": model, "rel_mae": 0.42}
