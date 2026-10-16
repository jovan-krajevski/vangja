"""Seeded fixture: metric reassignment + denominator clipping."""
import numpy as np

REL_MAE_EPSILON = 1e-6


def relative_mae(mae, persistence_mae, epsilon=REL_MAE_EPSILON):
    denom = np.clip(persistence_mae, 1e-6, None)
    return mae / denom
