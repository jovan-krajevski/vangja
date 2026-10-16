"""Fixture: per-group shape-contract violations (must be flagged)."""

import numpy as np


class BadTrend:
    def __init__(self, pool_type="partial", n_groups=1):
        self.pool_type = pool_type
        self.n_groups = n_groups

    def _get_initval(self, initvals, model):
        result = {}
        result["lt_0 - slope"] = 0.5            # scalar for a vector RV
        if self.pool_type == "complete":
            result["lt_0 - intercept"] = initvals[0]   # guarded: fine
        else:
            result["lt_0 - intercept"] = initvals[0]   # unguarded scalar
        return result

    def _predict_map(self, future, map_approx):
        predictions = np.zeros((len(future), 1))
        self._predict_columns = {}
        self._predict_columns["lt_0_0"] = predictions[:, 0]  # 2-D slice
        self._predict_columns["lt_0_1"] = predictions[-1]    # fine
        return predictions
