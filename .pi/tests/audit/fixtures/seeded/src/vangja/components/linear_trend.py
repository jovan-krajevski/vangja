"""Synthetic stand-in for the trend component (seeded defect D1)."""

import pymc as pm


class LinearTrend:
    """A trend with a transfer potential."""

    loss_factor = 1.0

    def transfer(self, slope_g, slope_transferred):
        # BUG (seeded): the wrong sign rewards deviation from the context slope.
        potential = +self.loss_factor * (slope_g - slope_transferred) ** 2
        return pm.Potential("transfer_trend", potential)
