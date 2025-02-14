import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from vangja_simple.time_series import TimeSeriesModel


class Constant(TimeSeriesModel):
    def __init__(self, lower, upper, allow_tune=False):
        self.lower = lower
        self.upper = upper
        self.allow_tune = allow_tune

    def definition(self, model, data, initvals, model_idxs):
        model_idxs["c"] = model_idxs.get("c", 0)
        self.model_idx = model_idxs["c"]
        model_idxs["c"] += 1

        with model:
            c = pm.Uniform(
                f"c_{self.model_idx} - c(l={self.lower},u={self.upper})",
                lower=self.lower,
                upper=self.upper,
            )

        return c

    def _tune(self, model, data, initvals, model_idxs, prev):
        return self.definition(model, data, initvals, model_idxs)

    def _set_initval(self, initvals, model: pm.Model):
        pass

    def _predict_map(self, future, map_approx):
        future[f"c_{self.model_idx}"] = (
            np.ones_like(future["t"])
            * map_approx[f"c_{self.model_idx} - c(l={self.lower},u={self.upper})"]
        )

        return future[f"c_{self.model_idx}"]

    def _predict_mcmc(self, future, trace):
        future[f"c_{self.model_idx}"] = (
            np.ones_like(future["t"])
            * trace["posterior"][
                f"c_{self.model_idx} - c(l={self.lower},u={self.upper})"
            ]
            .to_numpy()[:, :]
            .mean()
        )

        return future[f"c_{self.model_idx}"]

    def _plot(self, plot_params, future, data, y_max, y_true=None):
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"Constant({self.model_idx},l={self.lower},u={self.upper})")
        plt.bar(0, future[f"c_{self.model_idx}"][0])
        plt.axhline(0, c="k", linewidth=3)

    def __str__(self):
        return f"C(l={self.lower},u={self.upper},at={self.allow_tune})"
