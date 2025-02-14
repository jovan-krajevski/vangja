import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from vangja_simple.time_series import TimeSeriesModel


class BetaConstant(TimeSeriesModel):
    def __init__(
        self,
        lower,
        upper,
        alpha=0.5,
        beta=0.5,
        allow_tune=False,
    ):
        self.lower = lower
        self.upper = upper
        self.alpha = alpha
        self.beta = beta
        self.allow_tune = allow_tune

    def definition(self, model, data, initvals, model_idxs):
        model_idxs["c"] = model_idxs.get("c", 0)
        self.model_idx = model_idxs["c"]
        model_idxs["c"] += 1

        with model:
            beta = pm.Beta(
                f"bc_{self.model_idx} - beta(alpha={self.alpha},beta={self.beta})",
                alpha=self.alpha,
                beta=self.beta,
            )
            c = pm.Deterministic(
                f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})",
                beta * (self.upper - self.lower) + self.lower,
            )

        return c

    def _tune(self, model, data, initvals, model_idxs, prev):
        return self.definition(model, data, initvals, model_idxs)

    def _set_initval(self, initvals, model: pm.Model):
        pass

    def _predict_map(self, future, map_approx):
        future[f"bc_{self.model_idx}"] = (
            np.ones_like(future["t"])
            * map_approx[f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})"]
        )
        return future[f"bc_{self.model_idx}"]

    def _predict_mcmc(self, future, trace):
        future[f"bc_{self.model_idx}"] = (
            np.ones_like(future["t"])
            * trace["posterior"][
                f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})"
            ]
            .to_numpy()[:, :]
            .mean()
        )

        return future[f"bc_{self.model_idx}"]

    def _plot(self, plot_params, future, data, y_max, y_true=None):
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"BetaConstant({self.model_idx},l={self.lower},u={self.upper})")
        plt.bar(0, future[f"bc_{self.model_idx}"][0])
        plt.axhline(0, c="k", linewidth=3)

    def __str__(self):
        return f"BC(alpha={self.alpha},beta={self.beta},l={self.lower},u={self.upper},at={self.allow_tune})"
