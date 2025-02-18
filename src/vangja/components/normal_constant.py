import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from vangja.time_series import TimeSeriesModel


class NormalConstant(TimeSeriesModel):
    def __init__(
        self,
        mu=0,
        sd=1,
        allow_tune=False,
    ):
        self.mu = mu
        self.sd = sd
        self.allow_tune = allow_tune

    def definition(self, model, data, initvals, model_idxs):
        model_idxs["c"] = model_idxs.get("c", 0)
        self.model_idx = model_idxs["c"]
        model_idxs["c"] += 1

        with model:
            c = pm.Normal(
                f"nc_{self.model_idx} - normal(mu={self.mu},sd={self.sd})",
                mu=self.mu,
                sigma=self.sd,
            )

        return c

    def _tune(self, model, data, initvals, model_idxs, prev):
        return self.definition(model, data, initvals, model_idxs)

    def _get_initval(self, initvals, model: pm.Model):
        return {}

    def _predict_map(self, future, map_approx):
        future[f"nc_{self.model_idx}"] = (
            np.ones_like(future["t"])
            * map_approx[f"nc_{self.model_idx} - normal(mu={self.mu},sd={self.sd})"]
        )
        return future[f"nc_{self.model_idx}"]

    def _predict_mcmc(self, future, trace):
        future[f"nc_{self.model_idx}"] = (
            np.ones_like(future["t"])
            * trace["posterior"][
                f"nc_{self.model_idx} - normal(mu={self.mu},sd={self.sd})"
            ]
            .to_numpy()[:, :]
            .mean()
        )

        return future[f"nc_{self.model_idx}"]

    def _plot(self, plot_params, future, data, y_max, y_true=None):
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"NormalConstant({self.model_idx},mu={self.mu},sd={self.sd})")
        plt.bar(0, future[f"nc_{self.model_idx}"][0])
        plt.axhline(0, c="k", linewidth=3)

    def __str__(self):
        return f"NC(mu={self.mu},sd={self.sd},at={self.allow_tune})"
