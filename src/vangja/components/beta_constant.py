import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from vangja.time_series import TimeSeriesModel
from vangja.utils import get_group_definition


class BetaConstant(TimeSeriesModel):
    def __init__(
        self,
        lower,
        upper,
        alpha=0.5,
        beta=0.5,
        pool_cols=None,
        pool_type="complete",
        allow_tune=False,
    ):
        self.lower = lower
        self.upper = upper
        self.alpha = alpha
        self.beta = beta

        self.pool_cols = pool_cols
        self.pool_type = pool_type
        self.allow_tune = allow_tune

    def definition(self, model, data, initvals, model_idxs):
        model_idxs["c"] = model_idxs.get("c", 0)
        self.model_idx = model_idxs["c"]
        model_idxs["c"] += 1

        group, n_groups, self.groups_ = get_group_definition(
            data, self.pool_cols, self.pool_type
        )

        with model:
            if self.pool_type == "partial":
                mu_beta = pm.Beta(
                    f"bc_{self.model_idx} - mu_beta(alpha={self.alpha},beta={self.beta})",
                    alpha=self.alpha,
                    beta=self.beta,
                    shape=n_groups,
                )
                offset_beta = pm.Normal(
                    f"bc_{self.model_idx} - offset_beta(alpha={self.alpha},beta={self.beta})",
                    mu=0,
                    sigma=1,
                    shape=n_groups,
                )
                beta = pm.Deterministic(
                    f"bc_{self.model_idx} - beta(alpha={self.alpha},beta={self.beta})",
                    mu_beta + offset_beta,
                )
                c = pm.Deterministic(
                    f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})",
                    beta * (self.upper - self.lower) + self.lower,
                )
            else:
                beta = pm.Beta(
                    f"bc_{self.model_idx} - beta(alpha={self.alpha},beta={self.beta})",
                    alpha=self.alpha,
                    beta=self.beta,
                    shape=n_groups,
                )
                c = pm.Deterministic(
                    f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})",
                    beta * (self.upper - self.lower) + self.lower,
                )

        return c[group]

    def _tune(self, model, data, initvals, model_idxs, prev):
        return self.definition(model, data, initvals, model_idxs)

    def _predict_map(self, future, map_approx):
        forecasts = []
        for group_code in self.groups_.keys():
            forecasts.append(
                np.ones_like(future["t"])
                * map_approx[f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})"][
                    group_code
                ]
            )
            future[f"bc_{self.model_idx}_{group_code}"] = forecasts[-1]

        return np.vstack(forecasts)

    def _predict_mcmc(self, future, trace):
        forecasts = []
        for group_code in self.groups_.keys():
            forecasts.append(
                np.ones_like(future["t"])
                * trace["posterior"][
                    f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})"
                ]
                .to_numpy()[:, :, group_code]
                .mean()
            )
            future[f"bc_{self.model_idx}_{group_code}"] = forecasts[-1]

        return np.vstack(forecasts)

    def _plot(self, plot_params, future, data, y_max, y_true=None):
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"bc_{self.model_idx} - c(l={self.lower},u={self.upper})")

        plot_data = []
        for group_code, group_name in self.groups_.items():
            plot_data.append(
                (group_name, future[f"bc_{self.model_idx}_{group_code}"][0])
            )

        plt.bar(*zip(*plot_data))
        plt.axhline(0, c="k", linewidth=3)

    def __str__(self):
        return f"BC(alpha={self.alpha},beta={self.beta},l={self.lower},u={self.upper},at={self.allow_tune},{self.pool_type})"
