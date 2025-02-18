import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

from vangja.time_series import TimeSeriesModel


class LinearTrend(TimeSeriesModel):
    def __init__(
        self,
        n_changepoints=25,
        changepoint_range=0.8,
        slope_mean=0,
        slope_sd=5,
        intercept_mean=0,
        intercept_sd=5,
        delta_mean=0,
        delta_sd=0.05,
        allow_tune=False,
    ):
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.slope_mean = slope_mean
        self.slope_sd = slope_sd
        self.intercept_mean = intercept_mean
        self.intercept_sd = intercept_sd
        self.delta_mean = delta_mean
        self.delta_sd = delta_sd

        self.allow_tune = allow_tune

    def definition(self, model, data, initvals, model_idxs):
        model_idxs["lt"] = model_idxs.get("lt", 0)
        self.model_idx = model_idxs["lt"]
        model_idxs["lt"] += 1

        with model:
            t = np.array(data["t"])
            slope = pm.Normal(
                f"lt_{self.model_idx} - slope", self.slope_mean, self.slope_sd
            )
            intercept = pm.Normal(
                f"lt_{self.model_idx} - intercept",
                self.intercept_mean,
                self.intercept_sd,
            )

            if self.n_changepoints > 0:
                delta_sd = self.delta_sd
                if self.delta_sd is None:
                    delta_sd = pm.Exponential(f"lt_{self.model_idx} - tau", 1.5)

                delta = pm.Laplace(
                    f"lt_{self.model_idx} - delta",
                    self.delta_mean,
                    delta_sd,
                    shape=self.n_changepoints,
                )

                hist_size = int(np.floor(data.shape[0] * self.changepoint_range))
                cp_indexes = (
                    np.linspace(0, hist_size - 1, self.n_changepoints + 1)
                    .round()
                    .astype(int)
                )
                self.s = np.array(data.iloc[cp_indexes]["t"].tail(-1))
                A = (t[:, None] > self.s) * 1

                gamma = -self.s * delta
                trend = (slope + pm.math.sum(A * delta, axis=1)) * t + (
                    intercept + pm.math.sum(A * gamma, axis=1)
                )
            else:
                trend = slope * t + intercept

        return trend

    def _tune(self, model, data, initvals, model_idxs, prev):
        return self.definition(model, data, initvals, model_idxs)

    def _get_initval(self, initvals, model: pm.Model):
        slope_initval = initvals.get("slope", None)
        intercept_initval = initvals.get("intercept", None)
        delta_initval = initvals.get("delta", None)
        if delta_initval is not None:
            delta_initval = np.array([delta_initval] * self.n_changepoints)

        initval_dict = {
            model.named_vars[f"lt_{self.model_idx} - slope"]: slope_initval,
            model.named_vars[f"lt_{self.model_idx} - intercept"]: intercept_initval,
        }

        if f"lt_{self.model_idx} - delta" in model.named_vars:
            initval_dict[model.named_vars[f"lt_{self.model_idx} - delta"]] = (
                delta_initval
            )

        return initval_dict

    def _predict_map(self, future, map_approx):
        if f"lt_{self.model_idx} - delta" not in map_approx:
            future[f"lt_{self.model_idx}"] = np.array(
                map_approx[f"lt_{self.model_idx} - slope"] * future["t"]
                + map_approx[f"lt_{self.model_idx} - intercept"]
            )
        else:
            new_A = (np.array(future["t"])[:, None] > self.s) * 1

            future[f"lt_{self.model_idx}"] = np.array(
                (
                    map_approx[f"lt_{self.model_idx} - slope"]
                    + np.dot(new_A, map_approx[f"lt_{self.model_idx} - delta"])
                )
                * future["t"]
                + (
                    map_approx[f"lt_{self.model_idx} - intercept"]
                    + np.dot(
                        new_A, (-self.s * map_approx[f"lt_{self.model_idx} - delta"])
                    )
                )
            )

        return future[f"lt_{self.model_idx}"]

    def _predict_mcmc(self, future, trace):
        slope = (
            trace["posterior"][f"lt_{self.model_idx} - slope"].to_numpy()[:, :].mean(0)
        )
        intercept = (
            trace["posterior"][f"lt_{self.model_idx} - intercept"]
            .to_numpy()[:, :]
            .mean(0)
        )

        if f"lt_{self.model_idx} - delta" not in trace["posterior"]:
            future[f"lt_{self.model_idx}"] = (
                slope.mean() * future["t"].to_numpy() + intercept.mean()
            )
        else:
            new_A = (np.array(future["t"])[:, None] > self.s) * 1
            delta = (
                trace["posterior"][f"lt_{self.model_idx} - delta"]
                .to_numpy()[:, :]
                .mean(0)
            )

            future[f"lt_{self.model_idx}"] = (
                (slope + np.dot(new_A, delta.T)).T * future["t"].to_numpy()
                + (intercept + np.dot(new_A, (-self.s * delta).T)).T
            ).mean(0)

        return future[f"lt_{self.model_idx}"]

    def _plot(self, plot_params, future, data, y_max, y_true=None):
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(f"LinearTrend({self.model_idx})")
        plt.grid()

        plt.plot(future["ds"], future[f"lt_{self.model_idx}"], lw=1)

        plt.legend()

    def __str__(self):
        return f"LT(n={self.n_changepoints},r={self.changepoint_range},at={self.allow_tune})"
