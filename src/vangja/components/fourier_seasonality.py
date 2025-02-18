from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from vangja.time_series import TimeSeriesModel


class FourierSeasonality(TimeSeriesModel):
    def __init__(
        self,
        period,
        series_order,
        beta_mean=0,
        beta_sd=10,
        allow_tune=False,
        tune_method: Literal["simple"] = "simple",
    ):
        self.period = period
        self.series_order = series_order
        self.beta_mean = beta_mean
        self.beta_sd = beta_sd

        self.allow_tune = allow_tune
        self.tune_method = tune_method

    def _fourier_series(self, data):
        # convert to days since epoch
        NANOSECONDS_TO_SECONDS = 1000 * 1000 * 1000
        t = (
            data["ds"].to_numpy(dtype=np.int64)
            // NANOSECONDS_TO_SECONDS
            / (3600 * 24.0)
        )

        x_T = t * np.pi * 2
        fourier_components = np.empty((data["ds"].shape[0], 2 * self.series_order))
        for i in range(self.series_order):
            c = x_T * (i + 1) / self.period
            fourier_components[:, 2 * i] = np.sin(c)
            fourier_components[:, (2 * i) + 1] = np.cos(c)

        return fourier_components

    def definition(self, model, data, initvals, model_idxs):
        model_idxs["fs"] = model_idxs.get("fs", 0)
        self.model_idx = model_idxs["fs"]
        model_idxs["fs"] += 1

        x = self._fourier_series(data)

        with model:
            beta = pm.Normal(
                f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})",
                mu=self.beta_mean,
                sigma=self.beta_sd,
                shape=2 * self.series_order,
            )

        return pm.math.sum(x * beta, axis=1)

    def _tune(self, model, data, initvals, model_idxs, prev):
        if not self.allow_tune:
            return self.definition(model, data, initvals, model_idxs)

        model_idxs["fs"] = model_idxs.get("fs", 0)
        self.model_idx = model_idxs["fs"]
        model_idxs["fs"] += 1

        x = self._fourier_series(data)

        with model:
            beta_key = (
                f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
            )
            beta_mu_key = f"{beta_key} - beta_mu"
            beta_sd_key = f"{beta_key} - beta_sd"
            if beta_mu_key not in prev:
                prev[beta_mu_key] = (
                    prev["map_approx"][beta_key]
                    if prev["trace"] is None
                    else prev["trace"]["posterior"][beta_key]
                    .to_numpy()
                    .mean(axis=(1, 0))
                )

            if beta_sd_key not in prev:
                prev[beta_sd_key] = (
                    self.beta_sd
                    if prev["trace"] is None
                    else prev["trace"]["posterior"][beta_key]
                    .to_numpy()
                    .std(axis=(1, 0))
                )

            if self.tune_method == "simple":
                beta = pm.Normal(
                    beta_key,
                    mu=pt.as_tensor_variable(prev[beta_mu_key]),
                    sigma=pt.as_tensor_variable(prev[beta_sd_key]),
                    shape=2 * self.series_order,
                )

        return pm.math.sum(x * beta, axis=1)

    def _get_initval(self, initvals, model: pm.Model):
        return {}

    def _det_seasonality_posterior(self, beta, x):
        return np.dot(x, beta.T)

    def _predict_map(self, future, map_approx):
        future[f"fs_{self.model_idx}"] = self._det_seasonality_posterior(
            map_approx[
                f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
            ],
            self._fourier_series(future),
        )

        return future[f"fs_{self.model_idx}"]

    def _predict_mcmc(self, future, trace):
        future[f"fs_{self.model_idx}"] = self._det_seasonality_posterior(
            trace["posterior"][
                f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
            ]
            .to_numpy()[:, :]
            .mean(0),
            self._fourier_series(future),
        ).T.mean(0)

        return future[f"fs_{self.model_idx}"]

    def _plot(self, plot_params, future, data, y_max, y_true=None):
        date = future["ds"] if self.period > 7 else future["ds"].dt.day_name()
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(
            f"FourierSeasonality({self.model_idx},p={self.period},n={self.series_order})"
        )
        plt.grid()
        plt.plot(
            date[-int(self.period) :],
            future[f"fs_{self.model_idx}"][-int(self.period) :],
            lw=1,
        )

    def __str__(self):
        return f"FS(p={self.period},n={self.series_order},at={self.allow_tune})"
