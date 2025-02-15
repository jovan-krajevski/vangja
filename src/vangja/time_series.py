from pathlib import Path
from typing import Literal

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)


class TimeSeriesModel:
    def _scale_data(self):
        self.y_min = 0
        self.y_max = self.data["y"].abs().max()
        self.ds_min = self.data["ds"].min()
        self.ds_max = self.data["ds"].max()

        self.data["y"] = self.data["y"] / self.y_max
        self.data["t"] = (self.data["ds"] - self.ds_min) / (self.ds_max - self.ds_min)

    def _process_data(self):
        self.data["ds"] = pd.to_datetime(self.data["ds"])
        self.data.sort_values("ds", inplace=True)
        self._scale_data()

    def _get_model_initvals(self):
        i0, i1 = self.data["ds"].idxmin(), self.data["ds"].idxmax()
        T = self.data["t"].iloc[i1] - self.data["t"].iloc[i0]
        slope = (self.data["y"].iloc[i1] - self.data["y"].iloc[i0]) / T
        intercept = self.data["y"].iloc[i0] - slope * self.data["t"].iloc[i0]
        return {
            "slope": slope,
            "intercept": intercept,
            "delta": 0.0,
            "beta": 0.0,
            "sigma": 1.0,
        }

    def set_initval(self, initvals, model: pm.Model):
        model.set_initval(model.named_vars["sigma"], initvals.get("sigma", 1))
        self._set_initval(initvals, model)

    def fit(
        self,
        data,
        sigma_sd=0.5,
        mcmc_samples=0,
        chains=4,
        cores=4,
        use_prophet_initvals=True,
        progressbar=True,
        nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"] = "pymc",
    ):
        self.mcmc_samples = mcmc_samples

        self.data = data.reset_index(drop=True)
        self._process_data()

        self.initvals = {}
        if use_prophet_initvals:
            self.initvals = self._get_model_initvals()

        self.model = pm.Model()
        self.model_idxs = {}
        mu = self.definition(self.model, self.data, self.initvals, self.model_idxs)

        with self.model:
            sigma = pm.HalfNormal("sigma", sigma_sd)
            _ = pm.Normal("obs", mu=mu, sigma=sigma, observed=self.data["y"])

            self.set_initval(self.initvals, self.model)

            self.map_approx = None
            self.trace = None
            if self.mcmc_samples == 0:
                self.map_approx = pm.find_MAP(progressbar=progressbar, maxeval=1e4)
            else:
                self.trace = pm.sample(
                    self.mcmc_samples,
                    chains=chains,
                    cores=cores,
                    nuts_sampler=nuts_sampler,
                )

            self.fit_params = {"map_approx": self.map_approx, "trace": self.trace}
            self.tuned_model = None

    def load_trace(self, filepath: Path):
        if not hasattr(self, "fit_params"):
            self.fit_params = {"map_approx": None, "trace": None}

        self.fit_params["trace"] = az.from_netcdf(filepath)
        self.fit_params["map_approx"] = None
        self.tuned_model = None

    def tune(
        self,
        data,
        sigma_sd=0.5,
        mcmc_samples=0,
        chains=4,
        cores=4,
        use_prophet_initvals=True,
        progressbar=True,
        nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"] = "pymc",
    ):
        self.mcmc_samples = mcmc_samples

        self.data = data.reset_index(drop=True)
        self._process_data()

        self.initvals = {}
        if use_prophet_initvals:
            self.initvals = self._get_model_initvals()

        if self.tuned_model is None:
            model = pm.Model()
            self.model_idxs = {}
            mu = self._tune(
                model, self.data, self.initvals, self.model_idxs, self.fit_params
            )

            with model:
                observed = pm.Data("data", self.data["y"])
                sigma = pm.HalfNormal(
                    "sigma", sigma_sd, initval=self.initvals.get("sigma", 1)
                )
                _ = pm.Normal("obs", mu=mu, sigma=sigma, observed=observed)

            self.tuned_model = model

        self.model = self.tuned_model
        self.set_initval(self.initvals, self.model)
        with self.model:
            pm.set_data({"data": self.data["y"]})
            self.map_approx = None
            self.trace = None
            if self.mcmc_samples == 0:
                self.map_approx = pm.find_MAP(progressbar=progressbar, maxeval=1e4)
            else:
                self.trace = pm.sample(
                    self.mcmc_samples,
                    chains=chains,
                    cores=cores,
                    nuts_sampler=nuts_sampler,
                )

    def _make_future_df(self, days):
        future = pd.DataFrame(
            {
                "ds": pd.DatetimeIndex(
                    np.hstack(
                        (
                            self.data["ds"].unique().to_numpy(),
                            pd.date_range(
                                self.ds_max,
                                self.ds_max + pd.Timedelta(days, "D"),
                                inclusive="right",
                            ).to_numpy(),
                        )
                    )
                )
            }
        )
        future["t"] = (future["ds"] - self.ds_min) / (self.ds_max - self.ds_min)
        return future

    def predict(self, days):
        future = self._make_future_df(days)
        forecasts = self._predict(
            future, self.mcmc_samples, self.map_approx, self.trace
        )

        future["yhat"] = forecasts * self.y_max
        for model_type, model_cnt in self.model_idxs.items():
            if model_type.startswith("lt") is False:
                continue
            for model_idx in range(model_cnt):
                component = f"{model_type}_{model_idx}"
                if component in future.columns:
                    future[component] *= self.y_max

        return future

    def _predict(self, future, mcmc_samples, map_approx, trace):
        if mcmc_samples == 0:
            return self._predict_map(future, map_approx)

        return self._predict_mcmc(future, trace)

    def plot(self, future, y_true=None):
        plt.figure(figsize=(14, 100 * 6))
        plt.subplot(100, 1, 1)
        plt.title("Predictions")
        plt.grid()

        plt.scatter(
            self.data["ds"],
            self.data["y"] * self.y_max,
            s=0.5,
            color="C0",
            label="train y",
        )

        if y_true is not None:
            plt.scatter(y_true["ds"], y_true["y"], s=0.5, color="C1", label="y_true")

        plt.plot(future["ds"], future["yhat"], lw=1, label=r"$\widehat{y}$")

        plt.legend()
        plot_params = {"idx": 1}
        self._plot(plot_params, future, self.data, self.y_max, y_true)

    def metrics(self, y_true, future, label="y"):
        y = y_true["y"]
        yhat = future["yhat"][-len(y) :]
        return pd.DataFrame(
            {
                "mse": {f"{label}": mean_squared_error(y, yhat)},
                "rmse": {f"{label}": root_mean_squared_error(y, yhat)},
                "mae": {f"{label}": mean_absolute_error(y, yhat)},
                "mape": {f"{label}": mean_absolute_percentage_error(y, yhat)},
            }
        )

    def __add__(self, other):
        return AdditiveTimeSeries(self, other)

    def __pow__(self, other):
        return MultiplicativeTimeSeries(self, other)

    def __mul__(self, other):
        return SimpleMultiplicativeTimeSeries(self, other)


class AdditiveTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def definition(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.definition(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right.definition(*args, **kwargs)

        return left + right

    def _tune(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._tune(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._tune(*args, **kwargs)

        return left + right

    def _predict(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._predict(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._predict(*args, **kwargs)

        return left + right

    def _set_initval(self, *args, **kwargs):
        if not (type(self.left) is int or type(self.left) is float):
            self.left._set_initval(*args, **kwargs)

        if not (type(self.right) is int or type(self.right) is float):
            self.right._set_initval(*args, **kwargs)

    def _plot(self, *args, **kwargs):
        if not (type(self.left) is int or type(self.left) is float):
            self.left._plot(*args, **kwargs)

        if not (type(self.right) is int or type(self.right) is float):
            self.right._plot(*args, **kwargs)

    def __str__(self):
        return f"{self.left} + {self.right}"


class MultiplicativeTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def definition(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.definition(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right.definition(*args, **kwargs)

        return left * (1 + right)

    def _tune(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._tune(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._tune(*args, **kwargs)

        return left * (1 + right)

    def _predict(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._predict(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._predict(*args, **kwargs)

        return left * (1 + right)

    def _set_initval(self, *args, **kwargs):
        if not (type(self.left) is int or type(self.left) is float):
            self.left._set_initval(*args, **kwargs)

        if not (type(self.right) is int or type(self.right) is float):
            self.right._set_initval(*args, **kwargs)

    def _plot(self, *args, **kwargs):
        if not (type(self.left) is int or type(self.left) is float):
            self.left._plot(*args, **kwargs)

        if not (type(self.right) is int or type(self.right) is float):
            self.right._plot(*args, **kwargs)

    def __str__(self):
        left = f"{self.left}"
        if type(self.left) is AdditiveTimeSeries:
            left = f"({self.left})"

        return f"{left} * (1 + {self.right})"


class SimpleMultiplicativeTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def definition(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.definition(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right.definition(*args, **kwargs)

        return left * right

    def _tune(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._tune(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._tune(*args, **kwargs)

        return left * right

    def _predict(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._predict(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._predict(*args, **kwargs)

        return left * right

    def _set_initval(self, *args, **kwargs):
        if not (type(self.left) is int or type(self.left) is float):
            self.left._set_initval(*args, **kwargs)

        if not (type(self.right) is int or type(self.right) is float):
            self.right._set_initval(*args, **kwargs)

    def _plot(self, *args, **kwargs):
        if not (type(self.left) is int or type(self.left) is float):
            self.left._plot(*args, **kwargs)

        if not (type(self.right) is int or type(self.right) is float):
            self.right._plot(*args, **kwargs)

    def __str__(self):
        left = f"{self.left}"
        if type(self.left) is AdditiveTimeSeries:
            left = f"({self.left})"

        right = f"{self.right}"
        if type(self.right) is AdditiveTimeSeries:
            right = f"({self.right})"

        return f"{left} * {right}"
