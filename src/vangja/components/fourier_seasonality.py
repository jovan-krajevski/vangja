import math

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from vangja.time_series import TimeSeriesModel
from vangja.types import PoolType, TuneMethod
from vangja.utils import get_group_definition


class FourierSeasonality(TimeSeriesModel):
    model_idx: int | None = None

    def __init__(
        self,
        period: float,
        series_order: int,
        beta_mean: float = 0,
        beta_sd: float = 10,
        pool_type: PoolType = "partial",
        tune_method: TuneMethod | None = "parametric",
        override_beta_mean_for_tune: np.ndarray | None = None,
        override_beta_sd_for_tune: np.ndarray | None = None,
        shrinkage_strength: float = 1,
        shift_for_tune: bool = False,
        loss_factor_for_tune: float = 1,
    ):
        """
        Crate a Fourier Seasonality model component.

        Parameters
        ----------
        period: float
            The period of the seasonal effects.
        series_order: int
            Number of terms in the Fourier series.
        beta_mean: float
            The mean of the Normal prior for the Fourier series coefficients parameter.
        beta_sd: float
            The standard deviation of the Normal prior for the Fourier series
            coefficients parameter.
        pool_type: PoolType
            Type of pooling performed when sampling.
        tune_method: TuneMethod | None
            How the transfer learning is to be performed. One of "parametric" or
            "prior_from_idata". If set to None, this component will not be tuned even if
            idata is provided.
        override_beta_mean_for_tune: np.ndarray | None
            Override the mean of the Normal prior for the Fourier series coefficients
            parameter with this value.
        override_beta_sd_for_tune: np.ndarray | None
            Override the standard deviation of the Normal prior for the Fourier series
            coefficients parameter with this value.
        shrinkage_strength: float
            Shrinkage between groups for the hierarchical modeling.
        shift_for_tune: bool
            If true, a parameter determines how much the transfered posterior needs to
            be shifted along the time axis when using it as a prior.
        loss_factor_for_tune: float
            Regularization factor for transfer learning.
        """
        self.period = period
        self.series_order = series_order
        self.beta_mean = beta_mean
        self.beta_sd = beta_sd
        self.shrinkage_strength = shrinkage_strength
        self.pool_type = pool_type

        self.tune_method = tune_method
        self.override_beta_mean_for_tune = override_beta_mean_for_tune
        self.override_beta_sd_for_tune = override_beta_sd_for_tune
        self.shift_for_tune = shift_for_tune
        self.loss_factor_for_tune = loss_factor_for_tune

    def _fourier_series(self, data: pd.DataFrame, shift=None):
        # convert to days since epoch
        NANOSECONDS_TO_SECONDS = 1000 * 1000 * 1000
        t = (
            data["ds"].to_numpy(dtype=np.int64)
            // NANOSECONDS_TO_SECONDS
            / (3600 * 24.0)
        )
        if shift is not None:
            t += shift

        x_T = t * np.pi * 2
        fourier_components = np.empty((data["ds"].shape[0], 2 * self.series_order))
        if shift is not None and type(shift) is not np.ndarray:
            fourier_components = pt.as_tensor_variable(fourier_components)

        for i in range(self.series_order):
            c = x_T * (i + 1) / self.period
            if type(fourier_components) is np.ndarray:
                fourier_components[:, 2 * i] = np.sin(c)
                fourier_components[:, (2 * i) + 1] = np.cos(c)
            else:
                fourier_components = pt.set_subtensor(
                    fourier_components[:, 2 * i], np.sin(c)
                )
                fourier_components = pt.set_subtensor(
                    fourier_components[:, (2 * i) + 1], np.cos(c)
                )

        return fourier_components

    def _get_beta_params_from_idata(self, idata: az.InferenceData):
        """
        Calculate the mean and the standard deviation of the Normal prior for the
        Fourier series coefficients parameter from a provided posterior sample.

        Parameters
        ----------
        idata: az.InferenceData
            Sample from a posterior.
        """
        beta_key = f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"

        if self.override_beta_mean_for_tune is not None:
            beta_mean = self.override_beta_mean_for_tune
        else:
            beta_mean = idata["posterior"][beta_key].to_numpy().mean(axis=(1, 0))

        if self.override_beta_sd_for_tune is not None:
            beta_sd = self.override_beta_sd_for_tune
        else:
            beta_sd = idata["posterior"][beta_key].to_numpy().std(axis=(1, 0))

        return beta_mean, beta_sd

    def _complete_definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ):
        """
        Add the FourierSeasonality parameters to the model when pool_type is complete.

        Parameters
        ----------
        model: pm.Model
            The model to which the parameters are added.
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor), y
            (target) and series (name of time series).
        priors: dict[str, pt.TensorVariable] | None
            A dictionary of multivariate normal random variables approximating the
            posterior sample in idata.
        idata: az.InferenceData | None
            Sample from a posterior. If it is not None, Vangja will use this to set the
            parameters' priors in the model. If idata is not None, each component from
            the model should specify how idata should be used to set its parameters'
            priors.
        """
        with model:
            x = self._fourier_series(data)
            beta_key = (
                f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
            )

            if idata is not None and self.tune_method == "parametric":
                beta_mean, beta_sd = self._get_beta_params_from_idata(idata)
                beta = pm.Normal(
                    beta_key, beta_mean, beta_sd, shape=2 * self.series_order
                )
            elif priors is not None and self.tune_method == "prior_from_idata":
                beta = pm.Deterministic(beta_key, priors[f"prior_{beta_key}"])
            else:
                beta = pm.Normal(
                    beta_key, self.beta_mean, self.beta_sd, shape=2 * self.series_order
                )

            if idata is not None and self.tune_method is not None:
                reg_ds = pd.DataFrame(
                    {
                        "ds": pd.date_range(
                            "2000-01-01", periods=math.ceil(self.period), freq="D"
                        )
                    }
                )
                reg_x = self._fourier_series(reg_ds)
                old = pm.math.sum(reg_x * beta_mean, axis=1)
                new = pm.math.sum(reg_x * beta, axis=1)
                lam = (
                    2 * self.period / data.shape[0]
                    if self.period > 2 * data.shape[0]
                    else 0
                )
                pm.Potential(
                    f"{beta_key} - loss",
                    self.loss_factor_for_tune
                    * lam
                    * pm.math.minimum(0, pm.math.dot(old, old) - pm.math.dot(new, new)),
                )

            return pm.math.sum(x * beta, axis=1)

    def _partial_definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ):
        """
        Add the FourierSeasonality parameters to the model when pool_type is partial.

        Parameters
        ----------
        model: pm.Model
            The model to which the parameters are added.
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor), y
            (target) and series (name of time series).
        priors: dict[str, pt.TensorVariable] | None
            A dictionary of multivariate normal random variables approximating the
            posterior sample in idata.
        idata: az.InferenceData | None
            Sample from a posterior. If it is not None, Vangja will use this to set the
            parameters' priors in the model. If idata is not None, each component from
            the model should specify how idata should be used to set its parameters'
            priors.
        """
        with model:
            x = self._fourier_series(data)
            beta_key = (
                f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
            )

            beta_sd = self.beta_sd
            if idata is not None and self.tune_method == "parametric":
                beta_mean, beta_sd = self._get_beta_params_from_idata(idata)
                beta_shared = pm.Normal(
                    f"fs_{self.model_idx} - beta_shared",
                    beta_mean,
                    beta_sd,
                    shape=2 * self.series_order,
                )
            elif priors is not None and self.tune_method == "prior_from_idata":
                beta_mean, beta_sd = self._get_beta_params_from_idata(idata)
                beta_shared = pm.Deterministic(
                    f"fs_{self.model_idx} - beta_shared", priors[f"prior_{beta_key}"]
                )
            else:
                beta_shared = pm.Normal(
                    f"fs_{self.model_idx} - beta_shared",
                    self.beta_mean,
                    beta_sd,
                    shape=2 * self.series_order,
                )

            beta_sigma = pm.HalfNormal(
                f"fs_{self.model_idx} - beta_sigma(p={self.period},n={self.series_order})",
                sigma=beta_sd / self.shrinkage_strength,
                shape=2 * self.series_order,
            )
            beta_z_offset = pm.Normal(
                f"fs_{self.model_idx} - beta_z_offset(p={self.period},n={self.series_order})",
                mu=0,
                sigma=1,
                shape=(self.n_groups, 2 * self.series_order),
            )
            beta = pm.Deterministic(
                beta_key,
                beta_shared + beta_z_offset * beta_sigma,
            )

            if idata is not None and self.tune_method is not None:
                reg_ds = pd.DataFrame(
                    {
                        "ds": pd.date_range(
                            "2000-01-01", periods=math.ceil(self.period), freq="D"
                        )
                    }
                )
                reg_x = self._fourier_series(reg_ds)
                old = pm.math.sum(reg_x * beta_mean, axis=1)
                new = pm.math.dot(beta, reg_x.T)
                lam = np.array(
                    [
                        2 * data[self.group == group_code].shape[0]
                        if self.period > 2 * data[self.group == group_code].shape[0]
                        else 0
                        for group_code in self.groups_
                    ]
                )
                pm.Potential(
                    f"{beta_key} - loss",
                    self.loss_factor_for_tune
                    * pm.math.sum(
                        lam
                        * pm.math.minimum(
                            0, pm.math.dot(old, old) - pm.math.sum(new * new, axis=1)
                        )
                    ),
                )

            return pm.math.sum(x * beta[self.group], axis=1)

    def _individual_definition(
        self,
        model: pm.Model,
        data: pd.DataFrame,
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ):
        """
        Add the FourierSeasonality parameters to the model when pool_type is individual.

        Parameters
        ----------
        model: pm.Model
            The model to which the parameters are added.
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor), y
            (target) and series (name of time series).
        priors: dict[str, pt.TensorVariable] | None
            A dictionary of multivariate normal random variables approximating the
            posterior sample in idata.
        idata: az.InferenceData | None
            Sample from a posterior. If it is not None, Vangja will use this to set the
            parameters' priors in the model. If idata is not None, each component from
            the model should specify how idata should be used to set its parameters'
            priors.
        """
        with model:
            x = self._fourier_series(data)
            beta_key = (
                f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
            )

            if idata is not None and self.tune_method == "parametric":
                beta_mean, beta_sd = self._get_beta_params_from_idata(idata)
                beta = pm.Normal(
                    beta_key,
                    beta_mean,
                    beta_sd,
                    shape=(self.n_groups, 2 * self.series_order),
                )
            elif priors is not None and self.tune_method == "prior_from_idata":
                beta_mean, beta_sd = self._get_beta_params_from_idata(idata)
                # beta = pm.Deterministic(beta_key, priors[f"prior_{beta_key}"])
                # beta = pm.Deterministic(
                #     beta_key,
                #     pm.math.stack(
                #         [priors[f"prior_{beta_key}"] for _ in range(self.n_groups)]
                #     ),
                # )
                beta = pm.Deterministic(
                    beta_key, pt.tile(priors[f"prior_{beta_key}"], (self.n_groups, 1))
                )
            else:
                beta = pm.Normal(
                    beta_key,
                    self.beta_mean,
                    self.beta_sd,
                    shape=(self.n_groups, 2 * self.series_order),
                )

            if idata is not None and self.tune_method is not None:
                reg_ds = pd.DataFrame(
                    {
                        "ds": pd.date_range(
                            "2000-01-01", periods=math.ceil(self.period), freq="D"
                        )
                    }
                )
                reg_x = self._fourier_series(reg_ds)
                old = pm.math.sum(reg_x * beta_mean, axis=1)
                new = pm.math.dot(beta, reg_x.T)
                lam = np.array(
                    [
                        2 * data[self.group == group_code].shape[0]
                        if self.period > 2 * data[self.group == group_code].shape[0]
                        else 0
                        for group_code in self.groups_
                    ]
                )
                pm.Potential(
                    f"{beta_key} - loss",
                    self.loss_factor_for_tune
                    * pm.math.sum(
                        lam
                        * pm.math.minimum(
                            0, pm.math.dot(old, old) - pm.math.sum(new * new, axis=1)
                        )
                    ),
                )

            return pm.math.sum(x * beta[self.group], axis=1)

    def definition(
        self,
        model: TimeSeriesModel,
        data: pd.DataFrame,
        model_idxs: dict[str, int],
        priors: dict[str, pt.TensorVariable] | None,
        idata: az.InferenceData | None,
    ):
        """
        Add the FourierSeasonality parameters to the model.

        Parameters
        ----------
        model: TimeSeriesModel
            The model to which the parameters are added.
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor), y
            (target) and series (name of time series).
        model_idxs: dict[str, int]
            Count of the number of components from each type.
        priors: dict[str, pt.TensorVariable] | None
            A dictionary of multivariate normal random variables approximating the
            posterior sample in idata.
        idata: az.InferenceData | None
            Sample from a posterior. If it is not None, Vangja will use this to set the
            parameters' priors in the model. If idata is not None, each component from
            the model should specify how idata should be used to set its parameters'
            priors.
        """
        model_idxs["fs"] = model_idxs.get("fs", 0)
        self.model_idx = model_idxs["fs"]
        model_idxs["fs"] += 1

        self.group, self.n_groups, self.groups_ = get_group_definition(
            data, self.pool_type
        )

        with model:
            if self.pool_type == "complete":
                return self._complete_definition(model, data, priors, idata)
            elif self.pool_type == "partial":
                return self._partial_definition(model, data, priors, idata)
            elif self.pool_type == "individual":
                return self._individual_definition(model, data, priors, idata)

    def _get_initval(self, initvals: dict[str, float], model: pm.Model) -> dict:
        """Get the initval of the Fourier series coefficients parameters.

        Parameters
        ----------
        initvals: dict[str, float]
            Calculated initvals based on data.
        model: pm.Model
            The model for which the initvals will be set.
        """
        return {}

    def _det_seasonality_posterior(self, beta, x):
        return np.dot(x, beta.T)

    def _predict_map(self, future, map_approx):
        forecasts = []
        for group_code in self.groups_.keys():
            beta = map_approx[
                f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
            ]
            if self.pool_type != "complete":
                beta = beta[group_code]
            forecasts.append(
                self._det_seasonality_posterior(beta, self._fourier_series(future))
            )
            # breakpoint()
            future[f"fs_{self.model_idx}_{group_code}"] = forecasts[-1]

        return np.vstack(forecasts)

    def _predict_mcmc(self, future, trace, other_components):
        shift = trace["posterior"].get(f"fs_{self.model_idx} - shift", None)
        if shift is not None:
            shift = shift.mean()

        future[f"fs_{self.model_idx}"] = self._det_seasonality_posterior(
            trace["posterior"][
                f"fs_{self.model_idx} - beta(p={self.period},n={self.series_order})"
            ]
            .to_numpy()[:, :]
            .mean(0),
            self._fourier_series(future, shift),
        ).T.mean(0)

        return future[f"fs_{self.model_idx}"]

    def _plot(self, plot_params, future, data, scale_params, y_true=None, series=""):
        date = future["ds"] if self.period > 7 else future["ds"].dt.day_name()
        plot_params["idx"] += 1
        plt.subplot(100, 1, plot_params["idx"])
        plt.title(
            f"FourierSeasonality({self.model_idx},p={self.period},n={self.series_order})"
        )
        plt.grid()
        # Handle series parameter - use _0 as default for complete pooling
        if series == "":
            series_suffix = "_0"
        else:
            series_suffix = f"_{series}"
        plt.plot(
            date[-int(self.period) :],
            future[f"fs_{self.model_idx}{series_suffix}"][-int(self.period) :],
            lw=1,
        )

    def needs_priors(self, *args, **kwargs):
        return self.tune_method == "prior_from_idata"

    def __str__(self):
        return f"FS(p={self.period},n={self.series_order},tm={self.tune_method})"
