import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pymc_extras as pmx

from vangja.types import (
    FreqStr,
    Method,
    NutsSampler,
    PoolType,
    ScaleMode,
    Scaler,
    TScaleParams,
    YScaleParams,
)
from vangja.utils import get_group_definition


class TimeSeriesModel:
    data: pd.DataFrame
    y_scale_params: YScaleParams | dict[int, YScaleParams]
    t_scale_params: TScaleParams

    group: np.ndarray
    n_groups: int
    groups_: dict[int, str]

    model: pm.Model
    model_idxs: dict[str, int]
    samples: int
    method: Method
    initvals: dict[str, float]
    map_approx: dict[str, np.ndarray] | None
    trace: az.InferenceData | None

    def _get_scale_params(
        self, series: pd.DataFrame, scaler: Scaler, t_scale_params: TScaleParams | None
    ) -> tuple[TScaleParams, YScaleParams]:
        return (
            (
                t_scale_params
                if t_scale_params is not None
                else {
                    "ds_min": series["ds"].min(),
                    "ds_max": series["ds"].max(),
                }
            ),
            {
                "scaler": scaler,
                "y_min": 0 if scaler == "maxabs" else series["y"].min(),
                "y_max": series["y"].abs().max()
                if scaler == "maxabs"
                else series["y"].max(),
            },
        )

    def _process_data(
        self,
        data: pd.DataFrame,
        scaler: Scaler,
        scale_mode: ScaleMode,
        t_scale_params: TScaleParams | None,
    ) -> None:
        """Converts dataframe to correct format and scale dates and values.

        Parameters
        ----------
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor), y
            (target) and series (name of time series).
        scaler: Scaler
            Whether to use maxabs or minmax scaling of the y (target).
        scale_mode: ScaleMode
            Whether to scale each series individually or together.
        t_scale_params: TScaleParams | None
            Whether to override scale parameters for ds (predictor).
        """
        self.data = data.reset_index(drop=True)
        self.data["ds"] = pd.to_datetime(self.data["ds"])
        self.data["t"] = 0.0
        self.data.sort_values("ds", inplace=True)

        self.group, self.n_groups, self.groups_ = get_group_definition(
            self.data, "partial"
        )

        if scale_mode == "individual":
            self.t_scale_params, _ = self._get_scale_params(
                self.data, scaler, t_scale_params
            )
            self.y_scale_params = {}

            for group_code, group_name in self.groups_.items():
                _, y_params = self._get_scale_params(
                    self.data[self.data["series"] == group_name], scaler, t_scale_params
                )

                self.data.loc[self.data["series"] == group_name, "t"] = (
                    self.data.loc[self.data["series"] == group_name, "ds"]
                    - self.t_scale_params["ds_min"]
                ) / (self.t_scale_params["ds_max"] - self.t_scale_params["ds_min"])
                self.data.loc[self.data["series"] == group_name, "y"] = (
                    self.data.loc[self.data["series"] == group_name, "y"]
                    - y_params["y_min"]
                ) / (y_params["y_max"] - y_params["y_min"])

                self.y_scale_params[group_code] = y_params

            return

        self.t_scale_params, self.y_scale_params = self._get_scale_params(
            self.data, scaler, t_scale_params
        )
        self.data["t"] = (self.data["ds"] - self.t_scale_params["ds_min"]) / (
            self.t_scale_params["ds_max"] - self.t_scale_params["ds_min"]
        )
        self.data["y"] = (self.data["y"] - self.y_scale_params["y_min"]) / (
            self.y_scale_params["y_max"] - self.y_scale_params["y_min"]
        )

    def _get_model_initvals(self) -> dict[str, float]:
        """Calculate initvals based on data."""
        initvals: dict[str, float] = {"sigma": 1.0}
        for key in self.groups_.keys():
            series: pd.DataFrame = self.data[self.group == key]
            i0, i1 = series["ds"].idxmin(), series["ds"].idxmax()
            T = series["t"].loc[i1] - series["t"].loc[i0]
            slope = (series["y"].loc[i1] - series["y"].loc[i0]) / T
            intercept = series["y"].loc[i0] - slope * series["t"].loc[i0]
            initvals[f"slope_{key}"] = slope
            initvals[f"intercept_{key}"] = intercept

        return initvals

    def get_initval(self, initvals: dict[str, float], model: pm.Model) -> dict:
        """Get the initval of the standard deviation of the Normal prior of y (target).

        Parameters
        ----------
        initvals: dict[str, float]
            Calculated initvals based on data.
        model: pm.Model
            The model for which the initvals will be set.
        """
        return {
            model.named_vars["sigma"]: initvals.get("sigma", 1),
            **self._get_initval(initvals, model),
        }

    def fit(
        self,
        data: pd.DataFrame,
        scaler: Scaler = "maxabs",
        scale_mode: ScaleMode = "individual",
        t_scale_params: TScaleParams | None = None,
        sigma_sd: float = 0.5,
        sigma_pool_type: PoolType = "complete",
        sigma_shrinkage_strength: float = 1,
        method: Method = "mapx",
        samples: int = 0,
        chains: int = 4,
        cores: int = 4,
        nuts_sampler: NutsSampler = "pymc",
        progressbar: bool = True,
        idata: az.InferenceData | None = None,
    ):
        """
        Create and fit the model to the data.

        Parameters
        ----------
        data : pd.DataFrame
            A pandas dataframe that must at least have columns ds (predictor), y
            (target) and series (name of time series).
        scaler: Scaler
            Whether to use maxabs or minmax scaling of the y (target).
        scale_mode: ScaleMode
            Whether to scale each series individually or together.
        t_scale_params: TScaleParams | None
            Whether to override scale parameters for ds (predictor).
        sigma_sd: float
            The standard deviation of the Normal prior of y (target).
        sigma_pool_type: PoolType
            Type of pooling for the sigma parameter that is performed when sampling.
        sigma_shrinkage_strength: float
            Shrinkage between groups for the hierarchical modeling.
        method: Method
            The Bayesian inference method to be used. Either a point estimate MAP), a
            VI method (advi etc.) or full Bayesian sampling (MCMC).
        samples: int
            Denotes the number of samples to be drawn from the posterior for MCMC and
            VI methods.
        chains: int
            Denotes the number of independent chains drawn from the posterior. Only
            applicable to the MCMC methods.
        nuts_sampler: NutsSampler
            The sampler for the NUTS method.
        progressbar: bool
            Whether to show a progressbar while fitting the model.
        idata: az.InferenceData | None
            Sample from a posterior. If it is not None, Vangja will use this to set the
            parameters' priors in the model. If idata is not None, each component from
            the model should specify how idata should be used to set its parameters'
            priors.
        """
        self._process_data(data, scaler, scale_mode, t_scale_params)

        self.model = pm.Model()
        self.model_idxs = {}
        self.samples = samples
        self.method = method

        with self.model:
            priors = None
            if idata is not None and self.needs_priors():
                priors = pmx.utils.prior.prior_from_idata(
                    idata,
                    name="priors",
                    # add a "prior_" prefix to vars
                    **{
                        f"{var}": f"prior_{var}" for var in idata["posterior"].data_vars
                    },
                )

            mu = self.definition(self.model, self.data, self.model_idxs, priors, idata)
            if sigma_pool_type == "partial":
                sigma_sigma = pm.HalfCauchy(
                    "sigma_sigma", sigma_sd / sigma_shrinkage_strength
                )
                sigma_offset = pm.HalfNormal("sigma_offset", 1, shape=self.n_groups)
                sigma = pm.Deterministic("sigma", sigma_offset * sigma_sigma)
                _ = pm.Normal(
                    "obs", mu=mu, sigma=sigma[self.group], observed=self.data["y"]
                )
            elif sigma_pool_type == "individual":
                sigma = pm.HalfNormal("sigma", sigma_sd, shape=self.n_groups)
                _ = pm.Normal(
                    "obs", mu=mu, sigma=sigma[self.group], observed=self.data["y"]
                )
            else:
                sigma = pm.HalfNormal("sigma", sigma_sd)
                _ = pm.Normal("obs", mu=mu, sigma=sigma, observed=self.data["y"])

            self.map_approx = None
            self.trace = None

        self.initvals = self._get_model_initvals()
        initval_dict = self.get_initval(self.initvals, self.model)

        with self.model:
            if self.method == "mapx":
                self.map_approx = pmx.find_MAP(
                    method="L-BFGS-B",
                    use_grad=True,
                    initvals=initval_dict,
                    progressbar=progressbar,
                    gradient_backend="jax",
                    compile_kwargs={"mode": "JAX"},
                    options={"maxiter": 1e4},
                )
            elif self.method == "map":
                self.map_approx = pm.find_MAP(
                    start=initval_dict,
                    method="L-BFGS-B",
                    progressbar=progressbar,
                    maxeval=1e4,
                )
            elif self.method in ["fullrank_advi", "advi", "svgd", "asvgd"]:
                approx = pm.fit(
                    50000,
                    method=self.method,
                    start=initval_dict if self.method != "asvgd" else None,
                    progressbar=progressbar,
                )
                self.trace = approx.sample(draws=self.samples)
            elif self.method in ["nuts", "metropolis", "demetropolisz"]:
                step = pm.NUTS()
                if self.method == "metropolis":
                    step = pm.Metropolis()

                if self.method == "demetropolisz":
                    step = pm.DEMetropolisZ()

                self.trace = pm.sample(
                    self.samples,
                    chains=chains,
                    cores=cores,
                    nuts_sampler=nuts_sampler,
                    initvals=initval_dict,
                    step=step,
                    progressbar=progressbar,
                )
            else:
                raise NotImplementedError(
                    f"Method {self.method} is not supported at the moment!"
                )

    def _make_future_df(self, horizon: int, freq: FreqStr = "D"):
        """
        Create a dataframe for inference.

        Parameters
        ----------
        horizon: int
            The number of steps in the future that we are forecasting.
        freq: FreqStr
            The distance between the forecasting steps.
        """
        future = pd.DataFrame(
            {
                "ds": pd.DatetimeIndex(
                    np.hstack(
                        (
                            pd.date_range(
                                self.t_scale_params["ds_min"],
                                self.t_scale_params["ds_max"],
                                freq="D",
                            ).to_numpy(),
                            pd.date_range(
                                self.t_scale_params["ds_max"],
                                self.t_scale_params["ds_max"]
                                + pd.Timedelta(horizon, freq),
                                inclusive="right",
                            ).to_numpy(),
                        )
                    )
                )
            }
        )
        future["t"] = (future["ds"] - self.t_scale_params["ds_min"]) / (
            self.t_scale_params["ds_max"] - self.t_scale_params["ds_min"]
        )
        return future

    def predict(self, horizon: int, freq: FreqStr = "D"):
        """
        Perform out-of-sample inference.

        Parameters
        ----------
        horizon: int
            The number of steps in the future that we are forecasting.
        freq: FreqStr
            The distance between the forecasting steps.
        """
        future = self._make_future_df(horizon, freq)
        forecasts = self._predict(future, self.method, self.map_approx, self.trace)
        # TODO come up with a better way to check this
        is_individual = "scaler" not in self.y_scale_params

        for group_code in range(forecasts.shape[0]):
            if is_individual:
                future[f"yhat_{group_code}"] = (
                    forecasts[group_code] * self.y_scale_params[group_code]["y_max"]
                )
            else:
                future[f"yhat_{group_code}"] = (
                    forecasts[group_code] * self.y_scale_params["y_max"]
                )

            for model_type, model_cnt in self.model_idxs.items():
                if model_type.startswith("lt") is False:
                    continue
                for model_idx in range(model_cnt):
                    component = f"{model_type}_{model_idx}_{group_code}"
                    if component in future.columns:
                        if is_individual:
                            future[component] *= self.y_scale_params[group_code][
                                "y_max"
                            ]
                        else:
                            future[component] *= self.y_scale_params["y_max"]

        return future

    def _predict(
        self,
        future: pd.DataFrame,
        method: Method,
        map_approx: dict[str, np.ndarray] | None,
        trace: az.InferenceData | None,
    ):
        """
        Perform out-of-sample inference for each component.

        Parameters
        ----------
        future: pd.DataFrame
            Pandas dataframe containing the timestamps for which inference should be
            performed.
        method: Method
            The Bayesian inference method to be used. Either a point estimate MAP), a
            VI method (advi etc.) or full Bayesian sampling (MCMC).
        map_approx: dict[str, np.ndarray] | None
            The MAP posterior parameter estimate obtained with the Bayesian inference.
        trace: az.InferenceData | None
            Samples from the posterior obtained with the Bayesian inference.
        """
        if method in ["mapx", "map"]:
            return self._predict_map(future, map_approx)

        return self._predict_mcmc(future, trace)

    def plot(
        self, future: pd.DataFrame, series: str, y_true: pd.DataFrame | None = None
    ):
        """
        Plot the inference results for a given series.

        Parameters
        ----------
        future: pd.DataFrame
            Pandas dataframe containing the timestamps for which inference should be
            performed.
        series: str
            The name of the time series.
        y_true: pd.DataFrame | None
            A pandas dataframe containing the true values for the inference period that
            must at least have columns ds (predictor), y (target) and series (name of
            time series).
        """
        group_code: int | None = None
        for group_code_, group_name in self.groups_.items():
            if group_name == series:
                group_code = group_code_

        if group_code is None:
            raise ValueError(f"Time series {series} is not present in the dataset!")

        plt.figure(figsize=(14, 100 * 6))
        plt.subplot(100, 1, 1)
        plt.title("Predictions")
        plt.grid()

        plt.scatter(
            self.data["ds"],
            self.data["y"] * self.scale_params["y_max"],
            s=0.5,
            color="C0",
            label="train y",
        )

        if y_true is not None:
            plt.scatter(
                y_true["ds"],
                y_true[y_true["series"] == series]["y"],
                s=0.5,
                color="C1",
                label="y_true",
            )

        plt.plot(
            future["ds"], future[f"yhat_{group_code}"], lw=1, label=r"$\widehat{y}$"
        )

        plt.legend()
        plot_params = {"idx": 1}
        self._plot(
            plot_params, future, self.data, self.scale_params, y_true, group_code
        )

    def needs_priors(self, *args, **kwargs):
        return False

    def is_individual(self, *args, **kwargs):
        return self.pool_type == "individual"

    def __add__(self, other):
        return AdditiveTimeSeries(self, other)

    def __radd__(self, other):
        return AdditiveTimeSeries(other, self)

    def __pow__(self, other):
        return MultiplicativeTimeSeries(self, other)

    def __rpow__(self, other):
        return MultiplicativeTimeSeries(other, self)

    def __mul__(self, other):
        return SimpleMultiplicativeTimeSeries(self, other)

    def __rmul__(self, other):
        return SimpleMultiplicativeTimeSeries(other, self)


class CombinedTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def _get_initval(self, *args, **kwargs):
        left = {}
        right = {}
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._get_initval(*args, **kwargs)

        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._get_initval(*args, **kwargs)

        return {**left, **right}

    def _plot(self, *args, **kwargs):
        if not (type(self.left) is int or type(self.left) is float):
            self.left._plot(*args, **kwargs)

        if not (type(self.right) is int or type(self.right) is float):
            self.right._plot(*args, **kwargs)

    def needs_priors(self, *args, **kwargs):
        left = False
        right = False
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.needs_priors(*args, **kwargs)

        if not (type(self.right) is int or type(self.right) is float):
            right = self.right.needs_priors(*args, **kwargs)

        return left or right

    def is_individual(self, *args, **kwargs):
        left = True
        right = True
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.is_individual(*args, **kwargs)

        if not (type(self.right) is int or type(self.right) is float):
            right = self.right.is_individual(*args, **kwargs)

        return left and right


class AdditiveTimeSeries(CombinedTimeSeries):
    def definition(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.definition(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right.definition(*args, **kwargs)

        return left + right

    def _predict(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._predict(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._predict(*args, **kwargs)

        return left + right

    def __str__(self):
        return f"{self.left} + {self.right}"


class MultiplicativeTimeSeries(CombinedTimeSeries):
    def definition(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.definition(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right.definition(*args, **kwargs)

        return left * (1 + right)

    def _predict(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._predict(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._predict(*args, **kwargs)

        return left * (1 + right)

    def __str__(self):
        left = f"{self.left}"
        if type(self.left) is AdditiveTimeSeries:
            left = f"({self.left})"

        return f"{left} * (1 + {self.right})"


class SimpleMultiplicativeTimeSeries(CombinedTimeSeries):
    def definition(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left.definition(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right.definition(*args, **kwargs)

        return left * right

    def _predict(self, *args, **kwargs):
        left = self.left
        if not (type(self.left) is int or type(self.left) is float):
            left = self.left._predict(*args, **kwargs)

        right = self.right
        if not (type(self.right) is int or type(self.right) is float):
            right = self.right._predict(*args, **kwargs)

        return left * right

    def __str__(self):
        left = f"{self.left}"
        if type(self.left) is AdditiveTimeSeries:
            left = f"({self.left})"

        right = f"{self.right}"
        if type(self.right) is AdditiveTimeSeries:
            right = f"({self.right})"

        return f"{left} * {right}"
