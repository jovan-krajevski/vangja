"""Utility functions for vangja time series models.

This module provides helper functions for data processing and evaluation
of time series models.

Functions
---------
remove_random_gaps
    Remove random contiguous intervals from a time series to simulate missing data.
get_group_definition
    Assign group codes to different time series based on pooling type.
filter_predictions_by_series
    Filter predictions to dates relevant to a specific series.
metrics
    Calculate evaluation metrics for time series predictions.
compare_models
    Bayesian model comparison using WAIC or LOO-CV.
prior_sensitivity_analysis
    Evaluate how sensitive posteriors are to changes in prior specifications.
prior_predictive_coverage
    Calculate the fraction of prior predictive samples within a plausible range.
plot_prior_posterior
    Visualise prior-to-posterior updating for selected parameters.
plot_posterior_predictive
    Plot posterior predictive samples against observed data.
plot_prior_predictive
    Plot prior predictive samples against observed data.
"""

import itertools

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)

from vangja.types import PoolType


def remove_random_gaps(
    df: pd.DataFrame, n_gaps: int = 4, gap_fraction: float = 0.2
) -> pd.DataFrame:
    """Remove random continuous intervals (gaps) from a time series DataFrame.

    Creates realistic missing-data scenarios by removing ``n_gaps``
    non-overlapping contiguous blocks from the data. Each block removes
    approximately ``gap_fraction`` of the total data points.

    Parameters
    ----------
    df : pd.DataFrame
        A time series DataFrame. Must have at least a ``ds`` column.
    n_gaps : int, default 4
        Number of contiguous intervals to remove.
    gap_fraction : float, default 0.2
        Fraction of total data points removed per gap.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with the specified gaps removed,
        index reset.

    Raises
    ------
    ValueError
        If the total number of points to remove exceeds the length of the
        DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'ds': pd.date_range('2020-01-01', periods=100),
    ...     'y': range(100),
    ... })
    >>> df_with_gaps = remove_random_gaps(df, n_gaps=2, gap_fraction=0.1)
    >>> len(df_with_gaps) < len(df)
    True
    """
    n = len(df)
    gap_size = int(n * gap_fraction)
    total_gap_size = n_gaps * gap_size

    if total_gap_size >= n:
        raise ValueError(
            f"Cannot remove {n_gaps} gaps of {gap_fraction*100}% each from data"
        )

    # Generate non-overlapping gap start positions
    available_indices = list(range(n - gap_size))
    gap_starts = []

    for i in range(n_gaps):
        if not available_indices:
            break
        start = np.random.choice(available_indices)
        gap_starts.append(start)
        # Remove indices that would overlap with this gap
        available_indices = [
            idx
            for idx in available_indices
            if idx >= start + gap_size or idx + gap_size <= start
        ]

    # Create mask for rows to keep
    keep_mask = np.ones(n, dtype=bool)
    for start in gap_starts:
        keep_mask[start : start + gap_size] = False

    return df[keep_mask].reset_index(drop=True)


def get_group_definition(
    data: pd.DataFrame, pool_type: PoolType
) -> tuple[np.ndarray, int, dict[int, str]]:
    """Assign group codes to different series based on pooling type.

    This function processes a multi-series dataframe and assigns integer codes
    to each unique series. The behavior depends on the pool_type parameter:

    - "complete": All series share a single group (code 0)
    - "partial" or "individual": Each unique series gets its own code

    Parameters
    ----------
    data : pd.DataFrame
        A pandas dataframe that must at least have columns ds (predictor), y
        (target) and series (name of time series).
    pool_type : PoolType
        Type of pooling performed when sampling. One of "complete", "partial",
        or "individual".

    Returns
    -------
    group : np.ndarray
        Array of integer group codes, one for each row in data.
    n_groups : int
        Number of unique groups.
    group_mapping : dict[int, str]
        Dictionary mapping group codes (int) to series names (str).

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'ds': pd.date_range('2020-01-01', periods=6),
    ...     'y': [1, 2, 3, 4, 5, 6],
    ...     'series': ['A', 'A', 'A', 'B', 'B', 'B']
    ... })
    >>> group, n_groups, mapping = get_group_definition(data, 'partial')
    >>> print(n_groups)
    2
    >>> print(mapping)
    {0: 'A', 1: 'B'}
    """
    pool_cols = "series"
    if pool_type == "complete":
        group = np.zeros(len(data), dtype="int")
        group_mapping = {0: data.iloc[0][pool_cols]}
        n_groups = 1
    else:
        data[pool_cols] = pd.Categorical(data[pool_cols])
        group = data[pool_cols].cat.codes.values
        group_mapping = dict(enumerate(data[pool_cols].cat.categories))
        n_groups = data[pool_cols].nunique()

    return group, n_groups, group_mapping


def filter_predictions_by_series(
    future: pd.DataFrame,
    series_data: pd.DataFrame,
    yhat_col: str = "yhat_0",
    horizon: int = 0,
) -> pd.DataFrame:
    """Filter predictions to only include dates relevant to a specific series.

    When fitting multiple series simultaneously with different date ranges,
    the predict() method generates predictions for the entire combined time
    range. This function filters predictions to only include dates within a
    specific series' range, which is essential for correct metric calculation
    and plotting.

    Parameters
    ----------
    future : pd.DataFrame
        Predictions dataframe from model.predict() containing 'ds' and yhat columns.
    series_data : pd.DataFrame
        The original data for a specific series (train + test combined, or just
        the portion you want to filter to). Must have 'ds' column.
    yhat_col : str, default "yhat_0"
        The name of the prediction column to include in the output.
    horizon : int, default 0
        Additional days beyond the series' max date to include (for forecast period).

    Returns
    -------
    pd.DataFrame
        Filtered predictions with columns ['ds', 'yhat_0'] containing only dates
        within the series' range plus the specified horizon.

    Examples
    --------
    >>> # After fitting a multi-series model
    >>> future_combined = model.predict(horizon=365)
    >>> # Filter to only Air Passengers' relevant dates
    >>> future_passengers = filter_predictions_by_series(
    ...     future_combined,
    ...     air_passengers,  # full dataset (train + test)
    ...     yhat_col=f"yhat_{passengers_group}",
    ...     horizon=365
    ... )
    """
    date_min = series_data["ds"].min()
    date_max = series_data["ds"].max()

    filtered = future[["ds", yhat_col]].copy()
    filtered.columns = ["ds", "yhat_0"]
    filtered = filtered[
        (filtered["ds"] >= date_min)
        & (filtered["ds"] <= date_max + pd.Timedelta(days=horizon))
    ]
    return filtered.reset_index(drop=True)


def metrics(
    y_true: pd.DataFrame, future: pd.DataFrame, pool_type: PoolType = "complete"
) -> pd.DataFrame:
    """Calculate evaluation metrics for time series predictions.

    Computes Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
    Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE)
    for each time series in the dataset.

    Parameters
    ----------
    y_true : pd.DataFrame
        A pandas dataframe containing the true values for the inference period
        that must at least have columns ds (predictor), y (target) and series
        (name of time series).
    future : pd.DataFrame
        Pandas dataframe containing the timestamps and predictions. Must have
        columns named 'yhat_{group_code}' for each group. The 'ds' column is
        used to match predictions to test data by date.
    pool_type : PoolType
        Type of pooling performed when sampling. Used to determine group
        assignments in y_true.

    Returns
    -------
    pd.DataFrame
        A dataframe with series names as index and columns for each metric:
        'mse', 'rmse', 'mae', 'mape'.

    Examples
    --------
    >>> from vangja import LinearTrend
    >>> from vangja.utils import metrics
    >>> model = LinearTrend()
    >>> model.fit(train_data)
    >>> future = model.predict(horizon=30)
    >>> evaluation = metrics(test_data, future, pool_type="complete")
    >>> print(evaluation)
              mse     rmse      mae     mape
    series1  25.3    5.03    4.21    0.082

    Notes
    -----
    Predictions are matched to test data by merging on the 'ds' column. This
    correctly handles cases where predictions are at a different frequency
    than the test data (e.g., daily predictions vs monthly test data).
    """
    # Copy y_true and add a 'series' column if not present
    processed_y_true = y_true.copy()
    if "series" not in processed_y_true.columns:
        processed_y_true["series"] = "series"

    # Ensure ds columns are datetime for proper merging
    processed_y_true["ds"] = pd.to_datetime(processed_y_true["ds"])
    future = future.copy()
    future["ds"] = pd.to_datetime(future["ds"])

    metrics_dict = {"mse": {}, "rmse": {}, "mae": {}, "mape": {}}
    test_group, _, test_groups_ = get_group_definition(processed_y_true, pool_type)
    for group_code, group_name in test_groups_.items():
        group_idx = test_group == group_code
        y_true_group = processed_y_true[group_idx][["ds", "y"]]

        # Merge predictions with test data on ds to correctly align dates
        merged = y_true_group.merge(
            future[["ds", f"yhat_{group_code}"]],
            on="ds",
            how="inner",
        )

        if len(merged) == 0:
            raise ValueError(
                f"No matching dates found between test data and predictions for "
                f"series '{group_name}'. Ensure predictions cover the test period."
            )

        y = merged["y"]
        yhat = merged[f"yhat_{group_code}"]
        metrics_dict["mse"][group_name] = mean_squared_error(y, yhat)
        metrics_dict["rmse"][group_name] = root_mean_squared_error(y, yhat)
        metrics_dict["mae"][group_name] = mean_absolute_error(y, yhat)
        metrics_dict["mape"][group_name] = mean_absolute_percentage_error(y, yhat)

    return pd.DataFrame(metrics_dict)


def compare_models(
    model_dict: dict,
    ic: str = "loo",
) -> pd.DataFrame:
    """Compare multiple fitted models using information criteria.

    Wraps ``arviz.compare`` to produce a ranked table of models scored by
    WAIC or LOO-CV (PSIS).

    Parameters
    ----------
    model_dict : dict[str, az.InferenceData | object]
        Mapping of model names to either ``arviz.InferenceData`` objects or
        fitted vangja model objects that expose a ``.trace`` attribute.
    ic : {"loo", "waic"}, default "loo"
        Information criterion to use.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by the chosen criterion (best model first).

    Examples
    --------
    >>> from vangja.utils import compare_models
    >>> comparison = compare_models(
    ...     {"baseline": baseline_model, "transfer": transfer_model},
    ...     ic="loo",
    ... )
    """
    resolved: dict[str, az.InferenceData] = {}
    for name, obj in model_dict.items():
        if isinstance(obj, az.InferenceData):
            resolved[name] = obj
        elif hasattr(obj, "trace") and obj.trace is not None:
            resolved[name] = obj.trace
        else:
            raise ValueError(
                f"Model '{name}' does not have posterior samples. "
                "Fit with an MCMC or VI method."
            )
    return az.compare(resolved, ic=ic)  # type: ignore[arg-type]


def prior_predictive_coverage(
    prior_predictive,
    low: float = -2.0,
    high: float = 2.0,
) -> float:
    """Calculate the fraction of prior predictive samples within a plausible range.

    This is a quantitative complement to visual prior predictive checks.
    Because vangja scales the data so that :math:`y \\approx [-1, 1]` and
    :math:`t \\in [0, 1]`, comparing the prior predictive samples against a
    fixed plausible window (default ``[-2, 2]``) reveals how informative or
    diffuse the chosen priors are.

    **How to interpret the result:**

    * **< 5 %** — priors are too loose.  The sampler wastes time in
      physically impossible regions.  Reduce the prior standard deviations.
    * **> 95 %** — priors may be too tight.  The model risks being unable to
      capture sudden spikes or changepoints.  Increase the prior standard
      deviations.
    * **30–60 %** — a reasonable sweet spot for flexible models like Prophet.
      The prior covers the data range without encouraging absurd values.

    Parameters
    ----------
    prior_predictive : az.InferenceData
        Result of ``model.sample_prior_predictive()``.
    low : float, default -2.0
        Lower bound of the plausible range (in scaled space).
    high : float, default 2.0
        Upper bound of the plausible range (in scaled space).

    Returns
    -------
    float
        Fraction of individual sample values inside ``[low, high]``,
        between 0 and 1.

    Examples
    --------
    >>> model = LinearTrend() + FourierSeasonality(365.25, 10)
    >>> model.fit(data, method="mapx")
    >>> ppc = model.sample_prior_predictive(samples=500)
    >>> coverage = prior_predictive_coverage(ppc)
    >>> print(f"{coverage * 100:.1f}% of prior samples are within [-2, 2]")
    """
    obs = prior_predictive.prior_predictive["obs"].values
    mask = (obs >= low) & (obs <= high)
    return float(np.mean(mask))


def plot_prior_predictive(
    prior_predictive,
    series_idx: int | None = None,
    group: np.ndarray | None = None,
    data: pd.DataFrame | None = None,
    n_samples: int = 50,
    ax=None,
    title: str = "Prior Predictive Check",
    show_hdi: bool = False,
    hdi_prob: float = 0.9,
    show_ref_lines: bool = False,
    ref_values: tuple[float, float] = (-1.0, 1.0),
    t: np.ndarray | None = None,
):
    """Plot prior predictive samples, optionally overlaid on observed data.

    Draws a "spaghetti plot" of prior predictive traces and, optionally,
    an HDI envelope and horizontal reference lines to help judge whether
    the chosen priors are plausible in the scaled data space.

    Parameters
    ----------
    prior_predictive : az.InferenceData
        Result of ``model.sample_prior_predictive()``.
    series_idx : int or None
        If the prior predictive contains multiple series (e.g. from a hierarchical
        model), specify which one to plot. If None, plots everything.
        If series_idx is not None you must also pass the corresponding group array
        to the group parameter.
    group : np.ndarray or None
        If the prior predictive contains multiple groups (e.g. from a hierarchical
        model), specify which element belongs to which group.
    data : pd.DataFrame or None
        Observed data with columns ``ds`` and ``y``.
    n_samples : int, default 50
        Number of prior predictive traces to draw.
    ax : matplotlib axes or None
        Axes to plot on.  Created if ``None``.
    title : str
        Plot title.
    show_hdi : bool, default False
        If True, shade the Highest Density Interval across time.
    hdi_prob : float, default 0.9
        Probability mass for the HDI band (ignored when ``show_hdi=False``).
    show_ref_lines : bool, default False
        If True, draw horizontal dashed lines at the scaled-data bounds
        given by ``ref_values``.  Useful for checking whether the prior
        predictive concentrates within the plausible region of scaled data.
    ref_values : tuple[float, float], default (-1.0, 1.0)
        ``(lower, upper)`` values for the reference lines (ignored when
        ``show_ref_lines=False``).  The defaults correspond to the
        approximate extent of maxabs-scaled data.
    t : np.ndarray or None
        x-axis values.  When ``None`` the observation index is used.  Pass
        ``model.data["t"].values`` for the normalised time axis, or
        ``model.data["ds"].values`` for calendar dates.

    Returns
    -------
    matplotlib.axes.Axes

    Examples
    --------
    >>> model = LinearTrend() + FourierSeasonality(365.25, 10)
    >>> model.fit(data, method="mapx")
    >>> ppc = model.sample_prior_predictive(samples=200)
    >>> # Simple spaghetti plot
    >>> plot_prior_predictive(ppc)
    >>> # With HDI, reference lines and scaled time axis
    >>> plot_prior_predictive(
    ...     ppc,
    ...     data=data,
    ...     show_hdi=True,
    ...     show_ref_lines=True,
    ...     t=model.data["t"].values,
    ... )
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    pp = prior_predictive.prior_predictive
    if series_idx is not None and group is not None:
        # Find the observation dimension (the one aligned with smart_home_model.group)
        obs_dim = next(
            (dim for dim, size in pp.sizes.items() if size == len(group)), None
        )
        if obs_dim is None:
            raise ValueError(
                "Could not find an observation dimension matching group length."
            )

        # Indices where group == series_idx
        idx_group = [i for i, g in enumerate(group) if g == series_idx]

        # Subset prior_predictive to only those positions
        pp = pp.isel({obs_dim: idx_group})

    obs = pp["obs"].values
    # shape: (chains, draws, n_obs)
    obs_flat = obs.reshape(-1, obs.shape[-1])

    x_axis = np.arange(obs_flat.shape[1]) if t is None else t

    # HDI band
    if show_hdi:
        alpha_val = 1 - hdi_prob
        lo = np.percentile(obs_flat, 100 * alpha_val / 2, axis=0)
        hi = np.percentile(obs_flat, 100 * (1 - alpha_val / 2), axis=0)
        ax.fill_between(
            x_axis,
            lo,
            hi,
            color="C0",
            alpha=0.2,
            label=f"{hdi_prob:.0%} HDI",
        )

    # Spaghetti traces
    idx = np.random.choice(
        obs_flat.shape[0], size=min(n_samples, obs_flat.shape[0]), replace=False
    )
    for i in idx:
        ax.plot(x_axis, obs_flat[i], color="C0", alpha=0.1, lw=0.5)

    if data is not None:
        data_series = data
        if series_idx is not None and group is not None:
            # Filter data to the specified series
            data_series = data[group == series_idx]

        ax.plot(
            x_axis, data_series["y"].values, color="C1", lw=2, label="Observed data"
        )

    # Reference lines
    if show_ref_lines:
        ax.axhline(
            ref_values[0],
            color="red",
            linestyle="--",
            lw=1,
            label=f"Ref lower ({ref_values[0]})",
        )
        ax.axhline(
            ref_values[1],
            color="red",
            linestyle="--",
            lw=1,
            label=f"Ref upper ({ref_values[1]})",
        )

    if data is not None or show_hdi or show_ref_lines:
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("t" if t is not None else "Observation index")
    ax.set_ylabel("y")
    return ax


def plot_posterior_predictive(
    posterior_predictive,
    data: pd.DataFrame | None = None,
    n_samples: int = 50,
    ax=None,
    title: str = "Posterior Predictive Check",
    show_hdi: bool = False,
    hdi_prob: float = 0.9,
    show_ref_lines: bool = False,
    ref_values: tuple[float, float] = (-1.0, 1.0),
    t: np.ndarray | None = None,
):
    """Plot posterior predictive samples, overlaid on observed data.

    Parameters
    ----------
    posterior_predictive : az.InferenceData
        Result of ``model.sample_posterior_predictive()``.
    data : pd.DataFrame or None
        Observed data with columns ``ds`` and ``y``.
    n_samples : int, default 50
        Number of posterior predictive traces to draw.
    ax : matplotlib axes or None
        Axes to plot on.
    title : str
        Plot title.
    show_hdi : bool, default False
        If True, shade the Highest Density Interval across time.
    hdi_prob : float, default 0.9
        Probability mass for the HDI band (ignored when ``show_hdi=False``).
    show_ref_lines : bool, default False
        If True, draw horizontal dashed lines at the scaled-data bounds
        given by ``ref_values``.
    ref_values : tuple[float, float], default (-1.0, 1.0)
        ``(lower, upper)`` values for the reference lines.
    t : np.ndarray or None
        x-axis values.  When ``None`` the observation index is used.  Pass
        ``model.data["t"].values`` for the normalised time axis, or
        ``model.data["ds"].values`` for calendar dates.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))

    obs = posterior_predictive.posterior_predictive["obs"].values
    obs_flat = obs.reshape(-1, obs.shape[-1])

    x_axis = np.arange(obs_flat.shape[1]) if t is None else t

    # HDI band
    if show_hdi:
        alpha_val = 1 - hdi_prob
        lo = np.percentile(obs_flat, 100 * alpha_val / 2, axis=0)
        hi = np.percentile(obs_flat, 100 * (1 - alpha_val / 2), axis=0)
        ax.fill_between(
            x_axis,
            lo,
            hi,
            color="C0",
            alpha=0.2,
            label=f"{hdi_prob:.0%} HDI",
        )

    # Spaghetti traces
    idx = np.random.choice(
        obs_flat.shape[0], size=min(n_samples, obs_flat.shape[0]), replace=False
    )
    for i in idx:
        ax.plot(x_axis, obs_flat[i], color="C0", alpha=0.1, lw=0.5)

    if data is not None:
        ax.plot(x_axis, data["y"].values, color="C1", lw=2, label="Observed data")

    # Reference lines
    if show_ref_lines:
        ax.axhline(
            ref_values[0],
            color="red",
            linestyle="--",
            lw=1,
            label=f"Ref lower ({ref_values[0]})",
        )
        ax.axhline(
            ref_values[1],
            color="red",
            linestyle="--",
            lw=1,
            label=f"Ref upper ({ref_values[1]})",
        )

    if data is not None or show_hdi or show_ref_lines:
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("t" if t is not None else "Observation index")
    ax.set_ylabel("y")
    return ax


def plot_prior_posterior(
    trace,
    prior_params: dict[str, dict[str, float]],
    var_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
):
    """Plot prior and posterior densities on the same axes.

    Generates a grid of subplots, one per parameter, showing the prior
    density (from the analytic specification) and the posterior density
    (from MCMC/VI samples).

    Parameters
    ----------
    trace : az.InferenceData
        Posterior samples from a fitted model.
    prior_params : dict[str, dict[str, float]]
        Mapping of variable names to dicts describing the prior.  Each dict
        must contain ``"dist"`` (one of ``"normal"``, ``"halfnormal"``,
        ``"laplace"``) and the relevant parameters (``"mu"``/``"sigma"`` for
        Normal, ``"sigma"`` for HalfNormal, ``"mu"``/``"b"`` for Laplace).
    var_names : list[str] or None
        Subset of variables to include.  Defaults to all keys in
        ``prior_params``.
    figsize : tuple or None
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> plot_prior_posterior(
    ...     model.trace,
    ...     {
    ...         "fs_0 - beta(p=365.25,n=6)": {"dist": "normal", "mu": 0, "sigma": 10},
    ...     },
    ... )
    """
    if var_names is None:
        var_names = list(prior_params.keys())

    n = len(var_names)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, var in enumerate(var_names):
        ax = axes.flat[i]
        # Posterior
        post_samples = trace.posterior[var].values.flatten()
        ax.hist(
            post_samples,
            bins=50,
            density=True,
            alpha=0.5,
            color="C0",
            label="Posterior",
        )

        # Prior
        p = prior_params[var]
        x = np.linspace(post_samples.min() - 1, post_samples.max() + 1, 300)
        dist_name = str(p["dist"]).lower()
        if dist_name == "normal":
            pdf = stats.norm.pdf(x, loc=p.get("mu", 0), scale=p.get("sigma", 1))
        elif dist_name == "halfnormal":
            pdf = stats.halfnorm.pdf(x, scale=p.get("sigma", 1))
        elif dist_name == "laplace":
            pdf = stats.laplace.pdf(x, loc=p.get("mu", 0), scale=p.get("b", 1))
        else:
            pdf = np.zeros_like(x)

        ax.plot(x, pdf, "C3-", lw=2, label="Prior")
        ax.set_title(var, fontsize=9)
        ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes.flat[j].set_visible(False)

    fig.suptitle("Prior → Posterior Updating", fontsize=13)
    fig.tight_layout()
    return fig


def prior_sensitivity_analysis(
    model_factory,
    data: pd.DataFrame,
    param_grid: dict[str, list],
    fit_kwargs: dict | None = None,
    metric_data: pd.DataFrame | None = None,
    horizon: int = 0,
) -> pd.DataFrame:
    """Run prior sensitivity analysis by fitting a model under varied priors.

    Parameters
    ----------
    model_factory : callable
        A function that accepts keyword arguments from ``param_grid`` and
        returns an unfitted vangja model.  For example::

            def make_model(beta_sd=10):
                return FlatTrend() + FourierSeasonality(365.25, 6, beta_sd=beta_sd)

    data : pd.DataFrame
        Training data (columns ``ds``, ``y``).
    param_grid : dict[str, list]
        Dictionary mapping parameter names to lists of values to test.  A
        full Cartesian product is evaluated.
    fit_kwargs : dict or None
        Additional keyword arguments forwarded to ``model.fit()``.
    metric_data : pd.DataFrame or None
        If provided, compute forecast metrics against this test set for each
        configuration.
    horizon : int, default 0
        Forecast horizon used when computing metrics.

    Returns
    -------
    pd.DataFrame
        One row per configuration with the varied parameter values and, if
        ``metric_data`` is provided, the resulting forecast metrics.

    Examples
    --------
    >>> from vangja.utils import prior_sensitivity_analysis
    >>> results = prior_sensitivity_analysis(
    ...     model_factory=lambda beta_sd: (
    ...         FlatTrend() + FourierSeasonality(365.25, 6, beta_sd=beta_sd)
    ...     ),
    ...     data=train,
    ...     param_grid={"beta_sd": [1, 5, 10, 20]},
    ...     fit_kwargs={"method": "mapx", "scaler": "minmax"},
    ...     metric_data=test,
    ...     horizon=365,
    ... )
    """
    fit_kwargs = fit_kwargs or {}
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    rows: list[dict] = []
    for combo in combos:
        kw = dict(zip(keys, combo))
        model = model_factory(**kw)
        model.fit(data, **fit_kwargs)
        row = dict(kw)

        if metric_data is not None:
            pred = model.predict(horizon=horizon, freq="D")
            try:
                m = metrics(metric_data, pred, pool_type="complete")
                for col in m.columns:
                    row[col] = m.iloc[0][col]
            except Exception:
                for col in ["mse", "rmse", "mae", "mape"]:
                    row[col] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)
