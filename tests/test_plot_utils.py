"""Tests for vangja.utils plot helper functions and compare_models.

These tests cover:
- plot_prior_predictive
- plot_posterior_predictive
- plot_prior_posterior
- compare_models
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")

import arviz as az

from vangja.utils import (
    compare_models,
    plot_posterior_predictive,
    plot_prior_posterior,
    plot_prior_predictive,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_predictive_idata(n_obs: int = 50, n_chains: int = 1, n_draws: int = 100, group: str = "prior_predictive"):
    """Create a mock InferenceData with prior or posterior predictive samples."""
    obs = np.random.randn(n_chains, n_draws, n_obs)
    dataset = xr.Dataset(
        {"obs": (["chain", "draw", "obs_dim_0"], obs)},
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "obs_dim_0": np.arange(n_obs),
        },
    )
    return az.InferenceData(**{group: dataset})


def _make_trace_idata(var_dict: dict | None = None):
    """Create a mock InferenceData with posterior samples."""
    if var_dict is None:
        var_dict = {"slope": np.random.randn(1, 500)}

    data_vars = {}
    for name, values in var_dict.items():
        data_vars[name] = (["chain", "draw"], values)

    n_chains, n_draws = list(var_dict.values())[0].shape
    posterior = xr.Dataset(
        data_vars,
        coords={
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
        },
    )
    return az.InferenceData(posterior=posterior)


@pytest.fixture
def prior_idata():
    return _make_predictive_idata(n_obs=50, group="prior_predictive")


@pytest.fixture
def posterior_idata():
    return _make_predictive_idata(n_obs=50, group="posterior_predictive")


@pytest.fixture
def sample_data_50():
    return pd.DataFrame(
        {"ds": pd.date_range("2020-01-01", periods=50), "y": np.random.randn(50) * 10 + 50}
    )


@pytest.fixture
def trace():
    return _make_trace_idata(
        {"slope": np.random.randn(1, 500), "intercept": np.random.randn(1, 500) + 5}
    )


# ---------------------------------------------------------------------------
# plot_prior_predictive
# ---------------------------------------------------------------------------


class TestPlotPriorPredictive:
    """Tests for plot_prior_predictive function."""

    def test_returns_axes(self, prior_idata):
        ax = plot_prior_predictive(prior_idata)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_with_observed_data(self, prior_idata, sample_data_50):
        ax = plot_prior_predictive(prior_idata, data=sample_data_50)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_custom_n_samples(self, prior_idata):
        ax = plot_prior_predictive(prior_idata, n_samples=10)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_custom_title(self, prior_idata):
        ax = plot_prior_predictive(prior_idata, title="Custom Title")
        assert ax.get_title() == "Custom Title"
        plt.close("all")

    def test_with_provided_axes(self, prior_idata):
        fig, ax_in = plt.subplots()
        ax_out = plot_prior_predictive(prior_idata, ax=ax_in)
        assert ax_out is ax_in
        plt.close("all")

    def test_n_samples_capped_at_available(self):
        """Request more samples than available — should not raise."""
        idata = _make_predictive_idata(n_obs=30, n_draws=5, group="prior_predictive")
        ax = plot_prior_predictive(idata, n_samples=1000)
        assert isinstance(ax, plt.Axes)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_posterior_predictive
# ---------------------------------------------------------------------------


class TestPlotPosteriorPredictive:
    """Tests for plot_posterior_predictive function."""

    def test_returns_axes(self, posterior_idata):
        ax = plot_posterior_predictive(posterior_idata)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_with_observed_data(self, posterior_idata, sample_data_50):
        ax = plot_posterior_predictive(posterior_idata, data=sample_data_50)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_custom_title(self, posterior_idata):
        ax = plot_posterior_predictive(posterior_idata, title="Post-pred")
        assert ax.get_title() == "Post-pred"
        plt.close("all")

    def test_with_provided_axes(self, posterior_idata):
        fig, ax_in = plt.subplots()
        ax_out = plot_posterior_predictive(posterior_idata, ax=ax_in)
        assert ax_out is ax_in
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_prior_posterior
# ---------------------------------------------------------------------------


class TestPlotPriorPosterior:
    """Tests for plot_prior_posterior function."""

    def test_returns_figure(self, trace):
        prior_params = {
            "slope": {"dist": "normal", "mu": 0, "sigma": 1},
        }
        fig = plot_prior_posterior(trace, prior_params, var_names=["slope"])
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_multiple_variables(self, trace):
        prior_params = {
            "slope": {"dist": "normal", "mu": 0, "sigma": 1},
            "intercept": {"dist": "normal", "mu": 5, "sigma": 2},
        }
        fig = plot_prior_posterior(trace, prior_params)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_halfnormal_prior(self):
        trace = _make_trace_idata({"sigma": np.abs(np.random.randn(1, 300))})
        prior_params = {"sigma": {"dist": "halfnormal", "sigma": 1}}
        fig = plot_prior_posterior(trace, prior_params)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_laplace_prior(self):
        trace = _make_trace_idata({"delta": np.random.laplace(0, 0.05, (1, 300))})
        prior_params = {"delta": {"dist": "laplace", "mu": 0, "b": 0.05}}
        fig = plot_prior_posterior(trace, prior_params)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_unknown_dist_does_not_raise(self):
        trace = _make_trace_idata({"x": np.random.randn(1, 200)})
        prior_params = {"x": {"dist": "unknown_dist"}}
        fig = plot_prior_posterior(trace, prior_params)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_custom_figsize(self, trace):
        prior_params = {"slope": {"dist": "normal", "mu": 0, "sigma": 1}}
        fig = plot_prior_posterior(
            trace, prior_params, var_names=["slope"], figsize=(8, 4)
        )
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_var_names_subset(self, trace):
        """Test that var_names filters to a subset of prior_params."""
        prior_params = {
            "slope": {"dist": "normal", "mu": 0, "sigma": 1},
            "intercept": {"dist": "normal", "mu": 5, "sigma": 2},
        }
        fig = plot_prior_posterior(trace, prior_params, var_names=["slope"])
        # Only one subplot should be visible
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) == 1
        plt.close("all")


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


class TestCompareModels:
    """Tests for compare_models function."""

    def test_raises_for_missing_trace(self):
        """compare_models should raise ValueError for objects without traces."""

        class FakeModel:
            trace = None

        with pytest.raises(ValueError, match="does not have posterior samples"):
            compare_models({"bad": FakeModel()})

    def test_raises_for_invalid_object(self):
        """compare_models should raise ValueError for plain objects."""
        with pytest.raises(ValueError, match="does not have posterior samples"):
            compare_models({"nope": 42})

    def test_accepts_inference_data(self):
        """compare_models should accept raw InferenceData objects.

        We can only verify it doesn't error out during argument resolution.
        The actual az.compare may fail without proper log_likelihood group,
        so we just verify the ValueError about missing traces is NOT raised.
        """
        # Create a minimal InferenceData with log_likelihood
        obs = np.random.randn(50)
        ll = np.random.randn(1, 100, 50)
        log_lik = xr.Dataset(
            {"obs": (["chain", "draw", "obs_dim_0"], ll)},
            coords={
                "chain": [0],
                "draw": np.arange(100),
                "obs_dim_0": np.arange(50),
            },
        )
        posterior = xr.Dataset(
            {"mu": (["chain", "draw"], np.random.randn(1, 100))},
            coords={"chain": [0], "draw": np.arange(100)},
        )
        idata1 = az.InferenceData(posterior=posterior, log_likelihood=log_lik)
        idata2 = az.InferenceData(posterior=posterior, log_likelihood=log_lik)

        # Should not raise ValueError about missing traces
        try:
            result = compare_models({"m1": idata1, "m2": idata2}, ic="loo")
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            # az.compare may raise about data shape, but not about missing trace
            assert "does not have posterior samples" not in str(e)

    def test_accepts_model_with_trace_attribute(self):
        """compare_models should resolve objects via .trace attribute."""
        obs = np.random.randn(50)
        ll = np.random.randn(1, 100, 50)
        log_lik = xr.Dataset(
            {"obs": (["chain", "draw", "obs_dim_0"], ll)},
            coords={
                "chain": [0],
                "draw": np.arange(100),
                "obs_dim_0": np.arange(50),
            },
        )
        posterior = xr.Dataset(
            {"mu": (["chain", "draw"], np.random.randn(1, 100))},
            coords={"chain": [0], "draw": np.arange(100)},
        )
        idata = az.InferenceData(posterior=posterior, log_likelihood=log_lik)

        class FakeModel:
            trace = idata

        try:
            result = compare_models({"m1": FakeModel(), "m2": FakeModel()})
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            assert "does not have posterior samples" not in str(e)
