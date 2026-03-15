"""Tests for prior_from_idata transfer learning.

Verifies that the transfer learning pipeline works correctly:
- Variables passed to prior_from_idata are scoped to only what components need
- Individual pooling with prior_from_idata produces per-group free parameters
- Complete and partial pooling with prior_from_idata function correctly
- Initvals skip Deterministic variables gracefully
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from vangja.components import (
    FlatTrend,
    FourierSeasonality,
    LinearTrend,
    NormalConstant,
    UniformConstant,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def long_data():
    """Long time series for use as source / base model."""
    np.random.seed(0)
    dates = pd.date_range("2018-01-01", periods=730, freq="D")
    t = np.arange(730)
    y = 50 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 365.25) + np.random.randn(730) * 3
    return pd.DataFrame({"ds": dates, "y": y, "series": "src"})


@pytest.fixture
def short_data():
    """Short single-series target data."""
    np.random.seed(1)
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    t = np.arange(90)
    y = 55 + 0.04 * t + 8 * np.sin(2 * np.pi * t / 365.25) + np.random.randn(90) * 2
    return pd.DataFrame({"ds": dates, "y": y, "series": "tgt"})


@pytest.fixture
def short_multi_data():
    """Short multi-series target data (2 series)."""
    np.random.seed(2)
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    t = np.arange(90)
    frames = []
    for name, offset in [("s1", 55), ("s2", 60)]:
        y = offset + 0.04 * t + 8 * np.sin(2 * np.pi * t / 365.25) + np.random.randn(90) * 2
        frames.append(pd.DataFrame({"ds": dates, "y": y, "series": name}))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def base_trace(long_data):
    """Fit a base model on long_data and return the trace (InferenceData)."""
    model = LinearTrend(n_changepoints=5) + FourierSeasonality(365.25, 3)
    model.fit(long_data, method="advi", n=5000, samples=200, progressbar=False)
    return model.trace


# ---------------------------------------------------------------------------
# _assign_model_idx / _get_prior_var_names
# ---------------------------------------------------------------------------


class TestAssignModelIdx:
    """Verify that the pre-pass correctly assigns indices."""

    def test_single_component(self):
        lt = LinearTrend(tune_method="prior_from_idata", n_changepoints=5)
        idxs: dict[str, int] = {}
        lt._assign_model_idx(idxs)
        assert lt.model_idx == 0
        assert idxs["lt"] == 1

    def test_combined_model(self):
        model = (
            LinearTrend(tune_method="prior_from_idata", n_changepoints=5)
            + FourierSeasonality(365.25, 3, tune_method="prior_from_idata")
        )
        idxs: dict[str, int] = {}
        model._assign_model_idx(idxs)
        assert idxs["lt"] == 1
        assert idxs["fs"] == 1

    def test_prior_var_names_only_prior_from_idata(self):
        """Only components with tune_method='prior_from_idata' declare vars."""
        model = (
            LinearTrend(tune_method="prior_from_idata", n_changepoints=5)
            + FourierSeasonality(365.25, 3, tune_method="parametric")
        )
        idxs: dict[str, int] = {}
        model._assign_model_idx(idxs)
        names = model._get_prior_var_names()
        # LinearTrend declares slope + delta, FourierSeasonality returns nothing
        assert "lt_0 - slope" in names
        assert "lt_0 - delta" in names
        assert not any("fs_" in n for n in names)

    def test_prior_var_names_fourier(self):
        fs = FourierSeasonality(365.25, 5, tune_method="prior_from_idata")
        idxs: dict[str, int] = {}
        fs._assign_model_idx(idxs)
        names = fs._get_prior_var_names()
        assert names == ["fs_0 - beta(p=365.25,n=5)"]

    def test_prior_var_names_flat_trend(self):
        ft = FlatTrend(tune_method="prior_from_idata")
        idxs: dict[str, int] = {}
        ft._assign_model_idx(idxs)
        names = ft._get_prior_var_names()
        assert names == ["ft_0 - intercept"]

    def test_prior_var_names_normal_constant(self):
        nc = NormalConstant(mu=0, sd=1, tune_method="prior_from_idata")
        idxs: dict[str, int] = {}
        nc._assign_model_idx(idxs)
        names = nc._get_prior_var_names()
        assert names == ["nc_0 - c(mu=0,sd=1)"]

    def test_prior_var_names_uniform_constant(self):
        uc = UniformConstant(-1, 1, tune_method="prior_from_idata")
        idxs: dict[str, int] = {}
        uc._assign_model_idx(idxs)
        names = uc._get_prior_var_names()
        assert names == ["uc_0 - c(l=-1,u=1)"]


# ---------------------------------------------------------------------------
# Variable filtering in fit()
# ---------------------------------------------------------------------------


class TestVariableFiltering:
    """Ensure only needed variables are passed to prior_from_idata."""

    def test_sigma_not_in_prior_vars(self, base_trace):
        """sigma should NOT be included in the prior variable set."""
        model = (
            LinearTrend(tune_method="prior_from_idata", n_changepoints=5)
            + FourierSeasonality(365.25, 3, tune_method="prior_from_idata")
        )
        idxs: dict[str, int] = {}
        model._assign_model_idx(idxs)
        names = model._get_prior_var_names()
        assert "sigma" not in names

    def test_intercept_not_in_prior_vars(self, base_trace):
        """LinearTrend's intercept should NOT be transferred."""
        model = LinearTrend(tune_method="prior_from_idata", n_changepoints=5)
        idxs: dict[str, int] = {}
        model._assign_model_idx(idxs)
        names = model._get_prior_var_names()
        assert "lt_0 - intercept" not in names


# ---------------------------------------------------------------------------
# Individual pooling with prior_from_idata creates free per-group params
# ---------------------------------------------------------------------------


class TestIndividualPoolingPriorFromIdata:
    """Individual pooling with prior_from_idata must create free per-group RVs."""

    def test_linear_trend_individual_slope_is_free(
        self, short_multi_data, base_trace
    ):
        model = LinearTrend(
            tune_method="prior_from_idata",
            n_changepoints=5,
            pool_type="individual",
        ) + FourierSeasonality(365.25, 3, tune_method="prior_from_idata", pool_type="individual")
        model.fit(short_multi_data, method="map", idata=base_trace, progressbar=False)

        # slope should be a free RV (Normal), not a Deterministic
        slope_var = model.model.named_vars["lt_0 - slope"]
        assert slope_var in model.model.free_RVs, (
            "slope should be a free RV for individual pooling with prior_from_idata"
        )

    def test_fourier_beta_is_free(self, short_multi_data, base_trace):
        model = LinearTrend(
            tune_method="prior_from_idata",
            n_changepoints=5,
            pool_type="individual",
        ) + FourierSeasonality(365.25, 3, tune_method="prior_from_idata", pool_type="individual")
        model.fit(short_multi_data, method="map", idata=base_trace, progressbar=False)

        beta_var = model.model.named_vars["fs_0 - beta(p=365.25,n=3)"]
        assert beta_var in model.model.free_RVs, (
            "beta should be a free RV for individual pooling with prior_from_idata"
        )

    def test_flat_trend_individual_intercept_is_free(
        self, short_multi_data
    ):
        """FlatTrend individual + prior_from_idata → free intercept RV."""
        # First fit a base model
        base = FlatTrend() + FourierSeasonality(365.25, 3)
        # Use first series only as "long" data
        long = short_multi_data[short_multi_data["series"] == "s1"].copy()
        base.fit(long, method="advi", n=3000, samples=100, progressbar=False)

        target = FlatTrend(
            tune_method="prior_from_idata", pool_type="individual"
        ) + FourierSeasonality(365.25, 3, tune_method="prior_from_idata", pool_type="individual")
        target.fit(short_multi_data, method="map", idata=base.trace, progressbar=False)

        var = target.model.named_vars["ft_0 - intercept"]
        assert var in target.model.free_RVs


# ---------------------------------------------------------------------------
# Complete pooling with prior_from_idata still creates Deterministic
# ---------------------------------------------------------------------------


class TestCompletePoolingPriorFromIdata:
    """Complete pooling + prior_from_idata should produce Deterministic vars."""

    def test_complete_slope_is_deterministic(
        self, short_data, base_trace
    ):
        model = LinearTrend(
            tune_method="prior_from_idata", n_changepoints=5
        ) + FourierSeasonality(365.25, 3, tune_method="prior_from_idata")
        model.fit(short_data, method="map", idata=base_trace, progressbar=False)

        slope_var = model.model.named_vars["lt_0 - slope"]
        assert slope_var not in model.model.free_RVs, (
            "slope should be Deterministic for complete pooling with prior_from_idata"
        )


# ---------------------------------------------------------------------------
# End-to-end transfer learning produces valid predictions
# ---------------------------------------------------------------------------


class TestTransferLearningPredictions:
    """Verify that transfer learning produces reasonable predictions."""

    def test_prior_from_idata_complete_predicts(
        self, short_data, base_trace
    ):
        model = LinearTrend(
            tune_method="prior_from_idata", n_changepoints=5
        ) + FourierSeasonality(365.25, 3, tune_method="prior_from_idata")
        model.fit(short_data, method="map", idata=base_trace, progressbar=False)
        future = model.predict(horizon=30)
        assert "yhat_0" in future.columns
        assert not future["yhat_0"].isna().any()
        assert len(future) > 0

    def test_prior_from_idata_individual_predicts(
        self, short_multi_data, base_trace
    ):
        model = LinearTrend(
            tune_method="prior_from_idata",
            n_changepoints=5,
            pool_type="individual",
        ) + FourierSeasonality(
            365.25, 3, tune_method="prior_from_idata", pool_type="individual"
        )
        model.fit(
            short_multi_data,
            method="map",
            idata=base_trace,
            scale_mode="individual",
            progressbar=False,
        )
        future = model.predict(horizon=30)
        assert "yhat_0" in future.columns
        assert "yhat_1" in future.columns
        assert not future["yhat_0"].isna().any()

    def test_parametric_transfer_learning(self, short_data, base_trace):
        """Parametric transfer learning should also work correctly."""
        model = LinearTrend(
            tune_method="parametric",
            delta_tune_method="parametric",
            n_changepoints=5,
        ) + FourierSeasonality(365.25, 3, tune_method="parametric")
        model.fit(short_data, method="map", idata=base_trace, progressbar=False)
        future = model.predict(horizon=30)
        assert "yhat_0" in future.columns
        assert not future["yhat_0"].isna().any()

    def test_mixed_transfer_methods(self, short_data, base_trace):
        """Mix of prior_from_idata and parametric should work."""
        model = LinearTrend(
            tune_method="prior_from_idata", n_changepoints=5
        ) + FourierSeasonality(365.25, 3, tune_method="parametric")
        model.fit(short_data, method="map", idata=base_trace, progressbar=False)
        future = model.predict(horizon=30)
        assert "yhat_0" in future.columns
        assert not future["yhat_0"].isna().any()


# ---------------------------------------------------------------------------
# _get_initval skips Deterministic variables
# ---------------------------------------------------------------------------


class TestInitvalsSkipDeterministic:
    """Initvals should not include Deterministic variables."""

    def test_complete_pooling_prior_from_idata_initvals(
        self, short_data, base_trace
    ):
        model = LinearTrend(
            tune_method="prior_from_idata", n_changepoints=5
        ) + FourierSeasonality(365.25, 3, tune_method="prior_from_idata")
        model.fit(short_data, method="map", idata=base_trace, progressbar=False)

        # If we got here without error, initvals handled Deterministic correctly
        assert model.map_approx is not None
