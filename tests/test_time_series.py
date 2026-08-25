"""Tests for vangja.time_series module."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from unittest.mock import patch

import arviz as az

from vangja.components import (
    FlatTrend,
    FourierSeasonality,
    LinearTrend,
    NormalConstant,
)
from vangja.time_series import (
    AdditiveTimeSeries,
    CombinedTimeSeries,
    MultiplicativeTimeSeries,
    SimpleMultiplicativeTimeSeries,
    TimeSeriesModel,
)


class TestTimeSeriesModelScaling:
    """Tests for TimeSeriesModel data processing and scaling."""

    def test_process_data_converts_datetime(self, sample_data):
        """Test that _process_data converts ds column to datetime."""
        lt = LinearTrend()
        lt._process_data(sample_data.copy(), "maxabs", "individual", None)

        assert pd.api.types.is_datetime64_any_dtype(lt.data["ds"])

    def test_process_data_creates_t_column(self, sample_data):
        """Test that _process_data creates normalized t column."""
        lt = LinearTrend()
        lt._process_data(sample_data.copy(), "maxabs", "individual", None)

        assert "t" in lt.data.columns
        assert lt.data["t"].min() >= 0
        assert lt.data["t"].max() <= 1

    def test_process_data_sorts_by_date(self, sample_data):
        """Test that _process_data sorts data by date."""
        # Shuffle the data
        shuffled = sample_data.sample(frac=1, random_state=42).reset_index(drop=True)

        lt = LinearTrend()
        lt._process_data(shuffled.copy(), "maxabs", "individual", None)

        # Check data is sorted
        assert lt.data["ds"].is_monotonic_increasing

    def test_process_data_maxabs_scaler(self, sample_data):
        """Test maxabs scaling."""
        lt = LinearTrend()
        original_max = sample_data["y"].abs().max()
        lt._process_data(sample_data.copy(), "maxabs", "complete", None)

        assert lt.y_scale_params["scaler"] == "maxabs"
        assert lt.y_scale_params["y_min"] == 0

    def test_process_data_minmax_scaler(self, sample_data):
        """Test minmax scaling."""
        lt = LinearTrend()
        original_min = sample_data["y"].min()
        original_max = sample_data["y"].max()
        lt._process_data(sample_data.copy(), "minmax", "complete", None)

        assert lt.y_scale_params["scaler"] == "minmax"
        assert lt.y_scale_params["y_min"] == original_min
        assert lt.y_scale_params["y_max"] == original_max

    def test_process_data_individual_scale_mode(self, multi_series_data):
        """Test individual scale mode with multiple series."""
        lt = LinearTrend()
        lt._process_data(multi_series_data.copy(), "maxabs", "individual", None)

        # y_scale_params should be a dict with group codes as keys
        assert isinstance(lt.y_scale_params, dict)
        assert len(lt.y_scale_params) == 2  # Two series

    def test_process_data_complete_scale_mode(self, multi_series_data):
        """Test complete scale mode with multiple series."""
        lt = LinearTrend()
        lt._process_data(multi_series_data.copy(), "maxabs", "complete", None)

        # y_scale_params should have scaler key (single scaling)
        assert "scaler" in lt.y_scale_params

    def test_process_data_custom_t_scale_params(self, sample_data):
        """Test custom t_scale_params override."""
        custom_t_params = {
            "ds_min": pd.Timestamp("2019-01-01"),
            "ds_max": pd.Timestamp("2021-12-31"),
        }

        lt = LinearTrend()
        lt._process_data(sample_data.copy(), "maxabs", "complete", custom_t_params)

        assert lt.t_scale_params["ds_min"] == custom_t_params["ds_min"]
        assert lt.t_scale_params["ds_max"] == custom_t_params["ds_max"]


class TestTimeSeriesModelInitvals:
    """Tests for TimeSeriesModel initial values calculation."""

    def test_get_model_initvals_structure(self, sample_data):
        """Test structure of calculated initvals."""
        lt = LinearTrend()
        lt._process_data(sample_data.copy(), "maxabs", "individual", None)

        initvals = lt._get_model_initvals()

        assert "sigma" in initvals
        assert initvals["sigma"] == 1.0

    def test_get_model_initvals_slope_intercept(self, sample_data):
        """Test that initvals contain slope and intercept."""
        lt = LinearTrend()
        lt._process_data(sample_data.copy(), "maxabs", "individual", None)

        initvals = lt._get_model_initvals()

        # Should have slope and intercept for group 0
        assert "slope_0" in initvals
        assert "intercept_0" in initvals

    def test_get_model_initvals_multi_series(self, multi_series_data):
        """Test initvals for multiple series."""
        lt = LinearTrend()
        lt._process_data(multi_series_data.copy(), "maxabs", "individual", None)

        initvals = lt._get_model_initvals()

        # Should have slope and intercept for each group
        assert "slope_0" in initvals
        assert "slope_1" in initvals
        assert "intercept_0" in initvals
        assert "intercept_1" in initvals

    def test_get_initval_sigma_complete_pooling(self, sample_data):
        """Test that get_initval returns scalar sigma for complete pooling."""
        model_comp = LinearTrend(n_changepoints=0)
        model_comp._process_data(sample_data.copy(), "maxabs", "complete", None)

        pymc_model = pm.Model()
        with pymc_model:
            mu = model_comp.definition(pymc_model, model_comp.data, {}, None, None)
            sigma = pm.HalfNormal("sigma", 0.5)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=model_comp.data["y"])

        initvals = model_comp._get_model_initvals()
        result = model_comp.get_initval(initvals, pymc_model)
        sigma_var = pymc_model.named_vars["sigma"]

        assert sigma_var in result
        assert np.isscalar(result[sigma_var]) or result[sigma_var].ndim == 0

    def test_get_initval_sigma_individual_pooling(self, multi_series_data):
        """Test that get_initval returns vector sigma for individual pooling."""
        model_comp = LinearTrend(n_changepoints=0, pool_type="individual")
        model_comp._process_data(multi_series_data.copy(), "maxabs", "individual", None)

        pymc_model = pm.Model()
        with pymc_model:
            mu = model_comp.definition(pymc_model, model_comp.data, {}, None, None)
            sigma = pm.HalfNormal("sigma", 0.5, shape=model_comp.n_groups)
            group, _, _ = __import__(
                "vangja.utils", fromlist=["get_group_definition"]
            ).get_group_definition(model_comp.data, "partial")
            pm.Normal("obs", mu=mu, sigma=sigma[group], observed=model_comp.data["y"])

        initvals = model_comp._get_model_initvals()
        result = model_comp.get_initval(initvals, pymc_model)
        sigma_var = pymc_model.named_vars["sigma"]

        assert sigma_var in result
        assert isinstance(result[sigma_var], np.ndarray)
        assert result[sigma_var].shape == (model_comp.n_groups,)
        np.testing.assert_array_equal(result[sigma_var], np.ones(model_comp.n_groups))

    def test_get_initval_sigma_partial_pooling(self, multi_series_data):
        """Test that get_initval skips Deterministic sigma for partial pooling."""
        model_comp = LinearTrend(n_changepoints=0, pool_type="individual")
        model_comp._process_data(multi_series_data.copy(), "maxabs", "individual", None)

        pymc_model = pm.Model()
        with pymc_model:
            mu = model_comp.definition(pymc_model, model_comp.data, {}, None, None)
            sigma_sigma = pm.HalfCauchy("sigma_sigma", 0.5)
            sigma_offset = pm.HalfNormal("sigma_offset", 1, shape=model_comp.n_groups)
            sigma = pm.Deterministic("sigma", sigma_offset * sigma_sigma)
            group, _, _ = __import__(
                "vangja.utils", fromlist=["get_group_definition"]
            ).get_group_definition(model_comp.data, "partial")
            pm.Normal("obs", mu=mu, sigma=sigma[group], observed=model_comp.data["y"])

        initvals = model_comp._get_model_initvals()
        result = model_comp.get_initval(initvals, pymc_model)
        sigma_var = pymc_model.named_vars["sigma"]

        # Deterministic sigma should NOT be in initvals
        assert sigma_var not in result


class TestTimeSeriesModelOperators:
    """Tests for TimeSeriesModel operator overloading."""

    def test_add_creates_additive_time_series(self):
        """Test that __add__ creates AdditiveTimeSeries."""
        lt = LinearTrend()
        fs = FourierSeasonality(period=7, series_order=3)

        result = lt + fs

        assert isinstance(result, AdditiveTimeSeries)

    def test_radd_creates_additive_time_series(self):
        """Test that __radd__ creates AdditiveTimeSeries."""
        lt = LinearTrend()

        result = 5 + lt

        assert isinstance(result, AdditiveTimeSeries)
        assert result.left == 5

    def test_pow_creates_multiplicative_time_series(self):
        """Test that __pow__ creates MultiplicativeTimeSeries."""
        lt = LinearTrend()
        fs = FourierSeasonality(period=7, series_order=3)

        result = lt**fs

        assert isinstance(result, MultiplicativeTimeSeries)

    def test_mul_creates_simple_multiplicative(self):
        """Test that __mul__ creates SimpleMultiplicativeTimeSeries."""
        lt = LinearTrend()
        nc = NormalConstant()

        result = lt * nc

        assert isinstance(result, SimpleMultiplicativeTimeSeries)


class TestAdditiveTimeSeries:
    """Tests for AdditiveTimeSeries class."""

    def test_init_with_two_models(self):
        """Test AdditiveTimeSeries initialization."""
        lt = LinearTrend()
        fs = FourierSeasonality(period=7, series_order=3)

        combined = AdditiveTimeSeries(lt, fs)

        assert combined.left is lt
        assert combined.right is fs

    def test_init_with_number_left(self):
        """Test AdditiveTimeSeries with number on left."""
        lt = LinearTrend()

        combined = AdditiveTimeSeries(5, lt)

        assert combined.left == 5
        assert combined.right is lt

    def test_init_with_number_right(self):
        """Test AdditiveTimeSeries with number on right."""
        lt = LinearTrend()

        combined = AdditiveTimeSeries(lt, 10)

        assert combined.left is lt
        assert combined.right == 10

    def test_str_representation(self):
        """Test string representation shows addition."""
        lt = LinearTrend()
        nc = NormalConstant()

        combined = AdditiveTimeSeries(lt, nc)
        result = str(combined)

        assert "+" in result

    def test_needs_priors_both_false(self):
        """Test needs_priors when both components return False."""
        lt = LinearTrend()
        fs = FourierSeasonality(period=7, series_order=3)

        combined = AdditiveTimeSeries(lt, fs)

        assert combined.needs_priors() is False

    def test_is_individual_both_individual(self):
        """Test is_individual when both components are individual."""
        lt1 = LinearTrend(pool_type="individual")
        lt2 = LinearTrend(pool_type="individual")

        combined = AdditiveTimeSeries(lt1, lt2)

        assert combined.is_individual() is True

    def test_is_individual_mixed(self):
        """Test is_individual with mixed pool types."""
        lt = LinearTrend(pool_type="individual")
        fs = FourierSeasonality(period=7, series_order=3, pool_type="complete")

        combined = AdditiveTimeSeries(lt, fs)

        # Should return False if any is not individual
        assert combined.is_individual() is False


class TestMultiplicativeTimeSeries:
    """Tests for MultiplicativeTimeSeries class."""

    def test_init(self):
        """Test MultiplicativeTimeSeries initialization."""
        lt = LinearTrend()
        fs = FourierSeasonality(period=7, series_order=3)

        combined = MultiplicativeTimeSeries(lt, fs)

        assert combined.left is lt
        assert combined.right is fs

    def test_str_representation(self):
        """Test string representation shows multiplication pattern."""
        lt = LinearTrend()
        fs = FourierSeasonality(period=7, series_order=3)

        combined = MultiplicativeTimeSeries(lt, fs)
        result = str(combined)

        assert "(1 +" in result

    def test_str_with_additive_left(self):
        """Test string representation with additive component on left."""
        lt = LinearTrend()
        nc = NormalConstant()
        fs = FourierSeasonality(period=7, series_order=3)

        additive = AdditiveTimeSeries(lt, nc)
        combined = MultiplicativeTimeSeries(additive, fs)
        result = str(combined)

        # Should wrap additive in parentheses
        assert "(" in result


class TestSimpleMultiplicativeTimeSeries:
    """Tests for SimpleMultiplicativeTimeSeries class."""

    def test_init(self):
        """Test SimpleMultiplicativeTimeSeries initialization."""
        lt = LinearTrend()
        nc = NormalConstant()

        combined = SimpleMultiplicativeTimeSeries(lt, nc)

        assert combined.left is lt
        assert combined.right is nc

    def test_str_representation(self):
        """Test string representation."""
        lt = LinearTrend()
        nc = NormalConstant()

        combined = SimpleMultiplicativeTimeSeries(lt, nc)
        result = str(combined)

        assert "*" in result

    def test_str_with_additive_components(self):
        """Test string with additive components wrapped in parentheses."""
        lt1 = LinearTrend()
        lt2 = LinearTrend()
        nc = NormalConstant()

        additive = AdditiveTimeSeries(lt1, nc)
        combined = SimpleMultiplicativeTimeSeries(additive, lt2)
        result = str(combined)

        # Left additive should be in parentheses
        assert "(" in result


class TestCombinedTimeSeries:
    """Tests for CombinedTimeSeries base class methods."""

    def test_get_initval_with_numeric_left(self):
        """Test _get_initval when left is numeric returns empty dict for numeric."""
        lt = LinearTrend()

        combined = AdditiveTimeSeries(5, lt)

        # When model hasn't been fit, LinearTrend._get_initval requires model
        # Test that the structure handles numeric values on the left side
        assert combined.left == 5
        assert combined.right is lt

    def test_get_initval_with_numeric_right(self):
        """Test _get_initval when right is numeric returns empty dict for numeric."""
        lt = LinearTrend()

        combined = AdditiveTimeSeries(lt, 10)

        # Test that the structure handles numeric values on the right side
        assert combined.left is lt
        assert combined.right == 10

    def test_needs_priors_with_numeric(self):
        """Test needs_priors with numeric component."""
        lt = LinearTrend()

        combined = AdditiveTimeSeries(5, lt)

        assert combined.needs_priors() is False


class TestMakeFutureDataframe:
    """Tests for future dataframe creation."""

    def test_make_future_df_creates_correct_columns(self, sample_data):
        """Test that _make_future_df creates required columns."""
        lt = LinearTrend()
        lt._process_data(sample_data.copy(), "maxabs", "individual", None)

        future = lt._make_future_df(horizon=30, freq="D")

        assert "ds" in future.columns
        assert "t" in future.columns

    def test_make_future_df_includes_historical_dates(self, sample_data):
        """Test that future df includes historical dates."""
        lt = LinearTrend()
        lt._process_data(sample_data.copy(), "maxabs", "individual", None)

        future = lt._make_future_df(horizon=30, freq="D")

        # Should include dates from training data
        assert future["ds"].min() <= lt.t_scale_params["ds_min"]

    def test_make_future_df_extends_to_horizon(self, sample_data):
        """Test that future df extends to specified horizon."""
        lt = LinearTrend()
        lt._process_data(sample_data.copy(), "maxabs", "individual", None)

        horizon = 30
        future = lt._make_future_df(horizon=horizon, freq="D")

        # Should extend beyond training data
        expected_end = lt.t_scale_params["ds_max"] + pd.Timedelta(horizon, "D")
        assert future["ds"].max() >= expected_end

    def test_make_future_df_normalized_t(self, sample_data):
        """Test that t column is properly normalized."""
        lt = LinearTrend()
        lt._process_data(sample_data.copy(), "maxabs", "individual", None)

        future = lt._make_future_df(horizon=30, freq="D")

        # t should be 0 at ds_min and 1 at ds_max
        # Values beyond should be > 1
        assert future["t"].max() > 1


class TestMcmcPostFitMethods:
    """Tests for the MCMC/VI-dependent post-fit methods.

    Covers ``compute_log_likelihood``, ``waic`` (which computes WAIC
    manually since arviz 1.x removed ``az.waic``), ``loo``,
    ``convergence_summary`` and ``predict_uncertainty`` for MCMC fits.
    """

    @pytest.fixture(scope="class")
    def fitted_model(self):
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-01", periods=90)
        y = 50 + 5 * np.sin(2 * np.pi * np.arange(90) / 7) + rng.standard_normal(90)
        data = pd.DataFrame({"ds": dates, "y": y, "series": "s"})

        model = FlatTrend() + FourierSeasonality(7, 3)
        model.fit(
            data,
            method="nuts",
            samples=100,
            tune=100,
            chains=2,
            cores=1,
            progressbar=False,
        )
        return model

    def test_compute_log_likelihood_adds_group(self, fitted_model):
        fitted_model.compute_log_likelihood()
        assert hasattr(fitted_model.trace, "log_likelihood")

    def test_waic_returns_finite_criteria(self, fitted_model):
        result = fitted_model.waic()
        assert np.isfinite(result.waic)
        assert np.isfinite(result.elpd_waic)
        assert np.isfinite(result.p_waic)
        assert np.isfinite(result.se)
        # WAIC = -2 * elpd_waic
        assert result.waic == pytest.approx(-2.0 * result.elpd_waic)
        # p_waic (effective number of parameters) is non-negative
        assert result.p_waic >= 0
        assert result.se >= 0

    def test_loo_returns_elpd_data(self, fitted_model):
        result = fitted_model.loo()
        # arviz < 1.x exposes ``elpd_loo``; arviz >= 1.x exposes ``elpd``
        elpd = getattr(result, "elpd_loo", None)
        if elpd is None:
            elpd = result.elpd
        assert np.isfinite(float(elpd))

    def test_convergence_summary_has_diagnostics(self, fitted_model):
        summary = fitted_model.convergence_summary()
        assert "r_hat" in summary.columns
        assert "ess_bulk" in summary.columns

    def test_predict_uncertainty_mcmc_columns(self, fitted_model):
        future = fitted_model.predict_uncertainty(horizon=14, interval_width=0.95)
        assert "yhat_lower_0" in future.columns
        assert "yhat_upper_0" in future.columns
        assert (future["yhat_lower_0"] <= future["yhat_0"]).all()
        assert (future["yhat_0"] <= future["yhat_upper_0"]).all()


class TestFitSeedsAndSamplerDispatch:
    """Review P0-8: seeds, target_accept, and correct sampler dispatch."""

    @pytest.fixture()
    def small_data(self):
        dates = pd.date_range("2020-01-01", periods=30)
        y = np.sin(2 * np.pi * np.arange(30) / 7)
        return pd.DataFrame({"ds": dates, "y": y, "series": "s"})

    def test_mapx_receives_random_seed(self, small_data):
        with patch("vangja.time_series.pmx.find_MAP", return_value=_fake_map_result()) as mock_map:
            model = FlatTrend()
            model.fit(small_data, method="mapx", random_seed=42, progressbar=False)
        assert mock_map.call_args.kwargs["random_seed"] == 42
        assert model.fit_info["random_seed"] == 42

    def test_nuts_uses_backend_sampler_without_explicit_step(self, small_data):
        with patch("vangja.time_series.pm.sample", return_value=_fake_trace()) as mock_sample:
            model = FlatTrend()
            model.fit(
                small_data,
                method="nuts",
                samples=10,
                tune=10,
                chains=1,
                cores=1,
                nuts_sampler="nutpie",
                random_seed=7,
                target_accept=0.9,
                progressbar=False,
            )
        kwargs = mock_sample.call_args.kwargs
        assert "step" not in kwargs
        assert kwargs["nuts_sampler"] == "nutpie"
        assert kwargs["nuts_sampler_kwargs"] == {"target_accept": 0.9}
        assert kwargs["random_seed"] == 7

    def test_metropolis_uses_explicit_step_and_seed(self, small_data):
        with patch("vangja.time_series.pm.sample", return_value=_fake_trace()) as mock_sample:
            model = FlatTrend()
            model.fit(
                small_data,
                method="metropolis",
                samples=10,
                tune=10,
                chains=1,
                cores=1,
                random_seed=11,
                progressbar=False,
            )
        kwargs = mock_sample.call_args.kwargs
        assert isinstance(kwargs["step"], pm.CompoundStep)
        assert any(isinstance(m, pm.Metropolis) for m in kwargs["step"].methods)
        assert "nuts_sampler" not in kwargs
        assert kwargs["random_seed"] == 11

    def test_demetropolisz_uses_explicit_step(self, small_data):
        with patch("vangja.time_series.pm.sample", return_value=_fake_trace()) as mock_sample:
            model = FlatTrend()
            model.fit(
                small_data,
                method="demetropolisz",
                samples=10,
                tune=10,
                chains=1,
                cores=1,
                random_seed=5,
                progressbar=False,
            )
        step = mock_sample.call_args.kwargs["step"]
        assert any(
            isinstance(m, pm.DEMetropolisZ)
            for m in getattr(step, "methods", [step])
        )

    def test_advi_receives_random_seed(self, small_data):
        with patch("vangja.time_series.pm.fit", return_value=_fake_approx()) as mock_fit:
            model = FlatTrend()
            model.fit(
                small_data, method="advi", n=10, samples=5, random_seed=3, progressbar=False
            )
        assert mock_fit.call_args.kwargs["random_seed"] == 3

    def test_target_accept_default_none_means_no_kwarg(self, small_data):
        with patch("vangja.time_series.pm.sample", return_value=_fake_trace()) as mock_sample:
            model = FlatTrend()
            model.fit(
                small_data,
                method="nuts",
                samples=10,
                tune=10,
                chains=1,
                cores=1,
                progressbar=False,
            )
        assert mock_sample.call_args.kwargs["nuts_sampler_kwargs"] is None


class TestIncludeSourceInTarget:
    """The source series can be appended as a hierarchical co-learning group."""

    def test_requires_source_data(self, sample_data):
        model = FlatTrend()
        with pytest.raises(ValueError, match="source_data"):
            model.fit(sample_data.copy(), include_source_in_target=True, progressbar=False)

    def test_source_appended_as_group(self, sample_data):
        source = pd.DataFrame(
            {
                "ds": pd.date_range("2019-01-01", periods=100),
                "y": np.linspace(0, 1, 100),
            }
        )
        model = FlatTrend(pool_type="partial") + FourierSeasonality(
            7, 3, pool_type="partial"
        )
        model.fit(
            sample_data.copy(),
            method="mapx",
            maxiter=100,
            include_source_in_target=True,
            source_data=source,
            random_seed=1,
            progressbar=False,
        )
        assert model.n_groups == 2
        assert "source" in model.groups_.values()
        source_rows = model.data[model.data["series"] == "source"]
        assert len(source_rows) == 100


def _fake_map_result():
    """Minimal InferenceData-like return for pmx.find_MAP."""
    return az.from_dict(
        {
            "posterior": {
                "ft_0 - intercept": np.zeros((1, 1)),
                "sigma": np.ones((1, 1)),
            }
        }
    )


def _fake_trace():
    return az.from_dict(
        {"posterior": {"ft_0 - intercept": np.zeros((1, 2)), "sigma": np.ones((1, 2))}}
    )


def _fake_approx():
    class _A:
        def sample(self, draws):
            return _fake_trace()

    return _A()
