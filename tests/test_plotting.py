"""Tests for vangja plotting capabilities."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing

from vangja.components import (
    BetaConstant,
    FourierSeasonality,
    LinearTrend,
    NormalConstant,
    UniformConstant,
)


@pytest.fixture
def simple_data():
    """Create simple time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    values = (
        np.sin(np.arange(100) * 2 * np.pi / 30) * 10 + 50 + np.random.randn(100) * 2
    )

    return pd.DataFrame({"ds": dates, "y": values, "series": "test_series"})


@pytest.fixture
def multi_series_data():
    """Create multi-series time series data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")

    series1_values = (
        np.sin(np.arange(100) * 2 * np.pi / 30) * 10 + 50 + np.random.randn(100) * 2
    )
    series2_values = (
        np.cos(np.arange(100) * 2 * np.pi / 30) * 15 + 60 + np.random.randn(100) * 3
    )

    df1 = pd.DataFrame({"ds": dates, "y": series1_values, "series": "series_a"})
    df2 = pd.DataFrame({"ds": dates, "y": series2_values, "series": "series_b"})

    return pd.concat([df1, df2], ignore_index=True)


@pytest.fixture
def test_data():
    """Create test data for y_true validation."""
    np.random.seed(123)
    dates = pd.date_range(start="2020-04-10", periods=30, freq="D")
    values = np.sin(np.arange(30) * 2 * np.pi / 30) * 10 + 55 + np.random.randn(30) * 2

    return pd.DataFrame({"ds": dates, "y": values, "series": "test_series"})


class TestPlotBasicFunctionality:
    """Tests for basic plotting functionality."""

    def test_plot_linear_trend_complete_pooling(self, simple_data):
        """Test plotting with LinearTrend and complete pooling."""
        model = LinearTrend(n_changepoints=5, pool_type="complete")
        model.fit(simple_data, method="mapx", progressbar=False)
        future = model.predict(horizon=30, freq="D")

        # Should not raise an error
        try:
            model.plot(future, series="test_series")
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot() raised an unexpected exception: {e}")

    def test_plot_with_y_true(self, simple_data, test_data):
        """Test plotting with y_true data."""
        model = LinearTrend(n_changepoints=5, pool_type="complete")
        model.fit(simple_data, method="mapx", progressbar=False)
        future = model.predict(horizon=30, freq="D")

        # Should not raise an error
        try:
            model.plot(future, series="test_series", y_true=test_data)
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot() with y_true raised an unexpected exception: {e}")

    def test_plot_invalid_series_raises_error(self, simple_data):
        """Test that plotting with invalid series name raises ValueError."""
        model = LinearTrend(n_changepoints=5, pool_type="complete")
        model.fit(simple_data, method="mapx", progressbar=False)
        future = model.predict(horizon=30, freq="D")

        with pytest.raises(ValueError, match="is not present in the dataset"):
            model.plot(future, series="nonexistent_series")

        plt.close("all")


class TestPlotMultiSeries:
    """Tests for plotting with multiple series."""

    def test_plot_multi_series_individual_scaling(self, multi_series_data):
        """Test plotting with multiple series and individual scaling."""
        model = LinearTrend(n_changepoints=5, pool_type="partial")
        model.fit(
            multi_series_data,
            scale_mode="individual",
            method="mapx",
            progressbar=False,
        )
        future = model.predict(horizon=30, freq="D")

        # Should not raise an error for either series
        try:
            model.plot(future, series="series_a")
            plt.close("all")
            model.plot(future, series="series_b")
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot() for multi-series raised an unexpected exception: {e}")

    def test_plot_multi_series_complete_scaling(self, multi_series_data):
        """Test plotting with multiple series and complete scaling."""
        model = LinearTrend(n_changepoints=5, pool_type="partial")
        model.fit(
            multi_series_data,
            scale_mode="complete",
            method="mapx",
            progressbar=False,
        )
        future = model.predict(horizon=30, freq="D")

        try:
            model.plot(future, series="series_a")
            plt.close("all")
        except Exception as e:
            pytest.fail(f"plot() raised an unexpected exception: {e}")


class TestPlotCombinedModels:
    """Tests for plotting combined models."""

    def test_plot_additive_model(self, simple_data):
        """Test plotting an additive model."""
        model = LinearTrend(
            n_changepoints=5, pool_type="complete"
        ) + FourierSeasonality(period=30, series_order=3, pool_type="complete")
        model.fit(simple_data, method="mapx", progressbar=False)
        future = model.predict(horizon=30, freq="D")

        try:
            model.plot(future, series="test_series")
            plt.close("all")
        except Exception as e:
            pytest.fail(
                f"plot() for additive model raised an unexpected exception: {e}"
            )

    def test_plot_multiplicative_model(self, simple_data):
        """Test plotting a multiplicative model."""
        model = LinearTrend(
            n_changepoints=5, pool_type="complete"
        ) ** FourierSeasonality(period=30, series_order=3, pool_type="complete")
        model.fit(simple_data, method="mapx", progressbar=False)
        future = model.predict(horizon=30, freq="D")

        try:
            model.plot(future, series="test_series")
            plt.close("all")
        except Exception as e:
            pytest.fail(
                f"plot() for multiplicative model raised an unexpected exception: {e}"
            )

    def test_plot_with_constant_components(self, simple_data):
        """Test plotting with constant components."""
        model = LinearTrend(n_changepoints=5, pool_type="complete") + NormalConstant(
            mu=0, sd=1, pool_type="complete"
        )
        model.fit(simple_data, method="mapx", progressbar=False)
        future = model.predict(horizon=30, freq="D")

        try:
            model.plot(future, series="test_series")
            plt.close("all")
        except Exception as e:
            pytest.fail(
                f"plot() with NormalConstant raised an unexpected exception: {e}"
            )


class TestComponentPlotMethods:
    """Tests for individual component _plot methods."""

    def test_linear_trend_plot_method(self, simple_data):
        """Test LinearTrend._plot method."""
        model = LinearTrend(n_changepoints=5, pool_type="complete")
        model.fit(simple_data, method="mapx", progressbar=False)
        future = model.predict(horizon=30, freq="D")

        plot_params = {"idx": 0}
        plt.figure()

        try:
            model._plot(plot_params, future, model.data, model.y_scale_params, None, 0)
            plt.close("all")
        except Exception as e:
            pytest.fail(f"LinearTrend._plot raised an unexpected exception: {e}")

        assert plot_params["idx"] == 1  # Should increment

    def test_fourier_seasonality_plot_method(self, simple_data):
        """Test FourierSeasonality._plot method."""
        model = FourierSeasonality(period=30, series_order=3, pool_type="complete")

        # Need to fit a full model for this to work
        full_model = LinearTrend(n_changepoints=5, pool_type="complete") + model
        full_model.fit(simple_data, method="mapx", progressbar=False)
        future = full_model.predict(horizon=30, freq="D")

        plot_params = {"idx": 0}
        plt.figure()

        try:
            model._plot(
                plot_params,
                future,
                full_model.data,
                full_model.y_scale_params,
                None,
                "",
            )
            plt.close("all")
        except Exception as e:
            pytest.fail(f"FourierSeasonality._plot raised an unexpected exception: {e}")

    def test_normal_constant_plot_method(self, simple_data):
        """Test NormalConstant._plot method."""
        nc = NormalConstant(mu=0, sd=1, pool_type="complete")
        full_model = LinearTrend(n_changepoints=5, pool_type="complete") + nc
        full_model.fit(simple_data, method="mapx", progressbar=False)
        future = full_model.predict(horizon=30, freq="D")

        plot_params = {"idx": 0}
        plt.figure()

        try:
            nc._plot(
                plot_params, future, full_model.data, full_model.y_scale_params, None, 0
            )
            plt.close("all")
        except Exception as e:
            pytest.fail(f"NormalConstant._plot raised an unexpected exception: {e}")

        assert plot_params["idx"] == 1

    def test_uniform_constant_plot_method(self, simple_data):
        """Test UniformConstant._plot method."""
        uc = UniformConstant(lower=-5, upper=5, pool_type="complete")
        full_model = LinearTrend(n_changepoints=5, pool_type="complete") + uc
        full_model.fit(simple_data, method="mapx", progressbar=False)
        future = full_model.predict(horizon=30, freq="D")

        plot_params = {"idx": 0}
        plt.figure()

        try:
            uc._plot(
                plot_params, future, full_model.data, full_model.y_scale_params, None, 0
            )
            plt.close("all")
        except Exception as e:
            pytest.fail(f"UniformConstant._plot raised an unexpected exception: {e}")

        assert plot_params["idx"] == 1

    def test_beta_constant_plot_method(self, simple_data):
        """Test BetaConstant._plot method."""
        bc = BetaConstant(lower=0.5, upper=1.5, alpha=2, beta=2, pool_type="complete")
        full_model = LinearTrend(n_changepoints=5, pool_type="complete") * bc
        full_model.fit(simple_data, method="mapx", progressbar=False)
        future = full_model.predict(horizon=30, freq="D")

        plot_params = {"idx": 0}
        plt.figure()

        try:
            bc._plot(
                plot_params, future, full_model.data, full_model.y_scale_params, None, 0
            )
            plt.close("all")
        except Exception as e:
            pytest.fail(f"BetaConstant._plot raised an unexpected exception: {e}")

        assert plot_params["idx"] == 1


class TestPlotScaleParams:
    """Tests for correct handling of scale_params in plotting."""

    def test_plot_uses_y_scale_params_not_scale_params(self, simple_data):
        """Test that plot method uses y_scale_params attribute."""
        model = LinearTrend(n_changepoints=5, pool_type="complete")
        model.fit(simple_data, method="mapx", progressbar=False)

        # Verify y_scale_params exists and scale_params does not
        assert hasattr(model, "y_scale_params")
        assert not hasattr(model, "scale_params")

        future = model.predict(horizon=30, freq="D")

        # Should not raise AttributeError for scale_params
        try:
            model.plot(future, series="test_series")
            plt.close("all")
        except AttributeError as e:
            if "scale_params" in str(e):
                pytest.fail(
                    "plot() still uses 'scale_params' instead of 'y_scale_params'"
                )
            raise

    def test_plot_individual_scaling_correct_y_max(self, multi_series_data):
        """Test that plot uses correct y_max for individual scaling."""
        model = LinearTrend(n_changepoints=5, pool_type="partial")
        model.fit(
            multi_series_data,
            scale_mode="individual",
            method="mapx",
            progressbar=False,
        )

        # y_scale_params should be a dict without 'scaler' key
        assert isinstance(model.y_scale_params, dict)
        assert "scaler" not in model.y_scale_params

        future = model.predict(horizon=30, freq="D")

        # Should handle individual scaling correctly
        try:
            model.plot(future, series="series_a")
            plt.close("all")
        except Exception as e:
            pytest.fail(
                f"plot() with individual scaling raised an unexpected exception: {e}"
            )


class TestPlotParamsTypes:
    """Tests for plot_params dictionary handling."""

    def test_plot_params_idx_increments(self, simple_data):
        """Test that plot_params['idx'] increments for each component."""
        model = (
            LinearTrend(n_changepoints=5, pool_type="complete")
            + FourierSeasonality(period=30, series_order=3, pool_type="complete")
            + NormalConstant(mu=0, sd=1, pool_type="complete")
        )
        model.fit(simple_data, method="mapx", progressbar=False)
        future = model.predict(horizon=30, freq="D")

        # Create a custom plot_params to track
        plot_params = {"idx": 1}  # Start at 1 as in actual plot method

        plt.figure()
        model._plot(plot_params, future, model.data, model.y_scale_params, None, 0)
        plt.close("all")

        # Should have incremented for each component (LT + FS + NC = 3)
        assert plot_params["idx"] == 4


class TestPlotCleanup:
    """Tests for proper matplotlib cleanup."""

    def test_plot_does_not_leave_figures_open(self, simple_data):
        """Test that plot method handles figure cleanup."""
        initial_figures = len(plt.get_fignums())

        model = LinearTrend(n_changepoints=5, pool_type="complete")
        model.fit(simple_data, method="mapx", progressbar=False)
        future = model.predict(horizon=30, freq="D")

        model.plot(future, series="test_series")
        plt.close("all")

        final_figures = len(plt.get_fignums())
        assert final_figures == 0  # All figures should be closed
