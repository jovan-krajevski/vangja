"""Integration tests for vangja package.

These tests verify that components work together correctly.
Note: Some tests are marked as slow and may be skipped in CI.
"""

import numpy as np
import pandas as pd
import pytest

from vangja.components import (
    FourierSeasonality,
    LinearTrend,
    NormalConstant,
    UniformConstant,
)


class TestModelCombinations:
    """Tests for combining different model components."""

    def test_linear_trend_plus_seasonality(self):
        """Test combining LinearTrend with FourierSeasonality."""
        lt = LinearTrend(n_changepoints=5)
        fs = FourierSeasonality(period=7, series_order=3)

        model = lt + fs

        assert model.left is lt
        assert model.right is fs

    def test_complex_additive_model(self):
        """Test complex additive model with multiple components."""
        lt = LinearTrend()
        weekly = FourierSeasonality(period=7, series_order=3)
        yearly = FourierSeasonality(period=365.25, series_order=10)

        model = lt + weekly + yearly

        # Verify structure
        assert hasattr(model, "left")
        assert hasattr(model, "right")

    def test_multiplicative_seasonality(self):
        """Test multiplicative seasonality model."""
        lt = LinearTrend()
        fs = FourierSeasonality(period=7, series_order=3)

        model = lt**fs

        assert hasattr(model, "left")
        assert hasattr(model, "right")

    def test_additive_and_multiplicative_combination(self):
        """Test combining additive and multiplicative components."""
        lt = LinearTrend()
        weekly = FourierSeasonality(period=7, series_order=3)
        yearly = FourierSeasonality(period=365.25, series_order=5)

        # Trend with multiplicative weekly, plus additive yearly
        model = (lt**weekly) + yearly

        assert hasattr(model, "left")
        assert hasattr(model, "right")

    def test_scalar_multiplication(self):
        """Test multiplying model by scalar."""
        lt = LinearTrend()

        model = lt * 2

        assert model.right == 2


class TestModelStringRepresentations:
    """Tests for model string representations."""

    def test_additive_model_str(self):
        """Test string representation of additive model."""
        lt = LinearTrend()
        nc = NormalConstant(mu=0, sd=1)

        model = lt + nc
        result = str(model)

        assert "+" in result

    def test_multiplicative_model_str(self):
        """Test string representation of multiplicative model."""
        lt = LinearTrend()
        fs = FourierSeasonality(period=7, series_order=3)

        model = lt**fs
        result = str(model)

        assert "(1 +" in result

    def test_nested_model_str(self):
        """Test string representation of nested model."""
        lt = LinearTrend()
        nc = NormalConstant()
        fs = FourierSeasonality(period=7, series_order=3)

        model = (lt + nc) ** fs
        result = str(model)

        # The additive part should be in parentheses
        assert "(" in result


class TestModelPoolingBehavior:
    """Tests for model pooling behavior with different settings."""

    def test_complete_pooling_is_individual_false(self):
        """Test that complete pooling returns is_individual=False."""
        lt = LinearTrend(pool_type="complete")

        assert lt.is_individual() is False

    def test_individual_pooling_is_individual_true(self):
        """Test that individual pooling returns is_individual=True."""
        lt = LinearTrend(pool_type="individual")

        assert lt.is_individual() is True

    def test_combined_model_is_individual(self):
        """Test is_individual for combined model."""
        lt = LinearTrend(pool_type="individual")
        fs = FourierSeasonality(period=7, series_order=3, pool_type="individual")

        model = lt + fs

        # Both need to be individual for combined to be individual
        assert model.is_individual() is True

    def test_combined_model_mixed_pooling(self):
        """Test is_individual with mixed pooling types."""
        lt = LinearTrend(pool_type="individual")
        fs = FourierSeasonality(period=7, series_order=3, pool_type="complete")

        model = lt + fs

        # Should be False if any component is not individual
        assert model.is_individual() is False


class TestModelNeedsPriors:
    """Tests for needs_priors method."""

    def test_basic_model_needs_priors(self):
        """Test that basic model doesn't need priors by default."""
        lt = LinearTrend()

        assert lt.needs_priors() is False

    def test_combined_model_needs_priors(self):
        """Test combined model needs_priors."""
        lt = LinearTrend()
        fs = FourierSeasonality(period=7, series_order=3)

        model = lt + fs

        # Default should be False for both
        assert model.needs_priors() is False


class TestDataPreparation:
    """Tests for data preparation functionality."""

    def test_data_processing_preserves_series_column(self, sample_data):
        """Test that series column is preserved after processing."""
        lt = LinearTrend()
        lt._process_data(sample_data.copy(), "maxabs", "individual", None)

        assert "series" in lt.data.columns

    def test_group_assignment_single_series(self, sample_data):
        """Test group assignment for single series data."""
        lt = LinearTrend()
        lt._process_data(sample_data.copy(), "maxabs", "individual", None)

        assert lt.n_groups == 1
        assert len(lt.groups_) == 1

    def test_group_assignment_multi_series(self, multi_series_data):
        """Test group assignment for multi-series data."""
        lt = LinearTrend()
        lt._process_data(multi_series_data.copy(), "maxabs", "individual", None)

        assert lt.n_groups == 2
        assert len(lt.groups_) == 2


class TestScaleParameters:
    """Tests for scale parameter calculations."""

    def test_y_scale_params_maxabs(self, sample_data):
        """Test y scale parameters with maxabs scaler."""
        lt = LinearTrend()
        lt._process_data(sample_data.copy(), "maxabs", "complete", None)

        # y_min should be 0 for maxabs
        assert lt.y_scale_params["y_min"] == 0
        # y_max should be max absolute value
        assert lt.y_scale_params["y_max"] == sample_data["y"].abs().max()

    def test_y_scale_params_minmax(self, sample_data):
        """Test y scale parameters with minmax scaler."""
        lt = LinearTrend()
        lt._process_data(sample_data.copy(), "minmax", "complete", None)

        # Should capture actual min and max
        assert lt.y_scale_params["y_min"] == sample_data["y"].min()
        assert lt.y_scale_params["y_max"] == sample_data["y"].max()

    def test_t_scale_params(self, sample_data):
        """Test t scale parameters."""
        lt = LinearTrend()
        lt._process_data(sample_data.copy(), "maxabs", "complete", None)

        # Should capture date range
        assert lt.t_scale_params["ds_min"] == sample_data["ds"].min()
        assert lt.t_scale_params["ds_max"] == sample_data["ds"].max()

    def test_individual_y_scale_per_series(self, multi_series_data):
        """Test individual y scaling per series."""
        lt = LinearTrend()
        lt._process_data(multi_series_data.copy(), "maxabs", "individual", None)

        # Should have separate scale params for each group
        assert isinstance(lt.y_scale_params, dict)
        for group_code in lt.groups_.keys():
            assert group_code in lt.y_scale_params
            assert "y_max" in lt.y_scale_params[group_code]
