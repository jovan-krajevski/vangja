"""Tests for vangja.components module."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from vangja.components import (
    BetaConstant,
    FourierSeasonality,
    LinearTrend,
    NormalConstant,
    UniformConstant,
)


class TestLinearTrendInit:
    """Tests for LinearTrend initialization."""

    def test_default_initialization(self):
        """Test LinearTrend with default parameters."""
        lt = LinearTrend()

        assert lt.n_changepoints == 25
        assert lt.changepoint_range == 0.8
        assert lt.slope_mean == 0
        assert lt.slope_sd == 5
        assert lt.intercept_mean == 0
        assert lt.intercept_sd == 5
        assert lt.delta_mean == 0
        assert lt.delta_sd == 0.05
        assert lt.delta_side == "left"
        assert lt.pool_type == "complete"

    def test_custom_initialization(self):
        """Test LinearTrend with custom parameters."""
        lt = LinearTrend(
            n_changepoints=10,
            changepoint_range=0.9,
            slope_mean=1.0,
            slope_sd=2.0,
            intercept_mean=5.0,
            intercept_sd=10.0,
            delta_mean=0.1,
            delta_sd=0.1,
            delta_side="right",
            pool_type="partial",
        )

        assert lt.n_changepoints == 10
        assert lt.changepoint_range == 0.9
        assert lt.slope_mean == 1.0
        assert lt.slope_sd == 2.0
        assert lt.intercept_mean == 5.0
        assert lt.intercept_sd == 10.0
        assert lt.delta_mean == 0.1
        assert lt.delta_sd == 0.1
        assert lt.delta_side == "right"
        assert lt.pool_type == "partial"

    def test_tune_parameters(self):
        """Test LinearTrend tune method parameters."""
        lt = LinearTrend(
            tune_method="parametric",
            delta_tune_method="prior_from_idata",
            shrinkage_strength=50,
            loss_factor_for_tune=0.5,
        )

        assert lt.tune_method == "parametric"
        assert lt.delta_tune_method == "prior_from_idata"
        assert lt.shrinkage_strength == 50
        assert lt.loss_factor_for_tune == 0.5


class TestFourierSeasonalityInit:
    """Tests for FourierSeasonality initialization."""

    def test_weekly_seasonality(self):
        """Test FourierSeasonality for weekly pattern."""
        fs = FourierSeasonality(period=7, series_order=3)

        assert fs.period == 7
        assert fs.series_order == 3
        assert fs.beta_mean == 0
        assert fs.beta_sd == 10
        assert fs.pool_type == "partial"

    def test_yearly_seasonality(self):
        """Test FourierSeasonality for yearly pattern."""
        fs = FourierSeasonality(period=365.25, series_order=10)

        assert fs.period == 365.25
        assert fs.series_order == 10

    def test_custom_parameters(self):
        """Test FourierSeasonality with custom parameters."""
        fs = FourierSeasonality(
            period=30,
            series_order=5,
            beta_mean=1.0,
            beta_sd=5.0,
            pool_type="complete",
            tune_method="prior_from_idata",
            shrinkage_strength=2.0,
            shift_for_tune=True,
            loss_factor_for_tune=0.5,
        )

        assert fs.period == 30
        assert fs.series_order == 5
        assert fs.beta_mean == 1.0
        assert fs.beta_sd == 5.0
        assert fs.pool_type == "complete"
        assert fs.tune_method == "prior_from_idata"
        assert fs.shrinkage_strength == 2.0
        assert fs.shift_for_tune is True
        assert fs.loss_factor_for_tune == 0.5


class TestUniformConstantInit:
    """Tests for UniformConstant initialization."""

    def test_basic_initialization(self):
        """Test UniformConstant with basic parameters."""
        uc = UniformConstant(lower=0, upper=10)

        assert uc.lower == 0
        assert uc.upper == 10
        assert uc.allow_tune is False

    def test_with_tune_enabled(self):
        """Test UniformConstant with tuning enabled."""
        uc = UniformConstant(lower=-5, upper=5, allow_tune=True)

        assert uc.lower == -5
        assert uc.upper == 5
        assert uc.allow_tune is True

    def test_str_representation(self):
        """Test string representation of UniformConstant."""
        uc = UniformConstant(lower=0, upper=1, allow_tune=False)

        assert "UC" in str(uc)
        assert "l=0" in str(uc)
        assert "u=1" in str(uc)


class TestNormalConstantInit:
    """Tests for NormalConstant initialization."""

    def test_default_initialization(self):
        """Test NormalConstant with default parameters."""
        nc = NormalConstant()

        assert nc.mu == 0
        assert nc.sd == 1
        assert nc.allow_tune is False

    def test_custom_initialization(self):
        """Test NormalConstant with custom parameters."""
        nc = NormalConstant(mu=5, sd=2, allow_tune=True)

        assert nc.mu == 5
        assert nc.sd == 2
        assert nc.allow_tune is True

    def test_str_representation(self):
        """Test string representation of NormalConstant."""
        nc = NormalConstant(mu=10, sd=3, allow_tune=True)

        assert "NC" in str(nc)
        assert "mu=10" in str(nc)
        assert "sd=3" in str(nc)


class TestBetaConstantInit:
    """Tests for BetaConstant initialization."""

    def test_basic_initialization(self):
        """Test BetaConstant with basic parameters."""
        bc = BetaConstant(lower=0, upper=1)

        assert bc.lower == 0
        assert bc.upper == 1
        assert bc.alpha == 0.5
        assert bc.beta == 0.5

    def test_custom_alpha_beta(self):
        """Test BetaConstant with custom alpha and beta."""
        bc = BetaConstant(lower=0, upper=100, alpha=2, beta=5)

        assert bc.lower == 0
        assert bc.upper == 100
        assert bc.alpha == 2
        assert bc.beta == 5

    def test_pool_type_settings(self):
        """Test BetaConstant pool_type settings."""
        bc = BetaConstant(lower=0, upper=1, pool_type="partial")

        assert bc.pool_type == "partial"


class TestComponentOperators:
    """Tests for component operator overloading."""

    def test_addition_operator(self):
        """Test addition of two components."""
        lt = LinearTrend()
        fs = FourierSeasonality(period=7, series_order=3)

        combined = lt + fs

        assert hasattr(combined, "left")
        assert hasattr(combined, "right")
        assert combined.left is lt
        assert combined.right is fs

    def test_addition_with_number(self):
        """Test addition of component with number."""
        lt = LinearTrend()

        combined = lt + 5

        assert combined.left is lt
        assert combined.right == 5

    def test_radd_with_number(self):
        """Test reverse addition with number."""
        lt = LinearTrend()

        combined = 5 + lt

        assert combined.left == 5
        assert combined.right is lt

    def test_power_operator(self):
        """Test multiplicative combination with power operator."""
        lt = LinearTrend()
        fs = FourierSeasonality(period=7, series_order=3)

        combined = lt**fs

        assert hasattr(combined, "left")
        assert hasattr(combined, "right")

    def test_multiplication_operator(self):
        """Test simple multiplication."""
        lt = LinearTrend()
        uc = UniformConstant(lower=0, upper=1)

        combined = lt * uc

        assert hasattr(combined, "left")
        assert hasattr(combined, "right")

    def test_complex_combination(self):
        """Test complex combination of multiple components."""
        lt = LinearTrend()
        fs_weekly = FourierSeasonality(period=7, series_order=3)
        fs_yearly = FourierSeasonality(period=365.25, series_order=10)

        combined = lt + fs_weekly + fs_yearly

        # Should create nested AdditiveTimeSeries
        assert hasattr(combined, "left")
        assert hasattr(combined, "right")

    def test_str_representation_additive(self):
        """Test string representation of additive combination."""
        lt = LinearTrend()
        nc = NormalConstant()

        combined = lt + nc

        # AdditiveTimeSeries should have __str__ that shows the combination
        result = str(combined)
        assert "+" in result

    def test_str_representation_multiplicative(self):
        """Test string representation of multiplicative combination."""
        lt = LinearTrend()
        fs = FourierSeasonality(period=7, series_order=3)

        combined = lt**fs

        result = str(combined)
        assert "(1 +" in result


class TestComponentPoolType:
    """Tests for component pool_type settings."""

    def test_linear_trend_complete_pooling(self):
        """Test LinearTrend with complete pooling."""
        lt = LinearTrend(pool_type="complete")
        assert lt.pool_type == "complete"
        assert lt.is_individual() is False

    def test_linear_trend_individual_pooling(self):
        """Test LinearTrend with individual pooling."""
        lt = LinearTrend(pool_type="individual")
        assert lt.pool_type == "individual"
        assert lt.is_individual() is True

    def test_fourier_seasonality_partial_pooling(self):
        """Test FourierSeasonality with partial pooling (default)."""
        fs = FourierSeasonality(period=7, series_order=3, pool_type="partial")
        assert fs.pool_type == "partial"
