"""Tests for vangja.types module."""

import pytest

from vangja.types import (
    FreqStr,
    Method,
    NutsSampler,
    PoolType,
    ScaleMode,
    Scaler,
    TScaleParams,
    TuneMethod,
    YScaleParams,
)


class TestTypedDicts:
    """Tests for TypedDict definitions."""

    def test_y_scale_params_structure(self):
        """Test YScaleParams TypedDict structure."""
        params: YScaleParams = {
            "scaler": "maxabs",
            "y_min": 0.0,
            "y_max": 100.0,
        }
        assert params["scaler"] == "maxabs"
        assert params["y_min"] == 0.0
        assert params["y_max"] == 100.0

    def test_y_scale_params_minmax_scaler(self):
        """Test YScaleParams with minmax scaler."""
        params: YScaleParams = {
            "scaler": "minmax",
            "y_min": -10.0,
            "y_max": 50.0,
        }
        assert params["scaler"] == "minmax"
        assert params["y_min"] == -10.0

    def test_t_scale_params_structure(self):
        """Test TScaleParams TypedDict structure."""
        params: TScaleParams = {
            "ds_min": 0.0,
            "ds_max": 365.0,
        }
        assert params["ds_min"] == 0.0
        assert params["ds_max"] == 365.0


class TestLiteralTypes:
    """Tests for Literal type definitions."""

    def test_scaler_values(self):
        """Test valid Scaler values."""
        valid_scalers = ["maxabs", "minmax"]
        for scaler in valid_scalers:
            assert scaler in ["maxabs", "minmax"]

    def test_scale_mode_values(self):
        """Test valid ScaleMode values."""
        valid_modes = ["individual", "complete"]
        for mode in valid_modes:
            assert mode in ["individual", "complete"]

    def test_method_values(self):
        """Test valid Method values."""
        valid_methods = [
            "mapx",
            "map",
            "fullrank_advi",
            "advi",
            "svgd",
            "asvgd",
            "nuts",
            "metropolis",
            "demetropolisz",
        ]
        for method in valid_methods:
            assert method in valid_methods

    def test_nuts_sampler_values(self):
        """Test valid NutsSampler values."""
        valid_samplers = ["pymc", "nutpie", "numpyro", "blackjax"]
        for sampler in valid_samplers:
            assert sampler in valid_samplers

    def test_freq_str_values(self):
        """Test valid FreqStr values."""
        valid_freq = [
            "Y",
            "M",
            "W",
            "D",
            "h",
            "m",
            "s",
            "ms",
            "us",
            "ns",
            "ps",
            "minute",
            "second",
            "millisecond",
            "microsecond",
            "nanosecond",
            "picosecond",
        ]
        for freq in valid_freq:
            assert freq in valid_freq

    def test_tune_method_values(self):
        """Test valid TuneMethod values."""
        valid_methods = ["parametric", "prior_from_idata"]
        for method in valid_methods:
            assert method in valid_methods

    def test_pool_type_values(self):
        """Test valid PoolType values."""
        valid_types = ["partial", "complete", "individual"]
        for pool_type in valid_types:
            assert pool_type in valid_types
