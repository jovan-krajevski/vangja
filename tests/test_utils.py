"""Tests for vangja.utils module."""

import numpy as np
import pandas as pd
import pytest

from vangja.utils import get_group_definition


class TestGetGroupDefinition:
    """Tests for get_group_definition function."""

    def test_complete_pooling_single_series(self, sample_data):
        """Test complete pooling with a single series."""
        group, n_groups, group_mapping = get_group_definition(sample_data, "complete")

        assert n_groups == 1
        assert len(group) == len(sample_data)
        assert all(g == 0 for g in group)
        assert 0 in group_mapping
        assert group_mapping[0] == "test_series"

    def test_complete_pooling_multi_series(self, multi_series_data):
        """Test complete pooling with multiple series (should treat as one group)."""
        group, n_groups, group_mapping = get_group_definition(
            multi_series_data, "complete"
        )

        assert n_groups == 1
        assert all(g == 0 for g in group)

    def test_partial_pooling_single_series(self, sample_data):
        """Test partial pooling with a single series."""
        group, n_groups, group_mapping = get_group_definition(sample_data, "partial")

        assert n_groups == 1
        assert len(group) == len(sample_data)

    def test_partial_pooling_multi_series(self, multi_series_data):
        """Test partial pooling with multiple series."""
        group, n_groups, group_mapping = get_group_definition(
            multi_series_data, "partial"
        )

        assert n_groups == 2
        assert len(group) == len(multi_series_data)
        # Check that we have two different group codes
        unique_groups = np.unique(group)
        assert len(unique_groups) == 2

    def test_group_mapping_keys_match_unique_groups(self, multi_series_data):
        """Test that group_mapping keys match unique group codes."""
        group, n_groups, group_mapping = get_group_definition(
            multi_series_data, "partial"
        )

        unique_groups = np.unique(group)
        assert set(unique_groups) == set(group_mapping.keys())

    def test_group_mapping_values_are_series_names(self, multi_series_data):
        """Test that group_mapping values are original series names."""
        group, n_groups, group_mapping = get_group_definition(
            multi_series_data, "partial"
        )

        series_names = set(multi_series_data["series"].unique())
        mapping_values = set(group_mapping.values())
        assert mapping_values == series_names

    def test_group_array_dtype(self, sample_data):
        """Test that group array is integer type."""
        group, _, _ = get_group_definition(sample_data, "complete")
        assert group.dtype in [np.int64, np.int32, int]

    def test_group_array_length_matches_data(self, sample_data):
        """Test that group array length matches data length."""
        group, _, _ = get_group_definition(sample_data, "partial")
        assert len(group) == len(sample_data)


class TestGetGroupDefinitionEdgeCases:
    """Edge case tests for get_group_definition."""

    def test_empty_dataframe(self):
        """Test with empty dataframe - should handle gracefully or raise."""
        empty_df = pd.DataFrame({"ds": [], "y": [], "series": []})

        # Empty dataframe with complete pooling should raise IndexError
        # when trying to get the first row for mapping
        with pytest.raises(IndexError):
            get_group_definition(empty_df, "complete")

    def test_single_row_dataframe(self):
        """Test with single row dataframe."""
        single_row = pd.DataFrame(
            {"ds": [pd.Timestamp("2020-01-01")], "y": [100.0], "series": ["single"]}
        )
        group, n_groups, group_mapping = get_group_definition(single_row, "complete")

        assert len(group) == 1
        assert n_groups == 1

    def test_many_series(self):
        """Test with many different series."""
        np.random.seed(42)
        n_series = 10
        dfs = []
        for i in range(n_series):
            dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
            dfs.append(
                pd.DataFrame(
                    {"ds": dates, "y": np.random.randn(50), "series": f"series_{i}"}
                )
            )
        multi_df = pd.concat(dfs, ignore_index=True)

        group, n_groups, group_mapping = get_group_definition(multi_df, "partial")

        assert n_groups == n_series
        assert len(group_mapping) == n_series
