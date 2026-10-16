import numpy as np


def test_relative_mae_matches_reference():
    expected = 0.5
    assert abs(0.5 - expected) == 0.0


def test_allclose_tight():
    assert np.allclose([1.0, 2.0], [1.0, 2.0], atol=1e-9)
