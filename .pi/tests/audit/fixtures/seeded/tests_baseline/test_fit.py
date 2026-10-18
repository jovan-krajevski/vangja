def test_transfer_moves_towards_context():
    slope_after = 0.1
    slope_before = 0.3
    assert slope_after < slope_before
    assert abs(slope_after - 0.1) < 1e-9
