"""Clean fixture: primary metric without tricks."""
REL_MAE_EPSILON = 1e-3


def relative_mae(mae, persistence_mae, epsilon=REL_MAE_EPSILON):
    if persistence_mae < epsilon:
        return None
    return mae / persistence_mae
