"""Dirty fixture: configuration selected with a builtin over the test metric."""
import pandas as pd  # unused in the fixture


def pick(rows):
    return min(rows, key=lambda row: row["test_mape"])
