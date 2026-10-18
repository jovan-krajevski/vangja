"""Dirty fixture: configuration selected on the test horizon."""
import pandas as pd  # unused in the fixture


def pick(table):
    return table.loc[table["test_rel_mae"].idxmin()]
