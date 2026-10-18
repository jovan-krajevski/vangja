"""Clean fixture: selection on development metrics."""
import pandas as pd  # unused in the fixture


def pick(table):
    return table.loc[table["dev_rel_mae"].idxmin()]


def pick_builtin(rows):
    # Negative control: a builtin over a development metric is not selection.
    return min(rows, key=lambda row: row["dev_mape"])
