"""Clean fixture: selection on development metrics."""
import pandas as pd  # unused in the fixture


def pick(table):
    return table.loc[table["dev_rel_mae"].idxmin()]
