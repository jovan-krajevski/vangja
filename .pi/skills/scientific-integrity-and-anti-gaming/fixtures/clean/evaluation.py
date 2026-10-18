"""Clean fixture: selection on development metrics only."""


def best_dev_config(table):
    return table.loc[table["dev_rel_mae"].idxmin()]
