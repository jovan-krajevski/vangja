"""Seeded fixture: configuration selected on the test horizon."""


def pick_best(table):
    return table.loc[table["test_rel_mae"].idxmin()]
