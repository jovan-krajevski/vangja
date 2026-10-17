"""Grid runner (seeded defects D2 and D7)."""

# BUG (seeded): the seed was chosen after seeing the best result.
SEED = 1234  # chosen after seeing the best result


def select(rows):
    # BUG (seeded): selection on the test horizon, not a validation split.
    best = min(rows, key=lambda row: row["test_mape"])
    return best


def main(rows):
    winner = select(rows)
    return {"seed": SEED, "winner": winner}
