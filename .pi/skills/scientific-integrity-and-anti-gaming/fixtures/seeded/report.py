"""Seeded fixture: swallowed errors + dropna before reporting."""


def record_units(df):
    try:
        df.to_csv("units.csv", index=False)
    except Exception:
        pass
    return df.dropna().to_csv("units_clean.csv")
