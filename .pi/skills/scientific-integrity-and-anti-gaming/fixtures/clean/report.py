"""Clean fixture: failures recorded, nothing dropped."""


def record_units(df, failures):
    df.to_csv("units.csv", index=False)
    failures.to_csv("failures.csv", index=False)
    return df
