"""Chronological train / validation / test splits (never shuffled)."""
from __future__ import annotations

import pandas as pd


def time_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> dict[str, pd.DataFrame]:
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("train_frac and val_frac must be positive and sum to less than one")
    n = len(df)
    a, b = int(n * train_frac), int(n * (train_frac + val_frac))
    if a == 0 or b <= a or b >= n:
        raise ValueError("not enough rows for non-empty train, validation, and test partitions")
    return {"train": df.iloc[:a], "validation": df.iloc[a:b], "test": df.iloc[b:]}
