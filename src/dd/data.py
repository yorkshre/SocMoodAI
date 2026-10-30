from __future__ import annotations

import pandas as pd

REQUIRED = {"user_id", "timestamp", "text", "label"}


def load_posts(path: str) -> pd.DataFrame:
    """CSV with columns: user_id, timestamp, text, label (0/1, per user), optional lang."""
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["text"] = df["text"].fillna("").astype(str)
    bad = df.groupby("user_id")["label"].nunique()
    if (bad > 1).any():
        raise ValueError(f"Inconsistent labels for users: {list(bad[bad > 1].index[:5])}")
    return df
