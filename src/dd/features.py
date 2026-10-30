"""Interpretable features per post and aggregated per user."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lexicons import ABSOLUTIST, FIRST_PERSON_SINGULAR
from .preprocess import clean_text, detect_lang, tokenize

NUM_COLS = ["fp_ratio", "abs_ratio", "night_share", "log_n_posts", "mean_tokens"]
NIGHT_HOURS = {0, 1, 2, 3, 4, 5}


def post_ratios(text: str, lang: str) -> tuple[float, float, int]:
    """Return (first-person ratio, absolutist ratio, n_tokens) for one post."""
    lang = lang if lang in FIRST_PERSON_SINGULAR else "en"
    toks = tokenize(text)
    n = len(toks)
    if n == 0:
        return 0.0, 0.0, 0
    fp = sum(t in FIRST_PERSON_SINGULAR[lang] for t in toks)
    ab = sum(t in ABSOLUTIST[lang] for t in toks)
    return fp / n, ab / n, n


def build_user_table(posts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate post rows (user_id, timestamp, text, label[, lang]) to one row per user.

    Night share uses the hour in the timestamp's own timezone; if your source gives UTC,
    convert to the user's local time first, otherwise the feature is noise.
    """
    df = posts.copy()
    df["clean"] = df["text"].map(clean_text)
    if "lang" not in df:
        df["lang"] = df["clean"].map(detect_lang)
    ratios = [post_ratios(t, l) for t, l in zip(df["clean"], df["lang"])]
    df["fp"] = [r[0] for r in ratios]
    df["ab"] = [r[1] for r in ratios]
    df["ntok"] = [r[2] for r in ratios]
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["night"] = ts.dt.hour.isin(NIGHT_HOURS).astype(float).where(ts.notna())

    g = df.groupby("user_id", sort=True)
    out = pd.DataFrame(
        {
            "text": g["clean"].agg(" ".join),
            "label": g["label"].first().astype(int),
            "lang": g["lang"].agg(lambda s: s.mode().iloc[0]),
            "fp_ratio": g["fp"].mean(),
            "abs_ratio": g["ab"].mean(),
            "night_share": g["night"].mean().fillna(0.0),
            "log_n_posts": np.log1p(g.size()),
            "mean_tokens": g["ntok"].mean(),
        }
    )
    return out.reset_index()
