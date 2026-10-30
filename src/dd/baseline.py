"""Baselines + ablation. Split is by USER (one row per user), so no post-level leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import NUM_COLS

MODES = ("text", "features", "text+features")


def build_pipeline(mode: str, C: float = 1.0, seed: int = 42) -> Pipeline:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    parts = []
    if "text" in mode:
        parts.append(("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "text"))
    if "features" in mode:
        parts.append(("num", StandardScaler(), NUM_COLS))
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=seed)
    return Pipeline([("pre", ColumnTransformer(parts)), ("clf", clf)])


def evaluate(users: pd.DataFrame, mode: str, n_splits: int = 5, seed: int = 42) -> dict:
    y = users["label"].to_numpy()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(build_pipeline(mode, seed=seed), users, y, cv=cv, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "mode": mode,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba) if len(np.unique(y)) == 2 else float("nan"),
    }


def ablation(users: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    return pd.DataFrame([evaluate(users, m, n_splits, seed) for m in MODES]).set_index("mode")
