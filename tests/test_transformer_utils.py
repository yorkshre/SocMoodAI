import numpy as np
import pandas as pd
import pytest

from dd.transformer import aggregate_scores, best_threshold, chunk_users, split_users, user_metrics


def _users(n=20):
    return pd.DataFrame(
        {"user_id": [f"u{i}" for i in range(n)], "label": [i % 2 for i in range(n)],
         "text": [" ".join(f"w{i}_{k}" for k in range(300)) for i in range(n)]}
    )


def test_split_is_disjoint_stratified_and_complete():
    u = _users(40)
    tr, va, te = split_users(u, 0.15, 0.15, seed=1)
    ids = [set(x["user_id"]) for x in (tr, va, te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(i) for i in ids) == 40
    assert all(x["label"].nunique() == 2 for x in (tr, va, te))


def test_split_rejects_duplicate_users():
    u = pd.concat([_users(10), _users(10)])
    with pytest.raises(ValueError):
        split_users(u)


def test_chunking_sizes_and_cap():
    u = _users(2)
    c = chunk_users(u, words_per_chunk=100, max_chunks=None)
    assert (c.groupby("user_id").size() == 3).all()
    assert c["text"].map(lambda t: len(t.split())).max() == 100
    c2 = chunk_users(u, words_per_chunk=10, max_chunks=5)
    assert (c2.groupby("user_id").size() == 5).all()
    # evenly spread: first and last chunk both kept
    first = c2[c2.user_id == "u0"]["text"].iloc[0]
    last = c2[c2.user_id == "u0"]["text"].iloc[-1]
    assert first.startswith("w0_0") and "w0_299" in last


def test_chunking_empty_text_keeps_user():
    u = pd.DataFrame({"user_id": ["a"], "label": [1], "text": [""]})
    c = chunk_users(u)
    assert len(c) == 1 and c.loc[0, "user_id"] == "a"


def test_aggregate_mean_and_max():
    chunks = pd.DataFrame({"user_id": ["a", "a", "b"], "label": [1, 1, 0], "text": ["", "", ""]})
    m = aggregate_scores(chunks, [0.2, 0.6, 0.1], "mean").set_index("user_id")
    x = aggregate_scores(chunks, [0.2, 0.6, 0.1], "max").set_index("user_id")
    assert m.loc["a", "score"] == pytest.approx(0.4)
    assert x.loc["a", "score"] == pytest.approx(0.6)
    assert m.loc["b", "label"] == 0
    with pytest.raises(ValueError):
        aggregate_scores(chunks, [0, 0, 0], "median")


def test_best_threshold_and_metrics():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.6, 0.9])
    t = best_threshold(y, s)
    assert 0.4 < t <= 0.6
    m = user_metrics(y, s, t)
    assert m["f1"] == pytest.approx(1.0) and m["roc_auc"] == pytest.approx(1.0)


def test_metrics_single_class_does_not_crash():
    m = user_metrics(np.array([1, 1]), np.array([0.3, 0.8]), 0.5)
    assert np.isnan(m["roc_auc"])
