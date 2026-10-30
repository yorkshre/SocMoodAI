import numpy as np
import pandas as pd
import pytest

from dd.baseline import MODES, ablation, build_pipeline
from dd.data import load_posts
from dd.features import build_user_table


def _synthetic_users(n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = i % 2
        base = "I never feel okay nothing helps" if label else "we had a nice walk today"
        for k in range(3):
            rows.append(
                dict(user_id=f"u{i}", timestamp=f"2024-01-0{k+1} {rng.integers(0, 24):02d}:00",
                     text=f"{base} day{k}", label=label, lang="en")
            )
    return build_user_table(pd.DataFrame(rows))


def test_ablation_runs_and_separable_data_scores_high():
    res = ablation(_synthetic_users(), n_splits=4)
    assert list(res.index) == list(MODES)
    assert res.loc["text", "roc_auc"] > 0.9


def test_bad_mode():
    with pytest.raises(ValueError):
        build_pipeline("nope")


def test_load_posts_validates(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"user_id": [1], "text": ["a"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Missing"):
        load_posts(str(p))
    pd.DataFrame(
        {"user_id": [1, 1], "timestamp": ["2024-01-01"] * 2, "text": ["a", "b"], "label": [0, 1]}
    ).to_csv(p, index=False)
    with pytest.raises(ValueError, match="Inconsistent"):
        load_posts(str(p))
