import pandas as pd
import pytest

from dd.features import build_user_table, post_ratios


def test_ratios_en():
    fp, ab, n = post_ratios("I never feel like myself", "en")
    assert n == 5
    assert fp == pytest.approx(2 / 5)  # I, myself
    assert ab == pytest.approx(1 / 5)  # never


def test_ratios_pl():
    fp, ab, n = post_ratios("Ja nigdy nic nie czuję", "pl")
    assert n == 5
    assert fp == pytest.approx(1 / 5)
    assert ab == pytest.approx(2 / 5)


def test_empty_text_no_division_error():
    assert post_ratios("", "en") == (0.0, 0.0, 0)
    assert post_ratios("!!!", "pl") == (0.0, 0.0, 0)


def test_unknown_lang_falls_back():
    assert post_ratios("I always", "unk")[0] == pytest.approx(0.5)


def test_user_table_aggregation_and_night():
    df = pd.DataFrame(
        {
            "user_id": ["a", "a", "b"],
            "timestamp": ["2024-01-01 02:00", "2024-01-01 14:00", "not a date"],
            "text": ["I never sleep", "Nothing helps me", "Fine day https://x.co"],
            "label": [1, 1, 0],
            "lang": ["en", "en", "en"],
        }
    )
    t = build_user_table(df).set_index("user_id")
    assert t.loc["a", "night_share"] == pytest.approx(0.5)
    assert t.loc["b", "night_share"] == 0.0  # unparsable timestamp -> 0, not NaN
    assert t.loc["a", "label"] == 1 and t.loc["b", "label"] == 0
    assert "http" not in t.loc["b", "text"]
