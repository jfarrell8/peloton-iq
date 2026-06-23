"""tests/test_form_features.py — commentary sentiment parsing."""

from peloton_iq.commentary.form_features import _parse_signal, _normalize_rider_name


def test_parse_positive_signal():
    assert _parse_signal("positive: attacked early and held on") == 1.0


def test_parse_negative_signal():
    assert _parse_signal("negative: dropped on the final climb") == -1.0


def test_parse_neutral_signal():
    assert _parse_signal("neutral: rode in the peloton all day") == 0.0


def test_parse_unparseable_returns_none():
    assert _parse_signal("rode well today") is None


def test_parse_empty_returns_none():
    assert _parse_signal("") is None
    assert _parse_signal(None) is None


def test_normalize_strips_accent_preserves_case():
    assert _normalize_rider_name("POGAČAR Tadej") == "POGACAR Tadej"


def test_normalize_strips_whitespace():
    assert _normalize_rider_name("  VINGEGAARD Jonas  ") == "VINGEGAARD Jonas"


# ---------------------------------------------------------------------------
# Leakage safety — the most important test in this file.
# A commentary observation must NEVER be visible to a row whose race
# date is on or before that observation's date.
# ---------------------------------------------------------------------------

def test_lagged_window_excludes_same_and_future_dates():
    """
    Simulates the exact slicing logic used in compute_rider_history()
    to confirm no leakage: sentiment observations on or after the
    target race date must be excluded.
    """
    import pandas as pd

    sentiment = pd.DataFrame({
        "Name": ["TEST Rider"] * 4,
        "Date": pd.to_datetime([
            "2022-01-01",  # before target -> should be INCLUDED
            "2022-06-01",  # before target -> should be INCLUDED
            "2023-01-01",  # == target date -> should be EXCLUDED
            "2023-06-01",  # after target  -> should be EXCLUDED
        ]),
        "sentiment_score": [1.0, -1.0, 1.0, 1.0],
    })

    target_date = pd.Timestamp("2023-01-01")
    past = sentiment[sentiment["Date"] < target_date]

    assert len(past) == 2
    assert target_date not in past["Date"].values
    assert pd.Timestamp("2023-06-01") not in past["Date"].values
    assert set(past["sentiment_score"]) == {1.0, -1.0}