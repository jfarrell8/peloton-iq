"""tests/test_commentary.py — label utilities and video matching logic."""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("youtube_transcript_api", MagicMock())
sys.modules.setdefault("anthropic", MagicMock())

from peloton_iq.commentary.transcript import make_label, make_safe_name, _wrong_race_penalty
from peloton_iq.commentary.profiler import RiderProfiler


def test_make_label_stage_race():
    assert make_label("Tour de France", "2023-07-19", 17) == "2023 Tour de France Stage 17"


def test_make_label_one_day():
    assert make_label("Paris-Roubaix", "2022-04-17", None) == "2022 Paris-Roubaix"


def test_make_safe_name():
    assert make_safe_name("2023 Tour de France Stage 17") == "2023_tour_de_france_stage_17"


def test_wrong_race_penalty_giro_rosa():
    # Giro Rosa should never match Giro d'Italia
    penalty = _wrong_race_penalty("Giro d'Italia", "Giro Rosa 2019 Stage 5 Highlights")
    assert penalty < 0


def test_wrong_race_penalty_correct_race():
    penalty = _wrong_race_penalty("Tour de France", "Tour de France 2023 Stage 17 Highlights")
    assert penalty == 0.0


def test_rider_safe_name_accent_normalization():
    # POGAČAR and POGACAR should produce the same filename
    assert RiderProfiler._safe_name("POGAČAR Tadej") == RiderProfiler._safe_name("POGACAR Tadej")