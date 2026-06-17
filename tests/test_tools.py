"""tests/test_tools.py — agent dataframe tool functions."""

import pytest
import pandas as pd
from peloton_iq.agent.tools import get_stage_winner, get_race_results, dispatch_tool


def test_stage_winner(merged_df):
    result = get_stage_winner(merged_df, "Tour de France", 2023, stage=17)
    assert result["winner"] == "GALL Felix"
    assert result["team"] == "AG2R Citroën"


def test_stage_winner_unknown_race(merged_df):
    result = get_stage_winner(merged_df, "Tour of Nowhere", 2023)
    assert "error" in result


def test_race_results_excludes_dnf(merged_df):
    result = get_race_results(merged_df, "Tour de France", 2023, top_n=10)
    names = [r["Name"] for r in result["results"]]
    assert "RIDER DNF" not in names


def test_dispatch_tool_routes_correctly(merged_df, course_df):
    result = dispatch_tool(
        {"function": "get_stage_winner", "race_name": "Tour de France", "year": 2023, "stage": 17},
        merged_df, course_df,
    )
    assert result["winner"] == "GALL Felix"


def test_dispatch_tool_unknown_function(merged_df, course_df):
    result = dispatch_tool({"function": "nonexistent"}, merged_df, course_df)
    assert "error" in result