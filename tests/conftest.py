"""tests/conftest.py — shared fixtures for PelotonIQ tests."""

import pytest
import pandas as pd


@pytest.fixture
def merged_df():
    rows = [
        {"Race Name": "2023 Tour de France Stage 17", "Race_results": "Tour de France",
         "Year_results": 2023, "Stage_results": 17.0, "Date": "2023-07-19",
         "Rank": 1,  "Name": "GALL Felix",      "Team": "AG2R Citroën",      "Did_Finish": True,  "Top3": 1, "Top10": 1},
        {"Race Name": "2023 Tour de France Stage 17", "Race_results": "Tour de France",
         "Year_results": 2023, "Stage_results": 17.0, "Date": "2023-07-19",
         "Rank": 2,  "Name": "VINGEGAARD Jonas", "Team": "Jumbo-Visma",       "Did_Finish": True,  "Top3": 1, "Top10": 1},
        {"Race Name": "2023 Tour de France Stage 17", "Race_results": "Tour de France",
         "Year_results": 2023, "Stage_results": 17.0, "Date": "2023-07-19",
         "Rank": 99, "Name": "RIDER DNF",        "Team": "Some Team",         "Did_Finish": False, "Top3": 0, "Top10": 0},
        {"Race Name": "2022 Paris-Roubaix",           "Race_results": "Paris-Roubaix",
         "Year_results": 2022, "Stage_results": None,  "Date": "2022-04-17",
         "Rank": 1,  "Name": "VAN BAARLE Dylan", "Team": "INEOS Grenadiers",  "Did_Finish": True,  "Top3": 1, "Top10": 1},
    ]
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@pytest.fixture
def course_df():
    return pd.DataFrame([
        {"Race Name": "2023 Tour de France Stage 17", "Year": 2023,
         "Distance": 165.8, "Vertical Gain": 5570, "Highest Elevation": 2299,
         "Lowest Elevation": 780, "Cobblestones": 0, "Net Gain": 1000, "Downhill": 4570},
        {"Race Name": "2022 Paris-Roubaix", "Year": 2022,
         "Distance": 257.0, "Vertical Gain": 1200, "Highest Elevation": 180,
         "Lowest Elevation": 10, "Cobblestones": 54.5, "Net Gain": 170, "Downhill": 1030},
    ])