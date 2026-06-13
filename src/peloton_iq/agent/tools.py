"""
peloton_iq.agent.tools
~~~~~~~~~~~~~~~~~~~~~~~
Dataframe tool functions called by structured_node.

Pure dataframe operations — no embeddings, no API calls, no LLM.
Each function returns a plain dict consumed by the synthesizer.

All functions operate on the shared DataFrames injected by AgentDeps
at startup — no file I/O at query time.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def get_stage_winner(
    merged_df: pd.DataFrame,
    race_name: str,
    year: int,
    stage: Optional[int] = None,
) -> dict:
    """Return the winner of a specific race or stage."""
    df = merged_df[
        (merged_df["Year_results"] == year) &
        merged_df["Race_results"].str.contains(race_name, na=False, case=False)
    ]
    if stage is not None:
        df = df[
            (df["Stage_results"] == stage) |
            (df["Stage_results"] == float(stage))
        ]
    if df.empty:
        return {"error": f"No results found for {race_name} {year}"}

    winner = df[df["Rank"] == 1]
    if winner.empty:
        return {"error": f"No rank 1 found for {race_name} {year}"}

    row = winner.iloc[0]
    return {
        "race":   row["Race Name"],
        "winner": row["Name"],
        "team":   row["Team"],
        "year":   int(year),
    }


def get_race_results(
    merged_df: pd.DataFrame,
    race_name: str,
    year: int,
    top_n: int = 10,
) -> dict:
    """Return top-N finishers for a race or stage."""
    df = merged_df[
        merged_df["Race_results"].str.contains(race_name, na=False, case=False) &
        (merged_df["Year_results"] == year) &
        merged_df["Did_Finish"]
    ].copy()

    if df.empty:
        return {"error": f"No results found for {race_name} {year}"}

    one_day = df["Stage_results"].isna().all()
    if one_day:
        top = df.nsmallest(top_n, "Rank")[["Rank", "Name", "Team", "Race Name"]]
    else:
        top = (
            df.groupby("Name")["Rank"].min().reset_index()
            .nsmallest(top_n, "Rank")
            .merge(df[["Name", "Team"]].drop_duplicates("Name"), on="Name")
        )
    return {
        "race":    race_name,
        "year":    int(year),
        "results": top.to_dict(orient="records"),
    }


def get_rider_results(
    merged_df: pd.DataFrame,
    rider_name: str,
    year: int,
) -> dict:
    """Return a rider's season summary for a given year."""
    df = merged_df[
        merged_df["Name"].str.contains(rider_name, na=False, case=False) &
        (merged_df["Year_results"] == year)
    ].copy()

    if df.empty:
        return {"error": f"No results found for {rider_name} in {year}"}

    return {
        "rider":          df["Name"].iloc[0],
        "team":           df["Team"].iloc[0],
        "year":           int(year),
        "races_competed": int(df["Race_results"].nunique()),
        "wins":           df[df["Rank"] == 1]["Race Name"].tolist(),
        "podiums":        df[df["Top3"] == 1]["Race Name"].tolist(),
        "top10s":         int(df["Top10"].sum()),
        "dnfs":           int((~df["Did_Finish"]).sum()),
    }


def get_stage_profile(
    course_df: pd.DataFrame,
    race_name: str,
    year: int,
    stage: Optional[int] = None,
) -> dict:
    """Return course profile data for a race or stage."""
    df = course_df.copy()
    if stage is not None:
        pattern = f"{year} {race_name} Stage {stage}"
        df = df[df["Race Name"] == pattern]
    else:
        df = df[
            df["Race Name"].str.contains(race_name, na=False, case=False) &
            (df["Year"] == year)
        ]

    if df.empty:
        return {"error": f"No course data found for {race_name} {year}"}

    row = df.iloc[0]
    return {
        "race":            row["Race Name"],
        "distance_km":     row.get("Distance"),
        "vertical_gain_m": row.get("Vertical Gain"),
        "highest_elev_m":  row.get("Highest Elevation"),
        "lowest_elev_m":   row.get("Lowest Elevation"),
        "cobblestones_km": row.get("Cobblestones"),
        "net_gain_m":      row.get("Net Gain"),
        "downhill_m":      row.get("Downhill"),
    }


def get_best_mountain_riders(
    merged_df: pd.DataFrame,
    course_df: pd.DataFrame,
    min_vertical_gain: float = 4000.0,
    top_n: int = 10,
) -> dict:
    """
    Find riders with best historical performance on mountain stages
    using direct dataframe computation — more reliable than retrieval
    for numeric terrain queries.
    """
    mountain_stages = course_df[
        course_df["Vertical Gain"] >= min_vertical_gain
    ]["Race Name"].tolist()

    if not mountain_stages:
        return {"error": f"No stages found with >{min_vertical_gain}m vertical gain"}

    mountain_results = merged_df[
        merged_df["Race Name"].isin(mountain_stages) &
        merged_df["Did_Finish"]
    ].copy()

    if mountain_results.empty:
        return {"error": "No results found for mountain stages"}

    rider_stats = (
        mountain_results
        .groupby("Name")
        .agg(
            races=("Race Name", "count"),
            wins=("Rank", lambda x: (x == 1).sum()),
            top3=("Top3", "sum"),
            top10=("Top10", "sum"),
            avg_rank=("Rank", "mean"),
            team=("Team", "last"),
        )
        .reset_index()
    )

    rider_stats = rider_stats[rider_stats["races"] >= 3]
    rider_stats["top10_rate"] = rider_stats["top10"] / rider_stats["races"]
    rider_stats["win_rate"]   = rider_stats["wins"]  / rider_stats["races"]

    top_riders = (
        rider_stats
        .sort_values(["top10_rate", "avg_rank"], ascending=[False, True])
        .head(top_n)
    )

    return {
        "mountain_stage_count": len(mountain_stages),
        "min_vertical_gain":    min_vertical_gain,
        "top_riders": top_riders[[
            "Name", "team", "races", "wins", "top10", "top10_rate", "avg_rank"
        ]].to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(
    params: dict,
    merged_df: pd.DataFrame,
    course_df: pd.DataFrame,
) -> dict:
    """Route structured_params to the right tool function."""
    fn = params.get("function", "")

    if fn == "get_stage_winner":
        return get_stage_winner(
            merged_df,
            params.get("race_name", ""),
            params.get("year", 2023),
            params.get("stage"),
        )
    elif fn == "get_race_results":
        return get_race_results(
            merged_df,
            params.get("race_name", ""),
            params.get("year", 2023),
            params.get("top_n", 10),
        )
    elif fn == "get_rider_results":
        return get_rider_results(
            merged_df,
            params.get("rider_name", ""),
            params.get("year", 2023),
        )
    elif fn == "get_stage_profile":
        return get_stage_profile(
            course_df,
            params.get("race_name", ""),
            params.get("year", 2023),
            params.get("stage"),
        )
    elif fn == "get_best_mountain_riders":
        return get_best_mountain_riders(
            merged_df,
            course_df,
            min_vertical_gain=params.get("min_vertical_gain", 4000.0),
            top_n=params.get("top_n", 10),
        )

    log.warning("Unknown tool function: %s", fn)
    return {"error": f"Unknown function: {fn}"}