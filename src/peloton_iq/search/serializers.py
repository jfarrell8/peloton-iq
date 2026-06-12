"""
peloton_iq.search.serializers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Document serializers for embedding and search.

These functions convert structured DataFrames into natural language
documents suitable for embedding. They were duplicated across notebooks
02, 03, and 08 — this module is the single source of truth.

Usage:
    from peloton_iq.search.serializers import serialize_course_doc, serialize_rider_doc

    text = serialize_course_doc(course_data.iloc[0])
    text = serialize_rider_doc("POGAČAR Tadej", 2023, merged_df)
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Course document serializer
# ---------------------------------------------------------------------------

def serialize_course_doc(row: pd.Series) -> str:
    """
    Convert a course_data row into a natural language document
    suitable for embedding.

    Numbers alone embed poorly — prose context matters. This mirrors
    the serializer from notebooks 02 and 03 exactly.
    """
    parts = []

    # Race identity
    parts.append(f"Race: {row['Race Name']}.")

    # Distance
    if pd.notna(row.get("Distance")):
        parts.append(f"Distance: {row['Distance']:.1f} km.")

    # Elevation profile
    if pd.notna(row.get("Vertical Gain")) and pd.notna(row.get("Highest Elevation")):
        vg = row["Vertical Gain"]
        he = row["Highest Elevation"]
        le = row.get("Lowest Elevation", 0) or 0
        ng = row.get("Net Gain", 0) or 0

        if vg > 4000:
            stage_desc = "a high mountain stage"
        elif vg > 2000:
            stage_desc = "a hilly stage with significant climbing"
        elif vg > 800:
            stage_desc = "a moderately hilly stage"
        else:
            stage_desc = "a flat stage suited to sprinters"

        parts.append(
            f"This is {stage_desc} with {vg:.0f}m of vertical gain, "
            f"reaching a maximum elevation of {he:.0f}m "
            f"and a minimum elevation of {le:.0f}m. "
            f"Net elevation change: {ng:.0f}m."
        )

    # Surface composition
    surfaces = {
        "Asphalt":          "asphalt",
        "Road":             "road",
        "Cobblestones":     "cobblestones",
        "Compacted Gravel": "compacted gravel",
        "Unpaved":          "unpaved sections",
        "Paved":            "paved road",
    }
    surface_parts = []
    for col, label in surfaces.items():
        val = row.get(col)
        if pd.notna(val) and float(val) > 0.5:
            surface_parts.append(f"{val:.1f}km of {label}")

    if surface_parts:
        parts.append(f"Surface breakdown: {', '.join(surface_parts)}.")

    # Cobblestone flag — important for classics
    cob = row.get("Cobblestones")
    if pd.notna(cob) and float(cob) > 0:
        parts.append(
            f"Contains {cob:.1f}km of cobblestones — "
            f"a significant factor for classics specialists."
        )

    # Descending
    dh = row.get("Downhill")
    if pd.notna(dh) and float(dh) > 0:
        parts.append(f"Total descending: {dh:.0f}m.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Rider season document serializer
# ---------------------------------------------------------------------------

def serialize_rider_doc(rider_name: str, year: int, df: pd.DataFrame) -> str:
    """
    Build a natural language summary of a rider's season from their
    race results for that year.

    Args:
        rider_name: Rider name in ALL-CAPS SURNAME Firstname format.
        year:       Season year (matches Year_results in merged_df).
        df:         merged_df containing all race results.

    Returns:
        A prose document string, or empty string if no data found.
    """
    rider_df = df[
        (df["Name"] == rider_name) &
        (df["Year_results"] == year)
    ].copy()

    if rider_df.empty:
        return ""

    parts = []
    parts.append(f"Rider: {rider_name}. Season: {year}.")

    # Team
    team = rider_df["Team"].iloc[0]
    if pd.notna(team):
        parts.append(f"Team: {team}.")

    # Race count
    races = rider_df["Race_results"].unique()
    parts.append(f"Competed in {len(races)} races.")

    # Win / podium / top10 / DNF counts
    wins    = rider_df[rider_df["Rank"] == 1]
    podiums = rider_df[rider_df["Top3"] == 1]
    top10s  = rider_df[rider_df["Top10"] == 1]
    dnfs    = rider_df[~rider_df["Did_Finish"]]

    parts.append(
        f"Results: {len(wins)} wins, "
        f"{len(podiums)} podiums, "
        f"{len(top10s)} top-10 finishes, "
        f"{len(dnfs)} DNFs/DNS."
    )

    # List wins explicitly (cap at 5 to keep doc length reasonable)
    if not wins.empty:
        win_names = wins["Race Name"].tolist()
        parts.append(f"Won: {', '.join(win_names[:5])}.")

    # Best non-win results
    top_results = rider_df[
        (rider_df["Rank"] > 1) & (rider_df["Did_Finish"])
    ].nsmallest(3, "Rank")

    if not top_results.empty:
        best = [
            f"{row['Race Name']} (P{int(row['Rank'])})"
            for _, row in top_results.iterrows()
        ]
        parts.append(f"Other notable results: {', '.join(best)}.")

    # Grand Tour best results
    gc_races = ["Tour de France", "Giro d", "Vuelta a Espana"]
    for gc in gc_races:
        gc_results = rider_df[rider_df["Race_results"].str.contains(gc, na=False)]
        if not gc_results.empty:
            best_gc = gc_results[gc_results["Did_Finish"]]["Rank"].min()
            if best_gc < 999:
                parts.append(f"{gc}: best result P{int(best_gc)}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Document builder helpers  (used by EmbeddingStore)
# ---------------------------------------------------------------------------

def build_course_docs(course_df: pd.DataFrame) -> list[dict]:
    """
    Generate the full list of course document dicts ready for upsert.
    Each dict has: id, text, metadata.
    """
    docs = []
    for _, row in course_df.iterrows():
        text = serialize_course_doc(row)
        if not text.strip():
            continue
        docs.append({
            "id":   row["Race Name"],
            "text": text,
            "metadata": {
                "race_name":     row["Race Name"],
                "year":          int(row.get("Year", 0) or 0),
                "race":          str(row.get("Race", "") or ""),
                "stage":         str(row.get("Stage", "") or ""),
                "distance":      float(row["Distance"]) if pd.notna(row.get("Distance")) else None,
                "vertical_gain": float(row["Vertical Gain"]) if pd.notna(row.get("Vertical Gain")) else None,
                "doc_type":      "course_profile",
            },
        })
    return docs


def build_rider_docs(merged_df: pd.DataFrame) -> list[dict]:
    """
    Generate the full list of rider season document dicts ready for upsert.
    Each dict has: id, text, metadata.
    """
    docs = []
    rider_seasons = merged_df.groupby(["Name", "Year_results"]).size().reset_index()

    for _, row in rider_seasons.iterrows():
        text = serialize_rider_doc(row["Name"], row["Year_results"], merged_df)
        if not text.strip():
            continue
        docs.append({
            "id":   f"{row['Name']}_{row['Year_results']}",
            "text": text,
            "metadata": {
                "rider_name": row["Name"],
                "year":       int(row["Year_results"]),
                "doc_type":   "rider_season",
            },
        })
    return docs