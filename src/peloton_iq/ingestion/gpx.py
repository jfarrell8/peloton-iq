"""
peloton_iq.ingestion.gpx
~~~~~~~~~~~~~~~~~~~~~~~~~
Parse GPX files into elevation profile DataFrames.

Each GPX file from the Figshare dataset corresponds to a Race Name
in course_data_clean.csv via the convention:
    data/raw/gpx/{Race Name}.gpx

Usage:
    from peloton_iq.ingestion.gpx import load_elevation_profile, find_gpx_path

    df = load_elevation_profile("2023 Tour de France Stage 17")
    # Returns DataFrame with columns: distance_km, elevation_m, gradient_pct
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import pandas as pd

from peloton_iq.config import DATA_RAW_DIR

log = logging.getLogger(__name__)

GPX_DIR = DATA_RAW_DIR / "gpx"

# Downsample target — GPX files have thousands of points, we need ~200 for smooth chart
TARGET_POINTS = 200


def find_gpx_path(race_name: str) -> Optional[Path]:
    """
    Find the GPX file for a given race name.
    Tries exact match first, then normalized fallback.
    """
    # Exact match
    candidate = GPX_DIR / f"{race_name}.gpx"
    if candidate.exists():
        return candidate

    # Normalized match — handle minor whitespace/case differences
    race_norm = race_name.strip().lower()
    for path in GPX_DIR.glob("*.gpx"):
        if path.stem.strip().lower() == race_norm:
            return path

    return None


def load_elevation_profile(
    race_name: str,
    target_points: int = TARGET_POINTS,
) -> Optional[pd.DataFrame]:
    """
    Parse a GPX file into a distance/elevation DataFrame.

    Args:
        race_name:     Race Name as it appears in course_data_clean.csv
        target_points: Number of points to downsample to for charting

    Returns:
        DataFrame with columns: distance_km, elevation_m, gradient_pct
        or None if GPX file not found.
    """
    try:
        import gpxpy
    except ImportError:
        log.error("gpxpy not installed. Run: uv add gpxpy")
        return None

    path = find_gpx_path(race_name)
    if not path:
        log.debug("No GPX file found for: %s", race_name)
        return None

    try:
        with open(path, encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
    except Exception as e:
        log.warning("Failed to parse GPX for %s: %s", race_name, e)
        return None

    # Extract all track points
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    "lat":       point.latitude,
                    "lon":       point.longitude,
                    "elevation": point.elevation or 0.0,
                })

    # Fall back to route points if no tracks
    if not points:
        for route in gpx.routes:
            for point in route.points:
                points.append({
                    "lat":       point.latitude,
                    "lon":       point.longitude,
                    "elevation": point.elevation or 0.0,
                })

    if len(points) < 2:
        log.warning("GPX has fewer than 2 points: %s", race_name)
        return None

    df = pd.DataFrame(points)

    # Compute cumulative distance using Haversine
    df["distance_km"] = 0.0
    cumulative = 0.0
    for i in range(1, len(df)):
        cumulative += _haversine(
            df.iloc[i - 1]["lat"], df.iloc[i - 1]["lon"],
            df.iloc[i]["lat"],     df.iloc[i]["lon"],
        )
        df.at[i, "distance_km"] = cumulative

    # Smooth elevation with rolling window to reduce GPS noise
    df["elevation_m"] = (
        df["elevation"].rolling(window=5, center=True, min_periods=1).mean()
    )

    # Compute gradient
    dist_diff = df["distance_km"].diff().clip(lower=0.001)
    elev_diff = df["elevation_m"].diff()
    df["gradient_pct"] = (elev_diff / (dist_diff * 10)).clip(-25, 25)

    # Downsample to target_points for chart performance
    if len(df) > target_points:
        step = len(df) // target_points
        df   = df.iloc[::step].reset_index(drop=True)
        # Always include last point
        if df.iloc[-1]["distance_km"] < cumulative * 0.99:
            last = pd.DataFrame([{
                "distance_km": cumulative,
                "elevation_m": df["elevation"].iloc[-1],
                "gradient_pct": 0.0,
            }])
            df = pd.concat([df, last], ignore_index=True)

    return df[["distance_km", "elevation_m", "gradient_pct"]].reset_index(drop=True)


def get_climb_annotations(df: pd.DataFrame, min_gradient: float = 5.0) -> list[dict]:
    """
    Identify significant climbs from the elevation profile for chart annotation.
    Returns a list of dicts with: start_km, peak_km, peak_elevation, avg_gradient.
    """
    if df is None or df.empty:
        return []

    climbs      = []
    in_climb    = False
    climb_start = 0.0
    climb_elev  = []

    for _, row in df.iterrows():
        if row["gradient_pct"] >= min_gradient:
            if not in_climb:
                in_climb    = True
                climb_start = row["distance_km"]
                climb_elev  = []
            climb_elev.append((row["distance_km"], row["elevation_m"], row["gradient_pct"]))
        else:
            if in_climb and len(climb_elev) >= 3:
                peak_km, peak_elev, _ = max(climb_elev, key=lambda x: x[1])
                avg_grad = sum(c[2] for c in climb_elev) / len(climb_elev)
                climbs.append({
                    "start_km":      climb_start,
                    "peak_km":       peak_km,
                    "peak_elevation": peak_elev,
                    "avg_gradient":  round(avg_grad, 1),
                })
            in_climb = False

    return climbs


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two lat/lon points."""
    R    = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a    = (math.sin(dlat / 2) ** 2 +
            math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
            math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))