"""
peloton_iq.commentary.form_features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Builds a long-format (rider, race_date, sentiment_score) table from
Claude-extracted commentary, for use as a LAGGED feature in the
prediction model.

CRITICAL — leakage safety:
This module only produces the raw sentiment observations. It does NOT
decide which observations are "available" for which target race — that
temporal cutoff logic lives in ingestion/features.py, exactly like the
existing recent_avg_rank_5 / terrain_avg_rank features. A rider's
sentiment score for race X must never be computed using commentary
extracted FROM race X itself when X is the prediction target.

Sentiment encoding (from rider_signals in the raw Claude extraction):
    "positive: ..." -> +1.0
    "neutral: ..."  ->  0.0
    "negative: ..." -> -1.0
    unparseable      -> dropped (not included in the table)

Usage:
    from peloton_iq.commentary.form_features import build_sentiment_table

    sentiment_df = build_sentiment_table()
    # columns: Name, Date, sentiment_score, race_label
    sentiment_df.to_csv(COMMENTARY_SENTIMENT_PATH, index=False)
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

import pandas as pd

from peloton_iq.config import COMMENTARY_EXTRACTED_DIR

log = logging.getLogger(__name__)

SENTIMENT_MAP = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral":  0.0,
}


def _normalize_rider_name(name: str) -> str:
    """
    Strip accents only — preserves case/spacing so this matches the
    `Name` column in merged_uci_races.csv (e.g. "POGACAR Tadej").
    Mirrors RiderProfiler._normalize_rider_name exactly.
    """
    name = name.strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", name)
        if unicodedata.category(c) != "Mn"
    )


def _parse_signal(signal_text: str) -> float | None:
    """
    Parse a rider_signals value like "positive: attacked early and held on"
    into a numeric sentiment score. Returns None if unparseable.
    """
    if not signal_text or not isinstance(signal_text, str):
        return None
    prefix = signal_text.split(":", 1)[0].strip().lower()
    return SENTIMENT_MAP.get(prefix)


def build_sentiment_table(
    extracted_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Walk all extracted commentary JSON files and build a long-format
    sentiment table: one row per (rider, race) observation.

    Returns a DataFrame with columns:
        Name            — normalized rider name, matches merged_df["Name"]
        Date            — race date (datetime64)
        race_label      — the commentary label, for traceability/debugging
        sentiment_score — +1.0 / 0.0 / -1.0
    """
    extracted_dir = extracted_dir or COMMENTARY_EXTRACTED_DIR

    rows = []
    skipped_unusable  = 0
    skipped_no_signal = 0
    parse_errors      = 0

    for path in sorted(extracted_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            log.warning("Failed to read %s: %s", path.name, e)
            parse_errors += 1
            continue

        extraction = data.get("extraction", {})
        if not extraction:
            skipped_no_signal += 1
            continue

        # Respect the extraction's own usability flag when present
        if extraction.get("usable_for_rag") is False:
            skipped_unusable += 1
            continue

        race_date = data.get("race_date", "")
        label     = data.get("label", path.stem)
        if not race_date:
            skipped_no_signal += 1
            continue

        rider_signals = extraction.get("rider_signals", {})
        if not rider_signals:
            skipped_no_signal += 1
            continue

        for rider, signal_text in rider_signals.items():
            score = _parse_signal(signal_text)
            if score is None:
                continue
            rows.append({
                "Name":            _normalize_rider_name(rider),
                "Date":            race_date,
                "race_label":      label,
                "sentiment_score": score,
            })

    if not rows:
        log.warning(
            "No sentiment observations extracted (unusable=%d, no_signal=%d, errors=%d)",
            skipped_unusable, skipped_no_signal, parse_errors,
        )
        return pd.DataFrame(columns=["Name", "Date", "race_label", "sentiment_score"])

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Name", "Date"]).reset_index(drop=True)

    log.info(
        "Sentiment table built: %d observations, %d unique riders "
        "(skipped: %d unusable, %d no-signal, %d parse-errors)",
        len(df), df["Name"].nunique(),
        skipped_unusable, skipped_no_signal, parse_errors,
    )
    return df


def coverage_report(sentiment_df: pd.DataFrame | None = None) -> dict:
    """Quick summary stats on how much sentiment data is available."""
    if sentiment_df is None:
        sentiment_df = build_sentiment_table()

    if sentiment_df.empty:
        return {"total_observations": 0, "unique_riders": 0}

    by_rider = sentiment_df.groupby("Name").size().sort_values(ascending=False)

    return {
        "total_observations":      len(sentiment_df),
        "unique_riders":           sentiment_df["Name"].nunique(),
        "date_range":              (
            str(sentiment_df["Date"].min().date()),
            str(sentiment_df["Date"].max().date()),
        ),
        "sentiment_distribution":  sentiment_df["sentiment_score"].value_counts().to_dict(),
        "riders_with_5plus_obs":   int((by_rider >= 5).sum()),
        "riders_with_2plus_obs":   int((by_rider >= 2).sum()),
        "top_covered_riders":      by_rider.head(15).to_dict(),
    }