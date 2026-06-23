"""
peloton_iq.ingestion.features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Point-in-time rider feature engineering and GC proxy computation.

These are the most expensive operations in the pipeline — computing
rider history features for ~10,000 rider × race combinations takes
roughly 2 hours from scratch. Results are cached to CSV and only
recomputed when forced.

Usage:
    from peloton_iq.ingestion.features import compute_rider_history, add_gc_proxy

    merged_df  = add_gc_proxy(merged_df)
    rider_feats = compute_rider_history(merged_df)   # loads cache if available
"""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

from peloton_iq.config import RIDER_FEATURES_PATH, MODEL_DF_PATH
from peloton_iq.schemas import StageType
from peloton_iq.commentary.form_features import build_sentiment_table

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage type classification
# ---------------------------------------------------------------------------

def add_stage_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'stage_type' column derived from course metrics.
    Delegates to StageType.classify() — single source of truth.
    """
    def _classify(row) -> str:
        return StageType.classify(
            vertical_gain=row.get("Vertical Gain", 0) or 0,
            distance=row.get("Distance", 1) or 1,
            cobblestones=row.get("Cobblestones", 0) or 0,
        ).value

    df = df.copy()
    df["stage_type"] = df.apply(_classify, axis=1)
    log.debug("Stage type distribution:\n%s", df["stage_type"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# GC proxy
# ---------------------------------------------------------------------------

def add_gc_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'gc_proxy' column — the rider's average rank across all
    completed stages in the current race up to (but not including)
    the current stage.

    NaN for stage 1 and all one-day races — correct by design.
    No future leakage: only stages strictly before the current date are used.

    Column name detection: after the race/course merge, Race and Year get
    suffixed to Race_results / Year_results. Did_Finish and Rank are only
    in race_results so they keep their original names.
    """
    df = df.sort_values("Date").copy()
    df["gc_proxy"] = np.nan

    # Detect correct column names — handle both pre- and post-merge DataFrames
    race_col    = "Race_results" if "Race_results" in df.columns else "Race"
    year_col    = "Year_results" if "Year_results" in df.columns else "Year"
    finish_col  = "Did_Finish"
    rank_col    = "Rank"

    log.debug(
        "gc_proxy using columns: race=%s year=%s finish=%s rank=%s",
        race_col, year_col, finish_col, rank_col,
    )

    groups = df.groupby(["Name", race_col, year_col])
    total  = len(groups)

    for i, ((rider, race, year), group) in enumerate(groups):
        if len(group) < 2:
            continue   # one-day race — no GC proxy

        if i % 1000 == 0:
            log.debug("gc_proxy: %d/%d groups processed", i, total)

        group       = group.sort_values("Date")
        running_sum = 0
        count       = 0
        proxy_vals  = []

        for _, row in group.iterrows():
            proxy_vals.append(running_sum / count if count > 0 else np.nan)
            if row[finish_col] and row[rank_col] < 999:
                running_sum += row[rank_col]
                count       += 1

        df.loc[group.index, "gc_proxy"] = proxy_vals

    non_null = df["gc_proxy"].notna().sum()
    log.info(
        "gc_proxy: %d non-null rows, %d null (one-day races + stage 1s)",
        non_null,
        df["gc_proxy"].isna().sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Rider history features
# ---------------------------------------------------------------------------

def compute_rider_history(
    df: pd.DataFrame,
    cache_path=None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Compute point-in-time lag features for every rider × race row.

    Results are cached to CSV. On subsequent calls the cache is loaded
    unless force_recompute=True.

    Features computed (all strictly before the race date — no leakage):
      - Recent form: avg rank, top-10 rate, win rate, podium rate, DNF rate
        over last 5 races, 6 months, and 12 months
      - Workload: races in last 30 days / 12 months, days since last race
      - Terrain affinity: same stats filtered to the current stage type
      - Career: lifetime rates and counts

    Returns a DataFrame with one row per (Name, Race Name) combination,
    suitable for merging back onto the main DataFrame for model training.
    """
    cache_path = cache_path or RIDER_FEATURES_PATH

    if not force_recompute and cache_path.exists():
        log.info("Loading cached rider features from %s", cache_path)
        cached = pd.read_csv(cache_path, parse_dates=["Date"])
        log.info("Loaded rider features: %s", cached.shape)
        return cached

    log.info("Computing rider features from scratch (~2 hours)...")
    df = df.sort_values("Date").copy()

    # Load commentary sentiment table once — grouped per rider for fast
    # point-in-time slicing inside the loop below (same pattern as the
    # main rider_df groupby). Empty/missing gracefully degrades to NaN
    # features for riders with no commentary coverage.
    sentiment_df = build_sentiment_table()
    if not sentiment_df.empty:
        sentiment_by_rider = {
            name: grp.sort_values("Date")
            for name, grp in sentiment_df.groupby("Name")
        }
        log.info(
            "Commentary sentiment loaded: %d observations for %d riders",
            len(sentiment_df), len(sentiment_by_rider),
        )
    else:
        sentiment_by_rider = {}
        log.warning("No commentary sentiment data available — lagged sentiment features will be all-NaN")

    feature_rows = []
    riders       = df.groupby("Name")
    total        = df["Name"].nunique()

    for i, (rider, rider_df) in enumerate(riders):
        if i % 500 == 0:
            log.info("  %d / %d riders...", i, total)

        rider_df = rider_df.sort_values("Date").reset_index(drop=True)

        for _, row in rider_df.iterrows():
            current_date       = row["Date"]
            current_stage_type = row["stage_type"]

            # Time windows — all strictly before current race
            past      = rider_df[rider_df["Date"] < current_date]
            past_12mo = past[past["Date"] >= current_date - timedelta(days=365)]
            past_6mo  = past[past["Date"] >= current_date - timedelta(days=180)]
            past_30d  = past[past["Date"] >= current_date - timedelta(days=30)]
            past_5    = past.tail(5)

            # Finished-only subsets
            past_fin       = past[past["Did_Finish"]]
            past_12mo_fin  = past_12mo[past_12mo["Did_Finish"]]
            _past_6mo_fin  = past_6mo[past_6mo["Did_Finish"]]
            past_5_fin     = past_5[past_5["Did_Finish"]]

            # Terrain-specific subsets
            past_terrain      = past_fin[past_fin["stage_type"] == current_stage_type]
            past_terrain_12m  = past_12mo_fin[past_12mo_fin["stage_type"] == current_stage_type]

            # ── Lagged commentary sentiment ──────────────────────────────
            # CRITICAL: only sentiment observations strictly BEFORE the
            # current race's date are used — same leakage-safety rule as
            # every other feature in this function. A race's own commentary
            # is never used to predict that same race.
            rider_sentiment = sentiment_by_rider.get(rider)
            if rider_sentiment is not None:
                past_sentiment = rider_sentiment[rider_sentiment["Date"] < current_date]
                past_sentiment_5    = past_sentiment.tail(5)
                past_sentiment_12mo = past_sentiment[
                    past_sentiment["Date"] >= current_date - timedelta(days=365)
                ]
            else:
                past_sentiment      = pd.DataFrame(columns=["sentiment_score"])
                past_sentiment_5    = past_sentiment
                past_sentiment_12mo = past_sentiment

            def _mean(s: pd.Series, min_n: int = 1) -> float | None:
                return float(s.mean()) if len(s) >= min_n else None

            def _rate(s: pd.Series, min_n: int = 1) -> float | None:
                return float(s.mean()) if len(s) >= min_n else None

            feature_rows.append({
                "Name":                   rider,
                "Race Name":              row["Race Name"],
                "Date":                   current_date,
                # Recent form
                "recent_avg_rank_5":       _mean(past_5_fin["Rank"]),
                "recent_avg_rank_12mo":    _mean(past_12mo_fin["Rank"], 3),
                "recent_top10_rate_12mo":  _rate(past_12mo["Top10"], 3),
                "recent_top10_rate_6mo":   _rate(past_6mo["Top10"], 3),
                "recent_win_rate_12mo":    _rate((past_12mo["Rank"] == 1), 3),
                "recent_podium_rate_12mo": _rate(past_12mo["Top3"], 3),
                "recent_dnf_rate_12mo":    _rate((~past_12mo["Did_Finish"]), 3),
                # Workload
                "races_last_30d":          len(past_30d),
                "races_last_12mo":         len(past_12mo),
                "days_since_last_race":    (
                    float((current_date - past["Date"].max()).days)
                    if len(past) > 0 else 999.0
                ),
                # Terrain affinity
                "terrain_avg_rank":        _mean(past_terrain["Rank"], 3),
                "terrain_top10_rate":      _rate(past_terrain["Top10"], 3),
                "terrain_win_rate":        _rate((past_terrain["Rank"] == 1), 3),
                "terrain_podium_rate":     _rate(past_terrain["Top3"], 3),
                "terrain_dnf_rate":        _rate((~past_terrain["Did_Finish"]), 3),
                "terrain_avg_rank_12mo":   _mean(past_terrain_12m["Rank"], 3),
                "terrain_races_count":     len(past_terrain),
                # Career
                "career_top10_rate":       _rate(past_fin["Top10"], 5),
                "career_podium_rate":      _rate(past_fin["Top3"], 5),
                "career_win_rate":         _rate((past_fin["Rank"] == 1), 5),
                "career_races":            len(past),
                "career_avg_rank":         _mean(past_fin["Rank"], 5),
                # Commentary-derived form (lagged — strictly prior races only)
                "commentary_form_5":       _mean(past_sentiment_5["sentiment_score"], 2),
                "commentary_form_12mo":    _mean(past_sentiment_12mo["sentiment_score"], 2),
                "commentary_obs_count":    len(past_sentiment),
            })

    result = pd.DataFrame(feature_rows)
    result.to_csv(cache_path, index=False)
    log.info("Rider features saved to %s  shape=%s", cache_path, result.shape)
    return result


# ---------------------------------------------------------------------------
# Derived course features  (added to model_df, not rider_features)
# ---------------------------------------------------------------------------

def add_course_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add course-level derived columns used by the prediction model.
    Called after merging rider_features onto the main DataFrame.
    """
    df = df.copy()
    df["vg_per_km"]   = df["Vertical Gain"] / df["Distance"].clip(lower=1)
    df["cobble_pct"]  = df["Cobblestones"].fillna(0) / df["Distance"].clip(lower=1)
    df["asphalt_pct"] = df["Asphalt"].fillna(0) / df["Distance"].clip(lower=1)
    df["stage_num"]   = pd.to_numeric(df["Stage_results"], errors="coerce").fillna(0)
    return df


def build_model_df(
    merged_df: pd.DataFrame,
    rider_features: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Merge rider features onto the main DataFrame and add all derived
    course features. Returns the full model-ready DataFrame.

    Optionally saves to MODEL_DF_PATH for use by the agent notebook.
    """
    from sklearn.preprocessing import LabelEncoder

    feature_cols = [c for c in rider_features.columns if c != "Date"]
    model_df = merged_df.merge(
        rider_features[feature_cols],
        on=["Name", "Race Name"],
        how="left",
    )
    model_df = add_course_derived_features(model_df)

    # Encode stage type for the model
    le = LabelEncoder()
    model_df["stage_type_enc"] = le.fit_transform(
        model_df["stage_type"].fillna("flat")
    )

    # Encode tier as integer — mirrors TIER_TO_INT from notebook 05
    tier_order = ["winner", "podium", "top10", "top20", "finisher", "dnf"]
    tier_to_int = {t: i for i, t in enumerate(tier_order)}
    model_df["tier_int"] = model_df["tier"].map(tier_to_int)

    log.info("model_df built: %s", model_df.shape)

    if save:
        model_df.to_csv(MODEL_DF_PATH, index=False)
        log.info("model_df saved to %s", MODEL_DF_PATH)

    return model_df