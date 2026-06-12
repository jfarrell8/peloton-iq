"""
peloton_iq.ingestion.loaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CSV loading and cleaning for raw race data.

Each loader returns a validated, cleaned DataFrame ready for downstream
processing. The cleaning steps mirror notebook 01 exactly — the loaders
are the authoritative entry point for raw data into the system.

Usage:
    from peloton_iq.ingestion.loaders import load_race_results, load_course_data

    race_df   = load_race_results()
    course_df = load_course_data()
"""

from __future__ import annotations

import logging

import pandas as pd

from peloton_iq.config import (
    COURSE_DATA_PATH,
    COURSE_CLEAN_PATH,
    MERGED_RACES_PATH,
    RACE_RESULTS_PATH,
)
from peloton_iq.ingestion.filters import UCIFilter

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_race_name_parts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Year, Stage, and Race columns from the 'Race Name' column.
    Race Name format: "<YYYY> <Race> [Stage <N>]"
    """
    df = df.copy()
    df["Year"] = df["Race Name"].str.extract(r"^(\d{4})").astype(int)
    df["Stage"] = df["Race Name"].str.extract(r"Stage\s+(\d+)")
    df["Race"] = (
        df["Race Name"]
        .str.replace(r"^\d{4}\s+", "", regex=True)
        .str.replace(r"\s*Stage\s+\d+", "", regex=True)
        .str.strip()
    )
    return df



_NON_FINISHES = frozenset({"DNF", "DNS", "DSQ", "OTL", "DF", "NR"})
_BAD_RANK_VALUES = frozenset({"1006.0", "10077"})


def _clean_rank(val) -> tuple[int, bool]:
    """
    Parse a raw Rank value into (rank, did_finish).

    Non-finishes (DNF/DNS/DSQ/OTL/DF/NR) and known bad values
    map to (999, False) so they never score as top-N results.
    Mirrors clean_rank() from notebook 01.
    """
    s = str(val).strip()
    if s in _NON_FINISHES or s in _BAD_RANK_VALUES:
        return 999, False
    try:
        r = int(float(s))
        return r, True
    except (ValueError, TypeError):
        return 999, False


def _derive_result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Did_Finish, Top3, Top10 from the raw Rank column.
    Must be called before any filtering that uses these columns.
    """
    cleaned         = df["Rank"].apply(_clean_rank)
    df["Rank"]       = cleaned.apply(lambda x: x[0])
    df["Did_Finish"] = cleaned.apply(lambda x: x[1])
    df["Top10"]      = ((df["Rank"] <= 10) & df["Did_Finish"]).astype(int)
    df["Top3"]       = ((df["Rank"] <= 3)  & df["Did_Finish"]).astype(int)

    log.info(
        "race_results: %d finishers, %d non-finishers (DNF/DNS/etc)",
        df["Did_Finish"].sum(),
        (~df["Did_Finish"]).sum(),
    )
    return df


def _apply_uci_filter(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Apply UCIFilter to a DataFrame that has Year and Race columns.
    Drops non-UCI rows and logs a summary.
    """
    uci_filter = UCIFilter()
    before = len(df)

    results = df.apply(
        lambda row: uci_filter.is_uci_race(row["Year"], row["Race"]), axis=1
    )
    df["is_uci"]       = results.apply(lambda r: r.is_uci)
    df["match_reason"] = results.apply(lambda r: r.match_reason)

    fuzzy_matches = df[df["match_reason"].str.startswith("fuzzy", na=False)]
    if not fuzzy_matches.empty:
        log.debug(
            "%s: %d fuzzy matches — review if unexpected:\n%s",
            label,
            len(fuzzy_matches),
            fuzzy_matches[["Race Name", "Year", "match_reason"]]
            .drop_duplicates("Race Name")
            .to_string(),
        )

    df = df[df["is_uci"]].copy()
    log.info("%s: UCI filter %d → %d rows (dropped %d)", label, before, len(df), before - len(df))

    fuzzy_count = df["match_reason"].str.startswith("fuzzy", na=False).sum()
    if fuzzy_count:
        log.info("%s: %d rows matched via fuzzy (review match_reason column)", label, fuzzy_count)

    return df


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_race_results(path=None) -> pd.DataFrame:
    """
    Load and clean the raw race results CSV.

    Steps (matching notebook 01):
      1. Parse dates
      2. Derive Year / Stage / Race from Race Name
      3. Derive Did_Finish, Top3, Top10 from raw Rank
      4. Drop women's / junior races
      5. Apply UCI WorldTour filter
      6. Drop known-bad records

    Returns a cleaned DataFrame ready for feature engineering.
    """
    path = path or RACE_RESULTS_PATH
    log.info("Loading race results from %s", path)

    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = _parse_race_name_parts(df)
    df = _derive_result_columns(df)

    # Drop women's / junior races
    uci_filter = UCIFilter()
    before = len(df)
    df = df[~df["Race Name"].apply(uci_filter.is_excluded)]
    log.info("race_results: exclusion filter %d → %d rows", before, len(df))

    df = _apply_uci_filter(df, "race_results")
    # Drop filter audit columns — these are kept on course_data for auditing
    # but would create noisy suffixed duplicates in the merged DataFrame
    df = df.drop(columns=["is_uci", "match_reason"])
    return df


def load_course_data(path=None) -> pd.DataFrame:
    """
    Load and clean the raw course data CSV.

    Steps (matching notebook 01):
      1. Derive Year / Stage / Race from Race Name
      2. Drop women's / junior races
      3. Apply UCI WorldTour filter
      4. Drop known-bad records (Gent-Wevelgem distances, TdS 2023 summary row,
         Liege 2022 duplicate)

    Returns a cleaned DataFrame ready for serialization and embedding.
    """
    path = path or COURSE_DATA_PATH
    log.info("Loading course data from %s", path)

    df = pd.read_csv(path, index_col=0)
    # Drop stale integer index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = _parse_race_name_parts(df)

    uci_filter = UCIFilter()
    before = len(df)
    df = df[~df["Race Name"].apply(uci_filter.is_excluded)]
    log.info("course_data: exclusion filter %d → %d rows", before, len(df))

    df = _apply_uci_filter(df, "course_data")

    # ------------------------------------------------------------------
    # Drop known-bad records identified during EDA in notebook 01
    # ------------------------------------------------------------------

    # Gent-Wevelgem: wrong distances across multiple years
    df = df[~df["Race Name"].str.contains("Wevelgem", na=False)]

    # 2023 Tour de Suisse summary row with no stage number
    bad_tds = df[
        (df["Race Name"] == "2023 Tour de Suisse") & df["Stage"].isna()
    ].index
    df = df.drop(bad_tds)

    # Liege 2022 duplicate — keep the longer distance (257km is correct)
    liege_2022 = df[
        df["Race Name"].str.contains("Liege", na=False) &
        (df["Year"] == 2022)
    ]
    if len(liege_2022) > 1:
        keep_idx = liege_2022["Distance"].idxmax()
        drop_idx = liege_2022.index.difference([keep_idx])
        df = df.drop(drop_idx)
        log.debug("Dropped %d Liege 2022 duplicate row(s)", len(drop_idx))

    log.info("course_data: final shape %s", df.shape)
    return df


def merge_race_and_course(
    race_df: pd.DataFrame,
    course_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner join race results and course data on Race Name.

    Suffixes disambiguate columns that exist in both DataFrames
    (Year, Stage, Race) — matching the notebook convention:
      _results  -> from race_results
      _course   -> from course_data
    """
    before_race   = len(race_df)
    before_course = len(course_df)

    merged = pd.merge(
        race_df,
        course_df,
        on="Race Name",
        how="inner",
        suffixes=("_results", "_course"),
    )

    log.info(
        "merge: race_results(%d) x course_data(%d) -> merged(%d rows, %d cols)",
        before_race, before_course, len(merged), merged.shape[1],
    )

    dropped = before_race - len(merged)
    if dropped > 0:
        log.warning(
            "merge: %d race_result rows had no matching course profile (inner join)",
            dropped,
        )

    return merged


def load_merged_races(path=None) -> pd.DataFrame:
    """
    Load the pre-merged and cleaned races DataFrame from processed/.
    This is the output of a full ingestion run — used by downstream
    modules (search, prediction, agent) that don't need to re-run cleaning.
    """
    path = path or MERGED_RACES_PATH
    log.info("Loading merged races from %s", path)
    df = pd.read_csv(path, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_course_data_clean(path=None) -> pd.DataFrame:
    """
    Load the pre-cleaned course data from processed/.
    Used by downstream modules that don't need to re-run cleaning.
    """
    path = path or COURSE_CLEAN_PATH
    log.info("Loading clean course data from %s", path)
    return pd.read_csv(path)