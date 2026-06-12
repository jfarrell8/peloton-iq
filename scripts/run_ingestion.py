"""
scripts/run_ingestion.py
~~~~~~~~~~~~~~~~~~~~~~~~
CLI runner for the full data ingestion pipeline.

Reads from data/raw/, writes to data/test_outputs/ by default so
known-good processed files are never overwritten during validation runs.

Usage:
    # Test run — writes to data/test_outputs/
    python scripts/run_ingestion.py

    # Production run — writes to data/processed/
    python scripts/run_ingestion.py --prod

    # Skip the ~2hr rider features computation (use cached if available)
    python scripts/run_ingestion.py --skip-features

    # Force recompute rider features even if cache exists
    python scripts/run_ingestion.py --force-features
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make sure src/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from peloton_iq.config import (
    settings,
    DATA_PROCESSED_DIR,
    MERGED_RACES_PATH,
    COURSE_CLEAN_PATH,
    RIDER_FEATURES_PATH,
    MODEL_DF_PATH,
    TEST_OUTPUTS_DIR,
    TEST_MERGED_RACES_PATH,
    TEST_COURSE_CLEAN_PATH,
    TEST_RIDER_FEATURES_PATH,
    TEST_MODEL_DF_PATH,
)
from peloton_iq.ingestion.loaders import load_race_results, load_course_data, merge_race_and_course
from peloton_iq.ingestion.features import (
    add_stage_type,
    add_gc_proxy,
    compute_rider_history,
    build_model_df,
)
from peloton_iq.schemas import FinishTier

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_ingestion")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    log.info("=" * 60)
    log.info("  %s", title)
    log.info("=" * 60)


def _compare_to_known_good(
    test_df: pd.DataFrame,
    known_path: Path,
    label: str,
) -> None:
    """Load known-good CSV and print a basic diff summary."""
    if not known_path.exists():
        log.warning("  No known-good file at %s — skipping comparison", known_path)
        return

    known_df = pd.read_csv(known_path, low_memory=False)
    log.info("  [%s] test shape:  %s", label, test_df.shape)
    log.info("  [%s] known shape: %s", label, known_df.shape)

    row_diff = len(test_df) - len(known_df)
    col_diff = set(test_df.columns).symmetric_difference(set(known_df.columns))

    if row_diff == 0:
        log.info("  [%s] ✓ Row count matches (%d)", label, len(test_df))
    else:
        log.warning("  [%s] ✗ Row count differs by %+d", label, row_diff)

    if not col_diff:
        log.info("  [%s] ✓ Column set matches", label)
    else:
        log.warning("  [%s] ✗ Column diff: %s", label, col_diff)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_load_and_clean(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _print_section("STEP 1 — Load & Clean Raw Data")
    t0 = time.time()

    race_df   = load_race_results()
    course_df = load_course_data()
    merged_df = merge_race_and_course(race_df, course_df)

    log.info("race_results:  %d rows x %d cols", *race_df.shape)
    log.info("course_data:   %d rows x %d cols", *course_df.shape)
    log.info("merged_df:     %d rows x %d cols", *merged_df.shape)
    log.info("Years in merged_df: %s", sorted(merged_df["Year_results"].unique()))
    log.info("Elapsed: %.1fs", time.time() - t0)

    return race_df, course_df, merged_df


def step_enrich(race_df: pd.DataFrame) -> pd.DataFrame:
    _print_section("STEP 2 — Enrich: Stage Type + GC Proxy")
    t0 = time.time()

    race_df = add_stage_type(race_df)
    log.info("Stage type distribution:\n%s", race_df["stage_type"].value_counts().to_string())

    race_df = add_gc_proxy(race_df)
    log.info(
        "GC proxy: %d non-null  |  %d null (one-day + stage 1s)",
        race_df["gc_proxy"].notna().sum(),
        race_df["gc_proxy"].isna().sum(),
    )

    # Add tier column
    race_df["tier"] = race_df.apply(
        lambda r: FinishTier.from_rank(r["Rank"], r["Did_Finish"]).value, axis=1
    )
    log.info("Tier distribution:\n%s", race_df["tier"].value_counts().to_string())
    log.info("Elapsed: %.1fs", time.time() - t0)

    return race_df


def step_rider_features(
    race_df: pd.DataFrame,
    cache_path: Path,
    force_recompute: bool,
) -> pd.DataFrame:
    _print_section("STEP 3 — Rider History Features")
    t0 = time.time()

    rider_feats = compute_rider_history(
        race_df,
        cache_path=cache_path,
        force_recompute=force_recompute,
    )

    log.info("rider_features: %d rows x %d cols", *rider_feats.shape)
    log.info(
        "Null rates in key features:\n%s",
        rider_feats[["recent_avg_rank_5", "terrain_avg_rank", "career_top10_rate"]]
        .isna()
        .mean()
        .mul(100)
        .round(1)
        .to_string(),
    )
    log.info("Elapsed: %.1fs", time.time() - t0)

    return rider_feats


def step_build_model_df(
    race_df: pd.DataFrame,
    rider_feats: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    _print_section("STEP 4 — Build model_df")
    t0 = time.time()

    model_df = build_model_df(race_df, rider_feats, save=False)
    model_df.to_csv(out_path, index=False)
    log.info("model_df saved → %s", out_path)
    log.info("model_df: %d rows x %d cols", *model_df.shape)

    # Null rates on key model features
    key_feats = [
        "recent_avg_rank_5", "recent_top10_rate_12mo",
        "terrain_avg_rank", "terrain_top10_rate",
        "career_top10_rate", "gc_proxy",
    ]
    available = [f for f in key_feats if f in model_df.columns]
    log.info(
        "Null rates in model features:\n%s",
        model_df[available].isna().mean().mul(100).round(1).to_string(),
    )
    log.info("Elapsed: %.1fs", time.time() - t0)

    return model_df


def step_save_intermediates(
    race_df: pd.DataFrame,
    course_df: pd.DataFrame,
    out_dir: Path,
    merged_path: Path,
    course_path: Path,
) -> None:
    _print_section("STEP 5 — Save Intermediate CSVs")

    race_df.to_csv(merged_path, index=False)
    log.info("merged_uci_races saved  → %s", merged_path)

    course_df.to_csv(course_path, index=False)
    log.info("course_data_clean saved → %s", course_path)


def step_compare(
    race_df: pd.DataFrame,
    course_df: pd.DataFrame,
    model_df: pd.DataFrame,
    rider_feats: pd.DataFrame,
) -> None:
    _print_section("STEP 6 — Compare Against Known-Good Outputs")

    _compare_to_known_good(race_df,    MERGED_RACES_PATH,    "merged_uci_races")
    _compare_to_known_good(course_df,  COURSE_CLEAN_PATH,    "course_data_clean")
    _compare_to_known_good(rider_feats, RIDER_FEATURES_PATH, "rider_features")
    _compare_to_known_good(model_df,   MODEL_DF_PATH,        "model_df")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PelotonIQ ingestion pipeline")
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Write to data/processed/ instead of data/test_outputs/",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip rider feature computation (use cached CSV if available)",
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Force recompute rider features even if cache exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve output paths
    if args.prod:
        out_dir      = DATA_PROCESSED_DIR
        merged_path  = MERGED_RACES_PATH
        course_path  = COURSE_CLEAN_PATH
        features_path = RIDER_FEATURES_PATH
        model_path   = MODEL_DF_PATH
        log.info("MODE: PRODUCTION — writing to data/processed/")
    else:
        out_dir       = TEST_OUTPUTS_DIR
        merged_path   = TEST_MERGED_RACES_PATH
        course_path   = TEST_COURSE_CLEAN_PATH
        features_path = TEST_RIDER_FEATURES_PATH
        model_path    = TEST_MODEL_DF_PATH
        log.info("MODE: TEST — writing to data/test_outputs/")

    out_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # Run pipeline
    race_df, course_df, merged_df = step_load_and_clean(out_dir)
    merged_df          = step_enrich(merged_df)

    step_save_intermediates(merged_df, course_df, out_dir, merged_path, course_path)

    if args.skip_features:
        # Skip the ~2hr compute but still load from cache and rebuild model_df
        # so the comparison step can run
        if features_path.exists():
            log.info("--skip-features: loading rider features from cache %s", features_path)
            rider_feats = pd.read_csv(features_path, parse_dates=["Date"])
            log.info("rider_features loaded from cache: %s", rider_feats.shape)
            model_df = step_build_model_df(merged_df, rider_feats, model_path)
            step_compare(merged_df, course_df, model_df, rider_feats)
        else:
            log.warning(
                "--skip-features: no cache found at %s — skipping model_df and comparison",
                features_path,
            )
    else:
        rider_feats = step_rider_features(merged_df, features_path, args.force_features)
        model_df    = step_build_model_df(merged_df, rider_feats, model_path)
        step_compare(merged_df, course_df, model_df, rider_feats)

    log.info("=" * 60)
    log.info("  INGESTION COMPLETE  (%.1fs total)", time.time() - total_start)
    log.info("  Outputs in: %s", out_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()