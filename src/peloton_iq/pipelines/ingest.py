"""
peloton_iq.pipelines.ingest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prefect flow for the full data ingestion pipeline.

The flow wraps the same steps as scripts/run_ingestion.py but adds
Prefect task tracking, retries, and logging visible in the Prefect UI.

Run locally:
    python -m peloton_iq.pipelines.ingest
    python -m peloton_iq.pipelines.ingest --prod
    python -m peloton_iq.pipelines.ingest --skip-features
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from prefect import flow, task, get_run_logger

from peloton_iq.config import (
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

import pandas as pd


# ---------------------------------------------------------------------------
# Tasks  (each step becomes a tracked Prefect task)
# ---------------------------------------------------------------------------

@task(name="load-race-results", retries=1)
def task_load_race_results() -> pd.DataFrame:
    logger = get_run_logger()
    df = load_race_results()
    logger.info("race_results loaded: %d rows x %d cols", *df.shape)
    return df


@task(name="load-course-data", retries=1)
def task_load_course_data() -> pd.DataFrame:
    logger = get_run_logger()
    df = load_course_data()
    logger.info("course_data loaded: %d rows x %d cols", *df.shape)
    return df



@task(name="merge-race-course")
def task_merge(race_df: pd.DataFrame, course_df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    merged = merge_race_and_course(race_df, course_df)
    logger.info("merged_df: %d rows x %d cols", *merged.shape)
    return merged

@task(name="enrich-race-data")
def task_enrich(merged_df: pd.DataFrame) -> pd.DataFrame:
    logger = get_run_logger()
    merged_df = add_stage_type(merged_df)
    merged_df = add_gc_proxy(merged_df)
    merged_df["tier"] = merged_df.apply(
        lambda r: FinishTier.from_rank(r["Rank"], r["Did_Finish"]).value, axis=1
    )
    logger.info(
        "Enriched: stage_type + gc_proxy + tier added. gc_proxy non-null: %d",
        merged_df["gc_proxy"].notna().sum(),
    )
    return merged_df


@task(name="save-intermediates")
def task_save_intermediates(
    race_df: pd.DataFrame,
    course_df: pd.DataFrame,
    merged_path: Path,
    course_path: Path,
) -> None:
    logger = get_run_logger()
    race_df.to_csv(merged_path, index=False)
    course_df.to_csv(course_path, index=False)
    logger.info("Saved merged_uci_races  → %s", merged_path)
    logger.info("Saved course_data_clean → %s", course_path)


@task(name="compute-rider-features", retries=0)
def task_rider_features(
    race_df: pd.DataFrame,
    cache_path: Path,
    force_recompute: bool = False,
) -> pd.DataFrame:
    logger = get_run_logger()
    feats = compute_rider_history(race_df, cache_path=cache_path, force_recompute=force_recompute)
    logger.info("rider_features: %d rows x %d cols", *feats.shape)
    return feats


@task(name="build-model-df")
def task_build_model_df(
    race_df: pd.DataFrame,
    rider_feats: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    logger = get_run_logger()
    model_df = build_model_df(race_df, rider_feats, save=False)
    model_df.to_csv(out_path, index=False)
    logger.info("model_df saved → %s  shape=%s", out_path, model_df.shape)
    return model_df


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(name="pelotoniq-ingestion", log_prints=True)
def ingestion_flow(
    prod: bool = False,
    skip_features: bool = False,
    force_features: bool = False,
) -> None:
    """
    Full PelotonIQ ingestion pipeline.

    Args:
        prod:           Write to data/processed/ (default: data/test_outputs/).
        skip_features:  Skip the ~2hr rider feature computation.
        force_features: Force recompute rider features even if cache exists.
    """
    # Resolve output paths
    if prod:
        out_dir        = DATA_PROCESSED_DIR
        merged_path    = MERGED_RACES_PATH
        course_path    = COURSE_CLEAN_PATH
        features_path  = RIDER_FEATURES_PATH
        model_path     = MODEL_DF_PATH
    else:
        out_dir        = TEST_OUTPUTS_DIR
        merged_path    = TEST_MERGED_RACES_PATH
        course_path    = TEST_COURSE_CLEAN_PATH
        features_path  = TEST_RIDER_FEATURES_PATH
        model_path     = TEST_MODEL_DF_PATH

    out_dir.mkdir(parents=True, exist_ok=True)

    # Run tasks
    race_df   = task_load_race_results()
    course_df = task_load_course_data()
    merged_df = task_merge(race_df, course_df)
    merged_df = task_enrich(merged_df)

    task_save_intermediates(merged_df, course_df, merged_path, course_path)

    if not skip_features:
        rider_feats = task_rider_features(merged_df, features_path, force_features)
        task_build_model_df(merged_df, rider_feats, model_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PelotonIQ ingestion flow")
    parser.add_argument("--prod",           action="store_true")
    parser.add_argument("--skip-features",  action="store_true")
    parser.add_argument("--force-features", action="store_true")
    args = parser.parse_args()

    ingestion_flow(
        prod=args.prod,
        skip_features=args.skip_features,
        force_features=args.force_features,
    )