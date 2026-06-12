"""
peloton_iq.pipelines.embed
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prefect flow for building Qdrant vector indexes from scratch.

Loads processed CSVs, serializes documents, embeds with
all-MiniLM-L6-v2, and upserts into Qdrant — overwriting any
existing collections.

Run locally:
    python -m peloton_iq.pipelines.embed
"""

from __future__ import annotations

import time

import pandas as pd
from prefect import flow, task, get_run_logger

from peloton_iq.config import COURSE_CLEAN_PATH, MERGED_RACES_PATH, settings
from peloton_iq.search.embeddings import EmbeddingStore


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="load-processed-data", retries=1)
def task_load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    logger = get_run_logger()
    course_df = pd.read_csv(COURSE_CLEAN_PATH)
    merged_df = pd.read_csv(MERGED_RACES_PATH, low_memory=False)
    logger.info("course_df: %d rows x %d cols", *course_df.shape)
    logger.info("merged_df: %d rows x %d cols", *merged_df.shape)
    return course_df, merged_df


@task(name="build-course-index")
def task_build_course_index(course_df: pd.DataFrame, store: EmbeddingStore) -> int:
    logger = get_run_logger()
    t0 = time.time()
    store.build_course_index(course_df)
    count = store.collection_count(settings.qdrant_collection_courses)
    logger.info(
        "course_profiles: %d points upserted in %.1fs",
        count, time.time() - t0,
    )
    return count


@task(name="build-rider-index")
def task_build_rider_index(merged_df: pd.DataFrame, store: EmbeddingStore) -> int:
    logger = get_run_logger()
    t0 = time.time()
    store.build_rider_index(merged_df)
    count = store.collection_count(settings.qdrant_collection_riders)
    logger.info(
        "rider_seasons: %d points upserted in %.1fs",
        count, time.time() - t0,
    )
    return count


@task(name="validate-collections")
def task_validate(course_count: int, rider_count: int) -> None:
    logger = get_run_logger()

    expected_courses = 963
    expected_riders  = 5835

    course_ok = course_count == expected_courses
    rider_ok  = rider_count  == expected_riders

    logger.info(
        "course_profiles: %d points (expected %d) %s",
        course_count, expected_courses, "✓" if course_ok else "✗",
    )
    logger.info(
        "rider_seasons:   %d points (expected %d) %s",
        rider_count, expected_riders, "✓" if rider_ok else "✗",
    )

    if not course_ok or not rider_ok:
        logger.warning(
            "Point counts differ from notebook baseline — "
            "check serializers if unexpected"
        )


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(name="pelotoniq-embeddings", log_prints=True)
def embeddings_flow() -> None:
    """
    Build Qdrant vector indexes for course profiles and rider seasons.
    Overwrites existing collections on every run.
    """
    store = EmbeddingStore()

    course_df, merged_df = task_load_data()
    course_count = task_build_course_index(course_df, store)
    rider_count  = task_build_rider_index(merged_df, store)
    task_validate(course_count, rider_count)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    embeddings_flow()