"""
scripts/run_embeddings.py
~~~~~~~~~~~~~~~~~~~~~~~~~
CLI runner for building the Qdrant vector indexes from scratch.

Loads processed CSVs, serializes documents, embeds with
all-MiniLM-L6-v2, and upserts into Qdrant — overwriting any
existing collections.

Usage:
    python scripts/run_embeddings.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from peloton_iq.config import DATA_PROCESSED_DIR, settings
from peloton_iq.search.embeddings import EmbeddingStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_embeddings")


def main() -> None:
    total_start = time.time()

    # ------------------------------------------------------------------
    # Load processed data
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("  STEP 1 — Load processed data")
    log.info("=" * 60)

    course_df = pd.read_csv(DATA_PROCESSED_DIR / "course_data_clean.csv")
    merged_df = pd.read_csv(DATA_PROCESSED_DIR / "merged_uci_races.csv", low_memory=False)

    log.info("course_df:  %d rows x %d cols", *course_df.shape)
    log.info("merged_df:  %d rows x %d cols", *merged_df.shape)

    # ------------------------------------------------------------------
    # Build indexes
    # ------------------------------------------------------------------
    store = EmbeddingStore()

    log.info("=" * 60)
    log.info("  STEP 2 — Build course_profiles index")
    log.info("=" * 60)
    t0 = time.time()
    store.build_course_index(course_df)
    log.info("course_profiles built in %.1fs", time.time() - t0)
    log.info(
        "course_profiles: %d points in Qdrant",
        store.collection_count(settings.qdrant_collection_courses),
    )

    log.info("=" * 60)
    log.info("  STEP 3 — Build rider_seasons index")
    log.info("=" * 60)
    t0 = time.time()
    store.build_rider_index(merged_df)
    log.info("rider_seasons built in %.1fs", time.time() - t0)
    log.info(
        "rider_seasons: %d points in Qdrant",
        store.collection_count(settings.qdrant_collection_riders),
    )

    log.info("=" * 60)
    log.info("  EMBEDDINGS COMPLETE  (%.1fs total)", time.time() - total_start)
    log.info("=" * 60)


if __name__ == "__main__":
    main()