"""
peloton_iq.pipelines.commentary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prefect flow for the incremental commentary pipeline.

Each task is idempotent — safe to re-run, always skips already-processed work.

Run locally:
    python -m peloton_iq.pipelines.commentary
    python -m peloton_iq.pipelines.commentary --extract
    python -m peloton_iq.pipelines.commentary --status
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
from prefect import flow, task, get_run_logger

from peloton_iq.config import (
    COMMENTARY_EXTRACTED_DIR,
    COMMENTARY_RAW_DIR,
    MERGED_RACES_PATH,
    YOUTUBE_CACHE_PATH,
)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="load-race-index", retries=1)
def task_load_race_index() -> pd.DataFrame:
    logger = get_run_logger()
    merged_df = pd.read_csv(MERGED_RACES_PATH, low_memory=False)
    merged_df["Date"] = pd.to_datetime(merged_df["Date"])
    race_index = (
        merged_df[["Race Name", "Race_results", "Date", "Year_results", "Stage_results"]]
        .drop_duplicates("Race Name")
        .sort_values("Date", ascending=False)
        .reset_index(drop=True)
    )
    logger.info("Race index: %d unique races", len(race_index))
    return race_index


@task(name="refresh-youtube-cache", retries=1)
def task_refresh_cache(rebuild: bool = False) -> pd.DataFrame:
    logger = get_run_logger()
    from peloton_iq.commentary.youtube import YouTubeCacheManager

    mgr = YouTubeCacheManager()
    if rebuild or not YOUTUBE_CACHE_PATH.exists():
        logger.info("Building full cache...")
        df = mgr.build_cache(force_refresh=rebuild)
    else:
        logger.info("Running incremental refresh...")
        df = mgr.refresh_recent()

    logger.info("Cache: %d videos", len(df) if df is not None else 0)
    return df


@task(name="local-video-matching")
def task_local_matching(
    cache_df: pd.DataFrame,
    race_index: pd.DataFrame,
) -> dict:
    logger = get_run_logger()
    from peloton_iq.commentary.transcript import TranscriptFetcher

    fetcher = TranscriptFetcher(video_cache=cache_df)
    stats   = fetcher.run_local_matching(race_index, verbose=False)
    logger.info(
        "Matching — found: %d  not_found: %d  skipped: %d",
        stats["found"], stats["not_found"], stats["skipped"],
    )
    return stats


@task(
    name="fetch-transcripts",
    retries=2,
    retry_delay_seconds=300,  # wait 5 mins before retry (IP block cooldown)
)
def task_fetch_transcripts(
    cache_df: pd.DataFrame,
    race_index: pd.DataFrame,
    max_transcripts: int = 50,
    delay_seconds: float = 45.0,
) -> dict:
    logger = get_run_logger()
    from peloton_iq.commentary.transcript import TranscriptFetcher

    pending = sum(
        1 for p in COMMENTARY_RAW_DIR.glob("*.json")
        if json.load(open(p, encoding="utf-8")).get("status") == "video_found"
    )
    logger.info("Videos pending transcript: %d (fetching up to %d)", pending, max_transcripts)

    if pending == 0:
        logger.info("Nothing to fetch.")
        return {"success": 0, "ip_blocked": 0, "errors": 0}

    fetcher = TranscriptFetcher(video_cache=cache_df)
    stats   = fetcher.run_batch(
        race_index=race_index,
        max_transcripts=max_transcripts,
        delay_seconds=delay_seconds,
    )
    logger.info(
        "Fetch complete — saved: %d  ip_blocked: %d  errors: %d",
        stats["success"], stats["ip_blocked"], stats["errors"],
    )

    # Raise on IP block so Prefect retries after the delay
    if stats.get("ip_blocked", 0) > 0 and stats.get("success", 0) == 0:
        raise RuntimeError(
            "IP blocked after 0 successful fetches — "
            "Prefect will retry in 5 minutes"
        )

    return stats


@task(name="claude-extraction")
def task_extract(max_extractions: int = 3) -> dict:
    logger = get_run_logger()
    from peloton_iq.commentary.extractor import ClaudeExtractor

    pending = [
        p for p in COMMENTARY_RAW_DIR.glob("*.json")
        if not (COMMENTARY_EXTRACTED_DIR / p.name).exists()
        and json.load(open(p, encoding="utf-8")).get("status") == "transcript_saved"
    ]
    logger.info(
        "Pending extraction: %d (processing up to %d, ~$%.2f)",
        len(pending), max_extractions, max_extractions * 0.02,
    )

    if not pending:
        logger.info("Nothing to extract.")
        return {"success": 0, "skipped": 0, "errors": 0, "cost": 0.0}

    extractor = ClaudeExtractor()
    stats     = extractor.run_batch(max_extractions=max_extractions, verbose=True)
    logger.info(
        "Extraction complete — success: %d  skipped: %d  errors: %d  cost: $%.4f",
        stats["success"], stats["skipped"], stats["errors"], stats["cost"],
    )
    return stats


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------

@flow(name="pelotoniq-commentary", log_prints=True)
def commentary_flow(
    rebuild_cache: bool = False,
    skip_transcripts: bool = False,
    extract: bool = False,
    max_transcripts: int = 50,
    max_extractions: int = 3,
    delay_seconds: float = 45.0,
) -> None:
    """
    Incremental commentary pipeline.

    Args:
        rebuild_cache:    Force full YouTube cache rebuild.
        skip_transcripts: Skip transcript fetching step.
        extract:          Run Claude extraction on new transcripts.
        max_transcripts:  Max transcripts to fetch per run.
        max_extractions:  Max transcripts to extract per run.
        delay_seconds:    Seconds between transcript requests.
    """
    race_index = task_load_race_index()
    cache_df   = task_refresh_cache(rebuild=rebuild_cache)

    if cache_df is not None and not cache_df.empty:
        task_local_matching(cache_df, race_index)

        if not skip_transcripts:
            task_fetch_transcripts(
                cache_df=cache_df,
                race_index=race_index,
                max_transcripts=max_transcripts,
                delay_seconds=delay_seconds,
            )

    if extract:
        task_extract(max_extractions=max_extractions)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PelotonIQ commentary flow")
    parser.add_argument("--rebuild-cache",    action="store_true")
    parser.add_argument("--skip-transcripts", action="store_true")
    parser.add_argument("--extract",          action="store_true")
    parser.add_argument("--max-transcripts",  type=int,   default=50)
    parser.add_argument("--max-extractions",  type=int,   default=3)
    parser.add_argument("--delay",            type=float, default=45.0)
    args = parser.parse_args()

    commentary_flow(
        rebuild_cache=args.rebuild_cache,
        skip_transcripts=args.skip_transcripts,
        extract=args.extract,
        max_transcripts=args.max_transcripts,
        max_extractions=args.max_extractions,
        delay_seconds=args.delay,
    )