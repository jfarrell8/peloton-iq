"""
scripts/run_commentary.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
CLI runner for the incremental commentary pipeline.

Workflow (each step is idempotent — skips already-processed work):

  Step 1 — Refresh YouTube cache (new videos only)
  Step 2 — Local video matching against cache (zero quota)
  Step 3 — Fetch transcripts for newly matched videos
  Step 4 — Claude extraction for newly fetched transcripts (optional)

Usage:
    # Full incremental run (no extraction)
    python scripts/run_commentary.py

    # Full run including Claude extraction (costs ~$0.02/transcript)
    python scripts/run_commentary.py --extract

    # Run extraction only on N transcripts (for testing)
    python scripts/run_commentary.py --extract --max-extractions 3

    # Force rebuild full YouTube cache (weekly / first run)
    python scripts/run_commentary.py --rebuild-cache

    # Skip transcript fetching (just match videos)
    python scripts/run_commentary.py --skip-transcripts

    # Retry videos previously marked ip_blocked (genuine blocks only —
    # see scripts/relabel_mislabeled_ip_blocks.py for the one-time
    # cleanup of pre-fix mislabeled files)
    python scripts/run_commentary.py --retry-ip-blocked

    # Status report only
    python scripts/run_commentary.py --status
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from peloton_iq.config import (
    COMMENTARY_RAW_DIR,
    COMMENTARY_EXTRACTED_DIR,
    MERGED_RACES_PATH,
    YOUTUBE_CACHE_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_commentary")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    log.info("=" * 60)
    log.info("  %s", title)
    log.info("=" * 60)


def show_status() -> None:
    """Print a summary of the current commentary pipeline state."""
    raw_files       = list(COMMENTARY_RAW_DIR.glob("*.json"))
    extracted_files = list(COMMENTARY_EXTRACTED_DIR.glob("*.json"))
    cache_exists    = YOUTUBE_CACHE_PATH.exists()

    statuses: dict[str, int] = {}
    for path in raw_files:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            s = data.get("status", "unknown")
            bucket = (
                s if s in ("transcript_saved", "no_video_found", "video_found")
                else ("ip_blocked" if "ip_blocked" in s else "other_error")
            )
            statuses[bucket] = statuses.get(bucket, 0) + 1
        except Exception:
            statuses["parse_error"] = statuses.get("parse_error", 0) + 1

    log.info("══════ Commentary Pipeline Status ══════")
    log.info("  YouTube cache exists : %s", cache_exists)
    if cache_exists:
        try:
            cache_df = pd.read_parquet(YOUTUBE_CACHE_PATH)
            log.info("  Cache size           : %d videos", len(cache_df))
        except Exception:
            log.info("  Cache size           : (unreadable)")
    log.info("  Raw files            : %d", len(raw_files))
    log.info("    ✓ transcript_saved : %d", statuses.get("transcript_saved", 0))
    log.info("    → video_found      : %d", statuses.get("video_found", 0))
    log.info("    ✗ no_video_found   : %d", statuses.get("no_video_found", 0))
    log.info("    ⚠ ip_blocked       : %d", statuses.get("ip_blocked", 0))
    log.info("    ? other_error      : %d", statuses.get("other_error", 0))
    log.info("  Extracted (Claude)   : %d", len(extracted_files))

    pending_transcripts = statuses.get("video_found", 0)
    pending_extraction  = statuses.get("transcript_saved", 0) - len(extracted_files)
    if pending_transcripts > 0:
        log.info("  → %d videos pending transcript fetch", pending_transcripts)
    if pending_extraction > 0:
        log.info("  → %d transcripts pending Claude extraction (~$%.2f)",
                 pending_extraction, pending_extraction * 0.02)
    if statuses.get("ip_blocked", 0) > 0:
        log.info(
            "  → %d videos marked ip_blocked — use --retry-ip-blocked to retry them",
            statuses.get("ip_blocked", 0),
        )


def build_race_index() -> pd.DataFrame:
    """Load merged_df and build the race index for matching."""
    merged_df = pd.read_csv(MERGED_RACES_PATH, low_memory=False)
    merged_df["Date"] = pd.to_datetime(merged_df["Date"])
    race_index = (
        merged_df[["Race Name", "Race_results", "Date", "Year_results", "Stage_results"]]
        .drop_duplicates("Race Name")
        .sort_values("Date", ascending=False)
        .reset_index(drop=True)
    )
    log.info("Race index: %d unique races", len(race_index))
    return race_index


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_refresh_cache(rebuild: bool = False) -> pd.DataFrame:
    section("STEP 1 — Refresh YouTube Cache")
    from peloton_iq.commentary.youtube import YouTubeCacheManager

    mgr = YouTubeCacheManager()

    if rebuild or not YOUTUBE_CACHE_PATH.exists():
        log.info("Building full cache (this takes a few minutes)...")
        df = mgr.build_cache(force_refresh=rebuild)
    else:
        log.info("Running incremental refresh (last %d days)...", 30)
        df = mgr.refresh_recent()

    log.info("Cache: %d videos total", len(df) if df is not None else 0)
    return df


def step_local_matching(cache_df: pd.DataFrame, race_index: pd.DataFrame) -> dict:
    section("STEP 2 — Local Video Matching (zero quota)")
    from peloton_iq.commentary.transcript import TranscriptFetcher

    fetcher = TranscriptFetcher(video_cache=cache_df)
    stats   = fetcher.run_local_matching(race_index, verbose=False)

    log.info(
        "Matching complete — found: %d  not_found: %d  skipped: %d",
        stats["found"], stats["not_found"], stats["skipped"],
    )
    return stats


def step_fetch_transcripts(
    race_index: pd.DataFrame,
    cache_df: pd.DataFrame,
    max_transcripts: int = 50,
    delay_seconds: float = 45.0,
) -> dict:
    section("STEP 3 — Fetch Transcripts")
    from peloton_iq.commentary.transcript import TranscriptFetcher

    # Count pending
    pending = sum(
        1 for p in COMMENTARY_RAW_DIR.glob("*.json")
        if json.load(open(p, encoding="utf-8")).get("status") == "video_found"
    )
    log.info("Videos pending transcript: %d (fetching up to %d)", pending, max_transcripts)
    log.info(
        "Estimated time: %.0f minutes (%.0fs delay between requests)",
        max_transcripts * delay_seconds / 60, delay_seconds,
    )

    if pending == 0:
        log.info("Nothing to fetch — all matched videos already have transcripts.")
        return {"success": 0, "ip_blocked": 0, "errors": 0}

    fetcher = TranscriptFetcher(video_cache=cache_df)
    stats   = fetcher.run_batch(
        race_index=race_index,
        max_transcripts=max_transcripts,
        delay_seconds=delay_seconds,
    )
    return stats


def step_retry_ip_blocked(
    cache_df: pd.DataFrame,
    max_transcripts: int = 50,
    delay_seconds: float = 45.0,
) -> dict:
    """
    Retry videos currently marked ip_blocked. Resets their status back
    to 'video_found' so run_batch will pick them up normally, then runs
    a normal batch fetch limited to just those videos.

    NOTE: this is for retrying GENUINE blocks going forward. For the
    one-time cleanup of files mislabeled by the old substring-matching
    bug (see transcript.py's run_batch docstring), use
    scripts/relabel_mislabeled_ip_blocks.py instead — that script
    re-checks and relabels without assuming a retry will succeed.
    """
    section("STEP 3b — Retry IP-Blocked Videos")
    from peloton_iq.commentary.transcript import TranscriptFetcher

    ip_blocked_paths = []
    for path in sorted(COMMENTARY_RAW_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("status") == "transcript_error:ip_blocked":
            ip_blocked_paths.append((path, data))

    if not ip_blocked_paths:
        log.info("No videos currently marked ip_blocked.")
        return {"success": 0, "ip_blocked": 0, "errors": 0}

    log.info(
        "Found %d videos marked ip_blocked — resetting up to %d to video_found for retry",
        len(ip_blocked_paths), max_transcripts,
    )

    # Reset status so run_batch's normal pending-scan picks them up
    reset_count = 0
    for path, data in ip_blocked_paths[:max_transcripts]:
        data["status"] = "video_found"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        reset_count += 1

    log.info("Reset %d videos to video_found — running batch fetch...", reset_count)

    fetcher = TranscriptFetcher(video_cache=cache_df)
    stats = fetcher.run_batch(
        race_index=None,  # unused inside run_batch — it scans raw_dir directly
        max_transcripts=reset_count,
        delay_seconds=delay_seconds,
    )
    return stats


def step_extract(max_extractions: int = 3) -> dict:
    section(f"STEP 4 — Claude Extraction (up to {max_extractions} transcripts)")
    from peloton_iq.commentary.extractor import ClaudeExtractor

    # Count pending
    pending = [
        p for p in COMMENTARY_RAW_DIR.glob("*.json")
        if not (COMMENTARY_EXTRACTED_DIR / p.name).exists()
        and json.load(open(p, encoding="utf-8")).get("status") == "transcript_saved"
    ]
    log.info(
        "Transcripts pending extraction: %d (processing up to %d, est. $%.2f)",
        len(pending), max_extractions, max_extractions * 0.02,
    )

    if not pending:
        log.info("Nothing to extract.")
        return {"success": 0, "skipped": 0, "errors": 0, "cost": 0.0}

    extractor = ClaudeExtractor()
    stats     = extractor.run_batch(max_extractions=max_extractions, verbose=True)
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PelotonIQ commentary pipeline")
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="Force full rebuild of YouTube cache (slow, ~few hundred quota units)",
    )
    parser.add_argument(
        "--skip-transcripts", action="store_true",
        help="Skip transcript fetching (just refresh cache and match videos)",
    )
    parser.add_argument(
        "--extract", action="store_true",
        help="Run Claude extraction on newly fetched transcripts",
    )
    parser.add_argument(
        "--max-extractions", type=int, default=3,
        help="Max transcripts to extract in one run (default: 3)",
    )
    parser.add_argument(
        "--max-transcripts", type=int, default=50,
        help="Max transcripts to fetch in one run (default: 50)",
    )
    parser.add_argument(
        "--delay", type=float, default=45.0,
        help="Seconds between transcript requests (default: 45)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print pipeline status and exit",
    )
    parser.add_argument(
        "--fetch-nbc-older", action="store_true",
        help="Fetch older NBC Sports videos (pre-2021) by skipping playlist pages",
    )
    parser.add_argument(
        "--nbc-skip-pages", type=int, default=400,
        help="Number of playlist pages to skip before collecting (default: 400 = ~20,000 videos)",
    )
    parser.add_argument(
        "--retry-ip-blocked", action="store_true",
        help="Retry videos currently marked ip_blocked (genuine blocks going forward — "
             "for the one-time cleanup of pre-fix mislabeled files, use "
             "scripts/relabel_mislabeled_ip_blocks.py instead). Runs instead of the "
             "normal Step 3 fetch for new video_found videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0   = time.time()

    log.info("PelotonIQ Commentary Pipeline")

    if args.status:
        show_status()
        return

    # Status at start
    show_status()

    # NBC older fetch — run before cache refresh so new videos are included
    if args.fetch_nbc_older:
        section("NBC SPORTS — Fetch Older Videos")
        from peloton_iq.commentary.youtube import YouTubeCacheManager
        mgr = YouTubeCacheManager()
        cache_df = mgr.merge_nbc_older_into_cache(skip_pages=args.nbc_skip_pages)
        log.info("Cache now contains %d videos", len(cache_df) if cache_df is not None else 0)
        if not hasattr(args, "rebuild_cache"):
            args.rebuild_cache = False
        # Skip the normal cache refresh since we just updated it
        section("STEP 2 — Local Video Matching (zero quota)")
        race_index = build_race_index()
        step_local_matching(cache_df, race_index)
        if not args.skip_transcripts:
            step_fetch_transcripts(
                race_index=race_index,
                cache_df=cache_df,
                max_transcripts=args.max_transcripts,
                delay_seconds=args.delay,
            )
        if args.extract:
            step_extract(max_extractions=args.max_extractions)
        show_status()
        log.info("=" * 60)
        log.info("  COMMENTARY PIPELINE COMPLETE  (%.1fs total)", time.time() - t0)
        log.info("=" * 60)
        return

    # Retry-ip-blocked is a standalone mode — runs instead of the normal
    # cache-refresh/matching/fetch sequence, since it operates on already-
    # matched videos that just need their transcript fetch retried.
    if args.retry_ip_blocked:
        # Still need a cache_df for TranscriptFetcher's constructor, even
        # though retry doesn't do fresh matching — load whatever's on disk.
        from peloton_iq.commentary.youtube import YouTubeCacheManager
        mgr = YouTubeCacheManager()
        cache_df = mgr.load_cache()
        step_retry_ip_blocked(
            cache_df=cache_df,
            max_transcripts=args.max_transcripts,
            delay_seconds=args.delay,
        )
        if args.extract:
            step_extract(max_extractions=args.max_extractions)
        log.info("")
        show_status()
        log.info("=" * 60)
        log.info("  COMMENTARY PIPELINE COMPLETE  (%.1fs total)", time.time() - t0)
        log.info("=" * 60)
        return

    # Build race index
    race_index = build_race_index()

    # Step 1 — cache refresh
    cache_df = step_refresh_cache(rebuild=args.rebuild_cache)

    if cache_df is None or cache_df.empty:
        log.error("Cache is empty — cannot proceed with matching.")
        return

    # Step 2 — local matching
    step_local_matching(cache_df, race_index)

    # Step 3 — transcript fetching
    if not args.skip_transcripts:
        step_fetch_transcripts(
            race_index=race_index,
            cache_df=cache_df,
            max_transcripts=args.max_transcripts,
            delay_seconds=args.delay,
        )
    else:
        log.info("Skipping transcript fetching (--skip-transcripts)")

    # Step 4 — Claude extraction (opt-in)
    if args.extract:
        step_extract(max_extractions=args.max_extractions)
    else:
        log.info("Skipping extraction (pass --extract to enable)")

    # Status at end
    log.info("")
    show_status()

    log.info("=" * 60)
    log.info("  COMMENTARY PIPELINE COMPLETE  (%.1fs total)", time.time() - t0)
    log.info("=" * 60)


if __name__ == "__main__":
    main()