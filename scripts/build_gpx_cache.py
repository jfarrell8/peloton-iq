"""
scripts/build_gpx_cache.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Parse all 8,092 GPX files from data/raw/gpx/ into a single parquet file
at data/processed/gpx_profiles.parquet.

This is a one-time build step that makes deployment possible — the app
reads from the parquet at query time rather than parsing raw GPX files,
so only the parquet needs to be shipped (not the 500MB+ of raw GPX files).

The parquet schema:
    race_name    str        — matches Race Name in course_data_clean.csv
    distance_km  float64   — cumulative distance at each point
    elevation_m  float64   — smoothed elevation in metres
    gradient_pct float64   — gradient percentage (clipped -25 to +25)

Usage:
    python scripts/build_gpx_cache.py              # build all
    python scripts/build_gpx_cache.py --check      # verify existing cache
    python scripts/build_gpx_cache.py --rebuild    # force full rebuild
    python scripts/build_gpx_cache.py --sample 10  # parse first 10 (smoke test)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_gpx_cache")


def build_cache(
    gpx_dir: Path,
    output_path: Path,
    target_points: int = 200,
    sample: int | None = None,
    force_rebuild: bool = False,
) -> None:
    import pandas as pd
    from peloton_iq.ingestion.gpx import load_elevation_profile

    if output_path.exists() and not force_rebuild and sample is None:
        log.info("Cache already exists at %s", output_path)
        log.info("Pass --rebuild to regenerate. Running --check instead.")
        check_cache(output_path)
        return

    gpx_files = sorted(gpx_dir.glob("*.gpx"))
    if sample:
        gpx_files = gpx_files[:sample]
        log.info("Sample mode: processing first %d files", sample)

    total      = len(gpx_files)
    processed  = 0
    skipped    = 0
    errors     = 0
    all_frames = []

    log.info("Building GPX cache: %d files → %s", total, output_path)
    t0 = time.time()

    for i, path in enumerate(gpx_files):
        race_name = path.stem  # "2023 Tour de France Stage 17"

        try:
            df = load_elevation_profile(race_name, target_points=target_points)
        except Exception as e:
            log.debug("Error parsing %s: %s", race_name, e)
            errors += 1
            continue

        if df is None or df.empty:
            skipped += 1
            continue

        df["race_name"] = race_name
        all_frames.append(df[["race_name", "distance_km", "elevation_m", "gradient_pct"]])
        processed += 1

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (total - i - 1) / rate
            log.info(
                "  %d / %d  (%.0f/s  ETA %.0fs)  processed=%d  skipped=%d  errors=%d",
                i + 1, total, rate, eta, processed, skipped, errors,
            )

    if not all_frames:
        log.error("No GPX files parsed successfully — cache not written")
        return

    log.info("Concatenating %d profiles...", len(all_frames))
    result = pd.concat(all_frames, ignore_index=True)

    # Optimise storage — float32 is plenty for elevation data
    result["distance_km"]  = result["distance_km"].astype("float32")
    result["elevation_m"]  = result["elevation_m"].astype("float32")
    result["gradient_pct"] = result["gradient_pct"].astype("float32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False, compression="snappy")

    size_mb = output_path.stat().st_size / 1_000_000
    elapsed = time.time() - t0

    log.info("=" * 60)
    log.info("  GPX CACHE BUILD COMPLETE")
    log.info("  Processed : %d profiles", processed)
    log.info("  Skipped   : %d (no elevation data)", skipped)
    log.info("  Errors    : %d", errors)
    log.info("  Rows      : %d", len(result))
    log.info("  File size : %.1f MB", size_mb)
    log.info("  Time      : %.0fs", elapsed)
    log.info("  Output    : %s", output_path)
    log.info("=" * 60)


def check_cache(output_path: Path) -> None:
    import pandas as pd

    if not output_path.exists():
        log.error("Cache not found at %s — run without --check to build", output_path)
        return

    df       = pd.read_parquet(output_path)
    size_mb  = output_path.stat().st_size / 1_000_000
    n_races  = df["race_name"].nunique()
    avg_pts  = len(df) / n_races if n_races > 0 else 0

    log.info("=" * 60)
    log.info("  GPX CACHE STATUS")
    log.info("  Races     : %d", n_races)
    log.info("  Total rows: %d", len(df))
    log.info("  Avg points: %.0f per race", avg_pts)
    log.info("  File size : %.1f MB", size_mb)
    log.info("  Columns   : %s", list(df.columns))
    log.info("")
    log.info("  Elevation range: %.0fm – %.0fm",
             df["elevation_m"].min(), df["elevation_m"].max())
    log.info("  Distance range : 0 – %.0f km", df["distance_km"].max())
    log.info("")
    log.info("  Sample races:")
    for name in sorted(df["race_name"].unique())[:5]:
        race_df = df[df["race_name"] == name]
        log.info("    %-50s  %d pts  %.0fkm  %.0f-%.0fm",
                 name[:50], len(race_df),
                 race_df["distance_km"].max(),
                 race_df["elevation_m"].min(),
                 race_df["elevation_m"].max())
    log.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GPX elevation profile cache")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force full rebuild even if cache exists")
    parser.add_argument("--check", action="store_true",
                        help="Check existing cache without rebuilding")
    parser.add_argument("--sample", type=int, default=None,
                        help="Only process first N files (smoke test)")
    parser.add_argument("--points", type=int, default=200,
                        help="Target points per elevation profile (default: 200)")
    args = parser.parse_args()

    from peloton_iq.config import DATA_RAW_DIR, GPX_PROFILES_PATH

    gpx_dir = DATA_RAW_DIR / "gpx"

    if not gpx_dir.exists():
        log.error("GPX directory not found: %s", gpx_dir)
        log.error("Download the Figshare dataset and extract to data/raw/gpx/")
        sys.exit(1)

    if args.check:
        check_cache(GPX_PROFILES_PATH)
        return

    build_cache(
        gpx_dir=gpx_dir,
        output_path=GPX_PROFILES_PATH,
        target_points=args.points,
        sample=args.sample,
        force_rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()