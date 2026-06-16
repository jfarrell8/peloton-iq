"""
scripts/build_profiles.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Build tactical profiles for all riders with sufficient commentary coverage.

Reads from data/commentary/extracted/, calls Claude to synthesize
patterns, saves profiles to data/commentary/profiles/.

Run:
    # Build profiles for all riders with >= 2 race appearances
    python scripts/build_profiles.py

    # Force rebuild all existing profiles
    python scripts/build_profiles.py --rebuild

    # Build profile for a specific rider
    python scripts/build_profiles.py --rider "POGAČAR Tadej"

    # Show coverage report without building
    python scripts/build_profiles.py --report
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_profiles")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rider tactical profiles")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild all existing profiles")
    parser.add_argument("--rider", type=str, default=None,
                        help="Build profile for a specific rider name")
    parser.add_argument("--min-races", type=int, default=2,
                        help="Minimum race appearances to build profile (default: 2)")
    parser.add_argument("--report", action="store_true",
                        help="Show coverage report only, don't build profiles")
    args = parser.parse_args()

    from peloton_iq.commentary.profiler import RiderProfiler
    profiler = RiderProfiler()

    if args.report:
        report = profiler.coverage_report()
        log.info("=" * 60)
        log.info("  COMMENTARY COVERAGE REPORT")
        log.info("=" * 60)
        log.info("  Riders in commentary:    %d", report["total_riders_in_commentary"])
        log.info("  Total race observations: %d", report["total_race_observations"])
        log.info("  Riders with 2+ races:    %d", report["riders_with_2plus_races"])
        log.info("  Riders with 5+ races:    %d", report["riders_with_5plus_races"])
        log.info("  Profiles already built:  %d", report["riders_with_profiles"])
        log.info("")
        log.info("  Top covered riders:")
        for rider, n in report["top_covered"]:
            log.info("    %-35s  %d races", rider, n)
        return

    if args.rider:
        log.info("Building profile for: %s", args.rider)
        profile = profiler.build_profile_for_rider(
            args.rider, force_rebuild=args.rebuild
        )
        if profile:
            log.info("Profile built:")
            log.info("  Confidence:   %s", profile.get("confidence"))
            log.info("  Races:        %d", profile.get("races_analysed", 0))
            log.info("  Attack style: %s", profile.get("attacking_style", "")[:80])
            log.info("  Patterns:")
            for p in profile.get("key_patterns", []):
                log.info("    • %s", p)
        else:
            log.warning("No profile built for %s", args.rider)
        return

    # Build all profiles
    log.info("Building all rider profiles (min %d races)...", args.min_races)
    stats = profiler.build_all_profiles(
        min_races=args.min_races,
        force_rebuild=args.rebuild,
    )
    log.info("=" * 60)
    log.info("  PROFILE BUILD COMPLETE")
    log.info("  Built:        %d", stats["built"])
    log.info("  Skipped:      %d (already exist)", stats["skipped"])
    log.info("  Insufficient: %d (< %d races)", stats["insufficient"], args.min_races)
    log.info("  Errors:       %d", stats["errors"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()