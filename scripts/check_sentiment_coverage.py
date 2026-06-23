"""
scripts/check_sentiment_coverage.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Quick CLI to inspect how much rider sentiment data is extractable
from data/commentary/extracted/ before wiring it into model training.

Usage:
    python scripts/check_sentiment_coverage.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("check_sentiment")


def main() -> None:
    from peloton_iq.commentary.form_features import build_sentiment_table, coverage_report

    df     = build_sentiment_table()
    report = coverage_report(df)

    log.info("=" * 60)
    log.info("  COMMENTARY SENTIMENT COVERAGE")
    log.info("=" * 60)
    log.info("  Total observations    : %d", report["total_observations"])
    log.info("  Unique riders         : %d", report["unique_riders"])

    if report["total_observations"] == 0:
        log.warning("No sentiment observations found — check extraction output.")
        return

    log.info("  Date range            : %s to %s", *report["date_range"])
    log.info("  Riders with 2+ obs    : %d", report["riders_with_2plus_obs"])
    log.info("  Riders with 5+ obs    : %d", report["riders_with_5plus_obs"])
    log.info("")
    log.info("  Sentiment distribution:")
    for score, count in sorted(report["sentiment_distribution"].items()):
        label = {1.0: "positive", 0.0: "neutral", -1.0: "negative"}.get(score, str(score))
        log.info("    %-10s  %d", label, count)
    log.info("")
    log.info("  Top covered riders:")
    for rider, count in report["top_covered_riders"].items():
        log.info("    %-30s  %d observations", rider, count)
    log.info("=" * 60)


if __name__ == "__main__":
    main()