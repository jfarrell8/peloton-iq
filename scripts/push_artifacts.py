"""
scripts/push_artifacts.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Upload local model artifacts to S3 for deployment.

Run this after building all local artifacts:
    - models/tier_predictor.pkl       (run_training.py --prod)
    - models/bm25_course_index.pkl    (auto-built on first agent start)
    - models/bm25_rider_index.pkl     (auto-built on first agent start)
    - data/processed/gpx_profiles.parquet  (build_gpx_cache.py)

Prerequisites:
    uv add boto3
    Set PELOTON_S3_BUCKET=your-bucket-name in .env
    AWS credentials configured (aws configure or IAM role)

Usage:
    python scripts/push_artifacts.py           # upload missing artifacts
    python scripts/push_artifacts.py --force   # re-upload all
    python scripts/push_artifacts.py --status  # check local + S3 status
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("push_artifacts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Push artifacts to S3")
    parser.add_argument("--force",  action="store_true",
                        help="Re-upload even if already in S3")
    parser.add_argument("--status", action="store_true",
                        help="Show local and S3 status without uploading")
    args = parser.parse_args()

    from peloton_iq.artifacts import list_artifacts, push_artifacts
    from peloton_iq.config import settings

    if not settings.s3_bucket:
        log.error("PELOTON_S3_BUCKET not set in .env — cannot push artifacts")
        log.error("Add: PELOTON_S3_BUCKET=your-bucket-name")
        sys.exit(1)

    log.info("S3 bucket: s3://%s/%s", settings.s3_bucket, settings.s3_prefix)

    if args.status:
        rows = list_artifacts()
        log.info("")
        log.info("  %-45s  %8s  %8s  %8s", "Artifact", "Local", "Size MB", "S3")
        log.info("  " + "-" * 75)
        for r in rows:
            local = "✓" if r["local"] else "✗"
            s3    = "✓" if r["s3"]    else "✗"
            log.info("  %-45s  %8s  %8.1f  %8s",
                     r["artifact"], local, r["local_mb"], s3)
        return

    result = push_artifacts(force=args.force)

    if "error" in result:
        log.error("Push failed: %s", result["error"])
        sys.exit(1)

    log.info("Done — uploaded: %d  skipped: %d  errors: %d",
             result["uploaded"], result["skipped"], result["errors"])


if __name__ == "__main__":
    main()