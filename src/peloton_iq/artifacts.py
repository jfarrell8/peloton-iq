"""
peloton_iq.artifacts
~~~~~~~~~~~~~~~~~~~~~
S3 artifact management — upload and download model files and
pre-built caches so the app can start without the raw data pipeline.

Artifacts managed:
    models/tier_predictor.pkl      — XGBoost model
    models/bm25_course_index.pkl   — BM25 course index
    models/bm25_rider_index.pkl    — BM25 rider index
    data/processed/gpx_profiles.parquet — GPX elevation cache

Usage:
    # Upload after building locally
    python scripts/push_artifacts.py

    # Download at startup (called automatically by graph.py)
    from peloton_iq.artifacts import ensure_artifacts
    ensure_artifacts()
"""

from __future__ import annotations

import logging
from pathlib import Path

from peloton_iq.config import (
    COURSE_CLEAN_PATH,
    GPX_PROFILES_PATH,
    MERGED_RACES_PATH,
    MODEL_DF_PATH,
    MODELS_DIR,
    settings,
)

log = logging.getLogger(__name__)

# Artifacts to manage — (local_path, s3_key_suffix)
ARTIFACTS: list[tuple[Path, str]] = [
    (MODELS_DIR / "tier_predictor.pkl",     "models/tier_predictor.pkl"),
    (MODELS_DIR / "bm25_course_index.pkl",  "models/bm25_course_index.pkl"),
    (MODELS_DIR / "bm25_rider_index.pkl",   "models/bm25_rider_index.pkl"),
    (GPX_PROFILES_PATH,                      "data/gpx_profiles.parquet"),
    (MERGED_RACES_PATH,                      "data/merged_uci_races.csv"),
    (COURSE_CLEAN_PATH,                      "data/course_data_clean.csv"),
    (MODEL_DF_PATH,                          "data/model_df.csv"),
]


def _get_client():
    """Get a boto3 S3 client. Returns None if boto3 not installed."""
    try:
        import boto3
        return boto3.client("s3", region_name=settings.aws_region)
    except ImportError:
        log.warning("boto3 not installed — S3 artifact sync unavailable")
        return None


def _s3_key(suffix: str) -> str:
    return f"{settings.s3_prefix}/{suffix}"


def ensure_artifacts(force: bool = False) -> dict:
    """
    Download any missing artifacts from S3.
    Skips files that already exist locally unless force=True.
    Silently skips if S3 is not configured (s3_bucket is empty).

    Returns a summary dict with download counts.
    """
    if not settings.s3_bucket:
        log.debug("S3 not configured — skipping artifact sync")
        return {"skipped": True}

    client = _get_client()
    if not client:
        return {"error": "boto3 not available"}

    downloaded = 0
    skipped    = 0
    errors     = 0

    for local_path, suffix in ARTIFACTS:
        if local_path.exists() and not force:
            skipped += 1
            continue

        key = _s3_key(suffix)
        try:
            log.info("Downloading s3://%s/%s → %s",
                     settings.s3_bucket, key, local_path.name)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(settings.s3_bucket, key, str(local_path))
            size_mb = local_path.stat().st_size / 1_000_000
            log.info("  ✓ %.1f MB", size_mb)
            downloaded += 1
        except Exception as e:
            log.warning("  ✗ Failed to download %s: %s", suffix, e)
            errors += 1

    if downloaded:
        log.info("Artifact sync complete: %d downloaded, %d skipped, %d errors",
                 downloaded, skipped, errors)

    return {"downloaded": downloaded, "skipped": skipped, "errors": errors}


def push_artifacts(force: bool = False) -> dict:
    """
    Upload local artifacts to S3.
    Skips files that don't exist locally.
    Requires s3_bucket to be configured in settings.
    """
    if not settings.s3_bucket:
        log.error("S3 not configured — set PELOTON_S3_BUCKET in .env")
        return {"error": "s3_bucket not set"}

    client = _get_client()
    if not client:
        return {"error": "boto3 not available"}

    uploaded = 0
    skipped  = 0
    errors   = 0
    total_mb = 0.0

    for local_path, suffix in ARTIFACTS:
        if not local_path.exists():
            log.warning("Skipping %s — file not found locally", suffix)
            skipped += 1
            continue

        key      = _s3_key(suffix)
        size_mb  = local_path.stat().st_size / 1_000_000

        try:
            log.info("Uploading %s (%.1f MB) → s3://%s/%s",
                     local_path.name, size_mb, settings.s3_bucket, key)
            client.upload_file(str(local_path), settings.s3_bucket, key)
            log.info("  ✓ done")
            uploaded  += 1
            total_mb  += size_mb
        except Exception as e:
            log.error("  ✗ Failed to upload %s: %s", suffix, e)
            errors += 1

    log.info("Push complete: %d uploaded (%.1f MB total), %d skipped, %d errors",
             uploaded, total_mb, skipped, errors)
    return {"uploaded": uploaded, "total_mb": total_mb, "skipped": skipped, "errors": errors}


def list_artifacts() -> list[dict]:
    """List all artifacts and their local/S3 status."""
    client = _get_client() if settings.s3_bucket else None

    rows = []
    for local_path, suffix in ARTIFACTS:
        key        = _s3_key(suffix)
        local_ok   = local_path.exists()
        local_mb   = local_path.stat().st_size / 1_000_000 if local_ok else 0

        s3_ok = False
        if client:
            try:
                client.head_object(Bucket=settings.s3_bucket, Key=key)
                s3_ok = True
            except Exception:
                pass

        rows.append({
            "artifact":  suffix,
            "local":     local_ok,
            "local_mb":  round(local_mb, 1),
            "s3":        s3_ok,
        })
    return rows