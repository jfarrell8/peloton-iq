"""
peloton_iq.config
~~~~~~~~~~~~~~~~~
Central configuration for the PelotonIQ system.

Pattern:
  - Paths are plain Path constants — always derived from PROJECT_ROOT,
    never need to be overridden by environment variables.
  - Everything that varies by environment (API keys, service URLs,
    tunable constants) lives in Settings (Pydantic BaseSettings).
    These can be overridden via PELOTON_* env vars or a .env file.

Usage:
    from peloton_iq.config import settings, DATA_PROCESSED_DIR, MODELS_DIR
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Project root  (resolved once at import time)
# src/peloton_iq/config.py  →  src/peloton_iq/  →  src/  →  <root>
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Path constants  (plain Path objects — no env var overrides needed)
# ---------------------------------------------------------------------------

# Data
DATA_DIR            = PROJECT_ROOT / "data"
DATA_RAW_DIR        = DATA_DIR / "raw"
DATA_PROCESSED_DIR  = DATA_DIR / "processed"

# Commentary
COMMENTARY_DIR          = DATA_DIR / "commentary"
COMMENTARY_RAW_DIR      = COMMENTARY_DIR / "raw"
COMMENTARY_EXTRACTED_DIR = COMMENTARY_DIR / "extracted"
COMMENTARY_CACHE_DIR    = COMMENTARY_DIR / "cache"

# ML artifacts  (pkl files, Optuna trial CSVs)
MODELS_DIR = PROJECT_ROOT / "models"

# Key files
RACE_RESULTS_PATH    = DATA_RAW_DIR / "race_results_2017_2023.csv"
COURSE_DATA_PATH     = DATA_RAW_DIR / "structured_course_data.csv"
MERGED_RACES_PATH    = DATA_PROCESSED_DIR / "merged_uci_races.csv"
COURSE_CLEAN_PATH    = DATA_PROCESSED_DIR / "course_data_clean.csv"
RIDER_FEATURES_PATH  = DATA_PROCESSED_DIR / "rider_features.csv"
MODEL_DF_PATH        = DATA_PROCESSED_DIR / "model_df.csv"
TIER_PREDICTOR_PATH  = MODELS_DIR / "tier_predictor.pkl"
YOUTUBE_CACHE_PATH   = COMMENTARY_CACHE_DIR / "all_channel_videos.parquet"
GPX_PROFILES_PATH    = DATA_PROCESSED_DIR / "gpx_profiles.parquet"


# ---------------------------------------------------------------------------
# Settings  (env-sourced values with validation)
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Runtime configuration sourced from environment variables or a .env file.
    All variables are prefixed with PELOTON_ (e.g. PELOTON_QDRANT_URL).
    """

    model_config = SettingsConfigDict(
        env_prefix="PELOTON_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # External services
    # ------------------------------------------------------------------

    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Base URL for the Qdrant vector store.",
    )

    qdrant_api_key: str = Field(
        default="",
        description="Qdrant Cloud API key. Leave empty for local Qdrant.",
    )
    youtube_api_key: str = Field(
        default="",
        description="YouTube Data API v3 key.",
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key.",
    )

    # ------------------------------------------------------------------
    # Qdrant collection names
    # ------------------------------------------------------------------

    qdrant_collection_courses: str = Field(
        default="course_profiles",
        description="Qdrant collection for course profile documents.",
    )
    qdrant_collection_riders: str = Field(
        default="rider_seasons",
        description="Qdrant collection for rider season documents.",
    )

    # ------------------------------------------------------------------
    # Embedding model
    # ------------------------------------------------------------------

    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-Transformers model used for all embeddings.",
    )
    embedding_batch_size: int = Field(
        default=64,
        description="Documents per embedding batch.",
    )

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    rrf_k: int = Field(
        default=60,
        description="RRF constant k. Higher = smoother rank fusion.",
    )
    search_fetch_k: int = Field(
        default=20,
        description="Candidates fetched from each retriever before RRF.",
    )
    search_top_k: int = Field(
        default=5,
        description="Final results returned after RRF fusion.",
    )

    # ------------------------------------------------------------------
    # UCI filter
    # ------------------------------------------------------------------

    fuzzy_threshold: int = Field(
        default=88,
        description="Minimum RapidFuzz token_sort_ratio for a fuzzy race name match.",
    )

    # ------------------------------------------------------------------
    # Prediction model
    # ------------------------------------------------------------------

    prediction_top_n: int = Field(
        default=10,
        description="Top riders included in a pre-race prediction context block.",
    )
    cutoff_year: int = Field(
        default=2023,
        description="Test-set year for temporal train/test split.",
    )
    tier_order: list[str] = Field(
        default=["winner", "podium", "top10", "top20", "finisher", "dnf"],
        description="Finish tier labels ordered from best to worst.",
    )
    model_feature_cols: list[str] = Field(
        default=[
            # Recent form
            "recent_avg_rank_5",
            "recent_avg_rank_12mo",
            "recent_top10_rate_12mo",
            "recent_top10_rate_6mo",
            "recent_win_rate_12mo",
            "recent_podium_rate_12mo",
            "recent_dnf_rate_12mo",
            # Workload
            "races_last_30d",
            "races_last_12mo",
            "days_since_last_race",
            # Terrain affinity
            "terrain_avg_rank",
            "terrain_top10_rate",
            "terrain_win_rate",
            "terrain_podium_rate",
            "terrain_dnf_rate",
            "terrain_avg_rank_12mo",
            "terrain_races_count",
            # Career
            "career_top10_rate",
            "career_podium_rate",
            "career_win_rate",
            "career_races",
            "career_avg_rank",
            # Course profile
            "Vertical Gain",
            "Distance",
            "Highest Elevation",
            "Cobblestones",
            "vg_per_km",
            "cobble_pct",
            "asphalt_pct",
            "stage_type_enc",
            "stage_num",
            # GC proxy
            "gc_proxy",
        ],
        description="Ordered feature columns expected by the trained model.",
    )

    # ------------------------------------------------------------------
    # Commentary / YouTube
    # ------------------------------------------------------------------

    youtube_channels: list[dict] = Field(
        default=[
            {"id": "UCqZQlzSHbVJrwrn5XvzrzcA", "name": "NBC Sports",         "coverage": "Grand Tours, Monuments, WorldTour"},
            {"id": "UCu7phdCr-raU7OaJfEpHZww", "name": "GCN Racing",          "coverage": "All WorldTour"},
            {"id": "UCfDfvvMARk4TKcC62ALi6eA", "name": "TNT Sports Cycling",  "coverage": "Grand Tours, European classics"},
            # Official race channels
            {"id": "UCSpycUnuU0IVF7gGIhGojhg", "name": "Tour de France",      "coverage": "Tour de France official"},
            {"id": "UCe10BxbsFg9Kbmkg-ean_Dg", "name": "Giro d'Italia",       "coverage": "Giro d'Italia official"},
            {"id": "UCf7iHZIcKEhiN34-fETtNCA", "name": "La Vuelta",           "coverage": "Vuelta a España official"},
            # Additional WorldTour channels
            {"id": "UCm0Qjs5OBrv3-d6kKBshEbg", "name": "Tour Down Under",     "coverage": "Tour Down Under official"},
            {"id": "UCXgba6tOLghtJuXaD8LBHWg", "name": "inCycle",             "coverage": "All WorldTour"},
            {"id": "UCcbBlBEtCZ2lX7bodgi02Xg", "name": "Velon",               "coverage": "All WorldTour"},
            {"id": "UClhp9g6TPiqCTOlcw0ROfNg", "name": "TNT Sports",          "coverage": "All WorldTour"},
        ],
        description="YouTube channels to pull race commentary from.",
    )
    youtube_cache_max_pages: int = Field(
        default=500,
        description="Max playlist pages per channel (500 × 50 = 25,000 videos).",
    )
    youtube_refresh_days: int = Field(
        default=30,
        description="Look-back window in days for lightweight cache refreshes.",
    )
    transcript_retry_attempts: int = Field(
        default=3,
        description="Retry attempts for transcript fetch failures.",
    )
    transcript_retry_backoff: float = Field(
        default=2.0,
        description="Exponential backoff base in seconds between retries.",
    )

    # ------------------------------------------------------------------
    # Claude / Anthropic
    # ------------------------------------------------------------------

    claude_model: str = Field(
        default="claude-sonnet-4-5",
        description="Claude model used by the synthesizer and extractor nodes.",
    )
    claude_max_tokens: int = Field(
        default=1500,
        description="Max tokens for Claude responses in the agent synthesizer.",
    )

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    api_host: str  = Field(default="0.0.0.0")
    api_port: int  = Field(default=8000)
    api_reload: bool = Field(
        default=False,
        description="Enable uvicorn hot-reload. True for local dev only.",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Root log level.",
    )

    # S3 artifact storage
    s3_bucket: str = Field(
        default="",
        description="S3 bucket name (e.g. peloton-iq-artifacts).",
    )
    s3_prefix: str = Field(
        default="artifacts",
        description="S3 key prefix for artifacts.",
    )
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for S3 bucket.",
    )


# ---------------------------------------------------------------------------
# Test output paths  (write here during test runs to avoid overwriting
# known-good processed data)
# ---------------------------------------------------------------------------

TEST_OUTPUTS_DIR            = DATA_DIR / "test_outputs"
TEST_MERGED_RACES_PATH      = TEST_OUTPUTS_DIR / "merged_uci_races.csv"
TEST_COURSE_CLEAN_PATH      = TEST_OUTPUTS_DIR / "course_data_clean.csv"
TEST_RIDER_FEATURES_PATH    = TEST_OUTPUTS_DIR / "rider_features.csv"
TEST_MODEL_DF_PATH          = TEST_OUTPUTS_DIR / "model_df.csv"

# Test model artifacts  (separate from production models/)
TEST_MODELS_DIR             = PROJECT_ROOT / "models" / "test"
TEST_TIER_PREDICTOR_PATH    = TEST_MODELS_DIR / "tier_predictor.pkl"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

settings = Settings()