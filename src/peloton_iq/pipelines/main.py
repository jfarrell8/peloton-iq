"""
peloton_iq.pipelines.main
~~~~~~~~~~~~~~~~~~~~~~~~~~
Top-level PelotonIQ orchestration flow.

Chains all pipeline stages in the correct order:
  1. Ingestion       — raw CSVs → processed data
  2. Embeddings      — processed data → Qdrant vector store
  3. Training        — processed data → XGBoost model
  4. Commentary      — YouTube → transcripts → Claude extraction → profiles
  5. Artifact push   — upload models + caches to S3

Each stage is optional via flags so you can run only what's needed.
All stages are idempotent — safe to re-run.

Run locally:
    # Full pipeline (takes 2-3 hours)
    python -m peloton_iq.pipelines.main

    # Skip slow stages
    python -m peloton_iq.pipelines.main --skip-ingestion --skip-training

    # Commentary only (most common incremental run)
    python -m peloton_iq.pipelines.main --commentary-only --extract

    # Dry run — show what would run
    python -m peloton_iq.pipelines.main --dry-run

Deploy on Prefect Cloud:
    prefect deploy src/peloton_iq/pipelines/main.py:pelotoniq_pipeline
"""

from __future__ import annotations

import argparse
import time

from prefect import flow, get_run_logger, task


# ---------------------------------------------------------------------------
# Stage tasks — thin wrappers that call the individual flows
# ---------------------------------------------------------------------------

@task(name="ingestion", retries=1)
def task_ingestion(prod: bool = True, skip_features: bool = False) -> None:
    from peloton_iq.pipelines.ingest import ingestion_flow
    ingestion_flow(prod=prod, skip_features=skip_features)


@task(name="embeddings", retries=1)
def task_embeddings() -> None:
    from peloton_iq.pipelines.embed import embed_flow
    embed_flow()


@task(name="training", retries=1)
def task_training(n_trials: int = 50) -> None:
    from peloton_iq.pipelines.train import training_flow
    training_flow(n_trials=n_trials)


@task(name="commentary", retries=1)
def task_commentary(
    rebuild_cache: bool = False,
    max_transcripts: int = 100,
    extract: bool = False,
    max_extractions: int = 50,
    delay_seconds: float = 45.0,
) -> None:
    from peloton_iq.pipelines.commentary import commentary_flow
    commentary_flow(
        rebuild_cache=rebuild_cache,
        extract=extract,
        max_transcripts=max_transcripts,
        max_extractions=max_extractions,
        delay_seconds=delay_seconds,
    )


@task(name="build-profiles", retries=1)
def task_build_profiles(min_races: int = 2) -> None:
    from peloton_iq.commentary.profiler import RiderProfiler
    logger = get_run_logger()
    profiler = RiderProfiler()
    stats    = profiler.build_all_profiles(min_races=min_races)
    logger.info(
        "Profiles — built: %d  skipped: %d  errors: %d",
        stats["built"], stats["skipped"], stats["errors"],
    )


@task(name="push-artifacts")
def task_push_artifacts() -> None:
    from peloton_iq.artifacts import push_artifacts
    logger = get_run_logger()
    result = push_artifacts()
    if "error" in result:
        logger.warning("Artifact push skipped: %s", result["error"])
    else:
        logger.info(
            "Artifacts pushed — uploaded: %d  skipped: %d",
            result.get("uploaded", 0), result.get("skipped", 0),
        )


# ---------------------------------------------------------------------------
# Top-level flow
# ---------------------------------------------------------------------------

@flow(name="pelotoniq-pipeline", log_prints=True)
def pelotoniq_pipeline(
    # Stage toggles
    skip_ingestion:  bool = False,
    skip_embeddings: bool = False,
    skip_training:   bool = False,
    skip_commentary: bool = False,
    skip_profiles:   bool = False,
    skip_artifacts:  bool = False,

    # Ingestion options
    skip_features:   bool = False,

    # Commentary options
    rebuild_cache:    bool  = False,
    max_transcripts:  int   = 100,
    extract:          bool  = False,
    max_extractions:  int   = 50,
    delay_seconds:    float = 45.0,

    # Training options
    n_trials: int = 50,

    # Profile options
    min_races: int = 2,
) -> None:
    """
    Full PelotonIQ data pipeline.

    Runs all stages in dependency order. Each stage is idempotent —
    re-running won't duplicate work.

    Args:
        skip_ingestion:   Skip data ingestion (use existing processed CSVs)
        skip_embeddings:  Skip Qdrant vector build
        skip_training:    Skip model training (use existing pkl)
        skip_commentary:  Skip transcript fetch + extraction
        skip_profiles:    Skip rider profile synthesis
        skip_artifacts:   Skip S3 artifact push
        skip_features:    Skip rider feature recompute during ingestion
        rebuild_cache:    Rebuild YouTube channel cache from scratch
        max_transcripts:  Max transcripts to fetch per run
        extract:          Run Claude extraction on new transcripts
        max_extractions:  Max extractions per run
        delay_seconds:    Delay between transcript requests
        n_trials:         Optuna HPO trials for model training
        min_races:        Min race appearances to build a rider profile
    """
    logger   = get_run_logger()
    t0       = time.time()
    ran      = []

    logger.info("=" * 60)
    logger.info("PelotonIQ Pipeline — starting")
    logger.info("=" * 60)

    if not skip_ingestion:
        logger.info("Stage 1/6 — Ingestion")
        task_ingestion(prod=True, skip_features=skip_features)
        ran.append("ingestion")
    else:
        logger.info("Stage 1/6 — Ingestion (skipped)")

    if not skip_embeddings:
        logger.info("Stage 2/6 — Embeddings")
        task_embeddings()
        ran.append("embeddings")
    else:
        logger.info("Stage 2/6 — Embeddings (skipped)")

    if not skip_training:
        logger.info("Stage 3/6 — Training")
        task_training(n_trials=n_trials)
        ran.append("training")
    else:
        logger.info("Stage 3/6 — Training (skipped)")

    if not skip_commentary:
        logger.info("Stage 4/6 — Commentary")
        task_commentary(
            rebuild_cache=rebuild_cache,
            max_transcripts=max_transcripts,
            extract=extract,
            max_extractions=max_extractions,
            delay_seconds=delay_seconds,
        )
        ran.append("commentary")
    else:
        logger.info("Stage 4/6 — Commentary (skipped)")

    if not skip_profiles:
        logger.info("Stage 5/6 — Rider profiles")
        task_build_profiles(min_races=min_races)
        ran.append("profiles")
    else:
        logger.info("Stage 5/6 — Profiles (skipped)")

    if not skip_artifacts:
        logger.info("Stage 6/6 — Artifact push")
        task_push_artifacts()
        ran.append("artifacts")
    else:
        logger.info("Stage 6/6 — Artifact push (skipped)")

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.0fs", elapsed)
    logger.info("Stages run: %s", ", ".join(ran) or "none")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PelotonIQ full pipeline")

    # Skip flags
    parser.add_argument("--skip-ingestion",  action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-training",   action="store_true")
    parser.add_argument("--skip-commentary", action="store_true")
    parser.add_argument("--skip-profiles",   action="store_true")
    parser.add_argument("--skip-artifacts",  action="store_true")

    # Convenience shortcut
    parser.add_argument("--commentary-only", action="store_true",
                        help="Skip all stages except commentary and profiles")

    # Stage options
    parser.add_argument("--skip-features",  action="store_true")
    parser.add_argument("--rebuild-cache",  action="store_true")
    parser.add_argument("--extract",        action="store_true")
    parser.add_argument("--max-transcripts",type=int,   default=100)
    parser.add_argument("--max-extractions",type=int,   default=50)
    parser.add_argument("--delay",          type=float, default=45.0)
    parser.add_argument("--n-trials",       type=int,   default=50)
    parser.add_argument("--min-races",      type=int,   default=2)
    parser.add_argument("--dry-run",        action="store_true",
                        help="Print what would run without executing")

    args = parser.parse_args()

    if args.dry_run:
        skip_ing  = args.skip_ingestion  or args.commentary_only
        skip_emb  = args.skip_embeddings or args.commentary_only
        skip_tra  = args.skip_training   or args.commentary_only
        skip_com  = args.skip_commentary
        skip_pro  = args.skip_profiles
        skip_art  = args.skip_artifacts
        print("\nDry run — would execute:")
        for stage, skipped in [
            ("1. Ingestion",  skip_ing),
            ("2. Embeddings", skip_emb),
            ("3. Training",   skip_tra),
            ("4. Commentary", skip_com),
            ("5. Profiles",   skip_pro),
            ("6. Artifacts",  skip_art),
        ]:
            print(f"  {'SKIP' if skipped else 'RUN ':4}  {stage}")
        print()
    else:
        pelotoniq_pipeline(
            skip_ingestion  = args.skip_ingestion  or args.commentary_only,
            skip_embeddings = args.skip_embeddings or args.commentary_only,
            skip_training   = args.skip_training   or args.commentary_only,
            skip_commentary = args.skip_commentary,
            skip_profiles   = args.skip_profiles,
            skip_artifacts  = args.skip_artifacts,
            skip_features   = args.skip_features,
            rebuild_cache   = args.rebuild_cache,
            max_transcripts = args.max_transcripts,
            extract         = args.extract,
            max_extractions = args.max_extractions,
            delay_seconds   = args.delay,
            n_trials        = args.n_trials,
            min_races       = args.min_races,
        )