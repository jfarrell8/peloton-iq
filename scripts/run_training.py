"""
scripts/run_training.py
~~~~~~~~~~~~~~~~~~~~~~~~
CLI runner for training the tier prediction model.

Runs Optuna hyperparameter tuning for LightGBM and XGBoost,
evaluates on the 2023 holdout, and saves the best model as
tier_predictor.pkl.

Usage:
    # Test run — writes to models/test/
    python scripts/run_training.py

    # Production run — writes to models/
    python scripts/run_training.py --prod

    # Faster run with fewer Optuna trials
    python scripts/run_training.py --trials 20

    # Use test model_df from data/test_outputs/
    python scripts/run_training.py --test-data
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from peloton_iq.config import (
    MODEL_DF_PATH,
    MODELS_DIR,
    TEST_MODEL_DF_PATH,
    TEST_MODELS_DIR,
)
from peloton_iq.prediction.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_training")


def main() -> None:
    parser = argparse.ArgumentParser(description="PelotonIQ tier prediction training")
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Write to models/ instead of models/test/",
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of Optuna trials per model (default: 50)",
    )
    parser.add_argument(
        "--test-data",
        action="store_true",
        help="Read model_df from data/test_outputs/ instead of data/processed/",
    )
    args = parser.parse_args()

    # Resolve paths
    if args.prod:
        models_dir    = MODELS_DIR
        model_df_path = MODEL_DF_PATH
        log.info("MODE: PRODUCTION — writing to models/")
    else:
        models_dir    = TEST_MODELS_DIR
        model_df_path = TEST_MODEL_DF_PATH if args.test_data else MODEL_DF_PATH
        log.info("MODE: TEST — writing to models/test/")

    if args.test_data:
        log.info("DATA: reading model_df from data/test_outputs/")
    else:
        log.info("DATA: reading model_df from data/processed/")

    models_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  PelotonIQ — Tier Prediction Training")
    log.info("  Optuna trials per model: %d", args.trials)
    log.info("=" * 60)

    t0      = time.time()
    results = train(
        model_df_path=model_df_path,
        models_dir=models_dir,
        n_trials=args.trials,
    )

    best    = results["best"]
    ranking = results["ranking"]

    log.info("=" * 60)
    log.info("  TRAINING COMPLETE  (%.1fs total)", time.time() - t0)
    log.info("  Best model : %s", best)
    log.info("  Top-1      : %.1f%%", ranking[best]["top1"] * 100)
    log.info("  Top-3      : %.1f%%", ranking[best]["top3"] * 100)
    log.info("  Top-5      : %.1f%%", ranking[best]["top5"] * 100)
    log.info("  Top-10     : %.1f%%", ranking[best]["top10"] * 100)
    log.info("  Output dir : %s", models_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()