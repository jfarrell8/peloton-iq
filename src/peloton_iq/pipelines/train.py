"""
peloton_iq.pipelines.train
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prefect flow for training the tier prediction model.

Run locally:
    python -m peloton_iq.pipelines.train
    python -m peloton_iq.pipelines.train --trials 20
"""

from __future__ import annotations

import argparse

from prefect import flow, task, get_run_logger

from peloton_iq.prediction.trainer import train


@task(name="train-tier-predictor")
def task_train(n_trials: int) -> dict:
    logger = get_run_logger()
    logger.info("Starting training with %d Optuna trials per model", n_trials)
    results = train(n_trials=n_trials)
    best    = results["best"]
    top5    = results["ranking"][best]["top5"] * 100
    logger.info("Best model: %s  top-5=%.1f%%", best, top5)
    return results


@flow(name="pelotoniq-training", log_prints=True)
def training_flow(n_trials: int = 50) -> None:
    """Train the tier prediction model with Optuna hyperparameter tuning."""
    task_train(n_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()
    training_flow(n_trials=args.trials)