"""
peloton_iq.prediction
~~~~~~~~~~~~~~~~~~~~~~
Tier prediction for UCI WorldTour race finishes.

  trainer.py   — ModelTrainer: Optuna tuning, evaluation, artifact saving (one-time run)
  predictor.py — TierPredictor: load saved artifact and serve predictions (agent runtime)
"""

from peloton_iq.prediction.predictor import TierPredictor

__all__ = ["TierPredictor"]