"""
peloton_iq.prediction.predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TierPredictor — loads the saved tier_predictor.pkl and serves
finish tier probability predictions at inference time.

This is what the agent calls. It never retrains — it just loads
the artifact once and wraps it in a clean interface.

Usage:
    from peloton_iq.prediction.predictor import TierPredictor

    predictor = TierPredictor()
    context   = predictor.predict_stage("Tour de France", 2023, stage=17)
    print(context.to_prompt_text())
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd

from peloton_iq.config import MERGED_RACES_PATH, MODEL_DF_PATH, TIER_PREDICTOR_PATH, settings
from peloton_iq.schemas import PredictionContext, TierProbabilities

log = logging.getLogger(__name__)


class TierPredictor:
    """
    Wraps the saved tier_predictor.pkl for inference.

    Lazy initialization — the model and model_df are only loaded
    on first use so importing this class is cheap.
    """

    def __init__(self, artifact_path: Path | None = None) -> None:
        self._artifact_path = artifact_path or TIER_PREDICTOR_PATH
        self._artifact:  Optional[dict]         = None
        self._model_df:  Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    @property
    def artifact(self) -> dict:
        if self._artifact is None:
            log.info("Loading tier_predictor from %s", self._artifact_path)
            with open(self._artifact_path, "rb") as f:
                self._artifact = pickle.load(f)
            log.info(
                "Loaded: %s  top-5=%.1f%%",
                self._artifact["model_name"],
                self._artifact["metrics"].get(self._artifact["model_name"], {}).get("top5", 0) * 100,
            )
        return self._artifact

    @property
    def model(self):
        return self.artifact["model"]

    @property
    def model_name(self) -> str:
        return self.artifact["model_name"]

    @property
    def feature_cols(self) -> list[str]:
        return self.artifact["feature_cols"]

    @property
    def tier_order(self) -> list[str]:
        return self.artifact["tier_order"]

    @property
    def needs_imputation(self) -> bool:
        return self.model_name == "XGBoost"

    @property
    def medians(self) -> Optional[pd.Series]:
        raw = self.artifact.get("xgb_medians")
        return pd.Series(raw) if raw else None

    @property
    def model_df(self) -> pd.DataFrame:
        if self._model_df is None:
            path = MODEL_DF_PATH
            log.info("Loading model_df from %s", path)
            self._model_df = pd.read_csv(path, low_memory=False)
        return self._model_df

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_stage(
        self,
        race_name: str,
        year: int,
        stage: Optional[int] = None,
        top_n: int | None = None,
    ) -> PredictionContext:
        """
        Predict finish tier probabilities for all starters in a
        given race / stage.

        Args:
            race_name:  Partial or full race name (case-insensitive substring match).
            year:       Season year.
            stage:      Stage number for multi-stage races. None for one-day races.
            top_n:      Number of top riders to include in output.

        Returns:
            A PredictionContext with ranked TierProbabilities.
        """
        top_n = top_n or settings.prediction_top_n

        # Filter model_df to this race
        mask = (
            self.model_df["Race_results"].str.contains(
                race_name, na=False, case=False
            ) &
            (self.model_df["Year_results"] == year)
        )
        if stage is not None:
            mask &= (
                (self.model_df["Stage_results"] == stage) |
                (self.model_df["Stage_results"] == float(stage))
            )

        race_rows = self.model_df[mask].drop_duplicates("Name")

        if race_rows.empty:
            log.warning(
                "No rows found for '%s' %d%s",
                race_name, year,
                f" Stage {stage}" if stage else "",
            )
            available = (
                self.model_df[
                    self.model_df["Race_results"].str.contains(
                        race_name, na=False, case=False
                    ) &
                    (self.model_df["Year_results"] == year)
                ]["Stage_results"]
                .dropna()
                .unique()
            )
            log.info("Available stages: %s", sorted(available)[:20])
            return PredictionContext(
                race_name=race_name,
                year=year,
                stage=stage,
                model_name=self.model_name,
                top_riders=[],
            )

        # Run model
        X_race = race_rows[self.feature_cols].copy()
        if self.needs_imputation and self.medians is not None:
            X_race = X_race.fillna(self.medians)

        proba = self.model.predict_proba(X_race)

        # Build TierProbabilities for each rider
        tier_probs = []
        for i, (_, row) in enumerate(race_rows.iterrows()):
            tp = TierProbabilities.from_model_output(
                rider_name=row["Name"],
                proba_row=list(proba[i]),
                team=row.get("Team"),
            )
            tier_probs.append(tp)

        # Sort by win probability descending
        tier_probs.sort(key=lambda t: t.p_winner, reverse=True)

        # Infer stage type from first row
        stage_type = race_rows["stage_type"].iloc[0] if "stage_type" in race_rows.columns else None

        return PredictionContext(
            race_name=race_name,
            year=year,
            stage=stage,
            stage_type=stage_type,
            model_name=self.model_name,
            top_riders=tier_probs[:top_n],
        )

    def predict_race_context(
        self,
        race_name: str,
        year: int,
        stage: Optional[int] = None,
    ) -> str:
        """
        Convenience method returning the formatted prompt text directly.
        Used by the agent's predictor_node.
        """
        context = self.predict_stage(race_name, year, stage)
        return context.to_prompt_text()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def metrics_summary(self) -> str:
        """Return a formatted summary of model metrics."""
        metrics = self.artifact.get("metrics", {})
        lines   = [f"Model: {self.model_name}"]
        for name, m in metrics.items():
            lines.append(
                f"  {name:<20}  log_loss={m.get('log_loss', 0):.4f}  "
                f"auc={m.get('auc', 0):.4f}  "
                f"top5={m.get('top5', 0)*100:.1f}%"
            )
        return "\n".join(lines)