"""
peloton_iq.schemas.prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pydantic models for ML model outputs and formatted prediction context.

These are the data contracts between the prediction layer and the
agent's predictor_node — keeping the ML internals decoupled from
the agent state.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from peloton_iq.schemas.race import FinishTier


# ---------------------------------------------------------------------------
# Per-rider tier probability distribution
# ---------------------------------------------------------------------------

class TierProbabilities(BaseModel):
    """
    Probability distribution over finish tiers for a single rider
    in a single race. Probabilities sum to 1.0.
    """

    rider_name: str
    team:       Optional[str] = None

    # One field per tier — mirrors TIER_ORDER
    p_winner:   float = Field(ge=0.0, le=1.0)
    p_podium:   float = Field(ge=0.0, le=1.0)
    p_top10:    float = Field(ge=0.0, le=1.0)
    p_top20:    float = Field(ge=0.0, le=1.0)
    p_finisher: float = Field(ge=0.0, le=1.0)
    p_dnf:      float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def check_sums_to_one(self) -> "TierProbabilities":
        total = (
            self.p_winner + self.p_podium + self.p_top10
            + self.p_top20 + self.p_finisher + self.p_dnf
        )
        if not (0.98 <= total <= 1.02):
            raise ValueError(
                f"Tier probabilities must sum to ~1.0, got {total:.4f} for {self.rider_name}"
            )
        return self

    @property
    def most_likely_tier(self) -> FinishTier:
        probs = {
            FinishTier.WINNER:   self.p_winner,
            FinishTier.PODIUM:   self.p_podium,
            FinishTier.TOP10:    self.p_top10,
            FinishTier.TOP20:    self.p_top20,
            FinishTier.FINISHER: self.p_finisher,
            FinishTier.DNF:      self.p_dnf,
        }
        return max(probs, key=lambda t: probs[t])

    @property
    def win_probability_pct(self) -> float:
        return round(self.p_winner * 100, 1)

    @property
    def podium_or_better_pct(self) -> float:
        return round((self.p_winner + self.p_podium) * 100, 1)

    @classmethod
    def from_model_output(
        cls,
        rider_name: str,
        proba_row: list[float],
        team: Optional[str] = None,
    ) -> "TierProbabilities":
        """
        Construct from a raw model.predict_proba() row.
        Assumes column order matches TIER_ORDER:
          [winner, podium, top10, top20, finisher, dnf]
        """
        if len(proba_row) != 6:
            raise ValueError(f"Expected 6 class probabilities, got {len(proba_row)}")
        return cls(
            rider_name=rider_name,
            team=team,
            p_winner=proba_row[0],
            p_podium=proba_row[1],
            p_top10=proba_row[2],
            p_top20=proba_row[3],
            p_finisher=proba_row[4],
            p_dnf=proba_row[5],
        )


# ---------------------------------------------------------------------------
# Full prediction context for one race
# ---------------------------------------------------------------------------

class PredictionContext(BaseModel):
    """
    ML model output for a specific race, formatted for injection
    into the agent's synthesizer prompt.
    """

    race_name:   str
    year:        int
    stage:       Optional[int] = None
    stage_type:  Optional[str] = None  # StageType string value
    model_name:  str                   # "LightGBM" | "XGBoost"

    top_riders:  list[TierProbabilities] = Field(
        default_factory=list,
        description="Riders ranked by win probability, descending.",
    )

    def to_prompt_text(self) -> str:
        """
        Render as a formatted text block for the synthesizer system prompt.
        Mirrors predict_stage_context() output from notebook 08.
        """
        if not self.top_riders:
            return f"[NO PREDICTION] No rider data found for {self.race_name} {self.year}."

        header = (
            f"Pre-race ML predictions — {self.race_name} {self.year}"
            + (f" Stage {self.stage}" if self.stage else "")
            + (f" ({self.stage_type})" if self.stage_type else "")
            + f"\nModel: {self.model_name}\n"
        )

        lines = []
        for i, r in enumerate(self.top_riders, start=1):
            lines.append(
                f"  {i:>2}. {r.rider_name:<30}"
                f"  win: {r.win_probability_pct:>5.1f}%"
                f"  podium+: {r.podium_or_better_pct:>5.1f}%"
                f"  likely: {r.most_likely_tier.value}"
                + (f"  [{r.team}]" if r.team else "")
            )

        return header + "\n".join(lines)