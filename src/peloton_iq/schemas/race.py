"""
peloton_iq.schemas.race
~~~~~~~~~~~~~~~~~~~~~~~~~
Pydantic models for race results and course profiles.

These are pure data containers — no business logic, no I/O.
All field names match the column names in the processed CSVs
so that `Model(**row.to_dict())` works without remapping.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class StageType(str, Enum):
    """Terrain classification for a race or stage."""
    FLAT        = "flat"
    HILLY       = "hilly"
    MOUNTAIN    = "mountain"
    COBBLED     = "cobbled"
    TIME_TRIAL  = "time_trial"

    @classmethod
    def classify(cls, vertical_gain: float, distance: float, cobblestones: float = 0.0) -> "StageType":
        """
        Derive stage type from course metrics.
        Mirrors the logic in notebook 05 so classification is consistent
        across training and inference.
        """
        vg   = vertical_gain or 0.0
        dist = distance or 1.0
        cob  = cobblestones or 0.0

        if dist < 60 and vg < 500:
            return cls.TIME_TRIAL
        if cob > 10:
            return cls.COBBLED
        if vg > 4000:
            return cls.MOUNTAIN
        if vg > 2000:
            return cls.HILLY
        return cls.FLAT


class FinishTier(str, Enum):
    """Ordinal finish tier used as the prediction target."""
    WINNER   = "winner"
    PODIUM   = "podium"
    TOP10    = "top10"
    TOP20    = "top20"
    FINISHER = "finisher"
    DNF      = "dnf"

    @classmethod
    def from_rank(cls, rank: int, did_finish: bool) -> "FinishTier":
        """Assign a tier from a raw rank value. Mirrors notebook 05 assign_tier()."""
        if not did_finish:
            return cls.DNF
        if rank == 1:
            return cls.WINNER
        if rank <= 3:
            return cls.PODIUM
        if rank <= 10:
            return cls.TOP10
        if rank <= 20:
            return cls.TOP20
        return cls.FINISHER

    @property
    def ordinal(self) -> int:
        """Integer index matching TIER_ORDER in the trained model."""
        _order = [
            self.WINNER, self.PODIUM, self.TOP10,
            self.TOP20, self.FINISHER, self.DNF,
        ]
        return _order.index(self)


# ---------------------------------------------------------------------------
# Course profile
# ---------------------------------------------------------------------------

class CourseProfile(BaseModel):
    """
    Physical characteristics of a race or stage course.
    Sourced from structured_course_data.csv after cleaning.
    """

    race_name:         str
    year:              int
    race:              str                    # race series name, e.g. "Tour de France"
    stage:             Optional[str] = None   # "17", None for one-day races

    distance:          Optional[float] = None   # km
    vertical_gain:     Optional[float] = None   # metres
    highest_elevation: Optional[float] = None   # metres
    lowest_elevation:  Optional[float] = None   # metres
    net_gain:          Optional[float] = None   # metres

    # Surface breakdowns in km
    asphalt:           Optional[float] = None
    cobblestones:      Optional[float] = None
    compacted_gravel:  Optional[float] = None
    unpaved:           Optional[float] = None
    paved:             Optional[float] = None

    downhill:          Optional[float] = None   # total descent in metres

    @property
    def stage_type(self) -> StageType:
        return StageType.classify(
            vertical_gain=self.vertical_gain or 0.0,
            distance=self.distance or 1.0,
            cobblestones=self.cobblestones or 0.0,
        )

    @property
    def vg_per_km(self) -> Optional[float]:
        if self.vertical_gain is not None and self.distance:
            return self.vertical_gain / self.distance
        return None

    @property
    def cobble_pct(self) -> float:
        if self.cobblestones is not None and self.distance:
            return self.cobblestones / self.distance
        return 0.0

    @property
    def asphalt_pct(self) -> float:
        if self.asphalt is not None and self.distance:
            return self.asphalt / self.distance
        return 0.0

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Race result (one row = one rider's result in one race/stage)
# ---------------------------------------------------------------------------

class RaceResult(BaseModel):
    """
    A single rider's result in a single race or stage.
    Sourced from race_results_2017_2023.csv after cleaning.
    """

    race_name:   str
    name:        str            # rider name, ALL-CAPS SURNAME Firstname format
    team:        Optional[str] = None
    rank:        int            = Field(ge=1)
    did_finish:  bool
    date:        date
    year:        int
    race:        str            # series name
    stage:       Optional[str] = None

    # Derived flags (may be pre-computed in the CSV)
    top3:        Optional[bool] = None
    top10:       Optional[bool] = None

    @model_validator(mode="after")
    def derive_flags(self) -> "RaceResult":
        """Back-fill Top3/Top10 if not already set."""
        if self.top3 is None:
            self.top3 = self.did_finish and self.rank <= 3
        if self.top10 is None:
            self.top10 = self.did_finish and self.rank <= 10
        return self

    @property
    def finish_tier(self) -> FinishTier:
        return FinishTier.from_rank(self.rank, self.did_finish)

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# UCI filter result  (used by ingestion layer)
# ---------------------------------------------------------------------------

class UCIFilterResult(BaseModel):
    """
    Output of the UCI WorldTour race filter for a single row.
    Carries the match reason for audit logging.
    """
    is_uci:       bool
    match_reason: str   # "exact" | "fuzzy:<score>:<matched>" | "no_match:<score>" | etc.