"""
peloton_iq.schemas.rider
~~~~~~~~~~~~~~~~~~~~~~~~~~
Pydantic models for rider performance data and computed features.

RiderFeatures mirrors the exact column set written by
compute_rider_history() in notebook 05 so that downstream
code can validate loaded CSVs against a known schema.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Rider season summary  (used by the embedding / search layer)
# ---------------------------------------------------------------------------

class RiderSeason(BaseModel):
    """
    Aggregated results for one rider across one calendar year.
    Used to generate natural-language documents for embedding.
    """

    name:          str
    year:          int
    team:          Optional[str] = None
    races_count:   int = Field(default=0, ge=0)
    wins:          int = Field(default=0, ge=0)
    podiums:       int = Field(default=0, ge=0)
    top10s:        int = Field(default=0, ge=0)
    dnfs:          int = Field(default=0, ge=0)

    win_races:     list[str] = Field(default_factory=list)
    podium_races:  list[str] = Field(default_factory=list)

    # Best GC results in the three Grand Tours
    tdf_best:      Optional[int] = None   # rank, None if did not start
    giro_best:     Optional[int] = None
    vuelta_best:   Optional[int] = None

    @property
    def doc_id(self) -> str:
        return f"{self.name}_{self.year}"


# ---------------------------------------------------------------------------
# Per-race rider features  (used by the prediction layer)
# ---------------------------------------------------------------------------

class RiderFeatures(BaseModel):
    """
    Lag/window features for a single rider × race combination.
    All features are point-in-time safe (computed from data strictly
    before the race date).

    Field names match the CSV columns written by compute_rider_history()
    and the FEATURE_COLS list in the trained model.
    """

    name:       str
    race_name:  str
    date:       date

    # ------------------------------------------------------------------
    # Recent form
    # ------------------------------------------------------------------
    recent_avg_rank_5:       Optional[float] = None  # avg rank over last 5 races
    recent_avg_rank_12mo:    Optional[float] = None
    recent_top10_rate_12mo:  Optional[float] = None
    recent_top10_rate_6mo:   Optional[float] = None
    recent_win_rate_12mo:    Optional[float] = None
    recent_podium_rate_12mo: Optional[float] = None
    recent_dnf_rate_12mo:    Optional[float] = None

    # ------------------------------------------------------------------
    # Workload
    # ------------------------------------------------------------------
    races_last_30d:       int   = Field(default=0, ge=0)
    races_last_12mo:      int   = Field(default=0, ge=0)
    days_since_last_race: float = Field(default=999.0, ge=0)

    # ------------------------------------------------------------------
    # Terrain affinity  (for the stage type of the upcoming race)
    # ------------------------------------------------------------------
    terrain_avg_rank:     Optional[float] = None
    terrain_top10_rate:   Optional[float] = None
    terrain_win_rate:     Optional[float] = None
    terrain_podium_rate:  Optional[float] = None
    terrain_dnf_rate:     Optional[float] = None
    terrain_avg_rank_12mo: Optional[float] = None
    terrain_races_count:  int = Field(default=0, ge=0)

    # ------------------------------------------------------------------
    # Career stats
    # ------------------------------------------------------------------
    career_top10_rate:   Optional[float] = None
    career_podium_rate:  Optional[float] = None
    career_win_rate:     Optional[float] = None
    career_races:        int = Field(default=0, ge=0)
    career_avg_rank:     Optional[float] = None

    model_config = {"populate_by_name": True}