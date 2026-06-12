"""
peloton_iq.schemas
~~~~~~~~~~~~~~~~~~~~
Public re-exports for all domain models.

Import from here rather than from the submodules directly:

    from peloton_iq.schemas import (
        CourseProfile, RaceResult, StageType, FinishTier,
        RiderSeason, RiderFeatures,
        TierProbabilities, PredictionContext,
        TranscriptResult, CommentaryExtraction,
        PelotonState, empty_state,
    )
"""

from peloton_iq.schemas.race import (
    CourseProfile,
    FinishTier,
    RaceResult,
    StageType,
    UCIFilterResult,
)
from peloton_iq.schemas.rider import (
    RiderFeatures,
    RiderSeason,
)
from peloton_iq.schemas.prediction import (
    PredictionContext,
    TierProbabilities,
)
from peloton_iq.schemas.commentary import (
    CommentaryExtraction,
    TacticalInsight,
    TranscriptResult,
    TranscriptStatus,
    VideoMetadata,
)
from peloton_iq.schemas.agent import (
    PelotonState,
    PelotonStateModel,
    RaceContext,
    RouterOutput,
    empty_state,
)

__all__ = [
    # race
    "CourseProfile",
    "FinishTier",
    "RaceResult",
    "StageType",
    "UCIFilterResult",
    # rider
    "RiderFeatures",
    "RiderSeason",
    # prediction
    "PredictionContext",
    "TierProbabilities",
    # commentary
    "CommentaryExtraction",
    "TacticalInsight",
    "TranscriptResult",
    "TranscriptStatus",
    "VideoMetadata",
    # agent
    "PelotonState",
    "PelotonStateModel",
    "RaceContext",
    "RouterOutput",
    "empty_state",
]