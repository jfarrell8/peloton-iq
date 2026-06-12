"""
peloton_iq.schemas.commentary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pydantic models for YouTube transcript ingestion and
Claude tactical extraction outputs.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Transcript fetch status
# ---------------------------------------------------------------------------

class TranscriptStatus(str, Enum):
    SUCCESS             = "success"
    NO_TRANSCRIPT       = "no_transcript"
    TRANSCRIPTS_DISABLED = "transcripts_disabled"
    VIDEO_UNAVAILABLE   = "video_unavailable"
    NO_VIDEO_FOUND      = "no_video_found"
    IP_BLOCKED          = "ip_blocked"
    ERROR               = "error"


# ---------------------------------------------------------------------------
# YouTube video metadata  (from the channel cache)
# ---------------------------------------------------------------------------

class VideoMetadata(BaseModel):
    """
    Lightweight record for a YouTube video, stored in the parquet cache.
    """

    video_id:   str
    title:      str
    published:  datetime
    channel:    str
    channel_id: str


# ---------------------------------------------------------------------------
# Raw transcript result
# ---------------------------------------------------------------------------

class TranscriptResult(BaseModel):
    """
    Output of a single transcript fetch attempt.
    Stored as JSON in data/commentary/raw/<safe_name>.json.
    """

    label:      str             # e.g. "2023 Tour de France Stage 17"
    race_name:  str
    race_date:  str             # ISO date string "YYYY-MM-DD"
    stage:      Optional[int] = None

    video:      Optional[VideoMetadata] = None
    status:     TranscriptStatus = TranscriptStatus.NO_VIDEO_FOUND

    # Populated on success
    clean_text:     Optional[str]   = None
    snippet_count:  Optional[int]   = None
    raw_chars:      Optional[int]   = None
    clean_chars:    Optional[int]   = None
    duration_mins:  Optional[float] = None
    preview_start:  Optional[str]   = None
    preview_end:    Optional[str]   = None

    error_detail:   Optional[str] = None

    @property
    def success(self) -> bool:
        return self.status == TranscriptStatus.SUCCESS

    @property
    def safe_name(self) -> str:
        """Filesystem-safe version of the label."""
        import re
        return re.sub(r"[^a-z0-9]+", "_", self.label.lower()).strip("_")


# ---------------------------------------------------------------------------
# Claude tactical extraction output
# ---------------------------------------------------------------------------

class TacticalInsight(BaseModel):
    """
    A single tactical observation extracted from race commentary
    by the Claude extractor.
    """

    category:    str    # e.g. "attack", "team_tactics", "weather", "crash", "key_moment"
    description: str
    riders:      list[str] = Field(default_factory=list)
    km_to_go:    Optional[float] = None


class CommentaryExtraction(BaseModel):
    """
    Structured output of Claude's tactical extraction pass
    over a raw race transcript.
    Stored as JSON in data/commentary/extracted/<safe_name>.json.
    """

    label:         str
    race_name:     str
    race_date:     str
    stage:         Optional[int] = None
    video_id:      Optional[str] = None
    channel:       Optional[str] = None

    # Claude extraction outputs
    race_summary:  Optional[str] = None
    winner:        Optional[str] = None
    key_insights:  list[TacticalInsight] = Field(default_factory=list)
    raw_extraction: Optional[str] = None   # full Claude response text

    extraction_model: Optional[str] = None  # Claude model string used

    @property
    def safe_name(self) -> str:
        import re
        return re.sub(r"[^a-z0-9]+", "_", self.label.lower()).strip("_")

    def to_context_text(self) -> str:
        """
        Render as a short text block for injection into the agent's
        commentary_node context.
        """
        if not self.race_summary and not self.key_insights:
            return f"[NO COMMENTARY] No extracted context for {self.label}."

        parts = []
        if self.race_summary:
            parts.append(f"Race summary: {self.race_summary}")
        if self.winner:
            parts.append(f"Winner: {self.winner}")
        for ins in self.key_insights[:5]:   # cap at 5 to keep context tight
            rider_str = f" ({', '.join(ins.riders)})" if ins.riders else ""
            parts.append(f"[{ins.category}]{rider_str} {ins.description}")

        return "\n".join(parts)