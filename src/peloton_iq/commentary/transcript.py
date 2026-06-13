"""
peloton_iq.commentary.transcript
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TranscriptFetcher — local video matching and YouTube transcript fetching.

Two distinct responsibilities:
  1. Match a race label against the local video cache (zero quota)
  2. Fetch the transcript for a matched video via youtube_transcript_api

The matching logic mirrors the v3 notebook's score_local_video() and
find_best_video_local() functions, with channel priority tiers and
NBC/GCN fast-paths for known coverage patterns.

Usage:
    from peloton_iq.commentary.transcript import TranscriptFetcher
    from peloton_iq.commentary.youtube import YouTubeCacheManager

    mgr     = YouTubeCacheManager()
    cache   = mgr.load_cache()
    fetcher = TranscriptFetcher(cache)

    result  = fetcher.fetch("Tour de France", "2023-07-19", stage=17)
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from peloton_iq.config import (
    COMMENTARY_RAW_DIR,
    settings,
)
from peloton_iq.schemas import TranscriptResult, TranscriptStatus, VideoMetadata

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Matching constants  (mirrors notebook v3)
# ---------------------------------------------------------------------------

SKIP_KEYWORDS = [
    "preview", "interview", "training", "behind the scenes",
    "beginner", "guide", "how to", "shorts", "power data",
    "riders to watch", "ones to watch", "top 10",
    "what to watch", "race preview", "stage preview",
    "team time trial preparation", "kitchen", "mechanic",
    "neutral service", "french phrases", "injuries",
    "dutch corner", "how to watch",
]
WOMEN_KEYWORDS = ["femmes", "feminine", "women", "ladies"]

CHANNEL_TIER = {
    "NBC Sports":         3,
    "GCN Racing":         2,
    "TNT Sports Cycling": 1,
    "GCN":                1,
}

NBC_PRIORITY_RACES = ["tour de france", "vuelta a espana", "vuelta españa"]
GCN_PRIORITY_RACES = [
    "giro d'italia", "giro d italia",
    "paris-roubaix", "paris roubaix",
    "tour of flanders", "ronde van vlaanderen",
    "liege-bastogne-liege", "milan-san remo", "milan san remo",
    "il lombardia",
]


def _is_nbc_priority(race_name: str) -> bool:
    rn = race_name.lower()
    return any(r in rn for r in NBC_PRIORITY_RACES)


def _is_gcn_priority(race_name: str) -> bool:
    rn = race_name.lower()
    return any(r in rn for r in GCN_PRIORITY_RACES)


# ---------------------------------------------------------------------------
# Label helpers  (mirrors notebook v3 make_label / make_safe_name)
# ---------------------------------------------------------------------------

def make_label(race_name: str, race_date: str, stage: Optional[int]) -> str:
    label = f"{race_date[:4]} {race_name}"
    return label + (f" Stage {stage}" if stage else "")


def make_safe_name(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


# ---------------------------------------------------------------------------
# TranscriptFetcher
# ---------------------------------------------------------------------------

class TranscriptFetcher:
    """
    Matches races against the local YouTube cache and fetches transcripts.

    All video matching is done locally — zero YouTube quota cost.
    Transcript fetching uses youtube_transcript_api — no quota, but
    subject to IP rate limiting (hence the configurable delays and retries).
    """

    def __init__(
        self,
        video_cache: Optional[pd.DataFrame] = None,
        raw_dir: Path | None = None,
        match_threshold: float = 75.0,
        retry_attempts: int | None = None,
        retry_backoff: float | None = None,
    ) -> None:
        self._cache      = video_cache
        self._raw_dir    = raw_dir or COMMENTARY_RAW_DIR
        self._threshold  = match_threshold
        self._retries    = retry_attempts or settings.transcript_retry_attempts
        self._backoff    = retry_backoff  or settings.transcript_retry_backoff
        self._ytt        = YouTubeTranscriptApi()

        self._raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(
        self,
        race_name: str,
        race_date: str,
        stage: Optional[int] = None,
        delay_seconds: float = 45.0,
    ) -> TranscriptResult:
        """
        Full pipeline for one race: match video → fetch transcript → save JSON.

        Args:
            race_name:     Race name (e.g. "Tour de France")
            race_date:     ISO date string "YYYY-MM-DD"
            stage:         Stage number, or None for one-day races
            delay_seconds: Base delay between transcript fetches to avoid IP blocks.

        Returns:
            A TranscriptResult saved to data/commentary/raw/<safe_name>.json
        """
        label     = make_label(race_name, race_date, stage)
        safe_name = make_safe_name(label)
        out_path  = self._raw_dir / f"{safe_name}.json"

        # Already processed — load from disk
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                data = json.load(f)
            log.debug("Loaded cached result for %s (status=%s)", label, data.get("status"))
            return self._dict_to_result(data, label, race_name, race_date, stage)

        # Find video
        video_match = self._find_best_video(race_name, race_date, stage)
        if not video_match:
            result = TranscriptResult(
                label=label, race_name=race_name, race_date=race_date,
                stage=stage, status=TranscriptStatus.NO_VIDEO_FOUND,
            )
            self._save(result, out_path)
            return result

        # Fetch transcript with retries
        result = self._fetch_with_retry(
            label, race_name, race_date, stage,
            video_match, delay_seconds,
        )
        self._save(result, out_path)
        return result

    def run_batch(
        self,
        race_index: pd.DataFrame,
        max_transcripts: int = 50,
        delay_seconds: float = 45.0,
        skip_existing: bool = True,
    ) -> dict:
        """
        Batch fetch transcripts for all races in race_index that have
        been video-matched but not yet transcript-fetched.

        Args:
            race_index:       DataFrame with Race Name, Race_results, Date, Stage_results columns
            max_transcripts:  Max transcripts to fetch in this run
            delay_seconds:    Base delay between requests
            skip_existing:    Skip races already in raw_dir
        """
        # Find pending — video_found status
        pending = []
        for path in sorted(self._raw_dir.glob("*.json")):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") == "video_found":
                pending.append((path, data))

        total = min(len(pending), max_transcripts)
        log.info("Videos pending transcript: %d (fetching up to %d)", len(pending), total)

        success = errors = ip_blocked = 0

        for path, data in pending[:max_transcripts]:
            label    = data["label"]
            video    = data.get("video") or {}
            video_id = video.get("video_id")
            if not video_id:
                continue

            log.info("Fetching: %s", label)
            transcript = self._fetch_transcript_raw(video_id)

            if transcript["success"]:
                data["transcript"] = {k: transcript[k] for k in [
                    "snippet_count", "raw_chars", "clean_chars",
                    "duration_mins", "clean_text", "preview_start", "preview_end",
                ]}
                data["status"] = "transcript_saved"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                log.info("  ✓ %d chars | %.1f min", transcript["clean_chars"], transcript["duration_mins"])
                success += 1
            else:
                err         = transcript["error"]
                is_ip_block = any(
                    kw in err.lower()
                    for kw in ["blocked", "ip", "429", "too many"]
                )
                if is_ip_block:
                    data["status"] = "transcript_error:ip_blocked"
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    ip_blocked += 1
                    log.warning("IP blocking detected — stopping after %d saved.", success)
                    break
                else:
                    data["status"] = f"transcript_error:{err[:80]}"
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    errors += 1
                    log.warning("  ✗ %s", err[:80])

            actual_delay = max(20.0, delay_seconds + random.uniform(-5, 10))
            log.info("  Waiting %.0fs...", actual_delay)
            time.sleep(actual_delay)

        log.info(
            "Batch complete — saved: %d  ip_blocked: %d  errors: %d",
            success, ip_blocked, errors,
        )
        return {"success": success, "ip_blocked": ip_blocked, "errors": errors}

    def run_local_matching(
        self,
        race_index: pd.DataFrame,
        max_races: Optional[int] = None,
        threshold: float = 75.0,
        verbose: bool = False,
    ) -> dict:
        """
        Match all unprocessed races against the local video cache.
        Zero YouTube API quota — runs as fast as your CPU.
        Safe to run multiple times — skips already-processed races.
        """
        if self._cache is None or self._cache.empty:
            log.error("No video cache loaded. Call YouTubeCacheManager.load_cache() first.")
            return {"found": 0, "not_found": 0, "skipped": 0}

        to_process = []
        for _, row in race_index.iterrows():
            race_name, stage = self._parse_race_name(row["Race Name"])
            race_date        = str(row["Date"])[:10]
            label            = make_label(race_name, race_date, stage)
            out_path         = self._raw_dir / f"{make_safe_name(label)}.json"
            if not out_path.exists():
                to_process.append((race_name, race_date, stage, label, out_path))

        limit = max_races if max_races is not None else len(to_process)
        log.info(
            "Local cache: %d videos | Races to process: %d (limit: %d)",
            len(self._cache), len(to_process), limit,
        )

        found = not_found = 0
        for i, (race_name, race_date, stage, label, out_path) in enumerate(to_process[:limit]):
            if i % 100 == 0:
                log.info("  [%d new | %d found | %d no_video]", i, found, not_found)

            video = self._find_best_video(race_name, race_date, stage, threshold=threshold)

            if video:
                record = {
                    "label": label, "race_name": race_name,
                    "race_date": race_date, "stage": stage,
                    "video": video, "transcript": None, "status": "video_found",
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
                found += 1
                if verbose:
                    log.debug("✓ [%.0f] [%s] %s", video["match_score"], video["channel"], video["title"][:55])
            else:
                record = {
                    "label": label, "race_name": race_name,
                    "race_date": race_date, "stage": stage,
                    "video": None, "transcript": None, "status": "no_video_found",
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2)
                not_found += 1

        log.info("Matching complete — found: %d  not_found: %d", found, not_found)
        return {"found": found, "not_found": not_found, "skipped": len(to_process) - limit}

    def get_commentary_context(
        self,
        race_name: str,
        race_date: str,
        stage: Optional[int] = None,
        max_chars: int = 3000,
    ) -> str:
        """
        Return raw transcript text for injection into the agent context.
        Falls back gracefully if no transcript is available.
        Delegates to CommentaryExtractor.get_context() when extracted JSON exists.
        """
        label     = make_label(race_name, race_date, stage)
        safe_name = make_safe_name(label)
        raw_path  = self._raw_dir / f"{safe_name}.json"

        if raw_path.exists():
            try:
                with open(raw_path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("status") == "transcript_saved":
                    text    = data.get("transcript", {}).get("clean_text", "")
                    channel = data.get("video", {}).get("channel", "unknown")
                    title   = data.get("video", {}).get("title", "")[:55]
                    if text:
                        if len(text) > max_chars:
                            half = max_chars // 2
                            text = text[:half] + "\n[...]\n" + text[-half:]
                        return f"[RAW COMMENTARY: {channel} | {title}]\n\n{text}"
            except Exception as e:
                log.warning("Failed to load raw commentary for %s: %s", label, e)

        return f"[NO COMMENTARY for {label}] Analysis based on structured data only."

    # ------------------------------------------------------------------
    # Video matching
    # ------------------------------------------------------------------

    def _find_best_video(
        self,
        race_name: str,
        race_date: str,
        stage: Optional[int] = None,
        threshold: float | None = None,
    ) -> Optional[dict]:
        """Find the best matching video from the local cache."""
        threshold = threshold or self._threshold

        if self._cache is None or self._cache.empty:
            return None

        race_year  = race_date[:4]
        candidates = self._cache[
            self._cache["title"].str.contains(race_year, case=False, na=False)
        ].copy()

        if candidates.empty:
            return None

        # Narrow by first meaningful keywords
        keywords = [w for w in race_name.split() if len(w) > 3][:2]
        if keywords:
            pattern = "|".join(re.escape(k) for k in keywords)
            narrow  = candidates[candidates["title"].str.contains(pattern, case=False, na=False)]
            if not narrow.empty:
                candidates = narrow

        # NBC fast-path for grand tours
        if _is_nbc_priority(race_name):
            nbc = candidates[candidates["channel"] == "NBC Sports"].copy()
            if not nbc.empty:
                nbc["match_score"] = nbc.apply(
                    lambda r: self._score_video(r, race_name, race_date, stage), axis=1
                )
                nbc = nbc.sort_values("match_score", ascending=False)
                best = nbc.iloc[0]
                if best["match_score"] >= threshold:
                    return self._build_video_dict(best, nbc)

        # Full pool scoring
        candidates["match_score"] = candidates.apply(
            lambda r: self._score_video(r, race_name, race_date, stage), axis=1
        )
        candidates = candidates.sort_values("match_score", ascending=False)
        best       = candidates.iloc[0]

        if best["match_score"] < threshold:
            return None

        return self._build_video_dict(best, candidates)

    def _score_video(
        self,
        row: pd.Series,
        race_name: str,
        race_date: str,
        stage: Optional[int],
    ) -> float:
        """Score a candidate video row. Mirrors score_local_video() from notebook v3."""
        score    = 0.0
        title    = str(row["title"]).lower()
        channel  = str(row.get("channel", ""))
        race_dt  = pd.to_datetime(race_date, utc=True)
        pub_dt   = pd.to_datetime(row["published"], utc=True)
        days_diff = abs((pub_dt.date() - race_dt.date()).days)

        # Publish date proximity
        if days_diff == 0:    score += 60
        elif days_diff <= 1:  score += 50
        elif days_diff <= 3:  score += 35
        elif days_diff <= 7:  score += 15
        elif days_diff > 365: score -= 40 * abs(pub_dt.year - race_dt.year)

        # Race name fuzzy match
        name_score = fuzz.partial_ratio(race_name.lower(), title)
        score += name_score * 0.25
        if name_score < 40:
            score -= 40

        # Stage match
        if stage:
            if any(p in title for p in [
                f"stage {stage}", f"stage{stage}",
                f"étape {stage}", f"tappa {stage}",
            ]):
                score += 30
            else:
                score -= 10

        # Year in title
        if race_date[:4] in title:
            score += 15

        # Content type bonuses
        if "extended highlights" in title: score += 20
        elif "extended" in title:          score += 12
        elif "highlights" in title:        score += 5
        if "full race" in title:           score += 8

        # Channel tier bonus
        score += CHANNEL_TIER.get(channel, 0) * 5

        # NBC priority races
        if _is_nbc_priority(race_name) and channel == "NBC Sports":
            score += 20
            if "extended highlights" in title:
                score += 25

        # GCN priority races
        if _is_gcn_priority(race_name) and channel in ("GCN Racing", "TNT Sports Cycling"):
            score += 15

        # Skip penalties
        if any(k in title for k in SKIP_KEYWORDS):  score -= 30
        if any(k in title for k in WOMEN_KEYWORDS): score -= 30

        return round(score, 2)

    @staticmethod
    def _build_video_dict(best: pd.Series, candidates: pd.DataFrame) -> dict:
        return {
            "video_id":    best["video_id"],
            "title":       best["title"],
            "published":   str(best["published"]),
            "channel":     best["channel"],
            "url":         f"https://youtube.com/watch?v={best['video_id']}",
            "match_score": float(best["match_score"]),
            "all_candidates": [
                {
                    "video_id":    c["video_id"],
                    "title":       c["title"],
                    "channel":     c["channel"],
                    "published":   str(c["published"]),
                    "match_score": c["match_score"],
                }
                for c in candidates.head(10).to_dict("records")
            ],
        }

    # ------------------------------------------------------------------
    # Transcript fetching
    # ------------------------------------------------------------------

    def _fetch_with_retry(
        self,
        label: str,
        race_name: str,
        race_date: str,
        stage: Optional[int],
        video: dict,
        delay_seconds: float,
    ) -> TranscriptResult:
        """Fetch transcript with exponential backoff retries."""
        video_meta = VideoMetadata(
            video_id=video["video_id"],
            title=video["title"],
            published=pd.to_datetime(video["published"]),
            channel=video["channel"],
            channel_id=video.get("channel_id", ""),
        )

        for attempt in range(self._retries):
            result = self._fetch_transcript_raw(video["video_id"])

            if result["success"]:
                return TranscriptResult(
                    label=label,
                    race_name=race_name,
                    race_date=race_date,
                    stage=stage,
                    video=video_meta,
                    status=TranscriptStatus.SUCCESS,
                    clean_text=result["clean_text"],
                    snippet_count=result["snippet_count"],
                    raw_chars=result["raw_chars"],
                    clean_chars=result["clean_chars"],
                    duration_mins=result["duration_mins"],
                    preview_start=result["preview_start"],
                    preview_end=result["preview_end"],
                )

            err = result["error"]
            status = self._classify_error(err)

            if status == TranscriptStatus.IP_BLOCKED:
                log.warning("IP blocked on attempt %d/%d for %s", attempt + 1, self._retries, label)
                if attempt < self._retries - 1:
                    sleep_time = self._backoff ** (attempt + 1) * 60
                    log.info("  Backing off %.0fs...", sleep_time)
                    time.sleep(sleep_time)
            else:
                # Non-retriable error
                return TranscriptResult(
                    label=label, race_name=race_name, race_date=race_date,
                    stage=stage, video=video_meta,
                    status=status, error_detail=err,
                )

        return TranscriptResult(
            label=label, race_name=race_name, race_date=race_date,
            stage=stage, video=video_meta,
            status=TranscriptStatus.IP_BLOCKED,
            error_detail="Max retries exceeded",
        )

    def _fetch_transcript_raw(self, video_id: str) -> dict:
        """Raw transcript fetch — returns a plain dict."""
        try:
            transcript = self._ytt.fetch(video_id)
            raw_text   = " ".join([s.text for s in transcript])
            clean_text = re.sub(r"\[.*?\]", "", raw_text)
            clean_text = re.sub(r"\(.*?\)", "", clean_text)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()
            duration   = round(transcript[-1].start, 0) if transcript else 0
            return {
                "success":       True,
                "video_id":      video_id,
                "snippet_count": len(transcript),
                "raw_chars":     len(raw_text),
                "clean_chars":   len(clean_text),
                "duration_mins": round(duration / 60, 1),
                "clean_text":    clean_text,
                "preview_start": clean_text[:500],
                "preview_end":   clean_text[-500:],
                "error":         None,
            }
        except NoTranscriptFound:
            return {"success": False, "video_id": video_id, "error": "no_transcript"}
        except TranscriptsDisabled:
            return {"success": False, "video_id": video_id, "error": "transcripts_disabled"}
        except VideoUnavailable:
            return {"success": False, "video_id": video_id, "error": "video_unavailable"}
        except Exception as e:
            return {"success": False, "video_id": video_id, "error": str(e)}

    @staticmethod
    def _classify_error(error_msg: str) -> TranscriptStatus:
        msg = error_msg.lower()
        if any(k in msg for k in ["blocked", "ip", "429", "too many"]):
            return TranscriptStatus.IP_BLOCKED
        if "no_transcript" in msg:
            return TranscriptStatus.NO_TRANSCRIPT
        if "transcripts_disabled" in msg:
            return TranscriptStatus.TRANSCRIPTS_DISABLED
        if "video_unavailable" in msg:
            return TranscriptStatus.VIDEO_UNAVAILABLE
        return TranscriptStatus.ERROR

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_race_name(race_name_full: str) -> tuple[str, Optional[int]]:
        clean       = re.sub(r"^\d{4}\s+", "", race_name_full).strip()
        stage_match = re.search(r"Stage\s+(\d+)", clean, re.IGNORECASE)
        stage       = int(stage_match.group(1)) if stage_match else None
        race        = re.sub(r"\s*Stage\s+\d+", "", clean, flags=re.IGNORECASE).strip()
        return race, stage

    @staticmethod
    def _save(result: TranscriptResult, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def _dict_to_result(
        data: dict,
        label: str,
        race_name: str,
        race_date: str,
        stage: Optional[int],
    ) -> TranscriptResult:
        """Reconstruct a TranscriptResult from a saved JSON dict."""
        status_str = data.get("status", "")
        if status_str == "transcript_saved":
            status = TranscriptStatus.SUCCESS
        elif status_str == "no_video_found":
            status = TranscriptStatus.NO_VIDEO_FOUND
        elif "ip_blocked" in status_str:
            status = TranscriptStatus.IP_BLOCKED
        elif status_str == "video_found":
            status = TranscriptStatus.NO_TRANSCRIPT  # pending
        else:
            status = TranscriptStatus.ERROR

        transcript = data.get("transcript") or {}
        video_raw  = data.get("video")
        video_meta = None
        if video_raw:
            try:
                video_meta = VideoMetadata(
                    video_id=video_raw["video_id"],
                    title=video_raw["title"],
                    published=pd.to_datetime(video_raw.get("published", "2017-01-01")),
                    channel=video_raw.get("channel", ""),
                    channel_id=video_raw.get("channel_id", ""),
                )
            except Exception:
                pass

        return TranscriptResult(
            label=label, race_name=race_name, race_date=race_date,
            stage=stage, video=video_meta, status=status,
            clean_text=transcript.get("clean_text"),
            snippet_count=transcript.get("snippet_count"),
            raw_chars=transcript.get("raw_chars"),
            clean_chars=transcript.get("clean_chars"),
            duration_mins=transcript.get("duration_mins"),
            preview_start=transcript.get("preview_start"),
            preview_end=transcript.get("preview_end"),
        )