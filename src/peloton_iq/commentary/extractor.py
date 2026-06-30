"""
peloton_iq.commentary.extractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ClaudeExtractor — extracts structured tactical insights from race transcripts.

FIXED: EXTRACTION_PROMPT now includes a grounded roster of real rider names
(pulled from merged_df["Name"] for the relevant race/year) and instructs
Claude to use EXACT spelling from that list. Previously the prompt only
said "LASTNAME Firstname" as a format hint with no grounding at all —
Claude was transcribing names purely from noisy commentary audio with no
canonical reference, producing wildly inconsistent spellings across calls
for the same rider (e.g. "ala_philippe_julian" / "alaphilippe_julien" /
"colbrelli_sonny" / "cobrelli_sonny", and outright phonetic
misrecognitions like "gerheigh_ruben" for "guerreiro_ruben"). Grounding
the prompt in the real roster turns this into a matching problem against
known names instead of an open-ended transcription problem.

Usage:
    from peloton_iq.commentary.extractor import ClaudeExtractor

    extractor = ClaudeExtractor()
    result    = extractor.extract_from_file("2023 Tour de France Stage 17")
    context   = extractor.get_context("Tour de France", "2023-07-19", stage=17)
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import pandas as pd

from peloton_iq.config import (
    COMMENTARY_EXTRACTED_DIR,
    COMMENTARY_RAW_DIR,
    MERGED_RACES_PATH,
    settings,
)
from peloton_iq.commentary.transcript import make_label, make_safe_name
from peloton_iq.schemas import CommentaryExtraction, TacticalInsight

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction prompt  — now with a grounded roster placeholder
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a cycling tactics analyst extracting intelligence from race commentary for {race_name}.

Your output will be injected directly into pre-race briefings for future similar races.
Be concise and opinionated. Every sentence must contain a non-obvious tactical observation.
Do not repeat the same point across fields. Skip generic observations ("wet roads are slippery", "sprinters like flat stages").

CRITICAL — rider names: Use ONLY the exact spelling from this known rider roster
when referring to any rider. Do not invent, phonetically guess, or vary the
spelling of a name. If you genuinely cannot match a rider mentioned in the
commentary to any name in this roster, omit them rather than guessing a
spelling.

Known riders (LASTNAME Firstname format, exact spelling required):
{roster}

Focus on:
- HOW and WHERE decisive moves happened (km to go, climb number, terrain feature)
- Which riders/teams showed unexpected strength or weakness
- Team tactics that reveal strategic patterns
- Anything that would change how you'd handicap a future similar race

Return a JSON object with these exact fields:
{{
  "one_line_summary": "Single sentence: who won and how (e.g. POGACAR attacked 38km out on Col de la Loze, dropped VINGEGAARD on the descent)",
  "winner": "LASTNAME Firstname",
  "commentary_quality": "rich|moderate|thin",
  "tactical_patterns": [
    "Concise pattern observation directly useful for future race handicapping. Name the rider/team. Be specific about where/when. Max 20 words."
  ],
  "rider_signals": {{
    "LASTNAME Firstname": "positive|negative|neutral: one-line observation"
  }},
  "usable_for_rag": true
}}

Rules:
- tactical_patterns: 2-5 items max. If commentary is thin, return fewer rather than padding.
- rider_signals: only riders with meaningful signal AND an exact match in the roster above. Skip riders with no notable observation, and skip any rider you cannot confidently match to the roster.
- winner: must exactly match a name in the roster above.
- If the transcript covers only an intermediate sprint or non-decisive moment, set usable_for_rag to false.
- Return only valid JSON, no other text.

Transcript:
{transcript}"""


# Fallback prompt section used when no roster could be built for this
# race/year (e.g. merged_df has no rows for it) — same strict format
# instruction as before, since there's nothing to ground against.
NO_ROSTER_FALLBACK = (
    "(No verified roster available for this race — use your best judgment "
    "on LASTNAME Firstname format, and be conservative: omit a rider "
    "entirely if you are not confident of the spelling.)"
)


# ---------------------------------------------------------------------------
# ClaudeExtractor
# ---------------------------------------------------------------------------

class ClaudeExtractor:
    """
    Extracts structured tactical insights from race transcripts using Claude.

    Reads raw transcript JSON from data/commentary/raw/,
    calls Claude for extraction, and saves results to data/commentary/extracted/.
    """

    def __init__(
        self,
        raw_dir: Path | None = None,
        extracted_dir: Path | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        max_transcript_chars: int = 8000,
        merged_df: pd.DataFrame | None = None,
    ) -> None:
        self._raw_dir       = raw_dir       or COMMENTARY_RAW_DIR
        self._extracted_dir = extracted_dir or COMMENTARY_EXTRACTED_DIR
        self._model         = model         or settings.claude_model
        self._max_tokens    = max_tokens    or 3000
        self._max_chars     = max_transcript_chars
        self._client        = anthropic.Anthropic(
            api_key=settings.anthropic_api_key or None,
        )

        # Lazily-loaded roster source. Loading the full merged_df once
        # and slicing per-race is far cheaper than re-reading the CSV
        # on every single extraction call.
        self._merged_df = merged_df

        self._extracted_dir.mkdir(parents=True, exist_ok=True)

    @property
    def merged_df(self) -> Optional[pd.DataFrame]:
        """Lazy-load merged_df once, reused across all extraction calls."""
        if self._merged_df is None:
            try:
                self._merged_df = pd.read_csv(MERGED_RACES_PATH, low_memory=False)
            except Exception as e:
                log.warning(
                    "Could not load merged_df for roster grounding (%s) — "
                    "extraction will fall back to unconstrained name format.",
                    e,
                )
                self._merged_df = pd.DataFrame()  # sentinel: tried, empty
        return self._merged_df if not self._merged_df.empty else None

    def _build_roster(self, race_name: str, race_date: str) -> str:
        """
        Build the roster text block for the extraction prompt: every
        rider who appears in merged_df for this race name + year, as
        "LASTNAME Firstname", one per line. Falls back to a generic
        instruction if merged_df isn't available or has no matching rows.

        Year-only matching (not exact date) is intentional — stage-level
        rosters are usually stable across the whole race, and this keeps
        the lookup simple and tolerant of the race_date placeholder
        values we've seen show up for some manually-added races.
        """
        df = self.merged_df
        if df is None:
            return NO_ROSTER_FALLBACK

        year = str(race_date)[:4]
        try:
            year_int = int(year)
        except ValueError:
            return NO_ROSTER_FALLBACK

        mask = (
            df["Race_results"].str.contains(race_name, case=False, na=False)
            & (df["Year_results"] == year_int)
        )
        riders = df.loc[mask, "Name"].dropna().unique()

        if len(riders) == 0:
            return NO_ROSTER_FALLBACK

        return "\n".join(sorted(riders))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_from_file(self, label: str) -> Optional[CommentaryExtraction]:
        """
        Extract tactical insights for a race by label.
        Reads from raw/, saves to extracted/.
        Returns None if no transcript is available.
        """
        safe_name      = make_safe_name(label)
        raw_path       = self._raw_dir       / f"{safe_name}.json"
        extracted_path = self._extracted_dir / f"{safe_name}.json"

        # Already extracted
        if extracted_path.exists():
            return self._load_extraction(extracted_path)

        # Load raw transcript
        if not raw_path.exists():
            log.warning("No raw file for %s", label)
            return None

        with open(raw_path, encoding="utf-8") as f:
            data = json.load(f)

        if data.get("status") != "transcript_saved":
            log.debug("Skipping %s (status=%s)", label, data.get("status"))
            return None

        transcript_text = data.get("transcript", {}).get("clean_text", "")
        if not transcript_text.strip():
            log.warning("Empty transcript for %s", label)
            return None

        # Call Claude
        result = self._call_claude(label, data, transcript_text)
        if result:
            self._save_extraction(result, extracted_path)

        return result

    def run_batch(
        self,
        max_extractions: int = 50,
        verbose: bool = True,
    ) -> dict:
        """
        Extract all pending transcripts (transcript_saved but not yet extracted).
        """
        pending = [
            path for path in sorted(self._raw_dir.glob("*.json"))
            if not (self._extracted_dir / path.name).exists()
            and json.load(open(path, encoding="utf-8")).get("status") == "transcript_saved"
        ]

        total_cost = success = skipped = errors = 0
        log.info(
            "Transcripts pending extraction: %d (processing up to %d)",
            len(pending), max_extractions,
        )

        for path in pending[:max_extractions]:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            transcript_text = data.get("transcript", {}).get("clean_text", "")
            if not transcript_text.strip():
                skipped += 1
                continue

            label     = data.get("label", path.stem)
            chars     = len(transcript_text)
            est_cost  = min(chars, self._max_chars) / 4 / 1_000_000 * 3

            if verbose:
                log.info("Extracting: %s  (%d chars | ~$%.4f)", label, chars, est_cost)

            result = self._call_claude(label, data, transcript_text)
            if result:
                extracted_path = self._extracted_dir / path.name
                self._save_extraction(result, extracted_path)
                total_cost += est_cost
                success    += 1
                if verbose:
                    log.info(
                        "  ✓ events=%d  form_signals=%d",
                        len(result.key_insights), 0,
                    )
            else:
                errors += 1

            time.sleep(0.5)

        log.info(
            "Extraction complete — success: %d  skipped: %d  errors: %d  cost: $%.4f",
            success, skipped, errors, total_cost,
        )
        return {"success": success, "skipped": skipped, "errors": errors, "cost": total_cost}

    def get_context(
        self,
        race_name: str,
        race_date: str,
        stage: Optional[int] = None,
        max_chars: int = 3000,
    ) -> str:
        """
        Return commentary context string for the agent.

        Priority:
          1. Extracted JSON (structured) → rich context
          2. Raw transcript text → truncated raw context
          3. No data → placeholder string

        This is the single entry point the agent's commentary_node calls.
        """
        label     = make_label(race_name, race_date, stage)
        safe_name = make_safe_name(label)

        extracted_path = self._extracted_dir / f"{safe_name}.json"
        raw_path       = self._raw_dir       / f"{safe_name}.json"

        # Prefer extracted structured context
        if extracted_path.exists():
            result = self._load_extraction(extracted_path)
            if result:
                return result.to_context_text()

        # Fall back to raw transcript
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
    # Internal
    # ------------------------------------------------------------------

    def _call_claude(
        self,
        label: str,
        raw_data: dict,
        transcript_text: str,
    ) -> Optional[CommentaryExtraction]:
        """Call Claude and parse the JSON response into a CommentaryExtraction."""
        # Truncate transcript symmetrically if too long
        if len(transcript_text) > self._max_chars:
            half = self._max_chars // 2
            text = transcript_text[:half] + "\n[...middle omitted...]\n" + transcript_text[-half:]
        else:
            text = transcript_text

        race_name = raw_data.get("race_name", "")
        race_date = raw_data.get("race_date", "")
        roster    = self._build_roster(race_name, race_date)

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(
                        race_name=label, transcript=text, roster=roster,
                    ),
                }],
            )
            raw_json = re.sub(r"```json|```", "", response.content[0].text).strip()
            parsed   = json.loads(raw_json)
        except json.JSONDecodeError as e:
            log.error("JSON parse failed for %s: %s", label, e)
            return None
        except Exception as e:
            log.error("Claude call failed for %s: %s", label, e)
            return None

        # Map new compact schema to CommentaryExtraction
        insights = []

        # Tactical patterns -> TacticalInsight
        for pattern in parsed.get("tactical_patterns", []):
            insights.append(TacticalInsight(
                category="tactical_pattern",
                description=pattern,
                riders=[],
            ))

        # Rider signals -> TacticalInsight
        for rider, signal_str in parsed.get("rider_signals", {}).items():
            insights.append(TacticalInsight(
                category="rider_signal",
                description=f"{rider}: {signal_str}",
                riders=[rider],
            ))

        video_raw = raw_data.get("video") or {}
        return CommentaryExtraction(
            label=label,
            race_name=race_name,
            race_date=race_date,
            stage=raw_data.get("stage"),
            video_id=video_raw.get("video_id"),
            channel=video_raw.get("channel"),
            race_summary=parsed.get("one_line_summary"),
            winner=parsed.get("winner"),
            key_insights=insights,
            raw_extraction=json.dumps(parsed),
            extraction_model=self._model,
        )

    def _load_extraction(self, path: Path) -> Optional[CommentaryExtraction]:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            extraction = data.get("extraction", {})
            if "error" in extraction:
                return None

            insights = []
            for pattern in extraction.get("tactical_patterns", []):
                insights.append(TacticalInsight(
                    category="tactical_pattern",
                    description=pattern,
                    riders=[],
                ))
            for rider, signal_str in extraction.get("rider_signals", {}).items():
                insights.append(TacticalInsight(
                    category="rider_signal",
                    description=f"{rider}: {signal_str}",
                    riders=[rider],
                ))

            return CommentaryExtraction(
                label=data.get("label", ""),
                race_name=data.get("race_name", ""),
                race_date=data.get("race_date", ""),
                stage=data.get("stage"),
                video_id=None,
                channel=data.get("channel"),
                race_summary=extraction.get("one_line_summary"),
                winner=extraction.get("winner"),
                key_insights=insights,
                raw_extraction=json.dumps(extraction),
                extraction_model=None,
            )
        except Exception as e:
            log.warning("Failed to load extraction from %s: %s", path, e)
            return None

    @staticmethod
    def _save_extraction(result: CommentaryExtraction, path: Path) -> None:
        output = {
            "label":        result.label,
            "race_name":    result.race_name,
            "race_date":    result.race_date,
            "stage":        result.stage,
            "channel":      result.channel,
            "extraction":   json.loads(result.raw_extraction) if result.raw_extraction else {},
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)