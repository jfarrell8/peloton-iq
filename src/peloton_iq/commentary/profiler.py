"""
peloton_iq.commentary.profiler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RiderProfiler — aggregates tactical insights across multiple race
extractions to build a rider's tactical signature.

This is the differentiator. Where a single race extraction gives you
"Pogačar attacked on the Col de la Loze", the profiler gives you:
"Pogačar attacks on penultimate climbs in 7/9 mountain stages, never
from the final 2km, and UAE always controls tempo for 40+ km before
the move."

Profiles are cached to data/commentary/profiles/ so the agent can
load them at startup without re-reading every extraction file.

Usage:
    from peloton_iq.commentary.profiler import RiderProfiler

    profiler = RiderProfiler()
    profiler.build_all_profiles()           # one-time build
    profile  = profiler.get_profile("POGAČAR Tadej")
    context  = profiler.get_profile_context(["VINGEGAARD Jonas", "POGAČAR Tadej"])
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import anthropic

from peloton_iq.config import COMMENTARY_EXTRACTED_DIR, settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Profile storage
# ---------------------------------------------------------------------------

PROFILES_DIR = COMMENTARY_EXTRACTED_DIR.parent / "profiles"


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

PROFILE_PROMPT = """You are a professional cycling tactics analyst building a pre-race intelligence profile.

Below are tactical observations extracted from {n_races} race commentary appearances for {rider_name}.
These observations span {years} and include {race_types}.

Your job: synthesize these into a concise tactical signature for this rider.
Focus on PATTERNS that repeat across multiple races — not one-off events.
Be specific and opinionated. Skip obvious statements ("they like to win").

Return a JSON object:
{{
  "attacking_style": "1-2 sentence description of HOW and WHERE this rider typically attacks (km to go, terrain type, climb number)",
  "team_tactics": "How their team sets up their attack — tempo control, domestique usage, positioning",
  "terrain_preference": "Which specific terrain features suit them best, and which expose them",
  "vulnerability": "Specific conditions or scenarios where they have shown weakness",
  "key_patterns": [
    "Concise, specific pattern with evidence count where possible. Max 20 words each."
  ],
  "form_trajectory": "positive|negative|neutral|unknown: brief explanation based on most recent observations",
  "races_analysed": {n_races},
  "confidence": "high|medium|low based on number and quality of observations"
}}

Rules:
- key_patterns: 3-6 items. Each must be backed by at least 2 observations.
- If fewer than 3 races are available, set confidence to "low".
- Return only valid JSON, no other text.

Observations for {rider_name}:
{observations}"""


class RiderProfiler:
    """
    Aggregates tactical insights across race extractions to build
    rider tactical profiles. Profiles are cached to disk.
    """

    def __init__(
        self,
        extracted_dir: Path | None = None,
        profiles_dir: Path | None = None,
        model: str | None = None,
    ) -> None:
        self._extracted_dir = extracted_dir or COMMENTARY_EXTRACTED_DIR
        self._profiles_dir  = profiles_dir  or PROFILES_DIR
        self._model         = model or settings.claude_model
        self._client        = anthropic.Anthropic(
            api_key=settings.anthropic_api_key or None,
        )
        self._profiles_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache of loaded profiles
        self._cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_profile(self, rider_name: str) -> Optional[dict]:
        """
        Load a rider's tactical profile from disk cache.
        Returns None if no profile exists yet.
        """
        safe = self._safe_name(rider_name)
        if safe in self._cache:
            return self._cache[safe]

        path = self._profiles_dir / f"{safe}.json"
        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                profile = json.load(f)
            self._cache[safe] = profile
            return profile
        except Exception as e:
            log.warning("Failed to load profile for %s: %s", rider_name, e)
            return None

    def get_profile_context(
        self,
        rider_names: list[str],
        max_riders: int = 5,
    ) -> str:
        """
        Build a combined context string for multiple riders.
        Called by the agent's predictor_node to enrich pre-race briefings.

        Returns a formatted block like:
            TACTICAL PROFILES (from {N} race commentaries)
            ─────────────────────────────────────────────
            VINGEGAARD Jonas
              • Attacks on final climb only after UAE drops...
              • Vulnerable to multi-attack scenarios...
            ...
        """
        profiles_found = []
        for name in rider_names[:max_riders]:
            profile = self.get_profile(name)
            if profile:
                profiles_found.append((name, profile))

        if not profiles_found:
            return ""

        total_races = sum(p.get("races_analysed", 0) for _, p in profiles_found)
        lines = [
            f"TACTICAL PROFILES (from {total_races} race commentaries across {len(profiles_found)} riders)",
            "─" * 60,
        ]

        for rider_name, profile in profiles_found:
            conf     = profile.get("confidence", "low")
            n_races  = profile.get("races_analysed", 0)
            conf_str = f"[{conf} confidence · {n_races} races]"

            lines.append(f"\n{rider_name}  {conf_str}")

            if profile.get("attacking_style"):
                lines.append(f"  Attack style: {profile['attacking_style']}")

            if profile.get("vulnerability"):
                lines.append(f"  Vulnerability: {profile['vulnerability']}")

            for pattern in profile.get("key_patterns", [])[:3]:
                lines.append(f"  • {pattern}")

            traj = profile.get("form_trajectory", "unknown")
            if traj != "unknown":
                lines.append(f"  Form: {traj}")

        return "\n".join(lines)

    def build_all_profiles(
        self,
        min_races: int = 2,
        force_rebuild: bool = False,
    ) -> dict:
        """
        Build tactical profiles for all riders with sufficient
        commentary coverage. Skips riders with existing profiles
        unless force_rebuild=True.

        Args:
            min_races:     Minimum number of race extractions to build a profile.
            force_rebuild: Rebuild even if profile already exists.

        Returns:
            Summary dict with counts.
        """
        observations = self._collect_observations()

        built = skipped = insufficient = errors = 0
        qualifying = {
            rider: obs
            for rider, obs in observations.items()
            if len(obs) >= min_races
        }

        log.info(
            "Building profiles: %d riders qualify (>= %d races), %d insufficient",
            len(qualifying), min_races,
            len(observations) - len(qualifying),
        )

        for rider_name, obs_list in qualifying.items():
            safe = self._safe_name(rider_name)
            path = self._profiles_dir / f"{safe}.json"

            if path.exists() and not force_rebuild:
                skipped += 1
                continue

            try:
                profile = self._build_profile(rider_name, obs_list)
                if profile:
                    profile["rider_name"] = rider_name
                    profile["built_from"] = [o["label"] for o in obs_list]
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(profile, f, indent=2, ensure_ascii=False)
                    self._cache[safe] = profile
                    built += 1
                    log.info(
                        "Profile built: %-35s  %d races  confidence=%s",
                        rider_name, len(obs_list), profile.get("confidence", "?"),
                    )
                else:
                    errors += 1
            except Exception as e:
                log.error("Profile build failed for %s: %s", rider_name, e)
                errors += 1

        log.info(
            "Profiles complete — built: %d  skipped: %d  insufficient: %d  errors: %d",
            built, skipped, len(observations) - len(qualifying), errors,
        )
        return {
            "built":        built,
            "skipped":      skipped,
            "insufficient": len(observations) - len(qualifying),
            "errors":       errors,
            "total_riders": len(observations),
        }

    def build_profile_for_rider(
        self,
        rider_name: str,
        force_rebuild: bool = False,
    ) -> Optional[dict]:
        """Build or load a profile for a single rider."""
        safe = self._safe_name(rider_name)
        path = self._profiles_dir / f"{safe}.json"

        if path.exists() and not force_rebuild:
            return self.get_profile(rider_name)

        observations = self._collect_observations()
        obs_list = observations.get(rider_name, [])

        if len(obs_list) < 1:
            log.info("No observations found for %s", rider_name)
            return None

        profile = self._build_profile(rider_name, obs_list)
        if profile:
            profile["rider_name"]  = rider_name
            profile["built_from"]  = [o["label"] for o in obs_list]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
            self._cache[safe] = profile

        return profile

    def list_profiled_riders(self) -> list[str]:
        """Return names of all riders with existing profiles."""
        riders = []
        for path in sorted(self._profiles_dir.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                riders.append(data.get("rider_name", path.stem))
            except Exception:
                pass
        return riders

    def coverage_report(self) -> dict:
        """How many riders appear in extractions vs have profiles."""
        observations    = self._collect_observations()
        profiled        = self.list_profiled_riders()
        total_riders    = len(observations)
        total_races     = sum(len(v) for v in observations.values())
        well_covered    = sum(1 for v in observations.values() if len(v) >= 5)

        return {
            "total_riders_in_commentary": total_riders,
            "total_race_observations":    total_races,
            "riders_with_2plus_races":    sum(1 for v in observations.values() if len(v) >= 2),
            "riders_with_5plus_races":    well_covered,
            "riders_with_profiles":       len(profiled),
            "top_covered": sorted(
                [(r, len(v)) for r, v in observations.items()],
                key=lambda x: -x[1],
            )[:20],
        }

    # ------------------------------------------------------------------
    # Internal — observation collection
    # ------------------------------------------------------------------

    def _collect_observations(self) -> dict[str, list[dict]]:
        """
        Read all extracted JSON files and group observations by rider name.
        An observation is any tactical_pattern or rider_signal that
        mentions a specific rider.
        """
        by_rider: dict[str, list[dict]] = defaultdict(list)

        for path in sorted(self._extracted_dir.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            extraction = data.get("extraction", {})
            if not extraction or not extraction.get("usable_for_rag", True):
                continue

            label     = data.get("label", path.stem)
            race_name = data.get("race_name", "")
            race_date = data.get("race_date", "")

            # Collect rider signals — these are the richest per-rider observations
            for rider, signal in extraction.get("rider_signals", {}).items():
                rider_norm = self._normalize_rider_name(rider)
                by_rider[rider_norm].append({
                    "label":      label,
                    "race_name":  race_name,
                    "race_date":  race_date,
                    "type":       "rider_signal",
                    "text":       f"{signal}",
                    "source":     path.stem,
                })

            # Collect tactical patterns that mention rider names
            winner = extraction.get("winner", "")
            if winner:
                rider_norm = self._normalize_rider_name(winner)
                by_rider[rider_norm].append({
                    "label":     label,
                    "race_name": race_name,
                    "race_date": race_date,
                    "type":      "winner",
                    "text":      extraction.get("one_line_summary", f"Won {label}"),
                    "source":    path.stem,
                })

            for pattern in extraction.get("tactical_patterns", []):
                # Only add pattern to riders explicitly named in it
                named = self._extract_rider_names(pattern)
                for rider in named:
                    by_rider[rider].append({
                        "label":     label,
                        "race_name": race_name,
                        "race_date": race_date,
                        "type":      "tactical_pattern",
                        "text":      pattern,
                        "source":    path.stem,
                    })

        return dict(by_rider)

    # ------------------------------------------------------------------
    # Internal — profile building
    # ------------------------------------------------------------------

    def _build_profile(
        self,
        rider_name: str,
        observations: list[dict],
    ) -> Optional[dict]:
        """Call Claude to synthesize a tactical profile from observations."""
        # Sort observations by date (most recent first)
        obs_sorted = sorted(
            observations,
            key=lambda x: x.get("race_date", ""),
            reverse=True,
        )

        # Group by race for context
        races = list({o["label"] for o in obs_sorted})
        years = sorted({o["race_date"][:4] for o in obs_sorted if o.get("race_date")})
        race_types = self._summarize_race_types([o["race_name"] for o in obs_sorted])

        # Format observations for the prompt
        obs_lines = []
        for obs in obs_sorted:
            obs_lines.append(
                f"[{obs['race_date'][:10] if obs.get('race_date') else 'unknown'} | "
                f"{obs['race_name']} | {obs['type']}] {obs['text']}"
            )

        observations_text = "\n".join(obs_lines)

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": PROFILE_PROMPT.format(
                        rider_name=rider_name,
                        n_races=len(races),
                        years=", ".join(years) if years else "unknown",
                        race_types=race_types,
                        observations=observations_text,
                    ),
                }],
            )
            raw_json = re.sub(r"```json|```", "", response.content[0].text).strip()
            return json.loads(raw_json)
        except json.JSONDecodeError as e:
            log.error("Profile JSON parse failed for %s: %s", rider_name, e)
            return None
        except Exception as e:
            log.error("Profile build API call failed for %s: %s", rider_name, e)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_name(name: str) -> str:
        """
        Convert rider name to filesystem-safe string.
        Strips accents so POGAČAR and POGACAR map to the same file.
        """
        import unicodedata
        # Strip accents first
        name = "".join(
            c for c in unicodedata.normalize("NFD", name)
            if unicodedata.category(c) != "Mn"
        )
        return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    @staticmethod
    def _normalize_rider_name(name: str) -> str:
        """
        Normalize rider name for grouping across extractions.
        Strips accents so POGAČAR Tadej and POGACAR Tadej are treated
        as the same rider.
        """
        import unicodedata
        name = name.strip()
        return "".join(
            c for c in unicodedata.normalize("NFD", name)
            if unicodedata.category(c) != "Mn"
        )

    @staticmethod
    def _extract_rider_names(text: str) -> list[str]:
        """
        Extract rider names from a tactical pattern text.
        Looks for ALL-CAPS SURNAME Firstname pattern.
        """
        pattern = re.compile(r'\b([A-ZÁÉÍÓÚÀÈÙÂÊÎÔÛÄËÏÖÜÑ]{2,}(?:\s[A-ZÁÉÍÓÚÀÈÙÂÊÎÔÛÄËÏÖÜÑ]{2,})*)\s+([A-Z][a-záéíóúàèùâêîôûäëïöüñ]+(?:\s[A-Z][a-záéíóúàèùâêîôûäëïöüñ]+)*)\b')
        matches = pattern.findall(text)
        return [f"{last} {first}" for last, first in matches]

    @staticmethod
    def _summarize_race_types(race_names: list[str]) -> str:
        """Summarize the types of races in the observation set."""
        grand_tours = {"tour de france", "giro", "vuelta"}
        monuments   = {"paris-roubaix", "roubaix", "lombardia", "strade bianche",
                       "milan", "san remo", "liege", "amstel", "fleche"}
        classics    = {"classics", "ronde", "vlaanderen", "gent"}

        names_lower = [r.lower() for r in race_names]
        types = []
        if any(gt in n for n in names_lower for gt in grand_tours):
            types.append("Grand Tours")
        if any(mon in n for n in names_lower for mon in monuments):
            types.append("Monuments")
        if any(cl in n for n in names_lower for cl in classics):
            types.append("Classics")

        return ", ".join(types) if types else "WorldTour races"