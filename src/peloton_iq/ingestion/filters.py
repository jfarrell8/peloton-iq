"""
peloton_iq.ingestion.filters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
UCI WorldTour race filter.

The UCIFilter class encapsulates all race-name normalization, alias
resolution, and fuzzy matching logic from notebook 01. It is the single
source of truth for deciding whether a row belongs in the dataset.

Usage:
    from peloton_iq.ingestion.filters import UCIFilter

    f = UCIFilter()
    is_uci, reason = f.is_uci_race(year=2023, race_name="Giro d'Italia")
"""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz, process

from peloton_iq.config import settings
from peloton_iq.schemas import UCIFilterResult


# ---------------------------------------------------------------------------
# UCI WorldTour race calendars by year
# ---------------------------------------------------------------------------

_UCI_WORLD_TOUR_RACES: dict[int, set[str]] = {
    2017: {
        "Tour Down Under", "Great Ocean Road Race", "Abu Dhabi Tour",
        "Omloop Het Nieuwsblad", "Strade Bianche", "Paris-Nice",
        "Tirreno-Adriatico", "Milan-San Remo", "Volta a Catalunya",
        "Dwars door Vlaanderen", "E3 Harelbeke", "Gent-Wevelgem",
        "Ronde Van Vlaanderen", "Tour of the Basque Country", "Paris-Roubaix",
        "Amstel Gold Race", "La Fleche Wallonne", "Liege-Bastogne-Liege",
        "Tour de Romandie", "Eschborn-Frankfurt",
        "Giro d'Italia", "Tour of California", "Criterium du Dauphine",
        "Tour de Suisse", "Tour de France", "Clasica de San Sebastian",
        "Tour de Pologne", "RideLondon-Surrey Classic", "BinckBank Tour",
        "Vuelta a Espana", "EuroEyes Cyclassics", "Bretagne Classic Ouest-France",
        "GP de Quebec", "GP de Montreal", "Il Lombardia",
        "Presidential Tour of Turkey", "Tour of Guangxi", "Tour of Oman",
    },
    2018: {
        "Tour Down Under", "Great Ocean Road Race", "Abu Dhabi Tour",
        "Omloop Het Nieuwsblad", "Strade Bianche", "Paris-Nice",
        "Tirreno-Adriatico", "Milan-San Remo", "Volta a Catalunya",
        "E3 Harelbeke", "Gent-Wevelgem", "Dwars door Vlaanderen",
        "Ronde Van Vlaanderen", "Tour of the Basque Country", "Paris-Roubaix",
        "Amstel Gold Race", "La Fleche Wallonne", "Liege-Bastogne-Liege",
        "Tour de Romandie", "Eschborn-Frankfurt", "Giro d'Italia",
        "Tour of California", "Criterium du Dauphine", "Tour de Suisse",
        "Tour de France", "Clasica de San Sebastian", "Tour de Pologne",
        "RideLondon-Surrey Classic", "BinckBank Tour", "Vuelta a Espana",
        "EuroEyes Cyclassics", "Bretagne Classic Ouest-France",
        "GP de Quebec", "GP de Montreal", "Presidential Tour of Turkey",
        "Il Lombardia", "Tour of Guangxi", "Tour of Oman",
    },
    2019: {
        "Tour Down Under", "Great Ocean Road Race", "UAE Tour",
        "Omloop Het Nieuwsblad", "Strade Bianche", "Paris-Nice",
        "Tirreno-Adriatico", "Milan-San Remo", "Volta a Catalunya",
        "Classic Brugge-De Panne", "E3 BinckBank Classic",
        "Gent-Wevelgem", "Dwars door Vlaanderen", "Ronde Van Vlaanderen",
        "Tour of the Basque Country", "Paris-Roubaix",
        "Presidential Tour of Turkey", "Amstel Gold Race",
        "La Fleche Wallonne", "Liege-Bastogne-Liege", "Tour de Romandie",
        "Eschborn-Frankfurt", "Giro d'Italia", "Tour of California",
        "Criterium du Dauphine", "Tour de Suisse", "Tour de France",
        "Clasica de San Sebastian", "Tour de Pologne",
        "RideLondon-Surrey Classic", "BinckBank Tour", "Vuelta a Espana",
        "EuroEyes Cyclassics", "Bretagne Classic Ouest-France",
        "GP de Quebec", "GP de Montreal", "Il Lombardia", "Tour of Guangxi",
        "Tour of Oman",
    },
    2020: {
        "Tour Down Under", "Great Ocean Road Race", "UAE Tour",
        "Omloop Het Nieuwsblad", "Paris-Nice", "Strade Bianche",
        "Tour de Pologne", "Milan-San Remo", "Criterium du Dauphine",
        "Il Lombardia", "Bretagne Classic Ouest-France", "Tour de France",
        "Tirreno-Adriatico", "BinckBank Tour", "La Fleche Wallonne",
        "Giro d'Italia", "Liege-Bastogne-Liege", "Gent-Wevelgem",
        "Ronde van Vlaanderen", "Vuelta a Espana",
        "Classic Brugge-De Panne", "Volta a Catalunya",
    },
    2021: {
        "UAE Tour", "Omloop Het Nieuwsblad", "Strade Bianche", "Paris-Nice",
        "Tirreno-Adriatico", "Milan-San Remo", "Classic Brugge-De Panne",
        "E3 Saxo Bank Classic", "Gent-Wevelgem", "Dwars door Vlaanderen",
        "Ronde van Vlaanderen", "Itzulia Basque Country",
        "Paris-Roubaix", "Amstel Gold Race", "La Fleche Wallonne",
        "Liege-Bastogne-Liege", "Tour de Romandie", "Eschborn-Frankfurt",
        "Giro d'Italia", "Criterium du Dauphine", "Tour de Suisse",
        "Tour de France", "Clasica de San Sebastian", "Benelux Tour",
        "Vuelta a Espana", "Bretagne Classic Ouest-France",
        "GP de Quebec", "GP de Montreal", "Il Lombardia", "Tour of Guangxi",
        "Tour de Pologne", "Volta a Catalunya",
    },
    2022: {
        "UAE Tour", "Omloop Het Nieuwsblad", "Strade Bianche", "Paris-Nice",
        "Tirreno-Adriatico", "Milan-San Remo", "Classic Brugge-De Panne",
        "E3 Saxo Bank Classic", "Gent-Wevelgem", "Dwars door Vlaanderen",
        "Ronde van Vlaanderen", "Itzulia Basque Country",
        "Paris-Roubaix", "Amstel Gold Race", "La Fleche Wallonne",
        "Liege-Bastogne-Liege", "Tour de Romandie", "Eschborn-Frankfurt",
        "Giro d'Italia", "Criterium du Dauphine", "Tour de Suisse",
        "Tour de France", "Clasica de San Sebastian", "Vuelta a Espana",
        "Bretagne Classic Ouest-France", "GP de Quebec", "GP de Montreal",
        "Il Lombardia", "Tour de Pologne", "Volta a Catalunya",
    },
    2023: {
        "UAE Tour", "Omloop Het Nieuwsblad", "Strade Bianche", "Paris-Nice",
        "Tirreno-Adriatico", "Milan-San Remo", "Classic Brugge-De Panne",
        "E3 Saxo Bank Classic", "Gent-Wevelgem", "Dwars door Vlaanderen",
        "Ronde van Vlaanderen", "Itzulia Basque Country",
        "Paris-Roubaix", "Amstel Gold Race", "La Fleche Wallonne",
        "Liege-Bastogne-Liege", "Tour de Romandie", "Eschborn-Frankfurt",
        "Giro d'Italia", "Criterium du Dauphine", "Tour de Suisse",
        "Tour de France", "Clasica de San Sebastian", "Vuelta a Espana",
        "Bretagne Classic Ouest-France", "GP de Quebec", "GP de Montreal",
        "Il Lombardia", "Volta a Catalunya",
    },
}

# ---------------------------------------------------------------------------
# Name aliases  (all keys/values in plain ASCII after normalization)
# ---------------------------------------------------------------------------

_NAME_ALIASES: dict[str, str] = {
    "abu dhabi tour":                               "uae tour",
    "three days of bruges de panne":                "classic brugge de panne",
    "bruges de panne":                              "classic brugge de panne",
    "e3 harelbeke":                                 "e3 saxo bank classic",
    "e3 binckbank classic":                         "e3 saxo bank classic",
    "eschborn frankfurt rund um den finanzplatz":   "eschborn frankfurt",
    "euroeyes cyclassics":                          "hamburg cyclassics",
    "ronde van vlaanderen tour des flandres":       "ronde van vlaanderen",
    "la vuelta ciclista a espana":                  "vuelta a espana",
    "la vuelta a espana":                           "vuelta a espana",
    "vuelta espana":                                "vuelta a espana",
    "presidential cycling tour of turkey":          "presidential tour of turkey",
    "volta ciclista a catalunya":                   "volta a catalunya",
    "giro ditalia":                                 "giro d italia",
    "itzulia basque country":                       "tour of the basque country",
}

# Races that must never be fuzzy-matched (would produce false positives)
_FUZZY_BLOCKLIST: frozenset[str] = frozenset({
    "tour de normandie",
    "tour de wallonie",
    "tour du limousin",
    "tour du poitou charentes",
    "tour de bretagne",
    "tour de vendee",
    "tour alsace",
})

# Races to exclude entirely (women's, juniors, etc.)
_EXCLUSION_PATTERN = (
    r"Femme|Women|Woman|Ladies|Feminas|Feminin|Femminile|"
    r"Femenina|Femines|Femenino|Famenne|vroumen|Junior|Espoir"
)


# ---------------------------------------------------------------------------
# UCIFilter
# ---------------------------------------------------------------------------

class UCIFilter:
    """
    Determines whether a race belongs in the UCI WorldTour dataset.

    Applies, in order:
      1. Exclusion pattern (women's / junior races → always drop)
      2. Exact match against the year's UCI calendar
      3. Alias resolution then exact match
      4. Fuzzy match (RapidFuzz token_sort_ratio) with blocklist guard

    All normalization is handled internally — callers pass raw race names.
    """

    def __init__(self, fuzzy_threshold: int | None = None) -> None:
        self._threshold = fuzzy_threshold or settings.fuzzy_threshold

        # Pre-normalize the UCI lookup sets and alias table once at init
        self._normalized_uci: dict[int, set[str]] = {
            year: {self._normalize(r) for r in races}
            for year, races in _UCI_WORLD_TOUR_RACES.items()
        }
        self._normalized_aliases: dict[str, str] = {
            self._normalize(k): self._normalize(v)
            for k, v in _NAME_ALIASES.items()
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_uci_race(self, year: int, race_name: str) -> UCIFilterResult:
        """
        Return a UCIFilterResult indicating whether the race is a
        UCI WorldTour event for the given year.
        """
        if not race_name or not year:
            return UCIFilterResult(is_uci=False, match_reason="null")

        candidates = self._normalized_uci.get(int(year))
        if not candidates:
            return UCIFilterResult(is_uci=False, match_reason="year_not_in_lookup")

        norm = self._normalize(race_name)
        norm = self._normalized_aliases.get(norm, norm)

        # 1. Exact match
        if norm in candidates:
            return UCIFilterResult(is_uci=True, match_reason="exact")

        # 2. Blocklist guard before fuzzy
        if any(blocked in norm for blocked in _FUZZY_BLOCKLIST):
            return UCIFilterResult(is_uci=False, match_reason="blocklisted")

        # 3. Fuzzy match
        match, score, _ = process.extractOne(
            norm, candidates, scorer=fuzz.token_sort_ratio
        )
        if score >= self._threshold:
            return UCIFilterResult(
                is_uci=True,
                match_reason=f"fuzzy:{score:.0f}:{match}",
            )

        return UCIFilterResult(
            is_uci=False,
            match_reason=f"no_match:{score:.0f}",
        )

    def is_excluded(self, race_name: str) -> bool:
        """Return True if the race name matches the women's/junior exclusion pattern."""
        if not race_name:
            return False
        return bool(re.search(_EXCLUSION_PATTERN, race_name, re.IGNORECASE))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(name: str) -> str:
        """
        Lowercase, strip accents, normalize apostrophes,
        remove punctuation, collapse whitespace.
        Identical to normalize_name() in notebook 01.
        """
        name = name.lower()
        name = re.sub(r"[''`´']", " ", name)
        name = "".join(
            c for c in unicodedata.normalize("NFD", name)
            if unicodedata.category(c) != "Mn"
        )
        name = re.sub(r"[^a-z0-9]+", " ", name)
        return re.sub(r"\s+", " ", name).strip()