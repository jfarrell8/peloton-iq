"""
peloton_iq.agent.nodes
~~~~~~~~~~~~~~~~~~~~~~~
LangGraph node functions and conditional routing for the PelotonIQ agent.

Each node receives a PelotonState dict and returns an updated PelotonState.
Dependencies (DataFrames, searcher, predictor, extractor) are injected via
the AgentDeps dataclass rather than being module-level globals — this makes
each node independently testable.

Node graph:
    router → structured_tool → synthesizer
           → retriever → predictor → commentary → synthesizer
                       → commentary → synthesizer
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

import anthropic
import pandas as pd
from qdrant_client.models import FieldCondition, Filter, Range

from peloton_iq.agent.tools import dispatch_tool
from peloton_iq.commentary.extractor import ClaudeExtractor
from peloton_iq.config import settings
from peloton_iq.prediction.predictor import TierPredictor
from peloton_iq.schemas import PelotonState
from peloton_iq.search.hybrid import HybridSearcher

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent dependencies  (injected at startup, shared across all nodes)
# ---------------------------------------------------------------------------

@dataclass
class AgentDeps:
    """
    All runtime dependencies for the agent.

    Instantiated once at startup and passed through to each node via
    closure. Keeps nodes free of module-level state so they're testable.
    """
    merged_df:  pd.DataFrame
    course_df:  pd.DataFrame
    searcher:   HybridSearcher
    predictor:  TierPredictor
    extractor:  ClaudeExtractor
    client:     anthropic.Anthropic


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ROUTER_SYSTEM = """You are a query router for PelotonIQ, a professional cycling intelligence system.

Classify the query into exactly one type:

STRUCTURED — asks for a specific fact retrievable directly from a database.
  Examples: who won X race, what place did rider finish, top 10 of X race,
  which riders perform best on mountain stages, best climbers historically.
  NOTE: questions asking which riders perform best on specific terrain types
  should use get_best_mountain_riders, not semantic search.

  Functions available:
  - get_stage_winner
  - get_race_results
  - get_rider_results
  - get_stage_profile
  - get_best_mountain_riders  ← use for "best riders on terrain type X" queries

SEMANTIC_COURSE — asks about race/stage terrain, course profile, surface type, elevation.
  Examples: what makes Paris-Roubaix hard, hardest mountain stages, flat sprint stages

SEMANTIC_RIDER — asks about a rider's general performance, season, strengths.
  Examples: how did Pogacar perform in 2023, best climbers, Evenepoel season summary

PREDICTIVE — forward-looking analysis requiring ML model predictions.
  Examples: who should I watch on stage 17, pre-race analysis, stage preview

HYBRID — requires both terrain context AND rider performance to answer.
  Examples: who performs best on cobbled stages, climbers suited to this terrain

Return JSON only:
{
  "query_type": "STRUCTURED|SEMANTIC_COURSE|SEMANTIC_RIDER|PREDICTIVE|HYBRID",
  "reasoning": "one sentence",
  "structured_params": {
    "function": "get_stage_winner|get_race_results|get_rider_results|get_stage_profile|get_best_mountain_riders",
    "race_name": "...",
    "year": 2023,
    "stage": null,
    "rider_name": "...",
    "top_n": 10
  },
  "race_context": {
    "race_name": "...",
    "year": 2023,
    "stage": null,
    "race_date": "YYYY-MM-DD or null"
  }
}
Only include structured_params if query_type is STRUCTURED.
Always include race_context if a specific race is mentioned."""


SYNTHESIZER_SYSTEM = """You are PelotonIQ, a professional cycling intelligence assistant.

You have access to a dataset covering UCI WorldTour races from 2017-2023,
including detailed course profiles, race results, rider performance history,
ML-based finish probability predictions, and race commentary.

Answer using only the provided context. Be specific and cite the data.
For predictions, explain which factors drove the model's output.
If context is missing, say so clearly rather than guessing.
Keep responses concise and analytically focused."""


# ---------------------------------------------------------------------------
# Node factory  (returns node functions bound to deps)
# ---------------------------------------------------------------------------

def _lookup_race_date(
    merged_df,
    race_name: str,
    year: int,
    stage=None,
) -> str | None:
    """
    Look up the actual race date from merged_df given race name, year, and
    optional stage number. Returns ISO date string "YYYY-MM-DD" or None.

    Used by commentary_node when the router couldn't extract a race_date
    from the query (e.g. forward-looking queries like "before stage 17").
    """
    mask = (
        merged_df["Race_results"].str.contains(race_name, na=False, case=False) &
        (merged_df["Year_results"] == year)
    )
    if stage is not None:
        mask &= (
            (merged_df["Stage_results"] == stage) |
            (merged_df["Stage_results"] == float(stage))
        )

    rows = merged_df[mask]
    if rows.empty:
        return None

    date_val = rows["Date"].iloc[0]
    try:
        return str(date_val)[:10]  # "YYYY-MM-DD"
    except Exception:
        return None


def make_nodes(deps: AgentDeps):
    """
    Build all node functions bound to the injected AgentDeps.
    Returns a dict of {node_name: node_fn} for use in graph assembly.
    """

    # ------------------------------------------------------------------
    # router_node
    # ------------------------------------------------------------------
    def router_node(state: PelotonState) -> PelotonState:
        """Classify query type and extract structured parameters."""
        state["steps_taken"].append("router")

        response = deps.client.messages.create(
            model=settings.claude_model,
            max_tokens=800,
            system=ROUTER_SYSTEM,
            messages=[{"role": "user", "content": state["query"]}],
        )

        raw = re.sub(r"```json|```", "", response.content[0].text).strip()
        try:
            parsed                    = json.loads(raw)
            state["query_type"]       = parsed.get("query_type", "HYBRID")
            state["structured_params"] = parsed.get("structured_params", {})
            state["structured_params"]["race_context"] = parsed.get("race_context", {})
        except json.JSONDecodeError:
            state["query_type"] = "HYBRID"
            state["error"]      = f"Router parse error: {raw[:100]}"

        log.info("Router: %s → %s", state["query"][:50], state["query_type"])
        return state

    # ------------------------------------------------------------------
    # structured_node
    # ------------------------------------------------------------------
    def structured_node(state: PelotonState) -> PelotonState:
        """Execute dataframe tool call for structured queries."""
        state["steps_taken"].append("structured_tool")

        params = state.get("structured_params", {})
        result = dispatch_tool(params, deps.merged_df, deps.course_df)
        state["structured_data"] = result

        if "error" in result:
            log.warning("Tool error: %s", result["error"])
        else:
            log.info("Tool: %s → OK", params.get("function"))

        return state

    # ------------------------------------------------------------------
    # retriever_node
    # ------------------------------------------------------------------
    def retriever_node(state: PelotonState) -> PelotonState:
        """Run hybrid search over course and/or rider collections."""
        state["steps_taken"].append("retriever")
        qt          = state["query_type"]
        query_lower = state["query"].lower()

        # Elevation filter keywords — numeric filter is more precise
        # than semantic search for these queries
        elevation_keywords = [
            "2000m", "above 2000", "high altitude", "highest elevation",
            "most climbing", "most vertical", "hardest climb",
            "steepest", "highest point",
        ]
        use_elevation_filter = any(kw in query_lower for kw in elevation_keywords)

        if qt in ["SEMANTIC_COURSE", "PREDICTIVE", "HYBRID"]:
            if use_elevation_filter:
                try:
                    filtered, _ = deps.searcher._store.client.scroll(
                        collection_name=settings.qdrant_collection_courses,
                        scroll_filter=Filter(
                            must=[FieldCondition(
                                key="vertical_gain",
                                range=Range(gte=4000),
                            )]
                        ),
                        limit=5,
                        with_payload=True,
                    )
                    if filtered:
                        state["course_context"] = "\n".join(
                            f"[{r.payload.get('doc_id')}]\n{r.payload.get('text', '')[:400]}"
                            for r in filtered
                        )
                    else:
                        raise ValueError("No elevation-filtered results")
                except Exception:
                    results = deps.searcher.search_courses(state["query"], top_k=3)
                    state["course_context"] = "\n".join(
                        f"[{r['id']}]\n{r['text'][:400]}" for r in results
                    )
            else:
                results = deps.searcher.search_courses(state["query"], top_k=3)
                state["course_context"] = "\n".join(
                    f"[{r['id']}]\n{r['text'][:400]}" for r in results
                )
            log.info("Retriever: course_context set (%d chars)", len(state["course_context"]))

        if qt in ["SEMANTIC_RIDER", "PREDICTIVE", "HYBRID"]:
            results = deps.searcher.search_riders(state["query"], top_k=3)
            state["rider_context"] = "\n".join(
                f"[{r['id']}]\n{r['text'][:400]}" for r in results
            )
            log.info("Retriever: rider_context set (%d chars)", len(state["rider_context"]))

        return state

    # ------------------------------------------------------------------
    # predictor_node
    # ------------------------------------------------------------------
    def predictor_node(state: PelotonState) -> PelotonState:
        """Run ML model prediction for predictive/hybrid queries."""
        state["steps_taken"].append("predictor")

        structured_params = state.get("structured_params") or {}
        race_ctx  = structured_params.get("race_context") or {}
        race_name = race_ctx.get("race_name")
        year      = race_ctx.get("year")
        stage     = race_ctx.get("stage")

        if race_name and year:
            state["prediction_context"] = deps.predictor.predict_race_context(
                race_name, int(year), stage
            )
            log.info(
                "Predictor: %s %s%s → %d chars",
                race_name, year,
                f" Stage {stage}" if stage else "",
                len(state["prediction_context"]),
            )
        else:
            state["prediction_context"] = (
                "[NO PREDICTION] Query does not reference a specific race. "
                "Predictions require a named race and year."
            )

        return state

    # ------------------------------------------------------------------
    # commentary_node
    # ------------------------------------------------------------------
    def commentary_node(state: PelotonState) -> PelotonState:
        """Add race commentary context if available."""
        state["steps_taken"].append("commentary")

        structured_params = state.get("structured_params") or {}
        race_ctx  = structured_params.get("race_context") or {}
        race_name = race_ctx.get("race_name")
        race_date = race_ctx.get("race_date")
        stage     = race_ctx.get("stage")
        year      = race_ctx.get("year")

        # If race_date is missing but race_name + year + stage are known,
        # look it up from merged_df — handles forward-looking queries like
        # "before stage 17 of TDF 2023" where no explicit date is given
        if race_name and not race_date and year:
            race_date = _lookup_race_date(
                deps.merged_df, race_name, int(year), stage
            )
            if race_date:
                log.info("Commentary: resolved race_date=%s from merged_df", race_date)

        if race_name and race_date:
            state["commentary_context"] = deps.extractor.get_context(
                race_name, race_date, stage
            )
        else:
            state["commentary_context"] = (
                "[NO COMMENTARY] Could not identify race date for this query."
            )

        log.info(
            "Commentary: %s (%d chars)",
            state["commentary_context"][:60],
            len(state["commentary_context"]),
        )
        return state

    # ------------------------------------------------------------------
    # synthesizer_node
    # ------------------------------------------------------------------
    def synthesizer_node(state: PelotonState) -> PelotonState:
        """Generate final response using all available context."""
        state["steps_taken"].append("synthesizer")

        context = _build_context_prompt(state)
        response = deps.client.messages.create(
            model=settings.claude_model,
            max_tokens=settings.claude_max_tokens,
            system=SYNTHESIZER_SYSTEM,
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {state['query']}",
            }],
        )
        state["final_response"] = response.content[0].text
        log.info("Synthesizer: %d chars", len(state["final_response"]))
        return state

    return {
        "router":          router_node,
        "structured_tool": structured_node,
        "retriever":       retriever_node,
        "predictor":       predictor_node,
        "commentary":      commentary_node,
        "synthesizer":     synthesizer_node,
    }


# ---------------------------------------------------------------------------
# Conditional routing functions
# ---------------------------------------------------------------------------

def route_from_router(state: PelotonState) -> str:
    if state["query_type"] == "STRUCTURED":
        return "structured_tool"
    return "retriever"


def route_from_retriever(state: PelotonState) -> str:
    if state["query_type"] in ["PREDICTIVE", "HYBRID"]:
        return "predictor"
    return "commentary"


def route_from_structured(state: PelotonState) -> str:
    return "synthesizer"


def route_from_predictor(state: PelotonState) -> str:
    return "commentary"


def route_from_commentary(state: PelotonState) -> str:
    return "synthesizer"


# ---------------------------------------------------------------------------
# Context assembly helper
# ---------------------------------------------------------------------------

def _build_context_prompt(state: PelotonState) -> str:
    parts = []
    if state.get("structured_data"):
        parts.append("STRUCTURED DATA LOOKUP:")
        parts.append(json.dumps(state["structured_data"], indent=2, default=str))
    if state.get("course_context"):
        parts.append("\nCOURSE PROFILE CONTEXT:")
        parts.append(state["course_context"])
    if state.get("rider_context"):
        parts.append("\nRIDER PERFORMANCE CONTEXT:")
        parts.append(state["rider_context"])
    if state.get("prediction_context"):
        parts.append("\nML MODEL PREDICTIONS:")
        parts.append(state["prediction_context"])
    if state.get("commentary_context"):
        parts.append("\nRACE COMMENTARY:")
        parts.append(state["commentary_context"])
    return "\n".join(parts)