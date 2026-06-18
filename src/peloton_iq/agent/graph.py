"""
peloton_iq.agent.graph
~~~~~~~~~~~~~~~~~~~~~~~
LangGraph graph assembly and the public ask() interface.

The PelotonIQAgent class owns all dependencies and exposes a single
ask() method. Dependencies are initialized lazily on first query so
import is cheap.

Usage:
    from peloton_iq.agent.graph import PelotonIQAgent

    agent = PelotonIQAgent()
    response = agent.ask("Who won Tour de France Stage 17 in 2023?")
    print(response)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import anthropic
import pandas as pd
from langgraph.graph import END, StateGraph

from peloton_iq.agent.nodes import (
    AgentDeps,
    make_nodes,
    route_from_commentary,
    route_from_predictor,
    route_from_retriever,
    route_from_router,
    route_from_structured,
)
from peloton_iq.commentary.extractor import ClaudeExtractor
from peloton_iq.commentary.profiler import RiderProfiler
from peloton_iq.config import (
    COURSE_CLEAN_PATH,
    MERGED_RACES_PATH,
    settings,
)
from peloton_iq.prediction.predictor import TierPredictor
from peloton_iq.schemas import PelotonState, empty_state
from peloton_iq.search.embeddings import EmbeddingStore
from peloton_iq.search.hybrid import HybridSearcher

log = logging.getLogger(__name__)


class PelotonIQAgent:
    """
    The PelotonIQ agent — a LangGraph multi-node pipeline that combines
    structured data lookup, hybrid search, ML predictions, and race
    commentary into a unified pre-race intelligence platform.

    All dependencies are initialized lazily on first query.
    """

    def __init__(self) -> None:
        self._app     = None
        self._deps    = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Load all dependencies and compile the LangGraph.
        Called automatically on first ask() — can also be called
        explicitly to front-load startup time.
        """
        if self._initialized:
            return

        log.info("Initializing PelotonIQ agent...")
        t0 = time.time()

        # Sync artifacts from S3 if configured
        from peloton_iq.artifacts import ensure_artifacts
        ensure_artifacts()

        # Data
        log.info("  Loading DataFrames...")
        # Load only columns needed by agent tools with efficient dtypes
        # saves ~100MB vs full load
        MERGED_COLS = [
            "Race Name", "Race_results", "Year_results", "Stage_results",
            "Date", "Name", "Team", "Rank", "Did_Finish", "Top3", "Top10",
        ]
        MERGED_DTYPES = {
            "Race Name":     "category",
            "Race_results":  "category",
            "Year_results":  "int16",
            "Name":          "category",
            "Team":          "category",
            "Rank":          "int16",
            "Did_Finish":    "bool",
            "Top3":          "int8",
            "Top10":         "int8",
        }
        merged_df = pd.read_csv(
            MERGED_RACES_PATH,
            usecols=MERGED_COLS,
            dtype=MERGED_DTYPES,
            low_memory=True,
        )
        merged_df["Date"] = pd.to_datetime(merged_df["Date"])
        course_df = pd.read_csv(COURSE_CLEAN_PATH)

        # Search
        log.info("  Building BM25 indexes...")
        store    = EmbeddingStore()
        searcher = HybridSearcher(store)
        searcher.build_indexes(course_df, merged_df)

        # Prediction
        log.info("  Loading tier predictor...")
        predictor = TierPredictor()
        _ = predictor.artifact  # trigger lazy load now

        # Commentary
        extractor = ClaudeExtractor()

        # Tactical profiler
        log.info("  Loading rider profiles...")
        profiler = RiderProfiler()
        n_profiles = len(profiler.list_profiled_riders())
        log.info("  %d rider profiles loaded", n_profiles)

        # Anthropic client
        client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key or None,
        )

        # Bundle deps
        self._deps = AgentDeps(
            merged_df=merged_df,
            course_df=course_df,
            searcher=searcher,
            predictor=predictor,
            extractor=extractor,
            profiler=profiler,
            client=client,
        )

        # Build graph
        log.info("  Compiling LangGraph...")
        self._app = self._build_graph(self._deps)

        self._initialized = True
        log.info("Agent ready in %.1fs", time.time() - t0)

    def _build_graph(self, deps: AgentDeps):
        """Assemble and compile the LangGraph."""
        nodes = make_nodes(deps)
        graph = StateGraph(PelotonState)

        # Add nodes
        for name, fn in nodes.items():
            graph.add_node(name, fn)

        # Entry point
        graph.set_entry_point("router")

        # Conditional edges
        graph.add_conditional_edges(
            "router",
            route_from_router,
            {"structured_tool": "structured_tool", "retriever": "retriever"},
        )
        graph.add_conditional_edges(
            "retriever",
            route_from_retriever,
            {"predictor": "predictor", "commentary": "commentary"},
        )
        graph.add_conditional_edges(
            "structured_tool",
            route_from_structured,
            {"synthesizer": "synthesizer"},
        )
        graph.add_conditional_edges(
            "predictor",
            route_from_predictor,
            {"commentary": "commentary"},
        )
        graph.add_conditional_edges(
            "commentary",
            route_from_commentary,
            {"synthesizer": "synthesizer"},
        )

        # Terminal edge
        graph.add_edge("synthesizer", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        verbose: bool = True,
    ) -> str:
        """
        Run a query through the full PelotonIQ agent pipeline.

        Args:
            query:   Natural language question about cycling races.
            verbose: Print steps, query type, and response to stdout.

        Returns:
            The final response string.
        """
        if not self._initialized:
            self.initialize()

        if verbose:
            print(f"\n{'='*65}")
            print(f"Query: {query}")
            print("=" * 65)

        t0     = time.time()
        state  = empty_state(query)
        result = self._app.invoke(state)

        if verbose:
            print(f"\nSteps : {' → '.join(result['steps_taken'])}")
            print(f"Type  : {result['query_type']}")
            if result.get("error"):
                print(f"Error : {result['error']}")
            print(f"Time  : {time.time() - t0:.1f}s")
            print(f"\n{'-'*65}")
            print(result["final_response"])

        return result["final_response"]

    def sanity_check(self) -> dict:
        """
        Run a series of checks to verify all components are working.
        Returns a dict of {check_name: passed} without making any
        Claude API calls.
        """
        if not self._initialized:
            self.initialize()

        results = {}

        # DataFrames
        results["merged_df_loaded"] = len(self._deps.merged_df) > 0
        results["course_df_loaded"] = len(self._deps.course_df) > 0

        # BM25
        results["bm25_course_built"] = self._deps.searcher.course_doc_count > 0
        results["bm25_rider_built"]  = self._deps.searcher.rider_doc_count > 0

        # Qdrant
        store = self._deps.searcher._store
        results["qdrant_courses"] = store.collection_exists(settings.qdrant_collection_courses)
        results["qdrant_riders"]  = store.collection_exists(settings.qdrant_collection_riders)

        # Predictor
        results["predictor_loaded"] = self._deps.predictor._artifact is not None

        # Tool smoke test
        from peloton_iq.agent.tools import get_stage_winner
        winner = get_stage_winner(self._deps.merged_df, "Tour de France", 2023, stage=17)
        results["tool_stage_winner"] = "error" not in winner

        # Hybrid search smoke test
        course_results = self._deps.searcher.search_courses("mountain stage climbing", top_k=1)
        results["hybrid_search_course"] = len(course_results) > 0

        # Prediction smoke test
        ctx = self._deps.predictor.predict_race_context("Tour de France", 2023, stage=17)
        results["predictor_inference"] = "NO PREDICTION" not in ctx and "ERROR" not in ctx

        return results