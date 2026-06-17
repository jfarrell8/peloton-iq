"""
peloton_iq.api.app
~~~~~~~~~~~~~~~~~~~
FastAPI application for the PelotonIQ agent.

The agent is initialized once at startup and shared across all requests.
The Dash app calls this API via HTTP — the agent lives here, not in Dash.

Endpoints:
    POST /api/query   — run a query through the agent
    GET  /api/health  — health check with component status
    GET  /docs        — auto-generated OpenAPI docs

Run:
    python scripts/run_api.py
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent singleton — initialized once at startup
# ---------------------------------------------------------------------------

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        from peloton_iq.agent.graph import PelotonIQAgent
        _agent = PelotonIQAgent()
        _agent.initialize()
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting PelotonIQ API — initializing agent...")
    get_agent()
    log.info("API ready.")
    yield
    log.info("Shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PelotonIQ",
    description="UCI WorldTour race intelligence powered by ML + RAG",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query:            str
    response:         str
    query_type:       str
    steps:            list[str]
    elapsed_s:        float
    prediction_text:  Optional[str] = None
    race_context:     Optional[dict] = None
    error:            Optional[str]  = None


class HealthResponse(BaseModel):
    status:    str
    checks:    dict
    model:     str
    qdrant_ok: bool


class RaceResult(BaseModel):
    rank:      int
    rider:     str
    team:      str
    time_gap:  Optional[str] = None


class ResultsResponse(BaseModel):
    race_name:   str
    year:        int
    stage:       Optional[int] = None
    results:     list[RaceResult]
    found:       bool
    error:       Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Run a natural language query through the PelotonIQ agent."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    agent = get_agent()
    t0    = time.time()
    error = None
    state = {}

    try:
        from peloton_iq.schemas import empty_state
        state = agent._app.invoke(empty_state(req.query))
    except Exception as e:
        log.error("Query failed: %s", e)
        error = str(e)

    # Extract prediction text and race context for Dash charts
    pred_text = state.get("prediction_context", "")
    race_ctx  = (state.get("structured_params") or {}).get("race_context") or {}

    return QueryResponse(
        query=req.query,
        response=state.get("final_response", "An error occurred." if error else ""),
        query_type=state.get("query_type", ""),
        steps=state.get("steps_taken", []),
        elapsed_s=round(time.time() - t0, 2),
        prediction_text=pred_text or None,
        race_context={
            "race_name": race_ctx.get("race_name", ""),
            "year":      race_ctx.get("year"),
            "stage":     race_ctx.get("stage"),
        } if race_ctx else None,
        error=error,
    )


@app.get("/api/results", response_model=ResultsResponse)
async def results(
    race_name: str,
    year: int,
    stage: Optional[int] = None,
    top_n: int = 10,
) -> ResultsResponse:
    """
    Look up actual race results from the dataset.
    Returns top N finishers for a given race, year, and optional stage.
    Used by the Dash app to show actual outcomes alongside ML predictions.
    """
    try:
        agent  = get_agent()
        df     = agent._deps.merged_df

        # Filter to race
        mask = (
            df["Race_results"].str.contains(race_name, case=False, na=False) &
            (df["Year_results"] == year) &
            df["Did_Finish"]
        )
        if stage is not None:
            mask &= (
                (df["Stage_results"] == stage) |
                (df["Stage_results"] == float(stage))
            )

        race_df = df[mask].copy()

        if race_df.empty:
            return ResultsResponse(
                race_name=race_name, year=year, stage=stage,
                results=[], found=False,
                error=f"No results found for {race_name} {year}"
                      + (f" Stage {stage}" if stage else ""),
            )

        # Get top N by rank
        top = race_df.nsmallest(top_n, "Rank")

        # Build time gap — difference from winner's time if available
        results_list = []
        winner_rank  = top["Rank"].min()

        for _, row in top.iterrows():
            rank = int(row["Rank"])

            # Try to compute a time gap string
            time_gap = None
            if "Time" in row and "Time" in top.columns:
                try:
                    winner_time = top[top["Rank"] == winner_rank]["Time"].iloc[0]
                    rider_time  = row["Time"]
                    if pd.notna(rider_time) and pd.notna(winner_time) and rank > 1:
                        gap_secs = float(rider_time) - float(winner_time)
                        if gap_secs > 0:
                            mins  = int(gap_secs // 60)
                            secs  = int(gap_secs % 60)
                            time_gap = f"+{mins}:{secs:02d}" if mins > 0 else f"+{secs}s"
                    elif rank == 1:
                        time_gap = "winner"
                except Exception:
                    pass

            results_list.append(RaceResult(
                rank=rank,
                rider=str(row.get("Name", "")),
                team=str(row.get("Team", "")),
                time_gap=time_gap,
            ))

        return ResultsResponse(
            race_name=race_name,
            year=year,
            stage=stage,
            results=results_list,
            found=True,
        )

    except Exception as e:
        log.error("Results lookup failed: %s", e)
        return ResultsResponse(
            race_name=race_name, year=year, stage=stage,
            results=[], found=False,
            error=str(e),
        )


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — verifies all components are operational."""
    agent  = get_agent()
    checks = agent.sanity_check()
    return HealthResponse(
        status="ok" if all(checks.values()) else "degraded",
        checks=checks,
        model=agent._deps.predictor.model_name if agent._deps else "unknown",
        qdrant_ok=checks.get("qdrant_courses", False),
    )


@app.get("/")
async def root():
    return {
        "service": "PelotonIQ API",
        "docs":    "/docs",
        "health":  "/api/health",
        "query":   "POST /api/query",
    }