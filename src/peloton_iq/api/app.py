"""
peloton_iq.api.app
~~~~~~~~~~~~~~~~~~~
FastAPI application for the PelotonIQ agent.

The agent is initialized once at startup and shared across all requests.
The Dash app calls this API via HTTP — the agent lives here, not in Dash.

Endpoints:
    POST /api/query   — run a query through the agent
    GET  /api/results — look up actual race results from dataset
    GET  /api/health  — health check with component status
    GET  /docs        — auto-generated OpenAPI docs

Run:
    python scripts/run_api.py
"""

from __future__ import annotations

import logging
import threading
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
_agent_ready = threading.Event()
_agent_error: str | None = None


def _initialize_agent_background() -> None:
    """Initialize agent in background thread so port opens immediately."""
    global _agent, _agent_error
    try:
        log.info("Background agent initialization starting...")
        from peloton_iq.agent.graph import PelotonIQAgent
        agent = PelotonIQAgent()
        agent.initialize()
        _agent = agent
        log.info("Background agent initialization complete.")
    except Exception as e:
        _agent_error = str(e)
        log.error("Background agent initialization failed: %s", e)
    finally:
        _agent_ready.set()


def get_agent():
    """Get the agent, waiting for initialization if still in progress."""
    if not _agent_ready.is_set():
        log.info("Waiting for agent initialization...")
        _agent_ready.wait(timeout=300)  # wait up to 5 minutes
    if _agent_error:
        raise RuntimeError(f"Agent initialization failed: {_agent_error}")
    if _agent is None:
        raise RuntimeError("Agent not initialized")
    return _agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start agent initialization in background — port opens immediately
    # so Render doesn't time out waiting for it
    log.info("PelotonIQ API starting — initializing agent in background...")
    thread = threading.Thread(target=_initialize_agent_background, daemon=True)
    thread.start()
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


class RaceResult(BaseModel):
    rank:  int
    rider: str
    team:  str


class ResultsResponse(BaseModel):
    race_name: str
    year:      int
    stage:     Optional[int] = None
    results:   list[RaceResult]
    found:     bool
    error:     Optional[str] = None


class HealthResponse(BaseModel):
    status:    str
    checks:    dict
    model:     str
    qdrant_ok: bool


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
    """Look up actual race results from the dataset."""
    try:
        agent   = get_agent()
        df      = agent._deps.merged_df

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

        top = race_df.nsmallest(top_n, "Rank")
        results_list = [
            RaceResult(
                rank=int(row["Rank"]),
                rider=str(row.get("Name", "")),
                team=str(row.get("Team", "")),
            )
            for _, row in top.iterrows()
        ]

        return ResultsResponse(
            race_name=race_name, year=year, stage=stage,
            results=results_list, found=True,
        )

    except Exception as e:
        log.error("Results lookup failed: %s", e)
        return ResultsResponse(
            race_name=race_name, year=year, stage=stage,
            results=[], found=False, error=str(e),
        )


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check — verifies all components are operational."""
    if not _agent_ready.is_set():
        return HealthResponse(
            status="initializing",
            checks={"agent_ready": False},
            model="loading",
            qdrant_ok=False,
        )
    if _agent_error:
        return HealthResponse(
            status="error",
            checks={"agent_ready": False, "error": _agent_error},
            model="error",
            qdrant_ok=False,
        )
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