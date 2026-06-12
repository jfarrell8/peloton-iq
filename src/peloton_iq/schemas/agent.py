"""
peloton_iq.schemas.agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Typed state model for the PelotonIQ LangGraph agent.

In the notebook, PelotonState was a plain TypedDict with bare dicts
for structured_params and structured_data. This module replaces that
with validated Pydantic models so that missing fields and type errors
surface at the boundary rather than causing silent failures deep in
the graph.

The LangGraph-compatible TypedDict alias is exported alongside the
Pydantic model so the graph builder can use either form.
"""

from __future__ import annotations

from typing import Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

QUERY_TYPES = {
    "STRUCTURED",       # direct dataframe lookup — who won X?
    "SEMANTIC_COURSE",  # terrain / course profile questions
    "SEMANTIC_RIDER",   # rider performance / season questions
    "PREDICTIVE",       # pre-race analysis requiring ML model
    "HYBRID",           # requires both course context and rider context
}


# ---------------------------------------------------------------------------
# Race context extracted by the router
# ---------------------------------------------------------------------------

class RaceContext(BaseModel):
    """
    Structured parameters extracted from the user query by router_node.
    All fields are optional — the router may not always identify all of them.
    """

    race_name:  Optional[str] = None    # e.g. "Tour de France"
    year:       Optional[int] = None
    stage:      Optional[int] = None
    race_date:  Optional[str] = None    # ISO "YYYY-MM-DD" when known
    rider_name: Optional[str] = None


class RouterOutput(BaseModel):
    """
    Full output produced by router_node after classifying a query.
    """

    query_type:    str = Field(
        default="",
        description="One of: STRUCTURED | SEMANTIC_COURSE | SEMANTIC_RIDER | PREDICTIVE | HYBRID",
    )
    race_context:  RaceContext = Field(default_factory=RaceContext)
    raw_params:    dict = Field(
        default_factory=dict,
        description="Any additional key/value pairs extracted by the router.",
    )


# ---------------------------------------------------------------------------
# Agent state — Pydantic form (for validation at node boundaries)
# ---------------------------------------------------------------------------

class PelotonStateModel(BaseModel):
    """
    Validated state object for the PelotonIQ agent.

    Nodes receive and return a plain dict (the LangGraph TypedDict below);
    use PelotonStateModel.model_validate(state) at node entry points where
    you want explicit validation, and .model_dump() to convert back.
    """

    # Input
    query: str = ""

    # Routing — populated by router_node
    query_type:       str          = ""
    structured_params: RouterOutput = Field(default_factory=RouterOutput)

    # Retrieved context — each node adds to these
    structured_data:    dict = Field(default_factory=dict)
    course_context:     str  = ""
    rider_context:      str  = ""
    commentary_context: str  = ""
    prediction_context: str  = ""

    # Output
    final_response: str = ""
    error:          str = ""
    steps_taken:    list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# LangGraph-compatible TypedDict  (required by StateGraph)
# ---------------------------------------------------------------------------

class PelotonState(TypedDict, total=False):
    """
    LangGraph requires a TypedDict as the state schema.
    This mirrors PelotonStateModel exactly; use it when building the graph.
    """

    query:              str
    query_type:         str
    structured_params:  dict
    structured_data:    dict
    course_context:     str
    rider_context:      str
    commentary_context: str
    prediction_context: str
    final_response:     str
    error:              str
    steps_taken:        list


def empty_state(query: str) -> PelotonState:
    """
    Initialize a clean PelotonState for a new query.
    Drop-in replacement for the notebook's empty_state().
    """
    return PelotonState(
        query=query,
        query_type="",
        structured_params={},
        structured_data={},
        course_context="",
        rider_context="",
        commentary_context="",
        prediction_context="",
        final_response="",
        error="",
        steps_taken=[],
    )