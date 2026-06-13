"""
peloton_iq.agent
~~~~~~~~~~~~~~~~~
LangGraph multi-node agent for PelotonIQ race intelligence.

  tools.py  — Dataframe tool functions (get_stage_winner, get_race_results, etc.)
  nodes.py  — LangGraph node functions and routing logic
  graph.py  — PelotonIQAgent: graph assembly and ask() interface

All modules have heavy optional dependencies (anthropic, langgraph,
sentence-transformers). Import directly from submodules when needed:

    from peloton_iq.agent.graph import PelotonIQAgent
    from peloton_iq.agent.tools import get_stage_winner
"""