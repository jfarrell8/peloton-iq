"""
peloton_iq.evaluation.harness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Runs the golden test set through the live PelotonIQ agent, collects
the same context fields the synthesizer node itself uses, and scores
each example with RagasJudge.

The `contexts` list for each example is built by extracting
course_context, rider_context, prediction_context, commentary_context,
and structured_data from the agent's final state — i.e. exactly the
fields synthesizer_node()'s _build_context_prompt concatenates to build
its own prompt (see agent/nodes.py). This guarantees the eval harness
measures faithfulness against what the agent actually saw, not a
reconstruction that could silently drift out of sync with the real
pipeline.

NOTE: structured_data is the ONLY context field populated for STRUCTURED
queries — those route straight from structured_tool to synthesizer,
skipping retriever/predictor/commentary entirely (see route_from_structured
in agent/nodes.py). Without including it here, faithfulness and
context_precision return None for every STRUCTURED example, since there
would be nothing to check the answer against — even though the synthesizer
DID have grounding context (it's just JSON, not prose).

Usage:
    python scripts/run_ragas_eval.py
    python scripts/run_ragas_eval.py --query-type PREDICTIVE
    python scripts/run_ragas_eval.py --limit 5   # quick smoke test
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from peloton_iq.evaluation.golden_set import GoldenExample, get_golden_set
from peloton_iq.evaluation.ragas_metrics import RagasJudge

log = logging.getLogger(__name__)

CONTEXT_FIELDS = ["course_context", "rider_context", "prediction_context", "commentary_context"]


def collect_contexts(state: dict) -> list[str]:
    """
    Pull the same context fields synthesizer_node() uses to build its
    prompt — see agent/nodes.py _build_context_prompt for the
    authoritative list this mirrors.

    structured_data is included separately (JSON-serialized) because
    _build_context_prompt does the same thing for STRUCTURED queries —
    those skip retriever/predictor/commentary entirely, so structured_data
    is the ONLY context the synthesizer sees for that query type. Without
    this, faithfulness and context_precision silently return None for
    every STRUCTURED example, since collect_contexts would otherwise find
    nothing to check the answer against.
    """
    placeholder_prefixes = ("[NO COMMENTARY]", "[NO PREDICTION]")

    contexts = [
        state.get(field, "") for field in CONTEXT_FIELDS
        if state.get(field) and not state[field].startswith(placeholder_prefixes)
    ]

    if state.get("structured_data"):
        contexts.append(json.dumps(state["structured_data"], indent=2, default=str))

    return contexts


def run_one(agent, judge: RagasJudge, example: GoldenExample) -> dict:
    """Run a single golden example through the agent and score it."""
    from peloton_iq.schemas import empty_state

    t0    = time.time()
    state = empty_state(example.question)

    try:
        result = agent._app.invoke(state)
        error  = result.get("error", "")
    except Exception as e:
        log.error("Agent invocation failed for %r: %s", example.question, e)
        result = {}
        error  = str(e)

    elapsed     = round(time.time() - t0, 2)
    answer      = result.get("final_response", "")
    contexts    = collect_contexts(result)
    actual_type = result.get("query_type", "")
    steps_taken = result.get("steps_taken", [])

    scores = {}
    if answer and not error:
        try:
            scores = judge.evaluate_single(
                question=example.question,
                answer=answer,
                contexts=contexts,
                ground_truth=example.ground_truth,
            )
        except Exception as e:
            log.error("Judging failed for %r: %s", example.question, e)
            scores = {"judge_error": str(e)}

    return {
        "question":           example.question,
        "expected_query_type": example.query_type,
        "actual_query_type":   actual_type,
        "routing_correct":     actual_type == example.query_type,
        "answer":              answer,
        "ground_truth":        example.ground_truth,
        "n_contexts":          len(contexts),
        "context_chars":       sum(len(c) for c in contexts),
        "steps_taken":         steps_taken,
        "elapsed_s":           elapsed,
        "error":               error,
        "scores":              scores,
    }


def run_eval(
    query_type: str | None = None,
    limit: int | None = None,
) -> dict:
    """
    Run the (optionally filtered) golden set through the live agent
    and score every example.

    Returns a dict with per-example results and aggregate metrics,
    broken out both overall and per query_type — averaging STRUCTURED
    and PREDICTIVE results together would hide the fact that they have
    very different expected faithfulness ceilings.
    """
    from peloton_iq.agent.graph import PelotonIQAgent

    examples = get_golden_set(query_type)
    if limit:
        examples = examples[:limit]

    log.info("Running RAGAS-equivalent eval on %d examples...", len(examples))

    agent = PelotonIQAgent()
    agent.initialize()
    judge = RagasJudge()

    results = []
    for i, example in enumerate(examples):
        log.info("[%d/%d] %s", i + 1, len(examples), example.question)
        result = run_one(agent, judge, example)
        results.append(result)
        log.info(
            "  → query_type=%s (expected %s, %s) elapsed=%.1fs scores=%s",
            result["actual_query_type"], result["expected_query_type"],
            "✓" if result["routing_correct"] else "✗ MISMATCH",
            result["elapsed_s"], result["scores"],
        )

    aggregate = _aggregate(results)

    return {
        "run_at":     datetime.now(timezone.utc).isoformat(),
        "n_examples": len(results),
        "results":    results,
        "aggregate":  aggregate,
    }


def _aggregate(results: list[dict]) -> dict:
    """
    Compute mean scores overall and per query_type. None values
    (metric not applicable for that example — e.g. no ground_truth
    for context_recall) are excluded from the mean, not treated as 0.
    """
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def _mean_by(filter_fn) -> dict:
        subset = [r for r in results if filter_fn(r)]
        out = {"n": len(subset)}
        for metric in metric_names:
            values = [
                r["scores"].get(metric) for r in subset
                if r["scores"].get(metric) is not None
            ]
            out[metric] = round(sum(values) / len(values), 3) if values else None
            out[f"{metric}_n"] = len(values)
        routing_correct = sum(1 for r in subset if r["routing_correct"])
        out["routing_accuracy"] = round(routing_correct / len(subset), 3) if subset else None
        return out

    overall = _mean_by(lambda r: True)

    by_type = {}
    query_types = sorted({r["expected_query_type"] for r in results})
    for qt in query_types:
        by_type[qt] = _mean_by(lambda r, qt=qt: r["expected_query_type"] == qt)

    return {"overall": overall, "by_query_type": by_type}


def save_results(eval_result: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, default=str)
    log.info("Saved eval results to %s", output_path)