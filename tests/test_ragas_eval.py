"""tests/test_ragas_eval.py — pure-logic tests for the eval harness and judge helpers."""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("anthropic", MagicMock())

from peloton_iq.evaluation.ragas_metrics import _extract_json
from peloton_iq.evaluation.harness import collect_contexts, _aggregate
from peloton_iq.evaluation.golden_set import get_golden_set, summary, GOLDEN_SET


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

def test_extract_json_strips_code_fence():
    assert _extract_json("```json\n[1, 2, 3]\n```") == "[1, 2, 3]"


def test_extract_json_strips_whitespace():
    assert _extract_json("   [1, 2, 3]   ") == "[1, 2, 3]"


def test_extract_json_passthrough_when_clean():
    assert _extract_json('{"a": 1}') == '{"a": 1}'


# ---------------------------------------------------------------------------
# collect_contexts — must mirror synthesizer_node's own field list exactly
# ---------------------------------------------------------------------------

def test_collect_contexts_includes_nonempty_fields():
    state = {
        "course_context": "course info",
        "rider_context": "",
        "prediction_context": "prediction info",
        "commentary_context": "commentary info",
    }
    contexts = collect_contexts(state)
    assert contexts == ["course info", "prediction info", "commentary info"]


def test_collect_contexts_empty_state_returns_empty_list():
    assert collect_contexts({}) == []


def test_collect_contexts_all_empty_returns_empty_list():
    state = {"course_context": "", "rider_context": "", "prediction_context": "", "commentary_context": ""}
    assert collect_contexts(state) == []

def test_collect_contexts_includes_structured_data():
    state = {
        "course_context": "",
        "rider_context": "",
        "prediction_context": "",
        "commentary_context": "",
        "structured_data": {"winner": "Felix Gall", "year": 2023},
    }
    contexts = collect_contexts(state)
    assert len(contexts) == 1
    assert "Felix Gall" in contexts[0]

# ---------------------------------------------------------------------------
# _aggregate
# ---------------------------------------------------------------------------

def _fake_result(query_type, faithfulness=None, routing_correct=True):
    return {
        "expected_query_type": query_type,
        "routing_correct": routing_correct,
        "scores": {
            "faithfulness": faithfulness,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": None,
        },
    }


def test_aggregate_excludes_none_from_mean():
    results = [
        _fake_result("STRUCTURED", faithfulness=1.0),
        _fake_result("STRUCTURED", faithfulness=None),  # should be excluded, not treated as 0
        _fake_result("STRUCTURED", faithfulness=0.5),
    ]
    agg = _aggregate(results)
    # mean of [1.0, 0.5] = 0.75, NOT mean of [1.0, 0, 0.5] = 0.5
    assert agg["overall"]["faithfulness"] == 0.75
    assert agg["overall"]["faithfulness_n"] == 2


def test_aggregate_breaks_out_by_query_type():
    results = [
        _fake_result("STRUCTURED", faithfulness=1.0),
        _fake_result("PREDICTIVE", faithfulness=0.4),
    ]
    agg = _aggregate(results)
    assert agg["by_query_type"]["STRUCTURED"]["faithfulness"] == 1.0
    assert agg["by_query_type"]["PREDICTIVE"]["faithfulness"] == 0.4
    # overall mixes both, by_query_type does not
    assert agg["overall"]["faithfulness"] == 0.7


def test_aggregate_routing_accuracy():
    results = [
        _fake_result("STRUCTURED", routing_correct=True),
        _fake_result("STRUCTURED", routing_correct=False),
    ]
    agg = _aggregate(results)
    assert agg["overall"]["routing_accuracy"] == 0.5


# ---------------------------------------------------------------------------
# golden_set
# ---------------------------------------------------------------------------

def test_golden_set_nonempty():
    assert len(GOLDEN_SET) > 0


def test_golden_set_covers_all_five_query_types():
    covered = summary()
    expected_types = {"STRUCTURED", "SEMANTIC_COURSE", "SEMANTIC_RIDER", "PREDICTIVE", "HYBRID"}
    assert expected_types.issubset(covered.keys())


def test_get_golden_set_filters_by_type():
    structured = get_golden_set("STRUCTURED")
    assert all(ex.query_type == "STRUCTURED" for ex in structured)
    assert len(structured) > 0