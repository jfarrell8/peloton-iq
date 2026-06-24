"""
peloton_iq.evaluation.golden_set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Golden test set for RAGAS-style evaluation of the PelotonIQ agent.

Each entry is a (question, query_type, ground_truth) triple. ground_truth
is required for context_recall; the other three metrics (faithfulness,
answer_relevancy, context_precision) are reference-free and run
regardless.

query_type here is the EXPECTED router classification — used to check
routing accuracy as a side effect of evaluation, and to report metrics
broken out per query type (since STRUCTURED queries should score near-
perfect faithfulness, while PREDICTIVE queries are inherently less
certain — averaging them together would hide that distinction).

Keep ground_truth answers SHORT and FACTUAL — they get decomposed into
statements for context_recall, so a long, hedged ground truth produces
noisy, hard-to-attribute statements.

Extend this list over time; 20-30 well-chosen examples covering all
five query types is the right starting size per the original eval plan.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenExample:
    question:      str
    query_type:    str   # expected router classification
    ground_truth:  str | None = None  # required only for context_recall
    notes:         str = ""


GOLDEN_SET: list[GoldenExample] = [
    # ------------------------------------------------------------------
    # STRUCTURED — direct DataFrame lookups, should have near-perfect
    # faithfulness since there's no room for the LLM to embellish facts
    # that are just table lookups.
    # ------------------------------------------------------------------
    GoldenExample(
        question="Who won Tour de France Stage 17 in 2023?",
        query_type="STRUCTURED",
        ground_truth="Felix Gall of AG2R Citroën won Tour de France 2023 Stage 17.",
    ),
    GoldenExample(
        question="What were the top 5 finishers of Paris-Roubaix 2022?",
        query_type="STRUCTURED",
        ground_truth=(
            "Dylan van Baarle (INEOS Grenadiers) won Paris-Roubaix 2022, "
            "with Yves Lampaert finishing second."
        ),
    ),
    GoldenExample(
        question="How many races did Wout van Aert finish in 2022?",
        query_type="STRUCTURED",
    ),
    GoldenExample(
        question="What was Tadej Pogačar's result at the 2021 Tour de France?",
        query_type="STRUCTURED",
        ground_truth="Tadej Pogačar won the 2021 Tour de France overall.",
    ),
    GoldenExample(
        question="Did Primoz Roglic finish the 2020 Tour de France?",
        query_type="STRUCTURED",
    ),

    # ------------------------------------------------------------------
    # SEMANTIC_COURSE — course-profile retrieval, conceptual queries
    # that aren't direct row lookups.
    # ------------------------------------------------------------------
    GoldenExample(
        question="What are some of the hardest mountain stages in the Tour de France?",
        query_type="SEMANTIC_COURSE",
        notes="No single correct answer — evaluates retrieval relevance, not a fact lookup.",
    ),
    GoldenExample(
        question="Which races feature significant cobblestone sections?",
        query_type="SEMANTIC_COURSE",
        ground_truth="Paris-Roubaix is the most famous cobblestone race, featuring over 50km of pavé sectors.",
    ),
    GoldenExample(
        question="Tell me about the course profile of Strade Bianche.",
        query_type="SEMANTIC_COURSE",
        ground_truth="Strade Bianche features unpaved white gravel roads (strade bianche) in Tuscany.",
    ),
    GoldenExample(
        question="What kind of terrain does Liège-Bastogne-Liège cover?",
        query_type="SEMANTIC_COURSE",
        ground_truth="Liège-Bastogne-Liège is a hilly Ardennes classic with numerous short, steep climbs.",
    ),

    # ------------------------------------------------------------------
    # SEMANTIC_RIDER — rider-season / career retrieval, conceptual.
    # ------------------------------------------------------------------
    GoldenExample(
        question="Which riders have been strong in cobbled classics in recent years?",
        query_type="SEMANTIC_RIDER",
        notes="Open-ended — evaluates retrieval relevance.",
    ),
    GoldenExample(
        question="Tell me about Remco Evenepoel's results across different terrains.",
        query_type="SEMANTIC_RIDER",
    ),
    GoldenExample(
        question="Who are some strong time-trial specialists in the dataset?",
        query_type="SEMANTIC_RIDER",
    ),

    # ------------------------------------------------------------------
    # PREDICTIVE — ML model + tactical profile synthesis. Inherently
    # less certain than STRUCTURED; lower faithfulness/relevancy here
    # is expected and not necessarily a regression.
    # ------------------------------------------------------------------
    GoldenExample(
        question="Give me a pre-race briefing for Tour de France 2023 Stage 17.",
        query_type="PREDICTIVE",
        ground_truth=(
            "The stage favors strong climbers. Based on recent form and historical "
            "performance on similar terrain, top GC contenders and breakaway "
            "specialists are realistic contenders."
        ),
        notes="Ground truth is intentionally generic — there's no single correct prediction.",
    ),
    GoldenExample(
        question="Who is likely to win a mountain stage in next year's Giro d'Italia?",
        query_type="PREDICTIVE",
        notes="Tests handling of a genuinely unanswerable/future query — good faithfulness "
              "here means the model hedges appropriately rather than fabricating a confident pick.",
    ),
    GoldenExample(
        question="What's Jonas Vingegaard's win probability for a hilly classic?",
        query_type="PREDICTIVE",
    ),

    # ------------------------------------------------------------------
    # HYBRID — combines structured lookup + retrieval + prediction in
    # one query; tests whether the agent correctly fires multiple nodes.
    # ------------------------------------------------------------------
    GoldenExample(
        question=(
            "Compare Tadej Pogačar's recent Tour de France results to "
            "his likely performance on a mountain stage."
        ),
        query_type="HYBRID",
    ),
    GoldenExample(
        question=(
            "Given how Wout van Aert performed in Paris-Roubaix in past years, "
            "how do you think he'll do at the next one?"
        ),
        query_type="HYBRID",
    ),
    GoldenExample(
        question=(
            "Looking at the course profile of Strade Bianche and Tadej Pogačar's "
            "past results there, who would you favor this year?"
        ),
        query_type="HYBRID",
    ),
]


def get_golden_set(query_type: str | None = None) -> list[GoldenExample]:
    """Return the golden set, optionally filtered to one query type."""
    if query_type is None:
        return GOLDEN_SET
    return [ex for ex in GOLDEN_SET if ex.query_type == query_type]


def summary() -> dict[str, int]:
    """Count of examples per query type — sanity check on coverage balance."""
    counts: dict[str, int] = {}
    for ex in GOLDEN_SET:
        counts[ex.query_type] = counts.get(ex.query_type, 0) + 1
    return counts