"""
peloton_iq.evaluation.ragas_metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Direct implementation of the four core RAGAS metrics, following the
published methodology (Es et al., 2023) rather than depending on the
`ragas` package.

Why not the `ragas` package: as of this writing, `ragas` (0.2.x–0.4.x)
has a broken import chain (`ragas.llms.base` imports
`langchain_community.chat_models.vertexai.ChatVertexAI`, which has been
removed from current `langchain-community` releases) and its pinned
langchain ecosystem conflicts with this project's own langgraph/
langchain-anthropic versions. Rather than fight dependency resolution,
these four metrics are implemented directly against the same
`anthropic` client the agent already uses — same methodology, zero new
dependency-conflict risk, full control over the judge prompts.

Metrics (each returns a float in [0, 1], higher = better, except where noted):

  faithfulness(answer, contexts)
      Does every claim in the answer follow from the retrieved context?
      Reference-free. Two-step: extract atomic statements from the
      answer, then judge each against the context.

  answer_relevancy(question, answer)
      Does the answer actually address the question asked? Reference-free.
      Implemented as: generate N hypothetical questions the answer would
      best answer, then judge how well those align with the actual question.

  context_precision(question, answer, contexts)
      Of the retrieved context chunks, how many were actually useful for
      producing the response? Reference-free variant (no gold context
      list required) — judges each chunk against the response.

  context_recall(answer, contexts, ground_truth)
      Of the statements in the ground-truth answer, how many are
      supported by the retrieved context? REQUIRES a ground_truth
      reference answer — this is the one metric that needs labels.

Usage:
    from peloton_iq.evaluation.ragas_metrics import RagasJudge

    judge = RagasJudge()
    scores = judge.evaluate_single(
        question="Who won Tour de France Stage 17 in 2023?",
        answer=agent_response,
        contexts=[course_context, rider_context, commentary_context],
        ground_truth="Felix Gall (AG2R Citroën Team) won the stage.",
    )
    # {"faithfulness": 0.83, "answer_relevancy": 0.91,
    #  "context_precision": 0.67, "context_recall": 1.0}
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

import anthropic

from peloton_iq.config import settings

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge prompts — mirror the published RAGAS methodology
# ---------------------------------------------------------------------------

STATEMENT_EXTRACTION_PROMPT = """Given a question and an answer, break the answer down into one or more simple, standalone statements. Each statement should be a single factual claim that could be independently verified.

Return ONLY a JSON array of strings, no other text.

question: {question}
answer: {answer}

Statements (JSON array):"""

FAITHFULNESS_JUDGE_PROMPT = """Consider the given context and the following statements. For each statement, determine whether it can be directly inferred from the information present in the context. Provide a brief reason, then a verdict of "yes" or "no".

A statement should be marked "no" if it makes a claim not supported by the context, even if the claim happens to be true in general knowledge. A statement should be marked "yes" only if the context actually supports it.

context:
{context}

statements:
{statements}

Return ONLY a JSON array of objects, one per statement, in order, each with keys "statement", "reason", "verdict" ("yes" or "no"). No other text.

JSON array:"""

ANSWER_RELEVANCY_PROMPT = """Given an answer, generate {n} questions that this answer would be the ideal, direct response to. Focus on the core intent of the answer — avoid generating questions about peripheral details.

answer: {answer}

Return ONLY a JSON array of {n} question strings, no other text.

JSON array:"""

QUESTION_SIMILARITY_PROMPT = """Rate how similar in meaning and intent these two questions are, on a scale from 0.0 (completely unrelated) to 1.0 (asking exactly the same thing, even if worded differently).

original question: {original}
generated question: {generated}

Return ONLY a JSON object: {{"similarity": <float between 0.0 and 1.0>}}. No other text.

JSON:"""

CONTEXT_PRECISION_PROMPT = """Given a question, a generated response, and ONE retrieved context chunk, determine whether that context chunk was useful for producing the response — i.e. did the response actually rely on information from this chunk?

question: {question}
response: {response}
context chunk: {context_chunk}

Return ONLY a JSON object: {{"reason": "<brief reason>", "useful": "yes" or "no"}}. No other text.

JSON:"""

CONTEXT_RECALL_PROMPT = """Given a ground-truth answer and a set of retrieved context chunks, break the ground-truth answer into individual statements. For each statement, determine whether it can be attributed to (supported by) information in the retrieved context.

ground_truth answer: {ground_truth}

retrieved context:
{context}

Return ONLY a JSON array of objects, one per statement extracted from the ground truth, each with keys "statement", "reason", "attributed" ("yes" or "no"). No other text.

JSON array:"""


def _extract_json(text: str) -> str:
    """Strip markdown code fences and surrounding prose, if any, from a judge response."""
    text = re.sub(r"```json|```", "", text).strip()
    return text


class RagasJudge:
    """
    LLM-as-judge implementation of the four core RAGAS metrics, using
    the project's own Anthropic client and model setting.
    """

    def __init__(
        self,
        model: str | None = None,
        client: Optional["anthropic.Anthropic"] = None,
    ) -> None:
        self._model  = model or settings.claude_model
        self._client = client or anthropic.Anthropic(
            api_key=settings.anthropic_api_key or None,
        )

    # ------------------------------------------------------------------
    # Low-level judge call
    # ------------------------------------------------------------------

    def _call(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _call_json(self, prompt: str, max_tokens: int = 1024) -> dict | list:
        raw = self._call(prompt, max_tokens=max_tokens)
        cleaned = _extract_json(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            log.warning("Judge returned unparseable JSON: %s | raw=%r", e, raw[:300])
            return []

    # ------------------------------------------------------------------
    # Faithfulness
    # ------------------------------------------------------------------

    def faithfulness(self, answer: str, contexts: list[str], question: str = "") -> float:
        """
        Proportion of atomic statements in `answer` that are directly
        supported by the concatenated `contexts`. Reference-free.
        Returns None (not 0.0) if the answer has no extractable statements
        or no context was provided — these should be treated as "not
        applicable", not "failed", by the caller.
        """
        if not answer.strip():
            return None
        if not contexts or not any(c.strip() for c in contexts):
            log.debug("Faithfulness: no contexts provided — returning None")
            return None

        statements = self._call_json(
            STATEMENT_EXTRACTION_PROMPT.format(question=question, answer=answer)
        )
        if not statements:
            return None

        context_block = "\n\n---\n\n".join(c for c in contexts if c.strip())
        verdicts = self._call_json(
            FAITHFULNESS_JUDGE_PROMPT.format(
                context=context_block,
                statements="\n".join(f"- {s}" for s in statements),
            ),
            max_tokens=2048,
        )
        if not verdicts:
            return None

        supported = sum(1 for v in verdicts if str(v.get("verdict", "")).lower() == "yes")
        return supported / len(verdicts)

    # ------------------------------------------------------------------
    # Answer relevancy
    # ------------------------------------------------------------------

    def answer_relevancy(self, question: str, answer: str, n_questions: int = 3) -> float:
        """
        Generates n hypothetical questions the answer would ideally
        address, then scores their similarity to the actual question.
        High score = the answer is precisely on-topic for the question
        asked (not padded with irrelevant tangents, not under-answering).
        Reference-free.
        """
        if not answer.strip() or not question.strip():
            return None

        generated = self._call_json(
            ANSWER_RELEVANCY_PROMPT.format(answer=answer, n=n_questions)
        )
        if not generated:
            return None

        similarities = []
        for gen_q in generated:
            result = self._call_json(
                QUESTION_SIMILARITY_PROMPT.format(original=question, generated=gen_q),
                max_tokens=128,
            )
            if isinstance(result, dict) and "similarity" in result:
                similarities.append(float(result["similarity"]))

        if not similarities:
            return None
        return sum(similarities) / len(similarities)

    # ------------------------------------------------------------------
    # Context precision (reference-free variant)
    # ------------------------------------------------------------------

    def context_precision(self, question: str, answer: str, contexts: list[str]) -> float:
        """
        Of the retrieved context chunks, what fraction were actually
        useful for producing the response? Judges each chunk
        independently against the response (no gold context list
        required — this is the reference-free variant).
        """
        non_empty = [c for c in contexts if c.strip()]
        if not non_empty:
            return None

        useful_flags = []
        for chunk in non_empty:
            result = self._call_json(
                CONTEXT_PRECISION_PROMPT.format(
                    question=question, response=answer, context_chunk=chunk
                ),
                max_tokens=256,
            )
            if isinstance(result, dict):
                useful_flags.append(str(result.get("useful", "")).lower() == "yes")

        if not useful_flags:
            return None
        return sum(useful_flags) / len(useful_flags)

    # ------------------------------------------------------------------
    # Context recall (REQUIRES ground_truth)
    # ------------------------------------------------------------------

    def context_recall(self, contexts: list[str], ground_truth: str) -> float:
        """
        Of the statements in the ground-truth reference answer, what
        fraction are supported by the retrieved context? This is the
        one metric that requires a labelled ground_truth — it measures
        retrieval coverage against what a correct answer would need,
        independent of what the agent actually produced.
        """
        if not ground_truth.strip():
            raise ValueError("context_recall requires a non-empty ground_truth")
        non_empty = [c for c in contexts if c.strip()]
        if not non_empty:
            return None

        context_block = "\n\n---\n\n".join(non_empty)
        verdicts = self._call_json(
            CONTEXT_RECALL_PROMPT.format(ground_truth=ground_truth, context=context_block),
            max_tokens=2048,
        )
        if not verdicts:
            return None

        attributed = sum(1 for v in verdicts if str(v.get("attributed", "")).lower() == "yes")
        return attributed / len(verdicts)

    # ------------------------------------------------------------------
    # Convenience — evaluate one example across all applicable metrics
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> dict[str, float | None]:
        """
        Run all metrics for one (question, answer, contexts) triple.
        context_recall is only computed if ground_truth is provided —
        otherwise it's reported as None, not silently skipped, so it's
        visible in results which examples lack ground truth coverage.
        """
        scores = {
            "faithfulness":      self.faithfulness(answer, contexts, question=question),
            "answer_relevancy":  self.answer_relevancy(question, answer),
            "context_precision": self.context_precision(question, answer, contexts),
        }
        if ground_truth:
            scores["context_recall"] = self.context_recall(contexts, ground_truth)
        else:
            scores["context_recall"] = None
        return scores