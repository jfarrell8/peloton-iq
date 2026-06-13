"""
peloton_iq.search.hybrid
~~~~~~~~~~~~~~~~~~~~~~~~~
HybridSearcher — BM25 + semantic search with Reciprocal Rank Fusion.

Encapsulates the hybrid_search() pipeline from notebook 03 as a class
with persistent BM25 indexes so they are built once and reused across
queries, rather than rebuilt on every agent invocation.

Usage:
    from peloton_iq.search.hybrid import HybridSearcher
    from peloton_iq.search.embeddings import EmbeddingStore

    store    = EmbeddingStore()
    searcher = HybridSearcher(store)
    searcher.build_indexes(course_df, merged_df)

    results = searcher.search_courses("cobblestone classics like Paris-Roubaix")
    results = searcher.search_riders("best climbers 2023 Tour de France")
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pickle
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from peloton_iq.config import settings, MODELS_DIR
from peloton_iq.search.embeddings import EmbeddingStore
from peloton_iq.search.serializers import build_course_docs, build_rider_docs

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


# ---------------------------------------------------------------------------
# HybridSearcher
# ---------------------------------------------------------------------------

class HybridSearcher:
    """
    Hybrid search over course profiles and rider season documents.

    BM25 indexes are built once from the document corpora and held
    in memory. Semantic search is delegated to EmbeddingStore (Qdrant).
    Results are fused with Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        store: EmbeddingStore,
        rrf_k: int | None = None,
        fetch_k: int | None = None,
        top_k: int | None = None,
        cache_dir=None,
    ) -> None:
        self._store     = store
        self._rrf_k     = rrf_k   or settings.rrf_k
        self._fetch_k   = fetch_k or settings.search_fetch_k
        self._top_k     = top_k   or settings.search_top_k
        self._cache_dir = cache_dir or MODELS_DIR

        # BM25 indexes — populated by build_indexes()
        self._course_corpus:    list[str]   = []
        self._course_ids:       list[str]   = []
        self._bm25_course:      Optional[BM25Okapi] = None

        self._rider_corpus:     list[str]   = []
        self._rider_ids:        list[str]   = []
        self._bm25_rider:       Optional[BM25Okapi] = None

    # ------------------------------------------------------------------
    # Index builders
    # ------------------------------------------------------------------

    def build_indexes(
        self,
        course_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        force_rebuild: bool = False,
    ) -> None:
        """
        Build BM25 indexes for both course and rider corpora.
        Caches to disk so subsequent startups load in seconds.
        Pass force_rebuild=True to regenerate after data changes.
        """
        self._build_course_bm25(course_df, force_rebuild=force_rebuild)
        self._build_rider_bm25(merged_df, force_rebuild=force_rebuild)

    def _cache_path(self, name: str):
        return self._cache_dir / f"bm25_{name}_index.pkl"

    def _save_cache(self, name: str, payload: dict) -> None:
        path = self._cache_path(name)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        log.info("BM25 %s index cached to %s", name, path.name)

    def _load_cache(self, name: str):
        path = self._cache_path(name)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            log.warning("BM25 cache load failed for %s: %s", name, e)
            return None

    def _build_course_bm25(self, course_df: pd.DataFrame, force_rebuild: bool = False) -> None:
        cached = None if force_rebuild else self._load_cache("course")
        if cached:
            self._course_corpus = cached["corpus"]
            self._course_ids    = cached["ids"]
            self._bm25_course   = cached["index"]
            log.info("BM25 course index loaded from cache (%d docs)", len(self._course_corpus))
            return
        log.info("Building BM25 course index...")
        docs                = build_course_docs(course_df)
        self._course_corpus = [d["text"] for d in docs]
        self._course_ids    = [d["id"]   for d in docs]
        tokenized           = [_tokenize(t) for t in self._course_corpus]
        self._bm25_course   = BM25Okapi(tokenized)
        log.info("BM25 course index: %d documents", len(self._course_corpus))
        self._save_cache("course", {"corpus": self._course_corpus, "ids": self._course_ids, "index": self._bm25_course})

    def _build_rider_bm25(self, merged_df: pd.DataFrame, force_rebuild: bool = False) -> None:
        cached = None if force_rebuild else self._load_cache("rider")
        if cached:
            self._rider_corpus = cached["corpus"]
            self._rider_ids    = cached["ids"]
            self._bm25_rider   = cached["index"]
            log.info("BM25 rider index loaded from cache (%d docs)", len(self._rider_corpus))
            return
        log.info("Building BM25 rider index...")
        docs               = build_rider_docs(merged_df)
        self._rider_corpus = [d["text"] for d in docs]
        self._rider_ids    = [d["id"]   for d in docs]
        tokenized          = [_tokenize(t) for t in self._rider_corpus]
        self._bm25_rider   = BM25Okapi(tokenized)
        log.info("BM25 rider index: %d documents", len(self._rider_corpus))
        self._save_cache("rider", {"corpus": self._rider_corpus, "ids": self._rider_ids, "index": self._bm25_rider})

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_courses(
        self,
        query: str,
        top_k: int | None = None,
        fetch_k: int | None = None,
    ) -> list[dict]:
        """Hybrid search over course profile documents."""
        self._assert_built("course")
        return self._hybrid_search(
            query=query,
            collection=settings.qdrant_collection_courses,
            corpus=self._course_corpus,
            ids=self._course_ids,
            bm25_index=self._bm25_course,
            top_k=top_k or self._top_k,
            fetch_k=fetch_k or self._fetch_k,
        )

    def search_riders(
        self,
        query: str,
        top_k: int | None = None,
        fetch_k: int | None = None,
    ) -> list[dict]:
        """Hybrid search over rider season documents."""
        self._assert_built("rider")
        return self._hybrid_search(
            query=query,
            collection=settings.qdrant_collection_riders,
            corpus=self._rider_corpus,
            ids=self._rider_ids,
            bm25_index=self._bm25_rider,
            top_k=top_k or self._top_k,
            fetch_k=fetch_k or self._fetch_k,
        )

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _hybrid_search(
        self,
        query: str,
        collection: str,
        corpus: list[str],
        ids: list[str],
        bm25_index: BM25Okapi,
        top_k: int,
        fetch_k: int,
    ) -> list[dict]:
        """
        Full hybrid pipeline:
          1. Semantic search via Qdrant
          2. Lexical search via BM25
          3. Fuse with Reciprocal Rank Fusion
        """
        semantic = self._store.semantic_search(query, collection, top_k=fetch_k)
        lexical  = self._bm25_search(query, corpus, ids, bm25_index, top_k=fetch_k)
        fused    = self._rrf_fusion([semantic, lexical], top_k=top_k)
        return fused

    def _bm25_search(
        self,
        query: str,
        corpus: list[str],
        ids: list[str],
        bm25_index: BM25Okapi,
        top_k: int,
    ) -> list[dict]:
        tokens      = _tokenize(query)
        scores      = bm25_index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "id":    ids[i],
                "score": float(scores[i]),
                "text":  corpus[i],
            }
            for i in top_indices
            if scores[i] > 0  # only return docs with non-zero BM25 score
        ]

    def _rrf_fusion(
        self,
        result_lists: list[list[dict]],
        top_k: int,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.
        RRF score = sum(1 / (k + rank)) across all result lists.
        k=60 is the standard default from the original RRF paper.
        """
        scores:   dict[str, float] = {}
        doc_text: dict[str, str]   = {}

        for result_list in result_lists:
            for rank, doc in enumerate(result_list, start=1):
                doc_id = doc["id"]
                if doc_id not in scores:
                    scores[doc_id]   = 0.0
                    doc_text[doc_id] = doc["text"]
                scores[doc_id] += 1.0 / (self._rrf_k + rank)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            {
                "id":        doc_id,
                "rrf_score": round(score, 6),
                "text":      doc_text[doc_id],
            }
            for doc_id, score in sorted_docs[:top_k]
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assert_built(self, which: str) -> None:
        index = self._bm25_course if which == "course" else self._bm25_rider
        if index is None:
            raise RuntimeError(
                f"BM25 {which} index not built. Call build_indexes() first."
            )

    @property
    def course_doc_count(self) -> int:
        return len(self._course_corpus)

    @property
    def rider_doc_count(self) -> int:
        return len(self._rider_corpus)