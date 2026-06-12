"""
peloton_iq.search.embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
EmbeddingStore — a class wrapping Qdrant and SentenceTransformers
for building and querying vector collections.

Mirrors the embed_and_upsert() function from notebook 02 but as a
proper class with lazy initialization, so the model and client are
only loaded when first used.

Usage:
    from peloton_iq.search.embeddings import EmbeddingStore

    store = EmbeddingStore()
    store.build_course_index(course_df)
    store.build_rider_index(merged_df)
    results = store.semantic_search("mountain stages Tour de France", "course_profiles")
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from peloton_iq.config import settings
from peloton_iq.search.serializers import build_course_docs, build_rider_docs

log = logging.getLogger(__name__)


class EmbeddingStore:
    """
    Manages vector embeddings for course profiles and rider seasons
    in Qdrant.

    Lazy initialization — the SentenceTransformer model and Qdrant
    client are only instantiated on first use, so importing this
    class doesn't trigger heavy downloads.
    """

    def __init__(
        self,
        qdrant_url: str | None = None,
        embedding_model: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._qdrant_url      = qdrant_url      or settings.qdrant_url
        self._embedding_model = embedding_model or settings.embedding_model
        self._batch_size      = batch_size      or settings.embedding_batch_size

        self._client: Optional[QdrantClient]        = None
        self._model:  Optional[SentenceTransformer] = None

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            log.info("Connecting to Qdrant at %s", self._qdrant_url)
            self._client = QdrantClient(url=self._qdrant_url)
        return self._client

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            log.info("Loading embedding model: %s", self._embedding_model)
            self._model = SentenceTransformer(self._embedding_model)
        return self._model

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _recreate_collection(self, collection_name: str) -> None:
        """Drop and recreate a Qdrant collection."""
        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name in existing:
            self.client.delete_collection(collection_name)
            log.info("Deleted existing collection: %s", collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.model.get_embedding_dimension(),
                distance=Distance.COSINE,
            ),
        )
        log.info("Created collection: %s", collection_name)

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    def upsert_docs(self, docs: list[dict], collection_name: str) -> None:
        """
        Embed and upsert a list of document dicts into a Qdrant collection.
        Each doc must have: id, text, metadata.
        Recreates the collection from scratch on each call.
        """
        if not docs:
            log.warning("upsert_docs: empty doc list for %s — skipping", collection_name)
            return

        self._recreate_collection(collection_name)

        total          = len(docs)
        total_upserted = 0

        for i in range(0, total, self._batch_size):
            batch  = docs[i : i + self._batch_size]
            texts  = [d["text"] for d in batch]

            embeddings = self.model.encode(texts, show_progress_bar=False)

            points = [
                PointStruct(
                    id=i + j,
                    vector=embedding.tolist(),
                    payload={
                        "text": doc["text"],
                        "doc_id": doc["id"],
                        **doc.get("metadata", {}),
                    },
                )
                for j, (doc, embedding) in enumerate(zip(batch, embeddings))
            ]

            self.client.upsert(collection_name=collection_name, points=points)
            total_upserted += len(points)

            if (i // self._batch_size) % 10 == 0:
                log.info(
                    "  upsert %s: %d / %d docs",
                    collection_name, total_upserted, total,
                )

        log.info(
            "upsert_docs complete: %d docs → collection '%s'",
            total_upserted, collection_name,
        )

    # ------------------------------------------------------------------
    # Index builders  (called by the embed pipeline)
    # ------------------------------------------------------------------

    def build_course_index(self, course_df: pd.DataFrame) -> None:
        """Build the course_profiles Qdrant collection from course_data_clean."""
        log.info("Building course index from %d rows...", len(course_df))
        docs = build_course_docs(course_df)
        log.info("Generated %d course documents", len(docs))
        self.upsert_docs(docs, settings.qdrant_collection_courses)

    def build_rider_index(self, merged_df: pd.DataFrame) -> None:
        """Build the rider_seasons Qdrant collection from merged_df."""
        log.info("Building rider index from merged_df (%d rows)...", len(merged_df))
        docs = build_rider_docs(merged_df)
        log.info("Generated %d rider season documents", len(docs))
        self.upsert_docs(docs, settings.qdrant_collection_riders)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def semantic_search(
        self,
        query: str,
        collection: str,
        top_k: int | None = None,
        filter_condition=None,
    ) -> list[dict]:
        """
        Embed a query and search a Qdrant collection.

        Returns a list of dicts with keys: id, score, text.
        """
        top_k       = top_k or settings.search_fetch_k
        query_vec   = self.model.encode(query).tolist()

        results = self.client.query_points(
            collection_name=collection,
            query=query_vec,
            limit=top_k,
            with_payload=True,
            query_filter=filter_condition,
        )

        return [
            {
                "id":    point.payload.get("doc_id", str(point.id)),
                "score": point.score,
                "text":  point.payload.get("text", ""),
            }
            for point in results.points
        ]

    def collection_exists(self, collection_name: str) -> bool:
        """Return True if the collection is present in Qdrant."""
        existing = [c.name for c in self.client.get_collections().collections]
        return collection_name in existing

    def collection_count(self, collection_name: str) -> int:
        """Return the number of points in a collection."""
        info = self.client.get_collection(collection_name)
        return info.points_count or 0