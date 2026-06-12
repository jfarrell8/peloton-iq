"""
peloton_iq.search
~~~~~~~~~~~~~~~~~~
Vector search and hybrid retrieval for PelotonIQ.

  serializers.py — serialize_course_doc, serialize_rider_doc (single source of truth)
  embeddings.py  — EmbeddingStore (Qdrant wrapper with lazy init)
  hybrid.py      — HybridSearcher (BM25 + semantic + RRF fusion)

Note: EmbeddingStore and HybridSearcher import heavy dependencies
(sentence-transformers, qdrant-client, rank-bm25). Import them
directly from their modules to avoid loading them at package import time:

    from peloton_iq.search.serializers import serialize_course_doc
    from peloton_iq.search.embeddings import EmbeddingStore
    from peloton_iq.search.hybrid import HybridSearcher
"""

from peloton_iq.search.serializers import (
    serialize_course_doc,
    serialize_rider_doc,
    build_course_docs,
    build_rider_docs,
)

__all__ = [
    "serialize_course_doc",
    "serialize_rider_doc",
    "build_course_docs",
    "build_rider_docs",
    # Heavy deps — import directly from submodules when needed:
    # from peloton_iq.search.embeddings import EmbeddingStore
    # from peloton_iq.search.hybrid import HybridSearcher
]