"""
peloton_iq.commentary
~~~~~~~~~~~~~~~~~~~~~~
YouTube commentary ingestion and Claude tactical extraction.

  youtube.py    — YouTubeCacheManager: build/refresh local video cache
  transcript.py — TranscriptFetcher: local video matching + transcript fetching
  extractor.py  — ClaudeExtractor: tactical extraction + agent context retrieval

All three modules have heavy optional dependencies (google-api-python-client,
youtube-transcript-api, anthropic). Import directly from submodules:

    from peloton_iq.commentary.youtube import YouTubeCacheManager
    from peloton_iq.commentary.transcript import TranscriptFetcher
    from peloton_iq.commentary.extractor import ClaudeExtractor
"""