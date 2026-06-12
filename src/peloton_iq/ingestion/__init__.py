"""
peloton_iq.ingestion
~~~~~~~~~~~~~~~~~~~~~
Data ingestion, cleaning, and feature engineering.

  filters.py  — UCIFilter: race name normalization and WorldTour membership
  loaders.py  — CSV loading with cleaning steps matching notebook 01
  features.py — Point-in-time rider history features and GC proxy
"""

from peloton_iq.ingestion.filters import UCIFilter
from peloton_iq.ingestion.loaders import (
    load_course_data,
    load_course_data_clean,
    load_merged_races,
    load_race_results,
    merge_race_and_course,
)
from peloton_iq.ingestion.features import (
    add_gc_proxy,
    add_stage_type,
    build_model_df,
    compute_rider_history,
)

__all__ = [
    "UCIFilter",
    "load_race_results",
    "load_course_data",
    "load_merged_races",
    "load_course_data_clean",
    "merge_race_and_course",
    "add_stage_type",
    "add_gc_proxy",
    "compute_rider_history",
    "build_model_df",
]