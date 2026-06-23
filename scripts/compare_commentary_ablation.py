"""
scripts/compare_commentary_ablation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ablation study: does the lagged commentary form feature
(commentary_form_5, commentary_form_12mo, commentary_obs_count)
measurably improve the tier prediction model?

Trains two models with IDENTICAL hyperparameters, data split, and
random seed — the only difference is whether the three commentary
columns are included in FEATURE_COLS. Saves both sets of artifacts
to separate subdirectories so neither overwrites your production model.

Requires model_df.csv to already include the commentary_form_5,
commentary_form_12mo, and commentary_obs_count columns — i.e. run
this AFTER:
    python scripts/run_ingestion.py --prod --force-features

Usage:
    python scripts/compare_commentary_ablation.py
    python scripts/compare_commentary_ablation.py --trials 20   # faster, less optimal
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ablation")

COMMENTARY_COLS = [
    "commentary_form_5",
    "commentary_form_12mo",
    "commentary_obs_count",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Commentary feature ablation study")
    parser.add_argument("--trials", type=int, default=50,
                        help="Optuna trials per model (default: 50, same as production)")
    parser.add_argument("--output", type=str, default="ablation_results.json",
                        help="Where to save the comparison JSON")
    args = parser.parse_args()

    from peloton_iq.config import settings, MODELS_DIR, MODEL_DF_PATH
    from peloton_iq.prediction.trainer import train, load_model_df

    # Confirm the commentary columns actually exist in model_df
    df = load_model_df(MODEL_DF_PATH)
    missing = [c for c in COMMENTARY_COLS if c not in df.columns]
    if missing:
        log.error(
            "model_df.csv is missing commentary columns: %s\n"
            "Run: python scripts/run_ingestion.py --prod --force-features",
            missing,
        )
        sys.exit(1)

    coverage_pct = df["commentary_obs_count"].gt(0).mean() * 100
    log.info(
        "model_df has %d rows; %.1f%% have at least 1 prior commentary observation",
        len(df), coverage_pct,
    )

    baseline_cols  = [c for c in settings.model_feature_cols if c not in COMMENTARY_COLS]
    enriched_cols  = baseline_cols + COMMENTARY_COLS

    log.info("Baseline feature count : %d", len(baseline_cols))
    log.info("Enriched feature count : %d (+%d commentary)", len(enriched_cols), len(COMMENTARY_COLS))

    # ------------------------------------------------------------------
    # Train baseline (no commentary)
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("  TRAINING BASELINE (no commentary features)")
    log.info("=" * 60)
    baseline_dir = MODELS_DIR / "ablation_baseline"
    baseline_result = train(
        models_dir=baseline_dir,
        n_trials=args.trials,
        feature_cols=baseline_cols,
    )

    # ------------------------------------------------------------------
    # Train enriched (with commentary)
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("  TRAINING ENRICHED (with lagged commentary features)")
    log.info("=" * 60)
    enriched_dir = MODELS_DIR / "ablation_enriched"
    enriched_result = train(
        models_dir=enriched_dir,
        n_trials=args.trials,
        feature_cols=enriched_cols,
    )

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    comparison = _build_comparison(baseline_result, enriched_result, coverage_pct)

    log.info("=" * 60)
    log.info("  ABLATION RESULTS")
    log.info("=" * 60)
    for line in comparison["summary_lines"]:
        log.info("  %s", line)
    log.info("=" * 60)

    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)
    log.info("Full comparison saved to %s", out_path)


def _build_comparison(baseline: dict, enriched: dict, coverage_pct: float) -> dict:
    """
    Extract the best model from each run's results and build a clean
    side-by-side comparison.

    train() returns: {"clf": {name: {log_loss, auc, ...}},
                       "ranking": {name: {"top5": float, ...}},
                       "best": "LightGBM" | "XGBoost" | "Baseline"}

    We compare on whichever model was selected as best in EACH run
    independently, since that's what would actually be deployed as
    tier_predictor.pkl for that feature set. The two runs may pick
    different winning model types — that's reported explicitly rather
    than hidden.
    """
    b_best_name = baseline["best"]
    e_best_name = enriched["best"]

    b_top5 = baseline["ranking"][b_best_name]["top5"]
    e_top5 = enriched["ranking"][e_best_name]["top5"]
    b_ll   = baseline["clf"][b_best_name]["log_loss"]
    e_ll   = enriched["clf"][e_best_name]["log_loss"]
    b_auc  = baseline["clf"][b_best_name]["auc"]
    e_auc  = enriched["clf"][e_best_name]["auc"]

    summary_lines = [
        f"Commentary coverage in dataset: {coverage_pct:.1f}% of rows have 1+ prior observation",
        f"Baseline best model : {b_best_name}",
        f"Enriched best model  : {e_best_name}",
        "",
        f"{'Metric':<20}{'Baseline':<15}{'+ Commentary':<15}{'Delta':<12}",
    ]

    def _fmt_row(label, bv, ev, higher_is_better=True):
        delta = ev - bv
        sign  = "+" if delta >= 0 else ""
        better = (delta > 0) if higher_is_better else (delta < 0)
        verdict = "↑ better" if better and delta != 0 else ("↓ worse" if delta != 0 else "—")
        return f"{label:<20}{bv:<15.4f}{ev:<15.4f}{sign}{delta:<.4f}  {verdict}"

    summary_lines.append(_fmt_row("Top-5 accuracy", b_top5, e_top5, higher_is_better=True))
    summary_lines.append(_fmt_row("Log loss",       b_ll,   e_ll,   higher_is_better=False))
    summary_lines.append(_fmt_row("AUC (macro)",     b_auc,  e_auc,  higher_is_better=True))

    return {
        "coverage_pct":        coverage_pct,
        "baseline_best_model": b_best_name,
        "enriched_best_model": e_best_name,
        "baseline": {"top5_accuracy": b_top5, "log_loss": b_ll, "auc": b_auc},
        "enriched": {"top5_accuracy": e_top5, "log_loss": e_ll, "auc": e_auc},
        "summary_lines": summary_lines,
    }


if __name__ == "__main__":
    main()