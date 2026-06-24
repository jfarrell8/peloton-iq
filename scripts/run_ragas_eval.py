"""
scripts/run_ragas_eval.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Run the golden test set through the live PelotonIQ agent and score
every response with the four RAGAS-equivalent metrics (faithfulness,
answer_relevancy, context_precision, context_recall).
Saves results to data/eval/ragas_results.json (overwrites each run —
that file is what the Dash "System Health" tab will read from).
Cost note: this makes several Claude calls per example (1 for the
agent's own synthesis, plus ~2-6 for the judge metrics depending on how
many statements/contexts/questions get generated). With the ~18-example
golden set this is roughly $0.50-1.50 per full run at current Sonnet
pricing — cheap enough to re-run after every meaningful agent change.
Usage:
    python scripts/run_ragas_eval.py
    python scripts/run_ragas_eval.py --limit 5              # quick smoke test
    python scripts/run_ragas_eval.py --query-type PREDICTIVE
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_ragas_eval")
def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS-equivalent agent evaluation")
    parser.add_argument("--query-type", type=str, default=None,
                        choices=["STRUCTURED", "SEMANTIC_COURSE", "SEMANTIC_RIDER",
                                 "PREDICTIVE", "HYBRID"],
                        help="Only run examples of this query type")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N examples (smoke test)")
    parser.add_argument("--output", type=str, default="data/eval/ragas_results.json",
                        help="Where to save results (default: data/eval/ragas_results.json)")
    args = parser.parse_args()
    from peloton_iq.evaluation.harness import run_eval, save_results
    result = run_eval(query_type=args.query_type, limit=args.limit)
    log.info("=" * 70)
    log.info("  RAGAS-EQUIVALENT EVALUATION — RESULTS")
    log.info("=" * 70)
    overall = result["aggregate"]["overall"]
    log.info(
        "  OVERALL (n=%d)  routing_acc=%s  faithfulness=%s  relevancy=%s  "
        "precision=%s  recall=%s",
        overall["n"], overall["routing_accuracy"],
        overall["faithfulness"], overall["answer_relevancy"],
        overall["context_precision"], overall["context_recall"],
    )
    log.info("")
    log.info("  BY QUERY TYPE:")
    for qt, m in result["aggregate"]["by_query_type"].items():
        log.info(
            "    %-16s (n=%d)  routing_acc=%s  faithfulness=%s  relevancy=%s  "
            "precision=%s  recall=%s",
            qt, m["n"], m["routing_accuracy"],
            m["faithfulness"], m["answer_relevancy"],
            m["context_precision"], m["context_recall"],
        )
    log.info("=" * 70)
    save_results(result, Path(args.output))
if __name__ == "__main__":
    main()