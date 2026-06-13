"""
scripts/run_agent.py
~~~~~~~~~~~~~~~~~~~~~
CLI runner for the PelotonIQ agent.

Usage:
    # Interactive mode
    python scripts/run_agent.py

    # Single query
    python scripts/run_agent.py --query "Who won TDF Stage 17 in 2023?"

    # Sanity check only (no Claude API calls)
    python scripts/run_agent.py --check

    # Run all showcase queries
    python scripts/run_agent.py --showcase
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_agent")

# Showcase queries — one per query type
SHOWCASE_QUERIES = [
    # STRUCTURED
    "Who won Tour de France Stage 17 in 2023?",
    # SEMANTIC_COURSE
    "What makes Paris-Roubaix so different from other classics?",
    # SEMANTIC_RIDER
    "How did Remco Evenepoel perform across his seasons in our dataset?",
    # PREDICTIVE
    (
        "It's before stage 17 of the 2023 Tour de France. "
        "Who should I watch and why? Give me a pre-race briefing."
    ),
    # HYBRID
    "Which riders in our dataset have historically performed best on high mountain stages?",
]

# Routing accuracy tests
ROUTING_TESTS = [
    ("Who won TDF Stage 19 in 2023?",                       "STRUCTURED"),
    ("What is Paris-Roubaix like terrain-wise?",             "SEMANTIC_COURSE"),
    ("How did Vingegaard perform in 2023?",                  "SEMANTIC_RIDER"),
    ("Give me a pre-race briefing for TDF 2023 Stage 17",   "PREDICTIVE"),
    ("Who performs best on cobbled mountain stages?",        "HYBRID"),
    ("Top 10 results of Strade Bianche 2022",               "STRUCTURED"),
    ("Which stages in our dataset have the most climbing?",  "SEMANTIC_COURSE"),
]


def run_sanity_check(agent) -> None:
    log.info("Running sanity check (no Claude API calls)...")
    results = agent.sanity_check()
    all_passed = True
    for check, passed in results.items():
        icon = "✓" if passed else "✗"
        log.info("  %s  %s", icon, check)
        if not passed:
            all_passed = False
    if all_passed:
        log.info("All checks passed — agent ready.")
    else:
        log.warning("Some checks failed — see above.")


def run_routing_test(agent) -> None:
    """Test router classification accuracy without running the full pipeline."""
    from peloton_iq.schemas import empty_state

    log.info("Running routing accuracy test...")
    agent.initialize()
    nodes    = agent._app
    correct  = 0

    for query, expected in ROUTING_TESTS:
        # Just call the router node directly
        state  = empty_state(query)
        result = agent._deps.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            system=agent._app.nodes["router"].__doc__ or "",
            messages=[{"role": "user", "content": query}],
        )
        import json, re
        raw    = re.sub(r"```json|```", "", result.content[0].text).strip()
        parsed = json.loads(raw)
        actual = parsed.get("query_type", "?")
        match  = actual == expected
        correct += match
        icon   = "✓" if match else "✗"
        print(f"  {icon}  {query[:55]:<55}  expected={expected:<20}  got={actual}")

    print(f"\nRouting accuracy: {correct}/{len(ROUTING_TESTS)} ({correct/len(ROUTING_TESTS)*100:.0f}%)")


def run_showcase(agent) -> None:
    for query in SHOWCASE_QUERIES:
        agent.ask(query, verbose=True)
        print()


def run_interactive(agent) -> None:
    print("\nPelotonIQ Agent — Interactive Mode")
    print("Type 'quit' or 'exit' to stop, 'check' to run sanity check.\n")

    while True:
        try:
            query = input("Query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break
        if query.lower() == "check":
            run_sanity_check(agent)
            continue

        agent.ask(query, verbose=True)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="PelotonIQ Agent")
    parser.add_argument("--query",    type=str, help="Single query to run")
    parser.add_argument("--check",    action="store_true", help="Run sanity check only")
    parser.add_argument("--showcase", action="store_true", help="Run all showcase queries")
    args = parser.parse_args()

    from peloton_iq.agent.graph import PelotonIQAgent
    agent = PelotonIQAgent()

    if args.check:
        agent.initialize()
        run_sanity_check(agent)
    elif args.query:
        agent.ask(args.query, verbose=True)
    elif args.showcase:
        agent.initialize()
        run_showcase(agent)
    else:
        agent.initialize()
        run_interactive(agent)


if __name__ == "__main__":
    main()