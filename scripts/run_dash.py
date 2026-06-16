"""
scripts/run_dash.py
~~~~~~~~~~~~~~~~~~~~
Start the PelotonIQ Dash application.

Usage:
    python scripts/run_dash.py
    python scripts/run_dash.py --port 8051
    python scripts/run_dash.py --debug
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


def main() -> None:
    parser = argparse.ArgumentParser(description="PelotonIQ Dash app")
    parser.add_argument("--port",  type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    from peloton_iq.app import app
    app.run(debug=args.debug, port=args.port, host="0.0.0.0")


if __name__ == "__main__":
    main()