"""
scripts/run_api.py
~~~~~~~~~~~~~~~~~~
Start the PelotonIQ FastAPI server.

Usage:
    python scripts/run_api.py              # dev mode with reload
    python scripts/run_api.py --prod       # production mode
    python scripts/run_api.py --port 8080  # custom port
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="PelotonIQ API server")
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  type=int, default=8000)
    parser.add_argument("--prod",  action="store_true", help="Disable hot reload")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "peloton_iq.api.app:app",
        host=args.host,
        port=args.port,
        reload=not args.prod,
        log_level="info",
    )


if __name__ == "__main__":
    main()