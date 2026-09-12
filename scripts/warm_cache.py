"""Pre-fetch and cache the market frame for a window, so demos and tests run offline.

Usage: uv run python scripts/warm_cache.py 2023-12-08 2024-01-07
"""

import logging
import sys
import time

from copilot.config import load_settings
from copilot.investigate import load_window
from copilot.timeutil import helsinki


def main(start: str, end: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    settings = load_settings()
    t0 = time.time()
    frame = load_window(settings, helsinki(start), helsinki(end))  # exactly what the app fetches
    print(
        f"done in {time.time() - t0:.0f}s: {len(frame.data)} hours, {len(frame.data.columns)} columns"
    )
    print("missing:", frame.missing or "none")
    print(frame.data.describe().T[["count", "mean", "min", "max"]].round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
