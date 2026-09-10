"""Snapshot the cached hourly market frame for the demo window into tests/fixtures.

Usage: uv run python scripts/build_fixture.py   (needs data/cache warmed for Dec 2023 - Jan 2024)
"""

from pathlib import Path

from copilot.config import load_settings
from copilot.investigate import load_window
from copilot.timeutil import helsinki

OUT = Path("tests/fixtures/market_2023-12-08_2024-01-08.parquet")


def main() -> None:
    frame = load_window(load_settings(), helsinki("2023-12-08"), helsinki("2024-01-08"))
    if frame.missing:
        raise SystemExit(f"refusing to snapshot with missing series: {frame.missing}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.data.astype("float32").to_parquet(OUT)
    print(
        f"wrote {OUT} ({len(frame.data)} rows, {len(frame.data.columns)} cols, {OUT.stat().st_size // 1024} kB)"
    )


if __name__ == "__main__":
    main()
