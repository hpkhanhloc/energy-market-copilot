from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from copilot.data.cache import cache_key, cached_frame, ttl_for
from copilot.timeutil import helsinki, ts


def _frame(value: float) -> pd.DataFrame:
    idx = pd.DatetimeIndex(["2024-01-05T15:00Z", "2024-01-05T16:00Z"], name="time")
    return pd.DataFrame({"x": [value, value]}, index=idx)


def test_cache_key_is_filesystem_safe() -> None:
    key = cache_key("fingrid", 181, helsinki("2024-01-05 17:00"), "a/b c")
    assert key == "fingrid_181_20240105T1500_a-b-c"


def test_cached_frame_fetches_once(tmp_path: Path) -> None:
    calls: list[int] = []

    def fetch() -> pd.DataFrame:
        calls.append(1)
        return _frame(1.0)

    first = cached_frame("k", fetch, cache_dir=tmp_path)
    second = cached_frame("k", fetch, cache_dir=tmp_path)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)
    assert (tmp_path / "k.parquet").exists()


def test_cached_frame_refetches_after_ttl(tmp_path: Path) -> None:
    cached_frame("k", lambda: _frame(1.0), cache_dir=tmp_path)
    path = tmp_path / "k.parquet"
    old = path.stat().st_mtime - 7200
    import os

    os.utime(path, (old, old))
    fresh = cached_frame("k", lambda: _frame(2.0), cache_dir=tmp_path, ttl=timedelta(hours=1))
    assert fresh["x"].iloc[0] == 2.0


@pytest.mark.parametrize(
    ("end", "expected_ttl"),
    [
        (ts("2024-01-05"), None),
        (ts("2026-09-09"), timedelta(hours=1)),
    ],
)
def test_ttl_for(end: pd.Timestamp, expected_ttl: timedelta | None) -> None:
    now = ts("2026-09-10")
    assert ttl_for(end, now=now) == expected_ttl
