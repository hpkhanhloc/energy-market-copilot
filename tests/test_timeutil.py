import pandas as pd
import pytest

from copilot.timeutil import datetime_index, helsinki, to_utc, ts


def test_ts_localizes_naive_to_utc() -> None:
    assert str(ts("2024-01-05 15:00").tz) == "UTC"


def test_ts_keeps_aware_input() -> None:
    aware = ts("2024-01-05 17:00+02:00")
    assert aware == ts("2024-01-05 15:00")


def test_helsinki_localizes_and_converts() -> None:
    assert helsinki("2024-01-05 17:00") == ts("2024-01-05 15:00")
    assert str(helsinki("2024-01-05 17:00").tz) == "Europe/Helsinki"


def test_to_utc_rejects_naive() -> None:
    with pytest.raises(ValueError, match="tz-aware"):
        to_utc(pd.Timestamp("2024-01-05"))  # ty: ignore[invalid-argument-type]


def test_datetime_index_rejects_plain_index() -> None:
    with pytest.raises(TypeError):
        datetime_index(pd.Series([1, 2]))
