from pathlib import Path

import pandas as pd
import pytest

from copilot.data.fingrid import Dataset
from copilot.data.fingrid_source import FingridSource
from copilot.timeutil import helsinki, ts

START = helsinki("2024-01-05 00:00")
END = helsinki("2024-01-05 03:00")


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def fetch_hourly(
        self, dataset: Dataset | int, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        self.calls.append(int(dataset))
        idx = pd.date_range(start, periods=3, freq="1h", name="time")
        return pd.Series([1.0, 2.0, 3.0], index=idx, name="raw")


def test_series_is_named_cached_and_utc(tmp_path: Path) -> None:
    client = FakeClient()
    source = FingridSource(client, cache_dir=tmp_path)

    wind = source.series("wind_rt", START, END)
    again = source.series("wind_rt", START, END)

    assert wind.name == "wind_rt"
    assert wind.index[0] == ts("2024-01-04 22:00")
    assert wind.tolist() == [1.0, 2.0, 3.0]
    assert client.calls == [int(Dataset.WIND)]
    pd.testing.assert_series_equal(wind, again, check_freq=False)  # parquet drops freq


def test_unknown_column_raises(tmp_path: Path) -> None:
    source = FingridSource(FakeClient(), cache_dir=tmp_path)
    with pytest.raises(KeyError):
        source.series("nope", START, END)
