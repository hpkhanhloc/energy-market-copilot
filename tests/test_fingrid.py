from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from copilot.data.fingrid import Dataset, FingridClient, FingridError
from copilot.timeutil import datetime_index, helsinki, ts


@dataclass
class FakeResponse:
    status_code: int
    payload: dict
    text: str = ""

    def json(self) -> dict:
        return self.payload


@dataclass
class FakeSession:
    pages: list[FakeResponse]
    calls: list[dict] = field(default_factory=list)

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append(
            {"url": url, "params": dict(kwargs["params"]), "headers": kwargs["headers"]}
        )
        return self.pages.pop(0)


def _page(values: list[tuple[str, float]], next_page: int | None) -> FakeResponse:
    data = [{"datasetId": 181, "startTime": t, "endTime": t, "value": v} for t, v in values]
    return FakeResponse(200, {"data": data, "pagination": {"nextPage": next_page}})


START = helsinki("2024-01-05 17:00")
END = helsinki("2024-01-05 19:00")


def test_fetch_pages_and_returns_utc_series() -> None:
    session = FakeSession(
        [
            _page([("2024-01-05T15:00:00.000Z", 100.0), ("2024-01-05T15:03:00.000Z", 110.0)], 2),
            _page([("2024-01-05T15:06:00.000Z", 120.0)], None),
        ]
    )
    client = FingridClient("key", session=session, sleep=lambda _: None, clock=lambda: 0.0)
    series = client.fetch(Dataset.WIND, START, END)

    assert series.name == "wind"
    assert str(datetime_index(series).tz) == "UTC"
    assert series.tolist() == [100.0, 110.0, 120.0]
    assert [c["params"]["page"] for c in session.calls] == [1, 2]
    assert session.calls[0]["params"]["startTime"] == "2024-01-05T15:00:00Z"
    assert session.calls[0]["headers"] == {"x-api-key": "key"}


def test_fetch_hourly_averages_within_hour() -> None:
    session = FakeSession(
        [
            _page(
                [
                    ("2024-01-05T15:00:00.000Z", 100.0),
                    ("2024-01-05T15:30:00.000Z", 200.0),
                    ("2024-01-05T16:00:00.000Z", 50.0),
                ],
                None,
            )
        ]
    )
    client = FingridClient("key", session=session, sleep=lambda _: None, clock=lambda: 0.0)
    hourly = client.fetch_hourly(Dataset.WIND, START, END)
    assert hourly.tolist() == [150.0, 50.0]
    assert hourly.index[0] == ts("2024-01-05 15:00")


def test_throttles_between_calls() -> None:
    sleeps: list[float] = []
    ticks = iter([0.0, 0.0, 0.5, 0.5])
    session = FakeSession([_page([], 2), _page([], None)])
    client = FingridClient("key", session=session, sleep=sleeps.append, clock=lambda: next(ticks))
    client.fetch(Dataset.WIND, START, END)
    assert sleeps == [pytest.approx(1.5)]  # 2 s minimum gap minus 0.5 s already elapsed


def test_empty_result_is_empty_utc_series() -> None:
    session = FakeSession([_page([], None)])
    client = FingridClient("key", session=session, sleep=lambda _: None, clock=lambda: 0.0)
    series = client.fetch_hourly(Dataset.NUCLEAR, START, END)
    assert series.empty
    assert str(datetime_index(series).tz) == "UTC"


def test_non_200_raises() -> None:
    session = FakeSession([FakeResponse(401, {}, text="denied")])
    client = FingridClient("bad", session=session, sleep=lambda _: None, clock=lambda: 0.0)
    with pytest.raises(FingridError, match="401"):
        client.fetch(Dataset.WIND, START, END)


def test_naive_timestamp_rejected() -> None:
    client = FingridClient("key", session=FakeSession([]), sleep=lambda _: None, clock=lambda: 0.0)
    with pytest.raises(ValueError, match="tz-aware"):
        client.fetch(Dataset.WIND, pd.Timestamp("2024-01-05"), END)  # ty: ignore[invalid-argument-type]
