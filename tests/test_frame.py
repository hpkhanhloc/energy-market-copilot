from pathlib import Path

import pandas as pd

from copilot.data.frame import build_market_frame
from copilot.timeutil import helsinki, ts

START = helsinki("2024-01-05 00:00")
END = helsinki("2024-01-05 03:00")


def _hours(values: list[float], name: str) -> pd.Series:
    idx = pd.date_range(START.tz_convert("UTC"), periods=len(values), freq="1h", name="time")
    return pd.Series(values, index=idx, name=name)


class FakeEntsoe:
    def day_ahead_price(self, area: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        return _hours([1.0, 2.0, 3.0], "price_fi" if area == "FI" else f"price_{area.lower()}")

    def load(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        return _hours([10.0, 11.0], "load")  # one hour short: should become NaN, not crash

    def load_forecast(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        raise RuntimeError("entsoe down")

    def generation_by_type(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame({"nuclear": _hours([4.0, 4.0, 4.0], "nuclear")})

    def net_import(self, area: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        if area == "EE":
            raise RuntimeError("no data")
        return _hours([5.0, 5.0, 5.0], f"import_{area.lower()}")

    def wind_forecast(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        return _hours([], "wind_fc")


class FakeFingrid:
    def series(self, column: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        if column == "wind_rt":
            return _hours([7.0, 7.0, 7.0], column)
        raise RuntimeError("no key")


def test_frame_joins_and_reports_missing(tmp_path: Path) -> None:
    frame = build_market_frame(
        START, END, entsoe=FakeEntsoe(), fingrid=FakeFingrid(), sleep=lambda _: None
    )

    assert list(frame.data.index[:2]) == [ts("2024-01-04 22:00"), ts("2024-01-04 23:00")]
    assert len(frame.data) == 3
    assert frame.data["price_fi"].tolist() == [1.0, 2.0, 3.0]
    assert frame.data["price_se_3"].tolist() == [1.0, 2.0, 3.0]
    assert frame.data["nuclear"].tolist() == [4.0, 4.0, 4.0]
    assert frame.data["import_se_1"].tolist() == [5.0, 5.0, 5.0]
    assert frame.data["import_total"].tolist() == [15.0, 15.0, 15.0]  # 3 borders, EE missing
    assert "import_ee" in frame.missing
    assert frame.data["wind_rt"].tolist() == [7.0, 7.0, 7.0]
    assert frame.data["load"].isna().sum() == 1
    assert "load_fc" in frame.missing  # raised
    assert "wind_fc" in frame.missing  # empty
    assert "nuclear_rt" in frame.missing
    assert frame.has("price_fi", "nuclear")
    assert not frame.has("load_fc")


def test_frame_without_sources_is_empty_but_indexed() -> None:
    frame = build_market_frame(START, END, entsoe=None, fingrid=None)
    assert len(frame.data) == 3
    assert frame.data.columns.tolist() == []
    assert frame.missing == ()


def test_safe_retries_transient_errors() -> None:
    from copilot.data.frame import _safe

    attempts: list[int] = []

    def flaky() -> pd.Series:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("400 Bad Request")
        return _hours([1.0], "x")

    result = _safe(("x", flaky), sleep=lambda _: None)
    assert result is not None
    assert len(attempts) == 3


def test_safe_gives_up() -> None:
    from copilot.data.frame import _safe

    def broken() -> pd.Series:
        raise RuntimeError("nope")

    assert _safe(("x", broken), sleep=lambda _: None) is None
