from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from copilot.backtest import (
    compare,
    events_per_month,
    expected_hits,
    load_cached_price,
    naive_flags,
    our_flags,
    sweep,
)
from copilot.detect import DetectConfig, find_events
from copilot.timeutil import helsinki, ts

FIXTURE = Path(__file__).parent / "fixtures" / "market_2023-12-08_2024-01-08.parquet"


def _series(days: int = 45) -> pd.Series:
    idx = pd.date_range(ts("2023-12-01"), periods=days * 24, freq="1h", name="time")
    hours = np.arange(len(idx)) % 24
    rng = np.random.default_rng(1)
    values = 60 + 20 * np.sin(hours / 24 * 2 * np.pi) + rng.normal(0, 3, len(idx))
    return pd.Series(values, index=idx, name="price_fi")


@pytest.fixture(scope="module")
def fixture_price() -> pd.Series:
    return pd.read_parquet(FIXTURE)["price_fi"].astype("float64")


def test_naive_flags_planted_spike_after_thirty_days() -> None:
    price = _series()
    spike = ts("2024-01-10 12:00")
    price.loc[spike] = 900.0
    flags = naive_flags(price)
    assert flags.loc[spike]
    assert not flags.loc[spike - pd.Timedelta(hours=1)]
    assert not flags.loc[: ts("2023-12-30")].any()  # fewer than 30 prior same-hour values
    assert flags.dtype == bool


def test_our_flags_matches_find_events() -> None:
    price = _series()
    spike = ts("2024-01-10 12:00")
    price.loc[spike] = 900.0
    flags = our_flags(price)
    assert flags.loc[spike]
    assert flags.sum() == 1


def test_events_per_month_counts_fixture_events(fixture_price: pd.Series) -> None:
    table = events_per_month(find_events(fixture_price, top_n=None))
    assert list(table.columns) == ["spike", "crash", "negative"]
    assert table.loc["2024-01", "spike"] >= 1
    assert table.loc["2023-12", "negative"] >= 1
    assert events_per_month([]).empty


def test_sweep_counts_fall_as_thresholds_rise(fixture_price: pd.Series) -> None:
    table = sweep(fixture_price, z_values=(3.0, 5.0), abs_values=(30.0, 80.0), ramp_values=(100.0,))
    assert len(table) == 4
    loose = table[(table.z == 3.0) & (table.min_abs_deviation == 30.0)].events.iloc[0]
    tight = table[(table.z == 5.0) & (table.min_abs_deviation == 80.0)].events.iloc[0]
    assert loose >= tight >= 1


def test_compare_splits_hours_four_ways(fixture_price: pd.Series) -> None:
    result = compare(fixture_price, DetectConfig(), sample=3)
    total = result.both + result.only_ours + result.only_naive + result.neither
    assert total == len(fixture_price)
    assert len(result.only_ours_sample) <= 3
    assert result.only_ours_sample["naive"].eq(False).all()
    assert result.only_naive_sample["naive"].eq(True).all()


def test_expected_hits_rank_and_missing(fixture_price: pd.Series) -> None:
    events = find_events(fixture_price, top_n=None)
    hits = expected_hits(
        events,
        [
            (helsinki("2024-01-05 19:00"), "cold snap"),
            (helsinki("2023-12-29 03:00"), "quiet hour"),
            (helsinki("2030-01-01 00:00"), "future"),
        ],
        fixture_price,
    )
    assert hits[0].rank == 1
    assert hits[0].in_data
    assert hits[1].rank is None
    assert hits[1].in_data
    assert not hits[2].in_data


def test_load_cached_price_reads_months_and_reports_missing(tmp_path: Path) -> None:
    idx = pd.date_range(ts("2024-01-01"), periods=31 * 24, freq="1h", name="time")
    pd.DataFrame({"price_fi": 50.0}, index=idx).to_parquet(
        tmp_path / "entsoe_price_FI_202401.parquet"
    )

    cached = load_cached_price(tmp_path, ts("2024-01-10"), ts("2024-03-01"))

    assert cached.missing_months == ("202402",)
    assert cached.price.index[0] == ts("2024-01-10")
    assert len(cached.price) == (22 + 29) * 24  # every hour of the range, missing month as NaN
    assert cached.price.notna().sum() == 22 * 24
    assert cached.price.name == "price_fi"


def test_load_cached_price_turns_gaps_into_nan(tmp_path: Path) -> None:
    idx = pd.date_range(ts("2024-01-01"), periods=31 * 24, freq="1h", name="time")
    frame = pd.DataFrame({"price_fi": 50.0}, index=idx)
    frame = frame.drop(idx[100:110])  # ten hours the original fetch never returned
    frame.to_parquet(tmp_path / "entsoe_price_FI_202401.parquet")

    price = load_cached_price(tmp_path, ts("2024-01-01"), ts("2024-02-01")).price

    assert len(price) == 31 * 24
    assert price.isna().sum() == 10
    assert price.diff().abs().max() == 0  # no fake jump across the gap


def test_load_cached_price_empty_when_nothing_cached(tmp_path: Path) -> None:
    cached = load_cached_price(tmp_path, ts("2024-01-01"), ts("2024-02-01"))
    assert cached.price.empty
    assert cached.missing_months == ("202401",)
