"""Small helpers so timestamps are always tz-aware and typed as real Timestamps (never NaT)."""

import pandas as pd


def ts(value: object, tz: str = "UTC") -> pd.Timestamp:
    """Parse to a tz-aware Timestamp. Naive input is localized to `tz`; aware input is kept."""
    result = pd.Timestamp(value)  # ty: ignore[invalid-argument-type]
    if not isinstance(result, pd.Timestamp):
        raise ValueError(f"not a timestamp: {value!r}")
    if result.tzinfo is not None:
        return result
    # DST policy for naive local input: a repeated autumn hour takes its first (summer-time)
    # occurrence, a skipped spring hour is shifted forward. Documented, not raised.
    return result.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")


def helsinki(value: object) -> pd.Timestamp:
    """Parse to a Timestamp in Europe/Helsinki (localizing naive input to Helsinki)."""
    return ts(value, tz="Europe/Helsinki").tz_convert("Europe/Helsinki")


def to_utc(value: pd.Timestamp) -> pd.Timestamp:
    """Require tz-aware input and return it in UTC."""
    if value.tzinfo is None:
        raise ValueError("timestamps must be tz-aware")
    return value.tz_convert("UTC")


def datetime_index(obj: pd.Series | pd.DataFrame) -> pd.DatetimeIndex:
    """Return the index as a DatetimeIndex or raise. Keeps type checkers honest."""
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError("expected a DatetimeIndex")
    return obj.index
