"""Time utilities for aligning series to quarter-hour resolution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytz  # type: ignore


def floor_to_resolution(dt: datetime, minutes: int) -> datetime:
    delta = timedelta(minutes=minutes)
    epoch = datetime(1970, 1, 1, tzinfo=dt.tzinfo)
    seconds = (dt - epoch).total_seconds()
    bucket = int(seconds // (delta.total_seconds()))
    return epoch + bucket * delta


def generate_quarter_range(start: datetime, periods: int, resolution_minutes: int) -> list[datetime]:
    delta = timedelta(minutes=resolution_minutes)
    return [start + i * delta for i in range(periods)]


def ensure_timezone(dt: datetime, tz_name: str) -> datetime:
    tz = pytz.timezone(tz_name)
    if dt.tzinfo is None:
        return tz.localize(dt)
    return dt.astimezone(tz)


def to_utc_naive(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)
