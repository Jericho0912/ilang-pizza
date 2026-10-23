"""Rain on the event day, from Open-Meteo (free, no API key), with a JSON disk cache.

Which endpoint serves a day (all days are Asia/Manila dates):

* More than ``ARCHIVE_LAG_DAYS`` ago: the archive API. These values are settled, so they are
  cached for good.
* The last ``ARCHIVE_LAG_DAYS`` days and up to 15 days ahead: the forecast API, whose
  ``past_days`` covers the archive's lag. Forecasts change, so these values are re-fetched
  after ``CACHE_TTL_S``.
* Further ahead: nobody knows yet, so the answer is ``None``.

Only coordinates and dates are sent. Attendee data never leaves the machine.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

import requests

from ilang_pizza.config import CACHE_DIR
from ilang_pizza.schema import MANILA_TZ

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

ARCHIVE_START = date(1940, 1, 1)  # the archive's first day
ARCHIVE_LAG_DAYS = 7  # the archive's newest days are provisional; ask the forecast API instead
FORECAST_DAYS = 16  # today plus 15 days ahead, the forecast API's limit
MAX_ARCHIVE_SPAN_DAYS = 3 * 365  # longest date range asked for in one archive request
CACHE_TTL_S = 6 * 3600  # forecasts and missing values are re-fetched after this
RETRY_AFTER_S = 5 * 60  # after a failed request, wait this long before asking again

CACHE_FILE = "open-meteo-precipitation.json"
CACHE_VERSION = 1

Point = tuple[float, float, date]

_MANILA = ZoneInfo(MANILA_TZ)


@runtime_checkable
class WeatherProvider(Protocol):
    """Daily rain at a place. Implementations return None instead of raising."""

    def precipitation_mm(self, lat: float, lon: float, day: date) -> float | None:
        """Total rain in mm at ``lat, lon`` on ``day`` (a Manila date), or None if unknown."""
        ...


class NoWeather:
    """Offline provider: rain is always unknown."""

    def precipitation_mm(self, lat: float, lon: float, day: date) -> float | None:
        return None


class OpenMeteoWeather:
    """Daily ``precipitation_sum`` from Open-Meteo, cached as JSON under ``cache_dir``.

    Never raises on network or parse errors: it logs at debug level and returns None. If a
    refresh fails, the last cached value is returned instead. Coordinates are rounded to 2
    decimals (about 1 km) for cache keys and requests.

    ``session`` (anything with ``requests.Session.get``) and ``clock`` (returns an aware
    ``datetime``) let tests run without the network or the real time.
    """

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        timeout: float = 10.0,
        *,
        session: requests.Session | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.cache_path = Path(cache_dir).expanduser() / CACHE_FILE
        self.timeout = timeout
        self._session = session if session is not None else requests.Session()
        self._clock = clock if clock is not None else _manila_now
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] | None = None  # read from disk on first use
        self._failed_at: dict[str, float] = {}  # cache key -> when fetching it last failed
        self._paused_until = 0.0  # set when Open-Meteo looks unreachable

    def precipitation_mm(self, lat: float, lon: float, day: date) -> float | None:
        """Rain in mm at ``lat, lon`` on ``day``, fetched and cached if needed; None if unknown."""
        point = _normalize(lat, lon, day)
        if point is None:
            return None
        self.prefetch([point])
        with self._lock:
            entry = self._load().get(_key(*point))
        return None if entry is None else entry["mm"]

    def prefetch(self, points: Iterable[Point]) -> None:
        """Fetch every ``(lat, lon, day)`` not freshly cached, in as few requests as possible.

        Points are grouped by rounded location and endpoint, and each group is fetched as one
        date range (archive ranges longer than ``MAX_ARCHIVE_SPAN_DAYS`` are split). A backtest
        over 60 events at a few venues makes a handful of requests, not 60.
        """
        now = self._clock()
        now_s, today = now.timestamp(), now.astimezone(_MANILA).date()
        wanted: dict[tuple[str, float, float], set[date]] = defaultdict(set)
        with self._lock:
            entries = self._load()
            for raw in points:
                point = _normalize(*raw)
                if point is None:
                    continue
                key = _key(*point)
                source = _source_for(point[2], today)
                recently_failed = now_s - self._failed_at.get(key, -math.inf) < RETRY_AFTER_S
                if source and not recently_failed and not _is_fresh(entries.get(key), now_s):
                    wanted[(source, point[0], point[1])].add(point[2])

        fetched: dict[str, dict[str, Any]] = {}
        for (source, lat, lon), days in wanted.items():
            ordered = sorted(days)
            # One forecast request always covers every recent and upcoming day.
            runs = _split(ordered, MAX_ARCHIVE_SPAN_DAYS) if source == "archive" else [ordered]
            for run in runs:
                keys = [_key(lat, lon, day) for day in run]
                values = self._fetch(source, lat, lon, run, today, now_s)
                if values is None:
                    self._failed_at.update(dict.fromkeys(keys, now_s))
                    continue
                for key, day in zip(keys, run, strict=True):
                    fetched[key] = {"mm": values.get(day), "source": source, "fetched_at": now_s}
        if fetched:
            with self._lock:
                self._load().update(fetched)
                self._save()

    def _fetch(
        self, source: str, lat: float, lon: float, days: list[date], today: date, now_s: float
    ) -> dict[date, float | None] | None:
        """One request covering ``days`` (sorted); maps each returned date to mm, or None."""
        params: dict[str, Any] = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum",
            "timezone": MANILA_TZ,
        }
        if source == "archive":
            url = ARCHIVE_URL
            params.update(start_date=days[0].isoformat(), end_date=days[-1].isoformat())
        else:
            url = FORECAST_URL
            params.update(past_days=max(0, (today - days[0]).days), forecast_days=FORECAST_DAYS)
        payload = self._get_json(url, params, now_s)
        if payload is None:
            return None
        try:
            return _parse_daily(payload)
        except (KeyError, TypeError, ValueError) as exc:
            log.debug("unexpected Open-Meteo response from %s: %r", url, exc)
            return None

    def _get_json(self, url: str, params: dict[str, Any], now_s: float) -> Any:
        if now_s < self._paused_until:
            return None
        try:
            response = self._session.get(url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            # Offline or timing out: the next requests would only wait for their timeouts too.
            self._paused_until = now_s + RETRY_AFTER_S
            log.debug("Open-Meteo unreachable, pausing for %ss: %r", RETRY_AFTER_S, exc)
            return None
        if response.status_code != 200:
            if response.status_code == 429 or response.status_code >= 500:
                self._paused_until = now_s + RETRY_AFTER_S
            log.debug("Open-Meteo HTTP %s: %s", response.status_code, response.text[:200])
            return None
        try:
            return response.json()
        except ValueError as exc:
            log.debug("Open-Meteo sent invalid JSON: %r", exc)
            return None

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._entries is None:
            self._entries = _read_cache(self.cache_path)
        return self._entries

    def _save(self) -> None:
        data = {"version": CACHE_VERSION, "entries": self._entries}
        try:
            _atomic_write_json(self.cache_path, data)
        except OSError as exc:
            log.debug("could not write the weather cache %s: %r", self.cache_path, exc)


def default_provider(offline: bool = False) -> WeatherProvider:
    """``NoWeather()`` when offline, otherwise Open-Meteo cached under ``config.CACHE_DIR``."""
    return NoWeather() if offline else OpenMeteoWeather()


def _manila_now() -> datetime:
    return datetime.now(_MANILA)


def _normalize(lat: float, lon: float, day: date) -> Point | None:
    """Round the coordinates and reduce ``day`` to a date; None if the place is unknown."""
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    if isinstance(day, datetime):
        day = day.date()
    return round(lat, 2), round(lon, 2), day


def _key(lat: float, lon: float, day: date) -> str:
    return f"{lat:.2f},{lon:.2f},{day.isoformat()}"


def _source_for(day: date, today: date) -> str | None:
    """``"archive"`` for settled days, ``"forecast"`` for recent and upcoming ones, else None."""
    if day < ARCHIVE_START or (day - today).days >= FORECAST_DAYS:
        return None
    return "archive" if (today - day).days > ARCHIVE_LAG_DAYS else "forecast"


def _is_fresh(entry: dict[str, Any] | None, now_s: float) -> bool:
    if entry is None:
        return False
    if entry["source"] == "archive" and entry["mm"] is not None:
        return True  # settled history does not change
    return now_s - entry["fetched_at"] < CACHE_TTL_S


def _split(days: list[date], max_span: int) -> list[list[date]]:
    """Cut sorted days into runs whose first and last day are at most ``max_span`` days apart."""
    runs: list[list[date]] = []
    for day in days:
        if runs and (day - runs[-1][0]).days <= max_span:
            runs[-1].append(day)
        else:
            runs.append([day])
    return runs


def _parse_daily(payload: Any) -> dict[date, float | None]:
    """Map each date of an Open-Meteo daily response to its ``precipitation_sum`` in mm."""
    daily = payload["daily"]
    return {
        date.fromisoformat(day): _mm(value)
        for day, value in zip(daily["time"], daily["precipitation_sum"], strict=True)
    }


def _mm(value: Any) -> float | None:
    if value is None:
        return None
    mm = float(value)
    return mm if math.isfinite(mm) and mm >= 0 else None


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    """Cache entries from disk. A missing, corrupt or outdated file reads as empty."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
        return {}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {key: entry for key, entry in entries.items() if _well_formed(entry)}


def _well_formed(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and entry.get("source") in ("archive", "forecast")
        and isinstance(entry.get("fetched_at"), int | float)
        and (entry.get("mm") is None or isinstance(entry.get("mm"), int | float))
    )


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write via a temp file and a rename, so readers never see a half-written cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
