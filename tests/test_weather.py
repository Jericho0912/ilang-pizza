from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import requests

from ilang_pizza.features import weather as weather_mod
from ilang_pizza.features.weather import (
    ARCHIVE_URL,
    CACHE_FILE,
    CACHE_TTL_S,
    FORECAST_URL,
    RETRY_AFTER_S,
    NoWeather,
    OpenMeteoWeather,
    WeatherProvider,
    default_provider,
)
from ilang_pizza.schema import MANILA_TZ

TODAY = date(2026, 10, 23)
MAKATI = (14.5547, 121.0244)
BGC = (14.5509, 121.0503)

# A real archive response, verbatim (Manila during the July 2025 habagat floods).
REAL_ARCHIVE_RESPONSE = (
    '{"latitude":14.586995,"longitude":121.002785,"generationtime_ms":0.3038644790649414,'
    '"utc_offset_seconds":28800,"timezone":"Asia/Manila","timezone_abbreviation":"GMT+8",'
    '"elevation":12.0,"daily_units":{"time":"iso8601","precipitation_sum":"mm"},'
    '"daily":{"time":["2025-07-20","2025-07-21","2025-07-22","2025-07-23"],'
    '"precipitation_sum":[30.40,64.50,30.70,51.90]}}'
)


def fake_rain(day: date) -> float:
    """Distinct, predictable rain per day: 2025-07-21 -> 21.07 mm."""
    return day.day + day.month / 100


class FakeResponse:
    def __init__(self, status_code: int = 200, text: str = "{}") -> None:
        self.status_code = status_code
        self.text = text

    def json(self):
        return json.loads(self.text)


class FakeOpenMeteo:
    """Stands in for ``requests.Session``: answers like Open-Meteo and records every call."""

    def __init__(self, today: date = TODAY, rain=fake_rain) -> None:
        self.today = today
        self.rain = rain
        self.calls: list[tuple[str, dict]] = []
        self.reply: FakeResponse | Exception | None = None  # set to force a canned outcome

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        if isinstance(self.reply, Exception):
            raise self.reply
        if self.reply is not None:
            return self.reply
        if url == ARCHIVE_URL:
            first = date.fromisoformat(params["start_date"])
            last = date.fromisoformat(params["end_date"])
        else:
            first = self.today - timedelta(days=params["past_days"])
            last = self.today + timedelta(days=params["forecast_days"] - 1)
        days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
        daily = {
            "time": [d.isoformat() for d in days],
            "precipitation_sum": [self.rain(d) for d in days],
        }
        return FakeResponse(200, json.dumps({"timezone": MANILA_TZ, "daily": daily}))


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 23, 12, 0, tzinfo=ZoneInfo(MANILA_TZ))

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def server() -> FakeOpenMeteo:
    return FakeOpenMeteo()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def provider(tmp_path, server, clock) -> OpenMeteoWeather:
    return OpenMeteoWeather(cache_dir=tmp_path, session=server, clock=clock)


def test_parses_a_real_archive_response(provider, server):
    server.reply = FakeResponse(200, REAL_ARCHIVE_RESPONSE)
    assert provider.precipitation_mm(14.6, 120.98, date(2025, 7, 21)) == pytest.approx(64.5)


def test_old_days_come_from_the_archive(provider, server):
    assert provider.precipitation_mm(*MAKATI, date(2025, 7, 21)) == pytest.approx(21.07)
    [(url, params)] = server.calls
    assert url == ARCHIVE_URL
    assert params == {
        "latitude": 14.55,
        "longitude": 121.02,
        "daily": "precipitation_sum",
        "timezone": "Asia/Manila",
        "start_date": "2025-07-21",
        "end_date": "2025-07-21",
    }


def test_recent_and_upcoming_days_come_from_the_forecast(provider, server):
    week_ago, soon, horizon = (TODAY + timedelta(days=n) for n in (-7, 3, 15))
    provider.prefetch([(*MAKATI, week_ago), (*MAKATI, soon), (*MAKATI, horizon)])
    [(url, params)] = server.calls
    assert url == FORECAST_URL
    assert (params["past_days"], params["forecast_days"]) == (7, 16)
    for day in (week_ago, soon, horizon):
        assert provider.precipitation_mm(*MAKATI, day) == pytest.approx(fake_rain(day))
    assert len(server.calls) == 1  # the lookups after prefetch are cache hits


def test_eight_days_ago_is_settled_enough_for_the_archive(provider, server):
    provider.precipitation_mm(*MAKATI, TODAY - timedelta(days=8))
    assert server.calls[0][0] == ARCHIVE_URL


def test_beyond_the_forecast_horizon_is_unknown(provider, server):
    assert provider.precipitation_mm(*MAKATI, TODAY + timedelta(days=16)) is None
    assert server.calls == []


def test_unknown_coordinates_are_unknown_rain(provider, server):
    assert provider.precipitation_mm(math.nan, math.nan, date(2025, 7, 21)) is None
    assert server.calls == []


def test_prefetch_makes_a_handful_of_requests_for_a_backtest(provider, server):
    # Five years of monthly events rotating over three venues, plus two upcoming events.
    venues = [MAKATI, BGC, (14.6575, 121.0570)]
    past = [(*venues[i % 3], date(2021, 10, 18) + timedelta(days=30 * i)) for i in range(60)]
    upcoming = [(*MAKATI, TODAY + timedelta(days=6)), (*BGC, TODAY + timedelta(days=13))]
    provider.prefetch(past + upcoming)
    made = len(server.calls)
    assert made <= 8  # two archive ranges per venue, one forecast per venue with upcoming events
    assert all(provider.precipitation_mm(*point) is not None for point in past + upcoming)
    assert len(server.calls) == made


def test_nearby_coordinates_share_a_cache_entry(provider, server):
    provider.precipitation_mm(14.5547, 121.0244, date(2025, 7, 21))
    provider.precipitation_mm(14.5512, 121.0249, date(2025, 7, 21))
    assert len(server.calls) == 1


def test_cache_persists_across_instances(tmp_path, server, clock):
    first = OpenMeteoWeather(cache_dir=tmp_path, session=server, clock=clock)
    assert first.precipitation_mm(*MAKATI, date(2025, 7, 21)) == pytest.approx(21.07)

    offline = FakeOpenMeteo()
    offline.reply = requests.ConnectionError("no network")
    second = OpenMeteoWeather(cache_dir=tmp_path, session=offline, clock=clock)
    assert second.precipitation_mm(*MAKATI, date(2025, 7, 21)) == pytest.approx(21.07)
    assert offline.calls == []


def test_forecasts_expire_but_archive_values_do_not(provider, server, clock):
    old, soon = date(2025, 7, 21), TODAY + timedelta(days=2)
    provider.prefetch([(*MAKATI, old), (*MAKATI, soon)])
    assert len(server.calls) == 2

    clock.advance(CACHE_TTL_S - 60)
    provider.prefetch([(*MAKATI, old), (*MAKATI, soon)])
    assert len(server.calls) == 2

    clock.advance(120)
    server.rain = lambda day: 99.0  # the forecast was revised
    provider.prefetch([(*MAKATI, old), (*MAKATI, soon)])
    assert [url for url, _ in server.calls[2:]] == [FORECAST_URL]
    assert provider.precipitation_mm(*MAKATI, soon) == 99.0
    assert provider.precipitation_mm(*MAKATI, old) == pytest.approx(21.07)


def test_missing_values_are_none(provider, server):
    server.rain = lambda day: None
    assert provider.precipitation_mm(*MAKATI, date(2025, 7, 21)) is None
    provider.precipitation_mm(*MAKATI, date(2025, 7, 21))
    assert len(server.calls) == 1  # a known gap is cached for a while too


@pytest.mark.parametrize(
    "reply",
    [
        requests.ConnectionError("offline"),
        requests.Timeout("too slow"),
        FakeResponse(500, "oops"),
        FakeResponse(429, "slow down"),
        FakeResponse(400, '{"error":true,"reason":"Parameter \'end_date\' is out of range"}'),
        FakeResponse(200, "<html>captive portal</html>"),
        FakeResponse(200, '{"daily":{"time":["2025-07-21"]}}'),
        FakeResponse(200, '{"daily":{"time":["2025-07-21","2025-07-22"],"precipitation_sum":[1]}}'),
        FakeResponse(200, '{"daily":{"time":["21/07/2025"],"precipitation_sum":[1.0]}}'),
        FakeResponse(200, '[{"daily":{"time":["2025-07-21"],"precipitation_sum":[1.0]}}]'),
    ],
)
def test_failures_return_none(provider, server, reply):
    server.reply = reply
    assert provider.precipitation_mm(*MAKATI, date(2025, 7, 21)) is None


def test_a_failed_request_is_not_repeated_right_away(provider, server, clock):
    server.reply = FakeResponse(400, '{"error":true}')
    for _ in range(3):
        assert provider.precipitation_mm(*MAKATI, date(2025, 7, 21)) is None
    assert len(server.calls) == 1

    clock.advance(RETRY_AFTER_S + 1)
    server.reply = None
    assert provider.precipitation_mm(*MAKATI, date(2025, 7, 21)) == pytest.approx(21.07)
    assert len(server.calls) == 2


def test_an_unreachable_server_pauses_all_requests(provider, server, clock):
    server.reply = requests.Timeout("too slow")
    provider.prefetch([(*MAKATI, date(2025, 7, 21)), (*BGC, date(2025, 7, 21))])
    assert provider.precipitation_mm(*BGC, date(2024, 1, 5)) is None
    assert len(server.calls) == 1  # one timeout, then no more waiting

    clock.advance(RETRY_AFTER_S + 1)
    server.reply = None
    assert provider.precipitation_mm(*BGC, date(2024, 1, 5)) == pytest.approx(5.01)


def test_a_stale_value_beats_none_when_a_refresh_fails(provider, server, clock):
    soon = TODAY + timedelta(days=2)
    assert provider.precipitation_mm(*MAKATI, soon) == pytest.approx(fake_rain(soon))
    clock.advance(CACHE_TTL_S + 1)
    server.reply = requests.ConnectionError("offline")
    assert provider.precipitation_mm(*MAKATI, soon) == pytest.approx(fake_rain(soon))
    assert len(server.calls) == 2


def test_a_corrupt_cache_file_is_ignored_and_replaced(tmp_path, provider, server):
    (tmp_path / CACHE_FILE).write_text("{not json")
    assert provider.precipitation_mm(*MAKATI, date(2025, 7, 21)) == pytest.approx(21.07)
    saved = json.loads((tmp_path / CACHE_FILE).read_text())
    assert saved["entries"]["14.55,121.02,2025-07-21"]["mm"] == pytest.approx(21.07)


def test_malformed_cache_entries_are_dropped(tmp_path, server, clock):
    entries = {
        "14.55,121.02,2025-07-21": {"mm": "lots", "source": "archive", "fetched_at": 0},
        "14.55,121.02,2025-07-22": {"mm": 3.5, "source": "archive", "fetched_at": 0},
    }
    (tmp_path / CACHE_FILE).write_text(json.dumps({"version": 1, "entries": entries}))
    provider = OpenMeteoWeather(cache_dir=tmp_path, session=server, clock=clock)
    assert provider.precipitation_mm(*MAKATI, date(2025, 7, 22)) == 3.5
    assert server.calls == []
    assert provider.precipitation_mm(*MAKATI, date(2025, 7, 21)) == pytest.approx(21.07)


def test_cache_writes_leave_no_temp_files(tmp_path, provider):
    provider.prefetch([(*MAKATI, date(2025, 7, 21)), (*BGC, TODAY)])
    assert [p.name for p in tmp_path.iterdir()] == [CACHE_FILE]


def test_a_failed_cache_write_does_not_raise(tmp_path, provider, monkeypatch):
    def refuse(src, dst):
        raise PermissionError("read-only disk")

    monkeypatch.setattr(weather_mod.os, "replace", refuse)
    assert provider.precipitation_mm(*MAKATI, date(2025, 7, 21)) == pytest.approx(21.07)
    assert list(tmp_path.iterdir()) == []  # the temp file was cleaned up


def test_no_weather_and_default_provider():
    assert NoWeather().precipitation_mm(*MAKATI, TODAY) is None
    assert isinstance(default_provider(offline=True), NoWeather)
    online = default_provider()
    assert isinstance(online, OpenMeteoWeather)
    assert isinstance(online, WeatherProvider)
    assert isinstance(NoWeather(), WeatherProvider)


@pytest.mark.network
def test_real_open_meteo(tmp_path):
    provider = OpenMeteoWeather(cache_dir=Path(tmp_path))
    # Jul 21, 2025: the habagat floods that suspended classes across Metro Manila.
    assert provider.precipitation_mm(14.5995, 120.9842, date(2025, 7, 21)) > 20
    tomorrow = datetime.now(ZoneInfo(MANILA_TZ)).date() + timedelta(days=1)
    assert provider.precipitation_mm(14.5995, 120.9842, tomorrow) >= 0
