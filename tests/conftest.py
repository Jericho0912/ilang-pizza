"""Shared fixtures. `tiny_dataset` is a small, deterministic, synthetic dataset in canonical
form so every module can be tested without real exports, network, or model weights."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ilang_pizza.schema import MANILA_TZ, Dataset

_MARKER_ENV = {
    "tabpfn": "ILANG_PIZZA_TEST_TABPFN",
    "network": "ILANG_PIZZA_TEST_NETWORK",
    "ollama": "ILANG_PIZZA_TEST_OLLAMA",
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker, env in _MARKER_ENV.items():
            if marker in item.keywords and os.getenv(env) != "1":
                item.add_marker(pytest.mark.skip(reason=f"set {env}=1 to run"))


_VENUES = [
    ("Makati office", "Makati", 14.5547, 121.0244),
    ("BGC office", "Taguig", 14.5509, 121.0503),
    ("UP-Ayala TechnoHub", "Quezon City", 14.6575, 121.0570),
]
_SERIES_CYCLE = ["PyMeets", "PyWorks", "PyMeets", "WFH Pythonista"]


def make_tiny_dataset(n_past: int = 14, n_upcoming: int = 1, seed: int = 7) -> Dataset:
    """Monthly events from Jan 2025, ~20-45 RSVPs each, attendance known for past events."""
    rng = np.random.default_rng(seed)
    pool = [f"a{i:03d}" for i in range(120)]
    loyalty = rng.uniform(0.2, 0.9, size=len(pool))
    first = pd.Timestamp("2025-01-18 14:00", tz=MANILA_TZ)
    events, rsvps = [], []
    for i in range(n_past + n_upcoming):
        series = _SERIES_CYCLE[i % len(_SERIES_CYCLE)]
        online = series == "WFH Pythonista"
        start = first + pd.DateOffset(months=i)
        if series == "PyMeets":  # weekday evening after work
            start = (start + pd.Timedelta(days=(3 - start.weekday()) % 7)).replace(
                hour=18, minute=30
            )
        venue, city, lat, lon = ("", "", np.nan, np.nan) if online else _VENUES[i % len(_VENUES)]
        capacity = np.nan if online else float(rng.choice([25, 30, 40]))
        event_id = f"{series.lower().replace(' ', '-')}-{start:%Y-%m}"
        events.append(
            {
                "event_id": event_id,
                "title": f"Tiny {series} {start:%b %Y}",
                "series": series,
                "start": start,
                "venue": venue,
                "city": city,
                "latitude": lat,
                "longitude": lon,
                "is_online": online,
                "capacity": capacity,
                "food_provided": not online,
            }
        )
        n = int(rng.integers(20, 45))
        chosen = rng.choice(len(pool), size=n, replace=False)
        leads = np.sort(rng.uniform(0.1, 21.0, size=n))[::-1]  # earliest RSVP first
        for j, (idx, lead) in enumerate(zip(chosen, leads)):
            status = "waitlist" if (not np.isnan(capacity) and j >= capacity) else "going"
            p = loyalty[idx] * (0.6 if online else 1.0) * (0.85 if lead > 10 else 1.0)
            known = i < n_past and status == "going"
            rsvps.append(
                {
                    "event_id": event_id,
                    "attendee_id": pool[idx],
                    "rsvp_at": start - pd.Timedelta(days=float(lead)),
                    "status": status,
                    "guests": int(rng.random() < 0.1),
                    "attended": float(rng.random() < p) if known else np.nan,
                    "source": "synthetic",
                }
            )
    ev = pd.DataFrame(events)
    rs = pd.DataFrame(rsvps)
    ev["start"] = pd.to_datetime(ev["start"]).dt.tz_convert(MANILA_TZ)
    rs["rsvp_at"] = pd.to_datetime(rs["rsvp_at"]).dt.tz_convert(MANILA_TZ)
    ds = Dataset(events=ev, rsvps=rs, synthetic=True)
    ds.validate()
    return ds


@pytest.fixture(scope="session")
def tiny_dataset() -> Dataset:
    return make_tiny_dataset()
