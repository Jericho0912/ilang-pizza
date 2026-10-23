"""Synthetic demo data: deterministic, PythonPH-shaped, labeled, and with learnable signal."""

from __future__ import annotations

import datetime as dt
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ilang_pizza.data.loaders import canonical_events, canonical_rsvps, load_dataset
from ilang_pizza.data.synth import POOL_SIZE, make_synthetic_dataset, write_dataset
from ilang_pizza.schema import EVENT_COLUMNS, MANILA_TZ, RSVP_COLUMNS, Dataset

TODAY = pd.Timestamp("2026-10-01", tz=MANILA_TZ)


@pytest.fixture(scope="module")
def demo() -> Dataset:
    return make_synthetic_dataset(today=TODAY)


@pytest.fixture(scope="module")
def pooled() -> pd.DataFrame:
    """Going RSVPs with known attendance from three seeds, joined with event details and rain.

    Pooling keeps the effect checks from hinging on one lucky seed.
    """
    frames = []
    for seed in (2026, 7, 11):
        ds = make_synthetic_dataset(today=TODAY, seed=seed)
        rows = ds.rsvps.join(ds.events.set_index("event_id"), on="event_id")
        rain = ds.weather.set_index("date")["precipitation_mm"]
        rows["rain_mm"] = rows["start"].dt.date.map(rain)
        rows["lead_days"] = (rows["start"] - rows["rsvp_at"]).dt.total_seconds() / 86400
        rows["first_rsvp"] = ~rows.duplicated("attendee_id")  # RSVPs are in time order
        rows["seed"] = seed
        frames.append(rows[(rows["status"] == "going") & rows["attended"].notna()])
    return pd.concat(frames, ignore_index=True)


def show_rate(rows: pd.DataFrame) -> float:
    return float(rows["attended"].mean())


# --- determinism and shape --------------------------------------------------------------------


def test_same_seed_and_day_give_the_same_dataset(demo: Dataset) -> None:
    again = make_synthetic_dataset(today="2026-10-01")
    assert_frame_equal(demo.events, again.events)
    assert_frame_equal(demo.rsvps, again.rsvps)
    assert_frame_equal(demo.weather, again.weather)
    other = make_synthetic_dataset(today=TODAY, seed=1)
    assert not demo.rsvps["attendee_id"].equals(other.rsvps["attendee_id"])


def test_today_is_normalized_to_a_manila_date() -> None:
    from_date = make_synthetic_dataset(n_past=3, n_upcoming=1, today=dt.date(2026, 10, 1))
    from_evening = make_synthetic_dataset(
        n_past=3, n_upcoming=1, today=pd.Timestamp("2026-10-01 21:45", tz=MANILA_TZ)
    )
    assert_frame_equal(from_date.rsvps, from_evening.rsvps)


def test_dataset_is_labeled_and_canonical(demo: Dataset) -> None:
    demo.validate()
    assert demo.synthetic
    assert list(demo.events.columns) == EVENT_COLUMNS
    assert list(demo.rsvps.columns) == RSVP_COLUMNS
    assert str(demo.rsvps["rsvp_at"].dtype) == "datetime64[us, Asia/Manila]"
    assert set(demo.rsvps["source"]) == {"synthetic"}
    assert demo.rsvps["attendee_id"].str.fullmatch(r"[0-9a-f]{16}").all()
    assert_frame_equal(demo.events, canonical_events(demo.events))
    assert_frame_equal(demo.rsvps, canonical_rsvps(demo.rsvps, demo.events))


def test_events_span_five_years_and_end_near_today(demo: Dataset) -> None:
    starts = demo.events["start"]
    past = demo.events[starts < TODAY]
    upcoming = demo.events[starts >= TODAY]
    assert len(past) == 60 and len(upcoming) == 2
    assert 4.5 < (TODAY - past["start"].min()).days / 365.25 < 5.5
    assert (TODAY - past["start"].max()).days <= 35
    days_ahead = (upcoming["start"].dt.normalize() - TODAY).dt.days
    assert days_ahead.between(7, 21).all()
    assert demo.upcoming_event_ids() == list(upcoming["event_id"])
    gaps = past["start"].diff().dt.days.dropna()
    assert gaps.between(10, 55).all()  # roughly monthly


def test_event_ids_and_titles(demo: Dataset) -> None:
    for event in demo.events.itertuples():
        slug = event.series.lower().replace(" ", "-")
        assert event.event_id == f"{slug}-{event.start:%Y-%m-%d}"
        assert re.fullmatch(r"Demo .+ #\d+: .+", event.title), event.title
    assert demo.events["event_id"].is_unique


def test_each_series_keeps_its_usual_schedule(demo: Dataset) -> None:
    ev = demo.events
    weekday, clock = ev["start"].dt.day_of_week, ev["start"].dt.strftime("%H:%M")
    pymeets = ev["series"] == "PyMeets"
    assert weekday[pymeets].between(0, 4).all()
    assert clock[pymeets].isin(["18:30", "19:00"]).all()
    assert ev.loc[pymeets, "city"].isin(["Taguig", "Makati", "Pasig"]).all()  # BGC, Makati, Ortigas
    assert ev.loc[pymeets, "capacity"].between(40, 100).all()
    pyworks = ev["series"] == "PyWorks"
    assert (weekday[pyworks] == 5).all() and (clock[pyworks] == "13:00").all()
    assert ev.loc[pyworks, "capacity"].between(40, 80).all()
    wfh = ev["series"] == "WFH Pythonista"
    assert ev.loc[wfh, "is_online"].all() and ev.loc[wfh, "capacity"].isna().all()
    assert (ev.loc[wfh, "venue"] == "").all() and ev.loc[wfh, "latitude"].isna().all()
    assert (weekday[wfh] == 5).all() and (ev.loc[wfh, "start"].dt.hour >= 18).all()
    pykids = ev["series"] == "PyKids"
    assert (weekday[pykids] == 5).all() and ev.loc[pykids, "start"].dt.hour.between(12, 17).all()
    assert set(ev["series"]) == {"PyMeets", "PyWorks", "WFH Pythonista", "PyKids"}
    in_person = ev[~ev["is_online"]]
    assert in_person["latitude"].between(14.4, 14.8).all()  # Metro Manila
    assert in_person["longitude"].between(120.9, 121.2).all()
    assert ev.loc[pymeets, "food_provided"].mean() > 0.6


def test_rsvp_volumes_and_waitlists(demo: Dataset) -> None:
    rs = demo.rsvps.join(demo.events.set_index("event_id"), on="event_id")
    per_event = rs.groupby("event_id").agg(
        rsvps=("status", "size"),
        online=("is_online", "first"),
        series=("series", "first"),
        capacity=("capacity", "first"),
        waitlisted=("status", lambda s: int((s == "waitlist").sum())),
    )
    past = per_event[per_event.index.isin(demo.labeled_event_ids())]
    regular = past[~past["online"] & (past["series"] != "PyKids")]
    assert regular["rsvps"].between(25, 150).all()
    assert past.loc[past["online"], "rsvps"].median() > regular["rsvps"].median()
    assert (past["waitlisted"] > 0).sum() >= 10
    assert (per_event.loc[per_event["online"], "waitlisted"] == 0).all()
    # going heads never exceed capacity; the overflow is what the waitlist holds
    going = rs[rs["status"] == "going"]
    heads = (1 + going["guests"]).groupby(going["event_id"]).sum()
    capped = per_event["capacity"].notna()
    assert (heads[capped] <= per_event.loc[capped, "capacity"]).all()


def test_members_are_a_heavy_tailed_pool(demo: Dataset) -> None:
    per_member = demo.rsvps["attendee_id"].value_counts()
    assert 1200 < len(per_member) <= POOL_SIZE
    assert (per_member == 1).mean() > 0.4  # many one-timers
    core = per_member.head(len(per_member) // 20)
    assert core.sum() / per_member.sum() > 0.15  # a loyal core
    assert per_member.max() >= 20


def test_upcoming_events_have_rsvps_so_far_and_unknown_attendance(demo: Dataset) -> None:
    for event_id in demo.upcoming_event_ids():
        rows = demo.rsvps_for(event_id)
        assert len(rows) > 0
        assert rows["attended"].isna().all()
        assert (rows["rsvp_at"] < TODAY).all()
    first = demo.event(demo.upcoming_event_ids()[0])
    assert first["series"] == "PyMeets" and first["food_provided"]
    assert (demo.rsvps_for(first["event_id"])["status"] == "waitlist").any()


def test_past_waitlisted_rsvps_have_unknown_attendance(demo: Dataset) -> None:
    waitlisted = demo.rsvps[demo.rsvps["status"] == "waitlist"]
    assert len(waitlisted) > 0 and waitlisted["attended"].isna().all()


# --- ground truth -----------------------------------------------------------------------------


def test_show_rates_are_realistic(pooled: pd.DataFrame) -> None:
    in_person = pooled[~pooled["is_online"]]
    online = pooled[pooled["is_online"]]
    assert 0.45 <= show_rate(in_person) <= 0.60
    assert 0.35 <= show_rate(online) <= 0.45
    for _, rows in pooled.groupby("seed"):
        assert show_rate(rows[rows["is_online"]]) < show_rate(rows[~rows["is_online"]])


def test_documented_effects_show_up_in_the_data(pooled: pd.DataFrame) -> None:
    in_person = pooled[~pooled["is_online"]]
    dry = in_person[in_person["rain_mm"] < 2]
    heavy_rain = in_person[in_person["rain_mm"] >= 25]
    assert heavy_rain["event_id"].nunique() >= 5
    assert show_rate(dry) - show_rate(heavy_rain) > 0.08

    weekday = in_person["start"].dt.day_of_week < 5
    bgc_makati = in_person["city"].isin(["Taguig", "Makati"])
    saturday = in_person["start"].dt.day_of_week == 5
    assert show_rate(in_person[saturday]) > show_rate(in_person[weekday & bgc_makati])

    assert show_rate(pooled[~pooled["first_rsvp"]]) > show_rate(pooled[pooled["first_rsvp"]])
    assert show_rate(pooled[pooled["lead_days"] < 2]) > show_rate(pooled[pooled["lead_days"] > 14])


def test_weather_covers_every_event_day_with_a_wet_season(demo: Dataset) -> None:
    weather = demo.weather
    assert list(weather.columns) == ["date", "precipitation_mm"]
    assert all(isinstance(day, dt.date) for day in weather["date"])
    assert set(demo.events["start"].dt.date) <= set(weather["date"])
    assert (weather["precipitation_mm"] >= 0).all()
    month = pd.to_datetime(weather["date"]).dt.month
    wet = weather.loc[month.between(6, 11), "precipitation_mm"]
    dry = weather.loc[~month.between(6, 11), "precipitation_mm"]
    assert wet.mean() > 3 * dry.mean()
    assert weather["precipitation_mm"].max() >= 60  # the odd typhoon


def test_sixty_past_events_generate_quickly() -> None:
    started = time.perf_counter()
    make_synthetic_dataset(n_past=60, today=TODAY, seed=99)
    assert time.perf_counter() - started < 2.0


def test_rejects_an_empty_request() -> None:
    with pytest.raises(ValueError):
        make_synthetic_dataset(n_past=0, n_upcoming=0)


# --- writing ----------------------------------------------------------------------------------


def test_round_trip_through_files(demo: Dataset, tmp_path: Path) -> None:
    loaded = load_dataset(write_dataset(demo, tmp_path / "demo"))
    assert loaded.synthetic
    assert_frame_equal(loaded.events, demo.events)
    assert_frame_equal(loaded.rsvps, demo.rsvps)
    assert_frame_equal(loaded.weather, demo.weather)


def test_written_folder_is_labeled_and_canonical(demo: Dataset, tmp_path: Path) -> None:
    out = write_dataset(demo, tmp_path / "demo")
    assert (out / "SYNTHETIC").exists()
    readme = (out / "README.txt").read_text()
    assert "FAKE" in readme and "ilang-pizza demo" in readme
    events = pd.read_csv(out / "events.csv", dtype=str)
    assert list(events.columns) == EVENT_COLUMNS
    assert events["start"].str.endswith("+08:00").all()
    files = sorted(p.stem for p in (out / "rsvps").glob("*.csv"))
    assert files == sorted(demo.rsvps["event_id"].unique())
    one = pd.read_csv(out / "rsvps" / f"{files[0]}.csv", dtype=str)
    assert list(one.columns) == RSVP_COLUMNS and (one["event_id"] == files[0]).all()
    assert pd.read_csv(out / "weather.csv").shape == (len(demo.weather), 2)


def test_rewriting_a_demo_folder_replaces_old_files(tmp_path: Path) -> None:
    out = tmp_path / "demo"
    write_dataset(make_synthetic_dataset(n_past=6, today=TODAY), out)
    later = make_synthetic_dataset(n_past=6, today=TODAY + pd.Timedelta(days=90))
    write_dataset(later, out)
    assert sorted(p.stem for p in (out / "rsvps").glob("*.csv")) == sorted(
        later.rsvps["event_id"].unique()
    )
    assert_frame_equal(load_dataset(out).rsvps, later.rsvps)


def test_refuses_to_overwrite_a_folder_that_is_not_a_demo(demo: Dataset, tmp_path: Path) -> None:
    real = tmp_path / "data"
    real.mkdir()
    (real / "events.csv").write_text("event_id,start\n")
    with pytest.raises(FileExistsError, match="isn't a demo folder"):
        write_dataset(demo, real)
    assert (real / "events.csv").read_text() == "event_id,start\n"


def test_tiny_dataset_round_trips(tiny_dataset: Dataset, tmp_path: Path) -> None:
    loaded = load_dataset(write_dataset(tiny_dataset, tmp_path / "tiny"))
    events = canonical_events(tiny_dataset.events)
    assert_frame_equal(loaded.events, events)
    assert_frame_equal(loaded.rsvps, canonical_rsvps(tiny_dataset.rsvps, events))
    assert np.array_equal(
        loaded.rsvps["attended"].isna(),
        canonical_rsvps(tiny_dataset.rsvps, events)["attended"].isna(),
    )
