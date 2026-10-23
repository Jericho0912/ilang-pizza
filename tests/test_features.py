from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from conftest import make_tiny_dataset

from ilang_pizza.config import METRO_MANILA
from ilang_pizza.features.build import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    FEATURE_GROUPS,
    build_features,
)
from ilang_pizza.schema import MANILA_TZ, SERIES, Dataset

HISTORY = FEATURE_GROUPS["history"]
WEATHER = FEATURE_GROUPS["weather"]


def ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=MANILA_TZ)


def days_between(later: str, earlier: str) -> float:
    return (ts(later) - ts(earlier)) / pd.Timedelta(days=1)


def make_dataset(events: list[dict], rsvps: list[dict], weather=None) -> Dataset:
    """A canonical Dataset from minimal rows; unspecified columns get neutral defaults."""
    ev = pd.DataFrame(
        [
            {
                "title": e["event_id"],
                "series": "PyMeets",
                "venue": "Makati office",
                "city": "Makati",
                "latitude": 14.5547,
                "longitude": 121.0244,
                "is_online": False,
                "capacity": np.nan,
                "food_provided": True,
                **e,
            }
            for e in events
        ]
    )
    ev["start"] = pd.to_datetime(ev["start"]).dt.tz_localize(MANILA_TZ)
    rs = pd.DataFrame(
        [
            {"status": "going", "guests": 0, "attended": np.nan, "source": "canonical", **r}
            for r in rsvps
        ]
    )
    rs["rsvp_at"] = pd.to_datetime(rs["rsvp_at"]).dt.tz_localize(MANILA_TZ)
    return Dataset(events=ev, rsvps=rs, weather=weather)


# Four events and five people (ids "p1".."p5"), small enough to check every number by hand.
MINI_EVENTS = [
    {"event_id": "e1", "start": "2025-01-16 18:30", "capacity": 3.0},  # Thu
    {"event_id": "e2", "start": "2025-02-15 13:00", "series": "PyWorks",  # Sat
     "latitude": 14.5509, "longitude": 121.0503},
    {"event_id": "e3", "start": "2025-03-14 19:00", "series": "WFH Pythonista",  # Fri, payday
     "is_online": True, "venue": "", "city": "", "latitude": np.nan, "longitude": np.nan,
     "food_provided": False},
    {"event_id": "e4", "start": "2025-06-12 18:30", "capacity": 4.0,  # Independence Day
     "latitude": np.nan, "longitude": np.nan},
]  # fmt: skip
MINI_RSVPS = [
    {"event_id": "e1", "attendee_id": "p1", "rsvp_at": "2025-01-01 10:00", "attended": 1.0},
    {"event_id": "e1", "attendee_id": "p2", "rsvp_at": "2025-01-10 09:00", "attended": 0.0,
     "guests": 1},
    {"event_id": "e1", "attendee_id": "p3", "rsvp_at": "2025-01-16 08:00", "attended": 1.0,
     "guests": np.nan},
    {"event_id": "e1", "attendee_id": "p4", "rsvp_at": "2025-01-16 12:00", "status": "waitlist"},
    {"event_id": "e2", "attendee_id": "p1", "rsvp_at": "2025-02-01 21:00", "attended": 0.0},
    {"event_id": "e2", "attendee_id": "p4", "rsvp_at": "2025-02-14 20:00", "attended": 1.0},
    {"event_id": "e2", "attendee_id": "p2", "rsvp_at": None, "attended": 1.0},
    {"event_id": "e3", "attendee_id": "p1", "rsvp_at": "2025-03-01 12:00", "attended": 1.0},
    {"event_id": "e3", "attendee_id": "p2", "rsvp_at": "2025-03-02 12:00"},  # not tracked
    {"event_id": "e4", "attendee_id": "p1", "rsvp_at": "2025-06-01 12:00"},
    {"event_id": "e4", "attendee_id": "p2", "rsvp_at": "2025-06-02 12:00"},
    {"event_id": "e4", "attendee_id": "p3", "rsvp_at": "2025-06-03 12:00"},
    {"event_id": "e4", "attendee_id": "p4", "rsvp_at": "2025-06-04 12:00"},
    {"event_id": "e4", "attendee_id": "p5", "rsvp_at": "2025-06-05 12:00", "guests": 2},
]  # fmt: skip


@pytest.fixture
def mini() -> Dataset:
    return make_dataset(MINI_EVENTS, MINI_RSVPS)


def row(ds: Dataset, features: pd.DataFrame, event_id: str, attendee_id: str) -> pd.Series:
    match = ds.rsvps.index[
        (ds.rsvps["event_id"] == event_id) & (ds.rsvps["attendee_id"] == attendee_id)
    ]
    return features.loc[match[0]]


def flip_attendance(ds: Dataset, mask: pd.Series, seed: int = 0) -> Dataset:
    """Copy of ``ds`` with random labels (came, no-show, or unknown) on the masked RSVPs."""
    rng = np.random.default_rng(seed)
    rsvps = ds.rsvps.copy()
    rsvps.loc[mask, "attended"] = rng.choice([0.0, 1.0, np.nan], size=int(mask.sum()))
    return replace(ds, rsvps=rsvps)


def event_column(ds: Dataset, column: str) -> pd.Series:
    """An events column, looked up for every RSVP."""
    return ds.rsvps["event_id"].map(ds.events.set_index("event_id")[column])


def random_dataset(n_events: int = 100, per_event: int = 100, people: int = 2500) -> Dataset:
    """Random RSVPs to fortnightly events, labeled except for the last 10 events.

    People are drawn with replacement, so someone can RSVP twice to the same event. The
    defaults give 10k RSVPs over 100 events.
    """
    rng = np.random.default_rng(11)
    starts = pd.date_range("2022-01-13 18:30", periods=n_events, freq="14D", tz=MANILA_TZ)
    event_ids = np.array([f"ev{i:03d}" for i in range(n_events)])
    events = pd.DataFrame(
        {
            "event_id": event_ids,
            "title": event_ids,
            "series": rng.choice(SERIES[:4], size=n_events),
            "start": starts,
            "venue": "BGC office",
            "city": "Taguig",
            "latitude": 14.5509,
            "longitude": 121.0503,
            "is_online": False,
            "capacity": 80.0,
            "food_provided": True,
        }
    )
    which = np.repeat(np.arange(n_events), per_event)
    n = len(which)
    labeled = which < n_events - 10
    rsvps = pd.DataFrame(
        {
            "event_id": event_ids[which],
            "attendee_id": np.char.add("a", rng.integers(0, people, size=n).astype(str)),
            "rsvp_at": starts[which] - pd.to_timedelta(rng.uniform(0.05, 30, size=n), unit="D"),
            "status": np.where(rng.random(n) < 0.9, "going", "waitlist"),
            "guests": rng.integers(0, 2, size=n),
            "attended": np.where(labeled, rng.integers(0, 2, size=n), np.nan),
            "source": "synthetic",
        }
    )
    return Dataset(events=events, rsvps=rsvps, synthetic=True)


# --- the contract -----------------------------------------------------------------------------


def test_shape_dtypes_and_index(tiny_dataset):
    features = build_features(tiny_dataset)
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.index.equals(tiny_dataset.rsvps.index)
    assert (features.dtypes == np.float64).all()


def test_metadata_covers_every_column():
    assert [c for group in FEATURE_GROUPS.values() for c in group] == FEATURE_COLUMNS
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
    assert set(FEATURE_DESCRIPTIONS) == set(FEATURE_COLUMNS)
    assert all(FEATURE_DESCRIPTIONS.values())
    assert set(CATEGORICAL_FEATURES) <= set(FEATURE_COLUMNS)


def test_rows_follow_the_rsvp_index_not_its_order(tiny_dataset):
    shuffled = tiny_dataset.rsvps.sample(frac=1.0, random_state=3)
    shuffled.index = [f"r{i}" for i in shuffled.index]
    features = build_features(replace(tiny_dataset, rsvps=shuffled))
    assert features.index.equals(shuffled.index)
    expected = build_features(tiny_dataset)
    expected.index = [f"r{i}" for i in expected.index]
    pd.testing.assert_frame_equal(features, expected.loc[shuffled.index])


def test_no_nan_where_a_value_always_exists(tiny_dataset):
    features = build_features(tiny_dataset)
    sometimes_unknown = {"days_since_last_attended", "capacity_ratio", *WEATHER}
    always_known = [c for c in FEATURE_COLUMNS if c not in sometimes_unknown]
    assert not features[always_known].isna().any().any()
    online = event_column(tiny_dataset, "is_online")
    assert features.loc[online, "capacity_ratio"].isna().all()
    assert features[WEATHER].isna().all().all()  # no rain table and no provider


def test_categorical_codes_index_series(tiny_dataset):
    features = build_features(tiny_dataset)
    series = event_column(tiny_dataset, "series")
    assert (features["series_code"] == series.map(SERIES.index)).all()


# --- hand-checked values ----------------------------------------------------------------------


def test_rsvp_features(mini):
    f = build_features(mini)
    p1, p2, p3, p4 = (row(mini, f, "e1", p) for p in ("p1", "p2", "p3", "p4"))
    assert p1["lead_days"] == pytest.approx(days_between("2025-01-16 18:30", "2025-01-01 10:00"))
    assert p3["lead_days"] == pytest.approx(10.5 / 24)
    assert [p["is_late_rsvp"] for p in (p1, p2, p3, p4)] == [0.0, 0.0, 1.0, 1.0]
    assert [p["rsvp_order_pct"] for p in (p1, p2, p3, p4)] == pytest.approx([0, 1 / 3, 2 / 3, 1])
    assert [p["rsvp_hour"] for p in (p1, p2, p3, p4)] == [10.0, 9.0, 8.0, 12.0]
    assert [p["guests"] for p in (p1, p2, p3)] == [0.0, 1.0, 0.0]  # blank guests count as none


def test_unknown_rsvp_time_gives_nan_timing_only(mini):
    f = build_features(mini)
    unknown = row(mini, f, "e2", "p2")
    assert unknown[["lead_days", "rsvp_hour", "rsvp_order_pct", "is_late_rsvp"]].isna().all()
    assert unknown["prior_rsvps"] == 1.0
    # the other two RSVPs of e2 are still ranked between themselves
    assert row(mini, f, "e2", "p1")["rsvp_order_pct"] == 0.0
    assert row(mini, f, "e2", "p4")["rsvp_order_pct"] == 1.0


def test_history_features(mini):
    f = build_features(mini)
    p1 = row(mini, f, "e4", "p1")  # came to e1, skipped e2, came to e3
    assert (p1["prior_rsvps"], p1["prior_attended"], p1["prior_no_shows"]) == (3, 2, 1)
    assert p1["prior_show_rate"] == pytest.approx(3 / 5)
    assert p1["is_first_timer"] == 0.0
    assert p1["days_since_last_attended"] == pytest.approx(
        days_between("2025-06-12 18:30", "2025-03-14 19:00")
    )
    p2 = row(mini, f, "e4", "p2")  # skipped e1, came to e2, untracked at e3
    assert (p2["prior_rsvps"], p2["prior_attended"], p2["prior_no_shows"]) == (2, 1, 1)
    assert p2["days_since_last_attended"] == pytest.approx(
        days_between("2025-06-12 18:30", "2025-02-15 13:00")
    )
    p5 = row(mini, f, "e4", "p5")  # brand new
    assert (p5["prior_rsvps"], p5["prior_show_rate"], p5["is_first_timer"]) == (0, 0.5, 1)
    assert np.isnan(p5["days_since_last_attended"])

    never_came = row(mini, f, "e2", "p2")  # one no-show so far
    assert never_came["prior_show_rate"] == pytest.approx(1 / 3)
    assert np.isnan(never_came["days_since_last_attended"])


def test_a_waitlist_rsvp_is_not_a_record_but_ends_first_timer_status(mini):
    f = build_features(mini)
    p4 = row(mini, f, "e2", "p4")  # only waitlisted at e1 before
    assert p4["prior_rsvps"] == 0.0
    assert p4["is_first_timer"] == 0.0


def test_event_features(mini):
    f = build_features(mini)
    e1, e2, e3, e4 = (row(mini, f, e, "p1") for e in ("e1", "e2", "e3", "e4"))
    assert e1["event_rsvp_heads"] == 4.0  # p1, p2 + guest, p3; the waitlisted p4 is not going
    assert e1["capacity_ratio"] == pytest.approx(4 / 3)
    assert (e1["has_waitlist"], e2["has_waitlist"]) == (1.0, 0.0)
    assert np.isnan(e2["capacity_ratio"])  # unlimited
    assert e4["capacity_ratio"] == pytest.approx(7 / 4)  # p5 brings two guests
    assert (e1["is_weekend"], e2["is_weekend"]) == (0.0, 1.0)
    assert (e1["start_hour"], e2["start_hour"], e1["month"]) == (18.0, 13.0, 1.0)
    assert [e["series_code"] for e in (e1, e2, e3)] == [0.0, 1.0, 2.0]
    assert (e3["is_online"], e3["food_provided"], e1["food_provided"]) == (1.0, 0.0, 1.0)


def test_unknown_series_is_other():
    ds = make_dataset(
        [{"event_id": "x", "start": "2025-01-16 18:30", "series": "PyCon"}],
        [{"event_id": "x", "attendee_id": "p1", "rsvp_at": "2025-01-01 10:00"}],
    )
    assert build_features(ds)["series_code"].iloc[0] == SERIES.index("Other")


def test_calendar_features(mini):
    f = build_features(mini)
    e1, e2, e3, e4 = (row(mini, f, e, "p1") for e in ("e1", "e2", "e3", "e4"))
    assert e1["days_to_payday"] == 15.0  # Thu Jan 16 -> Fri Jan 31
    assert e2["days_to_payday"] == 13.0  # Sat Feb 15, paid Fri the 14th -> Fri Feb 28
    assert e3["days_to_payday"] == 0.0  # Fri Mar 14 is payday (the 15th is a Saturday)
    assert (e3["is_petsa_de_peligro"], e4["is_petsa_de_peligro"]) == (0.0, 1.0)
    assert (e1["is_holiday"], e4["is_holiday"]) == (0.0, 1.0)  # Independence Day
    assert (e2["is_long_weekend"], e4["is_long_weekend"]) == (0.0, 0.0)
    assert (e3["is_wet_season"], e4["is_wet_season"]) == (0.0, 1.0)


# --- leakage ----------------------------------------------------------------------------------


@pytest.mark.parametrize("position", [0, 5, 9, 14])
def test_attendance_at_the_event_or_later_never_changes_its_features(tiny_dataset, position):
    ds = tiny_dataset
    target = ds.events.sort_values("start")["event_id"].iloc[position]
    target_start = ds.event(target)["start"]
    rows = ds.rsvps["event_id"] == target
    base = build_features(ds)

    changed = build_features(flip_attendance(ds, event_column(ds, "start") >= target_start))
    pd.testing.assert_frame_equal(base[rows], changed[rows])

    later = event_column(ds, "start") > target_start
    if later.any():  # the flip does reach later events, so the check above has teeth
        assert not base[later].equals(changed[later])


def test_attendance_at_earlier_events_does_change_features(tiny_dataset):
    ds = tiny_dataset
    target_start = ds.events.sort_values("start")["start"].iloc[7]
    rows = event_column(ds, "start") == target_start
    changed = build_features(flip_attendance(ds, event_column(ds, "start") < target_start))
    assert not build_features(ds)[rows][HISTORY].equals(changed[rows][HISTORY])


def test_only_history_features_use_attendance(tiny_dataset):
    ds = tiny_dataset
    others = [c for c in FEATURE_COLUMNS if c not in HISTORY]
    changed = build_features(flip_attendance(ds, ds.rsvps["attended"].notna(), seed=1))
    pd.testing.assert_frame_equal(build_features(ds)[others], changed[others])


def test_everyone_is_a_first_timer_at_the_first_event(tiny_dataset):
    ds = tiny_dataset
    first = ds.rsvps["event_id"] == ds.events.sort_values("start")["event_id"].iloc[0]
    f = build_features(ds)[first]
    assert (f["is_first_timer"] == 1).all()
    assert (f["prior_rsvps"] == 0).all()
    assert (f["prior_show_rate"] == 0.5).all()
    assert f["days_since_last_attended"].isna().all()


def test_events_at_the_same_time_do_not_see_each_other():
    ds = make_dataset(
        [
            {"event_id": "talks", "start": "2025-01-16 18:30"},
            {"event_id": "stream", "start": "2025-01-16 18:30", "is_online": True},
            {"event_id": "next", "start": "2025-02-20 18:30"},
        ],
        [
            {"event_id": "talks", "attendee_id": "p1", "rsvp_at": "2025-01-01", "attended": 1.0},
            {"event_id": "stream", "attendee_id": "p1", "rsvp_at": "2025-01-02", "attended": 0.0},
            {"event_id": "next", "attendee_id": "p1", "rsvp_at": "2025-02-01"},
        ],
    )
    f = build_features(ds)
    assert f.loc[[0, 1], "is_first_timer"].tolist() == [1.0, 1.0]
    assert f.loc[[0, 1], "prior_rsvps"].tolist() == [0.0, 0.0]
    assert (f.loc[2, "prior_rsvps"], f.loc[2, "prior_attended"]) == (2.0, 1.0)


def reference_history(ds: Dataset) -> pd.DataFrame:
    """History features row by row, straight from their definitions. Slow; for checking."""
    start = event_column(ds, "start")
    rsvps = ds.rsvps
    rows = {}
    for i, rsvp in rsvps.iterrows():
        earlier = rsvps[(rsvps["attendee_id"] == rsvp["attendee_id"]) & (start < start[i])]
        record = earlier[(earlier["status"] == "going") & earlier["attended"].notna()]
        n, came = len(record), record["attended"].sum()
        attended_at = start[record.index[record["attended"] == 1]]
        rows[i] = {
            "prior_rsvps": n,
            "prior_attended": came,
            "prior_no_shows": n - came,
            "prior_show_rate": (came + 1) / (n + 2),
            "is_first_timer": float(earlier.empty),
            "days_since_last_attended": (
                (start[i] - attended_at.max()) / pd.Timedelta(days=1)
                if len(attended_at)
                else np.nan
            ),
        }
    return pd.DataFrame.from_dict(rows, orient="index").astype(np.float64)


@pytest.mark.parametrize(
    "make",
    [
        make_tiny_dataset,
        # repeat RSVPs to the same event, waitlists and unlabeled events, densely mixed
        lambda: random_dataset(n_events=20, per_event=30, people=60),
    ],
)
def test_history_matches_a_row_by_row_reference(make):
    ds = make()
    pd.testing.assert_frame_equal(build_features(ds)[HISTORY], reference_history(ds))


# --- robustness -------------------------------------------------------------------------------


def test_nothing_labeled_yet(tiny_dataset):
    unlabeled = replace(tiny_dataset, rsvps=tiny_dataset.rsvps.assign(attended=np.nan))
    f = build_features(unlabeled)
    assert (f["prior_rsvps"] == 0).all()
    assert (f["prior_show_rate"] == 0.5).all()
    assert f["days_since_last_attended"].isna().all()
    assert (f["is_first_timer"] == 0).any()  # repeat RSVPs are still seen


def test_no_rsvps_at_all(tiny_dataset):
    empty = replace(tiny_dataset, rsvps=tiny_dataset.rsvps.iloc[:0])
    f = build_features(empty)
    assert f.empty
    assert list(f.columns) == FEATURE_COLUMNS
    assert (f.dtypes == np.float64).all()


def test_anonymous_rsvps_have_unknown_history(mini):
    rsvps = mini.rsvps.copy()
    rsvps.loc[0, "attendee_id"] = np.nan
    rsvps.loc[1, "attendee_id"] = ""
    f = build_features(replace(mini, rsvps=rsvps))
    assert f.loc[[0, 1], HISTORY].isna().all().all()
    assert f.loc[[0, 1], "lead_days"].notna().all()
    assert f.loc[2:, ["prior_rsvps", "is_first_timer"]].notna().all().all()


def test_naive_timestamps_are_read_as_manila_time(mini):
    naive = replace(
        mini,
        events=mini.events.assign(start=mini.events["start"].dt.tz_localize(None)),
        rsvps=mini.rsvps.assign(rsvp_at=mini.rsvps["rsvp_at"].dt.tz_localize(None)),
    )
    pd.testing.assert_frame_equal(build_features(naive), build_features(mini))


def test_rsvps_for_unknown_events_are_rejected(mini):
    rsvps = mini.rsvps.copy()
    rsvps.loc[0, "event_id"] = "no-such-event"
    with pytest.raises(ValueError, match="no-such-event"):
        build_features(replace(mini, rsvps=rsvps))


# --- weather ----------------------------------------------------------------------------------


class FakeWeather:
    """Returns fixed rain per day and records what it was asked."""

    def __init__(self, rain: dict | None = None, default: float | None = 7.0) -> None:
        self.rain = rain or {}
        self.default = default
        self.asked: list[tuple] = []
        self.prefetched: list[list[tuple]] = []

    def prefetch(self, points) -> None:
        self.prefetched.append(list(points))

    def precipitation_mm(self, lat, lon, day):
        self.asked.append((lat, lon, day))
        return self.rain.get(day, self.default)


def rain_of(ds: Dataset, features: pd.DataFrame) -> dict[str, float | None]:
    by_event = features.groupby(ds.rsvps["event_id"])["precip_mm"].first()
    return {e: (None if np.isnan(v) else v) for e, v in by_event.items()}


def test_rain_lookup_order(mini):
    table = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-16").date(), pd.Timestamp("2025-03-14").date()],
            "precipitation_mm": [25.0, 30.0],
        }
    )
    provider = FakeWeather()
    ds = replace(mini, weather=table)
    rain = rain_of(ds, build_features(ds, weather=provider))
    assert rain == {"e1": 25.0, "e2": 7.0, "e3": None, "e4": 7.0}  # table, provider, online
    asked_days = [day.isoformat() for _, _, day in provider.asked]
    assert asked_days == ["2025-02-15", "2025-06-12"]  # never asked about e1 or the online e3
    assert provider.prefetched == [provider.asked]  # one batch up front


def test_rain_provider_gets_event_coordinates_or_metro_manila(mini):
    provider = FakeWeather()
    build_features(mini, weather=provider)
    coordinates = {day.isoformat(): (lat, lon) for lat, lon, day in provider.asked}
    assert coordinates["2025-02-15"] == (14.5509, 121.0503)
    assert coordinates["2025-06-12"] == METRO_MANILA  # in person, but no coordinates
    assert "2025-03-14" not in coordinates  # online


def test_a_gap_in_the_rain_table_falls_back_to_the_provider(mini):
    table = pd.DataFrame({"date": ["2025-01-16", "2025-02-15"], "precipitation_mm": [np.nan, 3.0]})
    ds = replace(mini, weather=table)
    rain = rain_of(ds, build_features(ds, weather=FakeWeather(default=11.0)))
    assert (rain["e1"], rain["e2"]) == (11.0, 3.0)


def test_unknown_rain_stays_nan(mini):
    ds = replace(mini, weather=pd.DataFrame({"date": [], "precipitation_mm": []}))
    rain = rain_of(ds, build_features(ds, weather=FakeWeather(default=None)))
    assert set(rain.values()) == {None}


def test_a_provider_without_prefetch_works(mini):
    class Minimal:
        def precipitation_mm(self, lat, lon, day):
            return 2.5

    rain = rain_of(mini, build_features(mini, weather=Minimal()))
    assert rain == {"e1": 2.5, "e2": 2.5, "e3": None, "e4": 2.5}


@pytest.mark.parametrize(("mm", "heavy"), [(0.0, 0.0), (19.9, 0.0), (20.0, 1.0), (64.5, 1.0)])
def test_heavy_rain_threshold(mini, mm, heavy):
    f = build_features(mini, weather=FakeWeather(default=mm))
    in_person = mini.rsvps["event_id"] != "e3"
    assert (f.loc[in_person, "is_heavy_rain"] == heavy).all()
    assert f.loc[~in_person, "is_heavy_rain"].isna().all()


# --- performance ------------------------------------------------------------------------------


def test_ten_thousand_rsvps_build_quickly():
    ds = random_dataset()
    started = time.perf_counter()
    features = build_features(ds, weather=FakeWeather())
    elapsed = time.perf_counter() - started
    assert len(features) == 10_000
    assert elapsed < 2.0, f"took {elapsed:.2f}s"


def test_a_hundred_tiny_events_build():
    ds = make_tiny_dataset(n_past=100)
    features = build_features(ds)
    assert len(features) == len(ds.rsvps)
    assert features["prior_rsvps"].max() > 10
