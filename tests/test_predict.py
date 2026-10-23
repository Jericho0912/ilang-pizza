from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ilang_pizza.features import build
from ilang_pizza.features.build import build_features
from ilang_pizza.model import predict
from ilang_pizza.model.predict import explain, forecast_event, informative_columns
from ilang_pizza.schema import MANILA_TZ, Dataset


def going(ds: Dataset, event_id: str) -> pd.DataFrame:
    rsvps = ds.rsvps_for(event_id)
    return rsvps[rsvps["status"] == "going"]


def heads(rsvps: pd.DataFrame) -> np.ndarray:
    return (1 + rsvps["guests"]).to_numpy()


def with_attendance(ds: Dataset, rows: pd.Index, values: np.ndarray) -> Dataset:
    rsvps = ds.rsvps.copy()
    rsvps.loc[rows, "attended"] = values
    return dataclasses.replace(ds, rsvps=rsvps)


def test_forecast_with_logistic(tiny_dataset):
    event_id = tiny_dataset.upcoming_event_ids()[0]
    fc = forecast_event(tiny_dataset, event_id, model_kind="logistic")
    rsvps = going(tiny_dataset, event_id)

    assert fc.event_id == event_id and fc.model_name == "logistic"
    assert fc.going_probs.shape == (len(rsvps),)
    assert np.all((fc.going_probs >= 0) & (fc.going_probs <= 1))
    np.testing.assert_array_equal(fc.going_heads, heads(rsvps))
    assert fc.pmf.sum() == pytest.approx(1.0)
    assert fc.pmf.size == fc.rsvp_heads + 1
    assert 0 <= fc.p10 <= fc.p50 <= fc.p90 <= fc.rsvp_heads
    assert fc.expected == pytest.approx(fc.going_probs @ fc.going_heads)
    assert fc.synthetic is True


def test_historical_show_rate_is_heads_weighted_over_earlier_events(tiny_dataset):
    event_id = tiny_dataset.upcoming_event_ids()[0]
    rs = tiny_dataset.rsvps
    known = rs[(rs["status"] == "going") & rs["attended"].notna()]
    expected = (known["attended"] * heads(known)).sum() / heads(known).sum()
    fc = forecast_event(tiny_dataset, event_id, model_kind="rule")
    assert fc.historical_show_rate == pytest.approx(expected)


def test_waitlisted_rsvps_are_scored_as_if_released(tiny_dataset):
    ds = tiny_dataset
    labeled = ds.labeled_event_ids()[3:]
    event_id = max(labeled, key=lambda e: (ds.rsvps_for(e)["status"] == "waitlist").sum())
    waiting = ds.rsvps_for(event_id).query("status == 'waitlist'")
    assert len(waiting) > 0

    fc = forecast_event(ds, event_id, model_kind="logistic")
    assert fc.waitlist_probs.shape == (len(waiting),)
    assert np.all((fc.waitlist_probs >= 0) & (fc.waitlist_probs <= 1))
    np.testing.assert_array_equal(fc.waitlist_heads, heads(waiting))
    assert fc.summary()["waitlist_heads"] == heads(waiting).sum()


def test_no_leakage_from_the_event_itself_or_later_ones(tiny_dataset):
    ds = tiny_dataset
    labeled = ds.labeled_event_ids()
    target = labeled[8]
    start = ds.event(target)["start"]
    starts = ds.rsvps["event_id"].map(ds.events.set_index("event_id")["start"])
    rng = np.random.default_rng(0)

    later = ds.rsvps.index[(starts >= start) & ds.rsvps["attended"].notna()]
    edited = with_attendance(ds, later, rng.integers(0, 2, len(later)).astype(float))
    assert not edited.rsvps["attended"].equals(ds.rsvps["attended"])

    before = forecast_event(ds, target, model_kind="logistic")
    after = forecast_event(edited, target, model_kind="logistic")
    np.testing.assert_array_equal(before.going_probs, after.going_probs)
    np.testing.assert_array_equal(before.waitlist_probs, after.waitlist_probs)
    np.testing.assert_array_equal(before.pmf, after.pmf)
    assert before.historical_show_rate == after.historical_show_rate
    assert before.drivers == after.drivers

    # The check is sensitive: changing an earlier event does move the forecast.
    earlier = ds.rsvps_for(labeled[2]).index
    flipped = with_attendance(ds, earlier, 1.0 - ds.rsvps.loc[earlier, "attended"])
    moved = forecast_event(flipped, target, model_kind="logistic")
    assert not np.allclose(moved.going_probs, before.going_probs)


def test_needs_enough_history(tiny_dataset):
    second = tiny_dataset.labeled_event_ids()[1]
    with pytest.raises(ValueError, match="not enough history"):
        forecast_event(tiny_dataset, second, model_kind="logistic")
    with pytest.raises(KeyError):
        forecast_event(tiny_dataset, "no-such-event", model_kind="logistic")


def test_precomputed_features_give_the_same_forecast(tiny_dataset):
    event_id = tiny_dataset.upcoming_event_ids()[0]
    features = build_features(tiny_dataset)
    a = forecast_event(tiny_dataset, event_id, model_kind="logistic")
    b = forecast_event(tiny_dataset, event_id, model_kind="logistic", features=features)
    np.testing.assert_array_equal(a.going_probs, b.going_probs)


def test_constant_columns_are_dropped(tiny_dataset):
    features = build_features(tiny_dataset).assign(always=1.0, never=np.nan)
    columns = informative_columns(features)
    assert "always" not in columns and "never" not in columns
    assert "lead_days" in columns


def test_event_without_rsvps(tiny_dataset):
    ds = tiny_dataset
    last = ds.events.iloc[-1]
    empty = last.copy()
    empty["event_id"], empty["start"] = "pymeets-empty", last["start"] + pd.Timedelta(days=30)
    ds = dataclasses.replace(ds, events=pd.concat([ds.events, empty.to_frame().T]))
    ds.events["start"] = pd.to_datetime(ds.events["start"]).dt.tz_convert(MANILA_TZ)

    fc = forecast_event(ds, "pymeets-empty", model_kind="logistic")
    assert fc.rsvp_heads == 0 and fc.expected == 0.0
    assert (fc.p10, fc.p50, fc.p90) == (0, 0, 0)
    np.testing.assert_array_equal(fc.pmf, [1.0])
    assert fc.going_probs.size == 0 and fc.waitlist_probs.size == 0


def test_drivers_are_short_aggregate_notes(tiny_dataset):
    event_id = tiny_dataset.upcoming_event_ids()[0]
    fc = forecast_event(tiny_dataset, event_id, model_kind="logistic")
    assert 1 <= len(fc.drivers) <= predict.MAX_DRIVERS
    people = set(tiny_dataset.rsvps["attendee_id"])
    for note in fc.drivers:
        assert isinstance(note, str) and len(note) < 160
        assert not any(person in note for person in people)


# --- drivers on a hand-made history ---------------------------------------------------------


def _history(n_events: int = 12, saturdays: bool = False) -> tuple[Dataset, pd.DataFrame]:
    """Fortnightly events, odd ones online, with show-ups that are easy to count by hand.

    Each event has 20 RSVPs: 5 regulars who always come, and 15 first-timers of whom 3 come
    online, 6 in heavy rain (in person, every 4th event) and 12 on dry days, minus 3 on a
    holiday (every 3rd event). With ``saturdays``, the dry in-person events are on Saturdays.
    """
    first = pd.Timestamp("2025-01-07 18:30", tz=MANILA_TZ)  # a Tuesday evening
    events, rsvps, features = [], [], []
    for i in range(n_events + 1):
        online, holiday = i % 2 == 1, i % 3 == 0
        rain = np.nan if online else (30.0 if i % 4 == 0 else 0.0)
        start = first + pd.Timedelta(weeks=2 * i)
        saturday = saturdays and rain == 0.0
        if saturday:
            start = start.replace(hour=14) + pd.Timedelta(days=4)
        first_timers_who_came = (3 if online else 6 if rain else 12) - 3 * holiday
        events.append(
            {
                "event_id": f"e{i:02d}",
                "title": f"Event {i}",
                "series": "WFH Pythonista" if online else "PyMeets",
                "start": start,
                "venue": "" if online else "Makati office",
                "city": "" if online else "Makati",
                "latitude": np.nan,
                "longitude": np.nan,
                "is_online": online,
                "capacity": np.nan,
                "food_provided": not online,
            }
        )
        for j in range(20):
            regular = j < 5
            came = regular or j - 5 < first_timers_who_came
            rsvps.append(
                {
                    "event_id": f"e{i:02d}",
                    "attendee_id": f"person-{i}-{j}",
                    "rsvp_at": start - pd.Timedelta(days=10),
                    "status": "going",
                    "guests": 0,
                    "attended": float(came) if i < n_events else np.nan,
                    "source": "synthetic",
                }
            )
            features.append(
                {
                    "prior_attended": 3.0 if regular else 0.0,
                    "lead_days": 20.0 if j >= 15 else 3.0,
                    "is_online": float(online),
                    "is_weekend": float(saturday),
                    "precip_mm": rain,
                    "is_holiday": float(holiday),
                }
            )
    ds = Dataset(events=pd.DataFrame(events), rsvps=pd.DataFrame(rsvps), synthetic=True)
    return ds, pd.DataFrame(features, index=ds.rsvps.index)


def _drivers(ds: Dataset, features: pd.DataFrame, event_id: str) -> list[str]:
    event = ds.event(event_id)
    past = predict.training_rsvps(ds, before=event["start"])
    return explain(ds, event, past, features[informative_columns(features.loc[past.index])])


def test_drivers_describe_the_history(monkeypatch):
    monkeypatch.setattr(build, "FEATURE_DESCRIPTIONS", {"is_holiday": "PH holiday"})
    ds, features = _history()
    notes = _drivers(ds, features, "e12")  # in person, heavy rain, a holiday
    assert len(notes) == predict.MAX_DRIVERS
    assert notes[0] == (
        "Regulars: 25% of these RSVPs have attended before; regulars show up 100% of the time "
        "vs 33% for first-timers"
    )
    assert "In person: past in-person events kept 65% of RSVPs vs 35% online (6 events)" in notes
    rain = "Rain: past events with ≥20 mm of rain kept 50% of RSVPs vs 80% on dry days (3 events)"
    assert rain in notes
    assert (
        "PH holiday: past events like this kept 40% of RSVPs vs 55% otherwise (4 events)" in notes
    )
    # The last 5 first-timers RSVP'd 20 days ahead. Only the two dry, non-holiday in-person
    # events had room for them, 2 each: 4 of 60 came, vs 116 of 180 who RSVP'd 3 days ahead.
    assert (
        "Lead time: 25% of these RSVPs came over 14 days ahead; early RSVPs show up 7% of the "
        "time vs 64% for last-week ones"
    ) in notes

    online = _drivers(ds, features, "e11")  # online, no rain data, not a holiday
    assert "Online: past online events kept 34% of RSVPs vs 65% in person (5 events)" in online
    assert not any(note.startswith(("Rain", "Dry day", "PH holiday")) for note in online)


def test_dry_days_and_saturdays():
    ds, features = _history(saturdays=True)
    notes = _drivers(ds, features, "e12")  # a rainy Tuesday evening
    slot = "Weekday evening: past weekday-evening events kept 50% of RSVPs vs 80% on Saturdays"
    assert f"{slot} (3 events)" in notes

    features.loc[ds.rsvps["event_id"] == "e12", "precip_mm"] = 0.0
    notes = _drivers(ds, features, "e12")
    dry = "Dry day: past dry days kept 80% of RSVPs vs 50% when it rained ≥20 mm (3 events)"
    assert dry in notes


def test_drivers_skip_features_the_model_lacks():
    ds, features = _history()
    notes = _drivers(ds, features.drop(columns=["prior_attended", "precip_mm"]), "e12")
    assert not any(note.startswith(("Regulars", "Rain", "Dry day")) for note in notes)
    assert any(note.startswith("In person") for note in notes)


def test_drivers_need_enough_events():
    ds, features = _history(n_events=4)
    notes = _drivers(ds, features, "e04")
    assert notes and all(note.startswith(("Regulars", "Lead time")) for note in notes)


@pytest.mark.tabpfn
def test_forecast_with_tabpfn_v2(tiny_dataset, monkeypatch):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    ds = tiny_dataset
    target = ds.labeled_event_ids()[8]
    fc = forecast_event(ds, target, model_kind="tabpfn")
    assert fc.model_name == "tabpfn-v2"
    assert np.all((fc.going_probs >= 0) & (fc.going_probs <= 1))
    assert fc.pmf.sum() == pytest.approx(1.0)
    assert 0 <= fc.p10 <= fc.p50 <= fc.p90 <= fc.rsvp_heads

    later = ds.rsvps.index[
        ds.rsvps["event_id"].map(ds.events.set_index("event_id")["start"])
        >= ds.event(target)["start"]
    ]
    flipped = with_attendance(ds, later, 1.0 - ds.rsvps.loc[later, "attended"])
    again = forecast_event(flipped, target, model_kind="tabpfn")
    np.testing.assert_array_equal(fc.going_probs, again.going_probs)
