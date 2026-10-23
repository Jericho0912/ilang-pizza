"""Forecast one event: P(show) for each RSVP, then the exact headcount distribution.

The show-up model only ever sees events that started strictly before the event being
forecast, so forecasting a past event replays what we would have said the night before.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ilang_pizza.features import build
from ilang_pizza.model import turnout
from ilang_pizza.model.backends import make_model
from ilang_pizza.schema import Dataset, TurnoutForecast

if TYPE_CHECKING:
    from ilang_pizza.features.weather import WeatherProvider

# With less history than this, a forecast is a guess dressed up as a number.
MIN_TRAIN_EVENTS = 3
MIN_TRAIN_RSVPS = 50


def forecast_event(
    ds: Dataset,
    event_id: str,
    *,
    model_kind: str = "tabpfn",
    weather: WeatherProvider | None = None,
    features: pd.DataFrame | None = None,
) -> TurnoutForecast:
    """Predict how many heads will show up to ``event_id``, from the events before it.

    Waitlisted RSVPs are scored as if released, so the planner can decide how many to let in.
    Pass ``features`` (from ``build_features``) to reuse them across calls.
    """
    event = ds.event(event_id)
    if features is None:
        features = build.build_features(ds, weather=weather)
    past = training_rsvps(ds, before=event["start"])
    _require_history(past, event)

    columns = informative_columns(features.loc[past.index])
    if not columns:
        raise ValueError(f"every feature is constant across the events before {event_id}")
    categorical = [c for c in build.CATEGORICAL_FEATURES if c in columns]
    model = make_model(model_kind, categorical_features=categorical)
    model.fit(features.loc[past.index, columns], past["attended"].to_numpy())

    rsvps = ds.rsvps_for(event_id)
    going = rsvps[rsvps["status"] == "going"]
    waiting = rsvps[rsvps["status"] == "waitlist"]
    # Score both groups in one call: every TabPFN prediction re-reads the whole context.
    scored = going.index.append(waiting.index)
    probs = model.predict_proba(features.loc[scored, columns]) if len(scored) else np.zeros(0)
    going_probs, waitlist_probs = probs[: len(going)], probs[len(going) :]
    going_heads = rsvp_heads(going)

    pmf = turnout.heads_pmf(going_probs, going_heads)
    return TurnoutForecast(
        event_id=event_id,
        model_name=model.name,
        going_probs=going_probs,
        going_heads=going_heads,
        waitlist_probs=waitlist_probs,
        waitlist_heads=rsvp_heads(waiting),
        pmf=pmf,
        expected=turnout.expected(pmf),
        p10=turnout.quantile(pmf, 0.10),
        p50=turnout.quantile(pmf, 0.50),
        p90=turnout.quantile(pmf, 0.90),
        historical_show_rate=show_rate(past),
        drivers=explain(ds, event, past, features[columns]),
        synthetic=ds.synthetic,
    )


def training_rsvps(ds: Dataset, *, before: pd.Timestamp) -> pd.DataFrame:
    """Going RSVPs with known attendance from events that started strictly before ``before``."""
    rs = ds.rsvps
    starts = rs["event_id"].map(ds.events.set_index("event_id")["start"])
    keep = (rs["status"] == "going") & rs["attended"].notna() & (starts < before)
    return rs[keep]


def informative_columns(frame: pd.DataFrame) -> list[str]:
    """Columns with more than one distinct value (NaN counts as a value)."""
    return [c for c in frame.columns if frame[c].nunique(dropna=False) > 1]


def rsvp_heads(rsvps: pd.DataFrame) -> np.ndarray:
    """Heads per RSVP: the person plus their guests."""
    return (1 + rsvps["guests"].fillna(0)).clip(lower=1).astype(int).to_numpy()


def show_rate(rsvps: pd.DataFrame) -> float:
    """Share of going heads that showed up (guests come and go with their RSVP)."""
    heads = rsvp_heads(rsvps)
    total = heads.sum()
    return float(rsvps["attended"].to_numpy() @ heads / total) if total else float("nan")


def _require_history(past: pd.DataFrame, event: pd.Series) -> None:
    n_events, n_rsvps = past["event_id"].nunique(), len(past)
    if n_events < MIN_TRAIN_EVENTS or n_rsvps < MIN_TRAIN_RSVPS:
        raise ValueError(
            f"not enough history before {event['title']}: a forecast needs check-ins from at "
            f"least {MIN_TRAIN_EVENTS} earlier events and {MIN_TRAIN_RSVPS} RSVPs, and there "
            f"are {n_events} events with {n_rsvps} RSVPs. Add attendance for more past events."
        )


# --- drivers: plain-English notes comparing this event with past ones ----------------------
#
# Each note splits past events (or past RSVPs) in two by a feature the model uses, compares
# their show-up rates, and says which side this event is on. A note needs enough events on
# both sides, reports aggregate rates only (never anything about a person), and is skipped
# when its feature is missing or constant in the training data. Biggest gaps come first.

MAX_DRIVERS = 5
MIN_EVENTS = 3  # per side of a comparison
MIN_RSVPS = 20  # per side of an RSVP-level comparison
MIN_GAP = 0.03  # smaller differences in show-up rate are noise, not a reason
HEAVY_RAIN_MM = 20.0  # same cut-off as the is_heavy_rain feature
DRY_MM = 1.0
EARLY_DAYS = 14  # an "early" RSVP came more than two weeks ahead,
LATE_DAYS = 7  # a "late" one in the final week


@dataclass(frozen=True)
class _Note:
    gap: float  # difference between the two show-up rates
    text: str


@dataclass(frozen=True)
class _History:
    """What the drivers compare: past RSVPs and events, and this event's going RSVPs."""

    event: pd.Series
    past: pd.DataFrame  # training RSVPs (going, labeled, before the event)
    past_x: pd.DataFrame  # their features (the columns the model uses)
    today_x: pd.DataFrame  # features of this event's going RSVPs
    events: pd.DataFrame  # one row per past event: shown, heads, start, is_online

    def kept(self, mask: pd.Series) -> float:
        """Share of going heads that showed up, pooled over the past events in ``mask``."""
        chosen = self.events[mask]
        return chosen["shown"].sum() / chosen["heads"].sum()

    def per_event(self, column: str) -> tuple[pd.Series, float]:
        """An event-level feature per past event, and its value for this event (may be NaN)."""
        values = self.past_x[column].groupby(self.past["event_id"]).median()
        return values.reindex(self.events.index), float(self.today_x[column].median())

    def compare(self, same: pd.Series, other: pd.Series) -> tuple[float, float, int] | None:
        """(rate where ``same``, rate where ``other``, events where ``same``) if supported."""
        if same.sum() < MIN_EVENTS or other.sum() < MIN_EVENTS:
            return None
        return self.kept(same), self.kept(other), int(same.sum())

    def compare_rsvps(self, same: pd.Series, other: pd.Series) -> tuple[float, float] | None:
        """Per-RSVP show-up rates of two groups of past RSVPs, if both are supported."""
        for group in (same, other):
            if group.sum() < MIN_RSVPS or self.past.loc[group, "event_id"].nunique() < MIN_EVENTS:
                return None
        return self.past.loc[same, "attended"].mean(), self.past.loc[other, "attended"].mean()


def explain(ds: Dataset, event: pd.Series, past: pd.DataFrame, features: pd.DataFrame) -> list[str]:
    """Up to MAX_DRIVERS notes on how conditions like this event's fared in the past.

    ``features`` holds the columns the model was given (``build_features`` output).
    """
    rs = ds.rsvps
    today = rs[(rs["event_id"] == event["event_id"]) & (rs["status"] == "going")]
    heads = pd.Series(rsvp_heads(past), index=past.index)
    events = pd.DataFrame({"shown": past["attended"] * heads, "heads": heads})
    events = events.groupby(past["event_id"]).sum()
    info = ds.events.set_index("event_id").reindex(events.index)
    events["start"], events["is_online"] = info["start"], info["is_online"].astype(bool)
    h = _History(event, past, features.loc[past.index], features.loc[today.index], events)

    notes = [
        note
        for column, driver in _DRIVERS
        if column in features.columns and (note := driver(h)) is not None
    ]
    notes = [note for note in notes + _flag_notes(h) if note.gap >= MIN_GAP]
    notes.sort(key=lambda note: note.gap, reverse=True)
    return [note.text for note in notes[:MAX_DRIVERS]]


def _regulars_note(h: _History) -> _Note | None:
    # "First-timers" here are people who have not come before, whether or not they RSVP'd.
    came = h.past_x["prior_attended"]
    rates = h.compare_rsvps(came > 0, came == 0)  # unknown people are in neither group
    today = h.today_x["prior_attended"].dropna()
    if rates is None or today.empty:
        return None
    a, b = rates
    text = (
        f"Regulars: {(today > 0).mean():.0%} of these RSVPs have attended before; regulars "
        f"show up {a:.0%} of the time vs {b:.0%} for first-timers"
    )
    return _Note(abs(a - b), text)


def _lead_note(h: _History) -> _Note | None:
    lead = h.past_x["lead_days"]
    rates = h.compare_rsvps(lead > EARLY_DAYS, lead <= LATE_DAYS)
    today = h.today_x["lead_days"].dropna()
    if rates is None or today.empty:
        return None
    a, b = rates
    text = (
        f"Lead time: {(today > EARLY_DAYS).mean():.0%} of these RSVPs came over {EARLY_DAYS} "
        f"days ahead; early RSVPs show up {a:.0%} of the time vs {b:.0%} for last-week ones"
    )
    return _Note(abs(a - b), text)


def _online_note(h: _History) -> _Note | None:
    online = bool(h.event["is_online"])
    rates = h.compare(h.events["is_online"] == online, h.events["is_online"] != online)
    if rates is None:
        return None
    a, b, n = rates
    if online:
        text = f"Online: past online events kept {a:.0%} of RSVPs vs {b:.0%} in person"
    else:
        text = f"In person: past in-person events kept {a:.0%} of RSVPs vs {b:.0%} online"
    return _Note(abs(a - b), f"{text} ({n} events)")


def _slot(start: pd.Timestamp) -> str | None:
    if start.weekday() == 5:
        return "Saturday"
    if start.weekday() < 5 and start.hour >= 17:
        return "weekday evening"
    return None


def _slot_note(h: _History) -> _Note | None:
    slot = _slot(h.event["start"])
    if slot is None:
        return None
    # Compare like with like: online and in-person events keep their RSVPs very differently.
    alike = h.events["is_online"] == bool(h.event["is_online"])
    slots = h.events["start"].map(_slot)
    other = "weekday evening" if slot == "Saturday" else "Saturday"
    rates = h.compare(alike & (slots == slot), alike & (slots == other))
    if rates is None:
        return None
    a, b, n = rates
    text = (
        f"{slot.capitalize()}: past {slot.replace(' ', '-')} events kept {a:.0%} of RSVPs "
        f"vs {b:.0%} on {other}s ({n} events)"
    )
    return _Note(abs(a - b), text)


def _rain_note(h: _History) -> _Note | None:
    rain, today = h.per_event("precip_mm")
    heavy, dry = rain >= HEAVY_RAIN_MM, rain < DRY_MM  # unknown rain is neither
    mm = f"≥{HEAVY_RAIN_MM:g} mm"
    if today >= HEAVY_RAIN_MM:
        rates = h.compare(heavy, dry)
        label, versus = f"Rain: past events with {mm} of rain", "on dry days"
    elif today < DRY_MM:
        rates = h.compare(dry, heavy)
        label, versus = "Dry day: past dry days", f"when it rained {mm}"
    else:  # light or unknown rain: no clear comparison
        return None
    if rates is None:
        return None
    a, b, n = rates
    return _Note(abs(a - b), f"{label} kept {a:.0%} of RSVPs vs {b:.0%} {versus} ({n} events)")


def _payday_note(h: _History) -> _Note | None:
    petsa, today = h.per_event("is_petsa_de_peligro")
    if today != 1:
        return None
    rates = h.compare(petsa == 1, petsa == 0)
    if rates is None:
        return None
    a, b, n = rates
    text = (
        f"Petsa de peligro: past events in the days before payday kept {a:.0%} of RSVPs "
        f"vs {b:.0%} otherwise ({n} events)"
    )
    return _Note(abs(a - b), text)


# (feature the note explains, note). A note is skipped when the model lacks its feature.
_DRIVERS: tuple[tuple[str, Callable[[_History], _Note | None]], ...] = (
    ("prior_attended", _regulars_note),
    ("lead_days", _lead_note),
    ("is_online", _online_note),
    ("is_weekend", _slot_note),
    ("precip_mm", _rain_note),
    ("is_petsa_de_peligro", _payday_note),
)
# Event-level features the notes above already cover, directly or by derivation.
_COVERED = (
    "is_online",
    "is_weekend",
    "start_hour",
    "precip_mm",
    "is_heavy_rain",
    "is_petsa_de_peligro",
    "days_to_payday",
)


def _flag_notes(h: _History) -> list[_Note]:
    """Notes for other yes/no event features that hold for this event (holiday, ...)."""
    descriptions = getattr(build, "FEATURE_DESCRIPTIONS", {})
    covered = [h.per_event(c)[0] for c in _COVERED if c in h.past_x.columns]
    notes = []
    for column in h.past_x.columns:
        if column in _COVERED:
            continue
        flag, today = h.per_event(column)
        within = h.past_x[column].groupby(h.past["event_id"]).nunique(dropna=False)
        if today != 1 or within.max() > 1 or not flag.dropna().isin((0.0, 1.0)).all():
            continue  # not an event-level yes/no feature that is "yes" for this event
        if any(flag.equals(c) or flag.equals(1 - c) for c in covered):
            continue  # says the same as a note above (e.g. food is only served in person)
        rates = h.compare(flag == 1, flag == 0)
        if rates is None:
            continue
        a, b, n = rates
        label = descriptions.get(column) or column.removeprefix("is_").replace("_", " ")
        text = f"{label[:1].upper()}{label[1:]}: past events like this kept {a:.0%} of RSVPs"
        notes.append(_Note(abs(a - b), f"{text} vs {b:.0%} otherwise ({n} events)"))
    return notes
