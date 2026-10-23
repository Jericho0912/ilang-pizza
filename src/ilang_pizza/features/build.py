"""Feature matrix for the show-up model: one row per RSVP, float64 columns, NaN allowed.

Everything here is known before the event starts:

* rsvp: when and how early the person signed up, and how many guests they bring.
* history: the attendee's record at events that started strictly earlier. An event's own
  attendance never feeds its features.
* event: series, timing, RSVP headcount, capacity and food, built from RSVPs only.
* calendar: paydays, PH holidays, long weekends and the rainy season (``calendar_ph``).
* weather: rain on the day, from ``Dataset.weather`` or a ``WeatherProvider``.

NaN means unknown: no RSVP time, an anonymous RSVP (no attendee id), no rain data, or no
capacity limit. Binary columns are 1.0 / 0.0.

Because history only looks backwards, features built once on the full dataset are also valid
for rolling-origin backtests: no row ever sees a label from its own event or a later one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ilang_pizza.config import METRO_MANILA
from ilang_pizza.features import calendar_ph
from ilang_pizza.schema import MANILA_TZ, SERIES, Dataset

if TYPE_CHECKING:
    from ilang_pizza.features.weather import WeatherProvider

# Feature columns by kind. Event, calendar and weather columns are the same for every RSVP of
# an event; rsvp and history columns vary within it.
FEATURE_GROUPS: dict[str, list[str]] = {
    "rsvp": ["lead_days", "rsvp_hour", "guests", "rsvp_order_pct", "is_late_rsvp"],
    "history": [
        "prior_rsvps",
        "prior_attended",
        "prior_no_shows",
        "prior_show_rate",
        "is_first_timer",
        "days_since_last_attended",
    ],
    "event": [
        "series_code",
        "is_online",
        "is_weekend",
        "start_hour",
        "month",
        "event_rsvp_heads",
        "capacity_ratio",
        "has_waitlist",
        "food_provided",
    ],
    "calendar": [
        "days_to_payday",
        "is_petsa_de_peligro",
        "is_holiday",
        "is_long_weekend",
        "is_wet_season",
    ],
    "weather": ["precip_mm", "is_heavy_rain"],
}

FEATURE_COLUMNS: list[str] = [column for group in FEATURE_GROUPS.values() for column in group]

# Integer-coded categorical columns (subset of FEATURE_COLUMNS). series_code indexes SERIES.
CATEGORICAL_FEATURES: list[str] = ["series_code"]

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "lead_days": "days between RSVP and event start",
    "rsvp_hour": "hour of day the RSVP came in",
    "guests": "extra guests on the RSVP",
    "rsvp_order_pct": "RSVP order (0 = first in, 1 = last in)",
    "is_late_rsvp": "RSVP'd less than a day before",
    "prior_rsvps": "past RSVPs with known attendance",
    "prior_attended": "past events attended",
    "prior_no_shows": "past no-shows",
    "prior_show_rate": "past show-up rate (smoothed)",
    "is_first_timer": "first-timer (no earlier RSVPs)",
    "days_since_last_attended": "days since they last showed up",
    "series_code": "event series",
    "is_online": "online event",
    "is_weekend": "weekend event",
    "start_hour": "start hour (24h)",
    "month": "month (1-12)",
    "event_rsvp_heads": "heads RSVP'd going (incl. guests)",
    "capacity_ratio": "RSVP heads per venue seat",
    "has_waitlist": "event has a waitlist",
    "food_provided": "food provided",
    "days_to_payday": "days until payday",
    "is_petsa_de_peligro": "petsa de peligro (the days just before payday)",
    "is_holiday": "PH holiday",
    "is_long_weekend": "part of a long weekend",
    "is_wet_season": "rainy season (June to November)",
    "precip_mm": "rain on the day (mm)",
    "is_heavy_rain": "heavy rain day (20 mm or more)",
}

LATE_RSVP_DAYS = 1.0
# A day with 20 mm of rain or more is a heavy-rain day: the ETCCDI "very heavy precipitation
# day" index (R20mm), a common daily proxy. PAGASA's colour-coded rainfall warnings are hourly
# rates, which a daily total cannot express.
HEAVY_RAIN_MM = 20.0

_SERIES_CODES = {name: code for code, name in enumerate(SERIES)}
_SECONDS_PER_DAY = 86_400.0
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def build_features(ds: Dataset, *, weather: WeatherProvider | None = None) -> pd.DataFrame:
    """Features for every RSVP in ``ds``, aligned with ``ds.rsvps.index``.

    Rain comes from ``ds.weather`` by date first, then from ``weather`` (at the event's
    coordinates, or Metro Manila for in-person events without them), else NaN. Online events
    always get NaN rain.
    """
    if ds.rsvps.empty:
        return pd.DataFrame(index=ds.rsvps.index, columns=FEATURE_COLUMNS, dtype=np.float64)
    rsvps = ds.rsvps.reset_index(drop=True)
    events = ds.events.drop_duplicates("event_id").set_index("event_id")
    events = events[events.index.isin(rsvps["event_id"])]  # skip events nobody RSVP'd to
    event_start = _to_manila(events["start"])
    start = rsvps["event_id"].map(event_start)
    if start.isna().any():
        missing = sorted(set(rsvps.loc[start.isna(), "event_id"].astype(str)))
        raise ValueError(f"RSVPs point at unknown events or events with no start: {missing[:5]}")

    per_event = _event_features(events, event_start, rsvps, ds.weather, weather)
    features = pd.concat(
        [
            _rsvp_features(rsvps, start),
            _history_features(rsvps, start),
            per_event.reindex(rsvps["event_id"]).reset_index(drop=True),
        ],
        axis=1,
    )
    features.index = ds.rsvps.index
    return features[FEATURE_COLUMNS].astype(np.float64)


def _rsvp_features(rsvps: pd.DataFrame, start: pd.Series) -> pd.DataFrame:
    """When each RSVP came in, relative to its event and to the event's other RSVPs."""
    rsvp_at = _to_manila(rsvps["rsvp_at"])
    lead_days = (start - rsvp_at).dt.total_seconds() / _SECONDS_PER_DAY
    signups = _epoch_seconds(rsvp_at).groupby(rsvps["event_id"])
    # Ties share a rank, unknown RSVP times stay NaN, and a lone RSVP counts as the first.
    order = (signups.rank(method="average") - 1) / np.maximum(signups.transform("count") - 1, 1)
    return pd.DataFrame(
        {
            "lead_days": lead_days,
            "rsvp_hour": rsvp_at.dt.hour,
            "guests": _guests(rsvps),
            "rsvp_order_pct": order,
            "is_late_rsvp": _flag(lead_days < LATE_RSVP_DAYS, known=lead_days.notna()),
        }
    )


def _history_features(rsvps: pd.DataFrame, start: pd.Series) -> pd.DataFrame:
    """Each attendee's record at events that started strictly before this RSVP's event.

    The record counts going RSVPs with known attendance. Running totals per person and event
    start are joined as-of each RSVP's own start, excluding that start itself.
    """
    person = _attendee_ids(rsvps["attendee_id"])
    known_person = person.notna()
    t = _epoch_seconds(start)  # event start
    came = pd.to_numeric(rsvps["attended"], errors="coerce")
    labeled = known_person & rsvps["status"].eq("going") & came.notna()

    # Running totals per person, one row per event start they have a record at.
    record = (
        pd.DataFrame({"person": person[labeled], "t": t[labeled], "came": came[labeled]})
        .groupby(["person", "t"], as_index=False)
        .agg(rsvps=("came", "size"), attended=("came", "sum"))
    )
    by_person = record.groupby("person")
    record["prior_rsvps"] = by_person["rsvps"].cumsum()
    record["prior_attended"] = by_person["attended"].cumsum()
    attended_t = record["t"].where(record["attended"] > 0)
    record["last_attended_t"] = attended_t.groupby(record["person"]).ffill()

    # Each RSVP takes the totals of its person's latest event strictly before its own start.
    # allow_exact_matches=False keeps out the event itself and any event at the same time.
    keys = pd.DataFrame({"person": person, "t": t, "row": np.arange(len(rsvps))})
    asof = pd.merge_asof(
        keys[known_person].sort_values("t"),
        record.drop(columns=["rsvps", "attended"]).sort_values("t"),
        on="t",
        by="person",
        allow_exact_matches=False,
    )
    asof = asof.set_index("row").reindex(keys["row"])

    prior_rsvps = asof["prior_rsvps"].fillna(0.0).where(known_person)
    prior_attended = asof["prior_attended"].fillna(0.0).where(known_person)
    first_seen = t.groupby(person).transform("min")
    return pd.DataFrame(
        {
            "prior_rsvps": prior_rsvps,
            "prior_attended": prior_attended,
            "prior_no_shows": prior_rsvps - prior_attended,
            # Laplace smoothing: 0.5 with no record, so no global prior is needed.
            "prior_show_rate": (prior_attended + 1) / (prior_rsvps + 2),
            # Any earlier RSVP counts here, labeled or not, going or waitlisted.
            "is_first_timer": _flag(first_seen >= t, known=known_person),
            "days_since_last_attended": (t - asof["last_attended_t"]) / _SECONDS_PER_DAY,
        }
    )


def _event_features(
    events: pd.DataFrame,
    start: pd.Series,
    rsvps: pd.DataFrame,
    rain_table: pd.DataFrame | None,
    provider: WeatherProvider | None,
) -> pd.DataFrame:
    """Event, calendar and weather columns, one row per event. Uses RSVPs, never attendance."""
    going_heads = (
        (1 + _guests(rsvps))
        .where(rsvps["status"].eq("going"), 0.0)
        .groupby(rsvps["event_id"])
        .sum()
        .reindex(events.index, fill_value=0.0)
    )
    has_waitlist = (
        rsvps["status"]
        .eq("waitlist")
        .groupby(rsvps["event_id"])
        .any()
        .reindex(events.index, fill_value=False)
    )
    capacity = pd.to_numeric(events["capacity"], errors="coerce")
    day = start.dt.date
    rain = _event_rain(events, day, rain_table, provider)
    return pd.DataFrame(
        {
            "series_code": events["series"].map(_SERIES_CODES).fillna(_SERIES_CODES["Other"]),
            "is_online": events["is_online"].astype(float),
            "is_weekend": start.dt.day_of_week >= 5,
            "start_hour": start.dt.hour,
            "month": start.dt.month,
            "event_rsvp_heads": going_heads,
            "capacity_ratio": going_heads / capacity.where(capacity > 0),  # NaN when unlimited
            "has_waitlist": has_waitlist,
            "food_provided": events["food_provided"].astype(float),
            "days_to_payday": day.map(calendar_ph.days_to_next_payday),
            "is_petsa_de_peligro": day.map(calendar_ph.is_petsa_de_peligro),
            "is_holiday": day.map(calendar_ph.is_ph_holiday),
            "is_long_weekend": day.map(calendar_ph.is_long_weekend),
            "is_wet_season": day.map(calendar_ph.is_wet_season),
            "precip_mm": rain,
            "is_heavy_rain": _flag(rain >= HEAVY_RAIN_MM, known=rain.notna()),
        },
        index=events.index,
    )


def _event_rain(
    events: pd.DataFrame,
    day: pd.Series,
    rain_table: pd.DataFrame | None,
    provider: WeatherProvider | None,
) -> pd.Series:
    """Rain per event in mm: the ``Dataset.weather`` table, then the provider, else NaN."""
    rain = pd.Series(np.nan, index=events.index)
    if rain_table is not None and len(rain_table):
        rain = day.map(_rain_by_date(rain_table)).astype(float)
    online = events["is_online"].eq(True)
    rain = rain.mask(online)  # online events never get rain
    ask = rain.isna() & ~online
    if provider is not None and ask.any():
        lat = pd.to_numeric(events.loc[ask, "latitude"], errors="coerce")
        lon = pd.to_numeric(events.loc[ask, "longitude"], errors="coerce")
        placeless = lat.isna() | lon.isna()
        lat, lon = lat.mask(placeless, METRO_MANILA[0]), lon.mask(placeless, METRO_MANILA[1])
        points = list(zip(lat, lon, day[ask], strict=True))
        prefetch = getattr(provider, "prefetch", None)
        if callable(prefetch):
            prefetch(points)  # one batched download instead of a request per event
        rain[ask] = [_as_float(provider.precipitation_mm(*point)) for point in points]
    return rain


def _rain_by_date(table: pd.DataFrame) -> pd.Series:
    """``Dataset.weather`` as a date -> mm lookup. Duplicate dates are averaged."""
    dates = pd.to_datetime(table["date"], errors="coerce").dt.date
    mm = pd.to_numeric(table["precipitation_mm"], errors="coerce")
    return mm.groupby(dates.to_numpy()).mean()


def _as_float(value: float | None) -> float:
    return np.nan if value is None else float(value)


def _guests(rsvps: pd.DataFrame) -> pd.Series:
    """Extra heads per RSVP; blank or invalid counts as none."""
    return pd.to_numeric(rsvps["guests"], errors="coerce").fillna(0.0).clip(lower=0.0)


def _attendee_ids(ids: pd.Series) -> pd.Series:
    """Attendee ids with blanks as missing: an anonymous RSVP has no history to link."""
    return ids.mask(ids.isna() | ids.astype(str).str.strip().eq(""))


def _flag(condition: pd.Series, *, known: pd.Series) -> pd.Series:
    """1.0 or 0.0 where ``known``, NaN elsewhere."""
    return condition.astype(float).where(known)


def _to_manila(values: pd.Series) -> pd.Series:
    """Datetimes in Asia/Manila. Naive values are read as Manila local time."""
    stamps = pd.to_datetime(values)
    if stamps.dt.tz is None:
        return stamps.dt.tz_localize(MANILA_TZ)
    return stamps.dt.tz_convert(MANILA_TZ)


def _epoch_seconds(stamps: pd.Series) -> pd.Series:
    """Seconds since 1970 as floats (NaN for NaT), whatever the datetime resolution."""
    return (stamps - _EPOCH).dt.total_seconds()
