"""Synthetic, PythonPH-shaped demo data: fake people, plausible turnout.

``make_synthetic_dataset`` builds about five years of roughly monthly events (PyMeets, PyWorks,
WFH Pythonista, PyKids) with RSVPs from a pool of 2,500 fake members and daily Metro Manila rain,
then draws attendance from the ground-truth model below, so the demo has real signal to learn.
The output is deterministic given ``today`` and ``seed``. Nothing here describes real people or
real past events: titles start with "Demo" and ``Dataset.synthetic`` is True.

Ground truth: a going RSVP shows up with probability ``sigmoid(logit)``, where the logit is a
base of +0.70 plus these effects (log-odds). In-person show rates land near 53%, online near 40%.

    effect                                      log-odds    why
    member reliability                          N(0, 0.8)   some people always come, some never do
    loyalty (per sd of RSVP frequency)          +0.35       regulars are the dependable core
    each earlier show-up (up to 5)              +0.12       habit
    each earlier no-show (up to 3)              -0.15
    first RSVP ever                             -0.60       first-timers flake more
    RSVP'd more than 14 days ahead              -0.40       very early RSVPs are cheap
    RSVP'd 7-14 days ahead                      -0.15
    RSVP'd in the last 24 hours                 +0.25       last-minute RSVPs mean it
    online event                                -0.95       a Zoom link is easy to skip
    weekday evening in BGC or Makati            -0.45       rush hour on EDSA
    weekday evening elsewhere                   -0.20
    rain that day: 10-25 / 25-50 / 50+ mm       -0.3 / -0.8 / -1.4, x1.3 at rush hour
    petsa de peligro (1-5 days before payday)   -0.15       transport money is tight
    food provided                               +0.15
    holiday, long weekend, or the eve of one    -0.45       people go home to the province
                                                            (online: -0.20)
    event noise (speaker, topic)                N(0, 0.25)
    RSVP noise                                  N(0, 0.35)

Rain, traffic, payday and food effects apply to in-person events only. Paydays are the 15th and
the last day of the month (*kinsenas at katapusan*). Rain follows Manila's climate: a wet season
from June to November, a dry season from December to May, and a couple of typhoon spells a year.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from ilang_pizza.data.loaders import (
    SYNTHETIC_MARKER,
    canonical_events,
    canonical_rsvps,
    canonical_weather,
    hash_identity,
)
from ilang_pizza.schema import EVENT_COLUMNS, MANILA_TZ, RSVP_COLUMNS, Dataset

POOL_SIZE = 2500

# Log-odds effects, documented in the module docstring.
_BASE = 0.70
_ONLINE = -0.95
_LOYALTY = 0.35
_PER_SHOW, _MAX_SHOWS = 0.12, 5
_PER_NO_SHOW, _MAX_NO_SHOWS = -0.15, 3
_FIRST_TIMER = -0.60
_VERY_EARLY, _EARLY, _LAST_MINUTE = -0.40, -0.15, 0.25  # >14 days, 7-14 days, <1 day ahead
_TRAFFIC_BGC_MAKATI, _TRAFFIC_ELSEWHERE = -0.45, -0.20
_RAIN_STEPS = ((50.0, -1.4), (25.0, -0.8), (10.0, -0.3))  # (at least mm, effect)
_RUSH_HOUR_RAIN = 1.3
_PETSA_DE_PELIGRO = -0.15
_FOOD = 0.15
_HOLIDAY_IN_PERSON, _HOLIDAY_ONLINE = -0.45, -0.20
_EVENT_NOISE, _RSVP_NOISE = 0.25, 0.35


@dataclass(frozen=True)
class _Venue:
    name: str
    city: str
    area: str
    latitude: float
    longitude: float


_PYMEETS_VENUES = (
    _Venue("Sponsor office, 5th Avenue", "Taguig", "BGC", 14.5509, 121.0503),
    _Venue("Coworking space, Bonifacio High Street", "Taguig", "BGC", 14.5508, 121.0525),
    _Venue("Sponsor office, Ayala Avenue", "Makati", "Makati", 14.5547, 121.0244),
    _Venue("Coworking space, Salcedo Village", "Makati", "Makati", 14.5605, 121.0212),
    _Venue("Sponsor office, Ortigas Center", "Pasig", "Ortigas", 14.5866, 121.0614),
)
_PYWORKS_VENUES = (
    _Venue("UP-Ayala Land TechnoHub", "Quezon City", "Quezon City", 14.6575, 121.0570),
    _Venue("Coworking space, Eastwood City", "Quezon City", "Quezon City", 14.6091, 121.0780),
)
_PYKIDS_VENUES = (
    _Venue("Community learning center, Diliman", "Quezon City", "Quezon City", 14.6549, 121.0645),
)


@dataclass(frozen=True)
class _Style:
    """How a series usually runs."""

    weekdays: tuple[int, ...]  # Monday is 0
    times: tuple[str, ...]
    venues: tuple[_Venue, ...]  # empty: online
    capacities: tuple[int, ...]  # empty: unlimited
    food_rate: float
    topics: tuple[str, ...]


_STYLES = {
    "PyMeets": _Style(
        weekdays=(1, 2, 3),
        times=("18:30", "18:30", "19:00"),
        venues=_PYMEETS_VENUES,
        capacities=(40, 50, 60, 80, 100),
        food_rate=0.9,
        topics=(
            "Typing in Practice",
            "Async Without Tears",
            "Packaging Your First Library",
            "Faster Pandas",
            "Testing with pytest",
            "APIs with FastAPI",
            "Polars for Pandas Users",
            "Django in Production",
            "Profiling Slow Code",
            "Data Validation with Pydantic",
            "Rust Extensions for Python",
            "Local LLMs on a Laptop",
            "Geospatial Python",
            "Observability for Python Services",
            "Writing Friendly CLIs",
            "Python in Fintech",
            "Free-threaded Python",
            "Open-Sourcing a Side Project",
            "SQL for Pythonistas",
        ),
    ),
    "PyWorks": _Style(
        weekdays=(5,),
        times=("13:00",),
        venues=_PYWORKS_VENUES,
        capacities=(40, 50, 60, 80),
        food_rate=0.7,
        topics=(
            "Hands-on Data Cleaning",
            "Build a Web Scraper",
            "Your First Django App",
            "Machine Learning Basics",
            "Git and GitHub for Pythonistas",
            "Dashboards with Streamlit",
            "Automate the Office Spreadsheet",
            "Jupyter from Zero",
            "Deploy a Flask App",
        ),
    ),
    "WFH Pythonista": _Style(
        weekdays=(5,),
        times=("19:00", "20:00"),
        venues=(),
        capacities=(),
        food_rate=0.0,
        topics=(
            "Live Coding a CLI",
            "Lightning Talks Night",
            "Python Careers AMA",
            "Code Review Clinic",
            "Data Viz Show and Tell",
            "Remote Pairing Session",
            "Notebook Tips and Tricks",
        ),
    ),
    "PyKids": _Style(
        weekdays=(5,),
        times=("14:00",),
        venues=_PYKIDS_VENUES,
        capacities=(40, 50),
        food_rate=0.85,
        topics=(
            "Make a Game with Turtle",
            "Code Your First Story",
            "Drawing with Code",
            "Python and Micro:bit",
        ),
    ),
}
_PAST_MIX = {"PyMeets": 0.5, "PyWorks": 0.2, "WFH Pythonista": 0.2, "PyKids": 0.1}
_UPCOMING_CYCLE = ("PyMeets", "PyWorks", "WFH Pythonista", "PyKids")

# Manila (Port Area) climate: share of rainy days and mean mm per rainy day, January first.
_RAIN_DAY_SHARE = np.array([0.12, 0.08, 0.07, 0.10, 0.32, 0.57, 0.74, 0.74, 0.73, 0.58, 0.43, 0.26])
_RAIN_MEAN_MM = np.array([4, 4, 5, 7, 14, 15, 18, 20, 16, 13, 10, 7], dtype=float)
_TYPHOON_START = np.array([0, 0, 0, 0, 0.002, 0.006, 0.012, 0.015, 0.012, 0.010, 0.006, 0.002])


@dataclass(frozen=True)
class _Event:
    event_id: str
    title: str
    series: str
    start: pd.Timestamp
    venue: _Venue | None
    capacity: float
    food: bool
    past: bool


@dataclass
class _Members:
    """The fake member pool and what each person has done so far."""

    affinity: np.ndarray  # how often they RSVP: heavy-tailed, a loyal core and many one-timers
    reliability: np.ndarray  # latent log-odds of showing up
    rsvps: np.ndarray
    shows: np.ndarray
    no_shows: np.ndarray
    last_rsvp: np.ndarray  # day number of their latest RSVP, NaN if none
    joined: int = 0  # members [0, joined) have RSVPed at least once


def make_synthetic_dataset(
    n_past: int = 60,
    n_upcoming: int = 2,
    seed: int = 2026,
    today: pd.Timestamp | None = None,
) -> Dataset:
    """Build a deterministic demo :class:`Dataset` ending near ``today`` (default: now in Manila).

    Past events have known attendance. Upcoming events fall 1-3 weeks after ``today`` and hold
    the RSVPs collected so far, with ``attended`` NaN.
    """
    if n_past < 0 or n_upcoming < 0 or n_past + n_upcoming == 0:
        raise ValueError("need at least one event, and no negative counts")
    today = _manila_midnight(today)
    plan_rng, weather_rng, people_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    )
    events = _plan_events(n_past, n_upcoming, today, plan_rng)
    weather = _make_weather(events[0].start.date(), events[-1].start.date(), weather_rng)
    rain = dict(zip(weather["date"], weather["precipitation_mm"]))
    calendar = _Calendar(range(events[0].start.year - 1, events[-1].start.year + 2))
    salt = f"ilang-pizza-demo-{seed}"
    rsvps = _simulate_rsvps(events, rain, calendar, today, people_rng, salt)

    rows = pd.DataFrame([_event_row(e) for e in events], columns=EVENT_COLUMNS)
    events_frame = canonical_events(rows)
    dataset = Dataset(
        events=events_frame,
        rsvps=canonical_rsvps(rsvps, events_frame),
        synthetic=True,
        weather=canonical_weather(weather),
    )
    dataset.validate()
    return dataset


def _manila_midnight(today: pd.Timestamp | str | dt.date | None) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz=MANILA_TZ) if today is None else pd.Timestamp(today)
    stamp = stamp.tz_localize(MANILA_TZ) if stamp.tzinfo is None else stamp.tz_convert(MANILA_TZ)
    return stamp.normalize()


# --- events -----------------------------------------------------------------------------------


def _plan_events(
    n_past: int, n_upcoming: int, today: pd.Timestamp, rng: np.random.Generator
) -> list[_Event]:
    """Roughly monthly past events ending ~2 weeks before today, then 1-3 weeks of upcoming."""
    today_date = today.date()
    names = list(_PAST_MIX)
    plans: list[tuple[str, dt.date, bool]] = []
    for k in range(n_past):
        series = names[rng.choice(len(names), p=list(_PAST_MIX.values()))]
        days_back = (n_past - k - 0.5) * 30.44 + rng.uniform(-6, 6)
        target = today_date - dt.timedelta(days=round(days_back))
        plans.append((series, _nearest_day(target, _STYLES[series].weekdays), True))
    taken = {day for _, day, _ in plans}
    first, last = today_date + dt.timedelta(days=7), today_date + dt.timedelta(days=21)
    for j in range(n_upcoming):
        series = _UPCOMING_CYCLE[j % len(_UPCOMING_CYCLE)]
        target = first + dt.timedelta(days=round(14 * j / max(n_upcoming - 1, 1)))
        day = _nearest_day(target, _STYLES[series].weekdays, lo=first, hi=last, taken=taken)
        taken.add(day)
        plans.append((series, day, False))
    plans.sort(key=lambda plan: plan[1])

    topics = {name: list(rng.permutation(style.topics)) for name, style in _STYLES.items()}
    numbers = dict.fromkeys(_STYLES, 0)
    events = []
    for series, day, past in plans:
        style = _STYLES[series]
        numbers[series] += 1
        topic = topics[series][(numbers[series] - 1) % len(topics[series])]
        clock = dt.time.fromisoformat(str(rng.choice(style.times)))
        venue = style.venues[rng.integers(len(style.venues))] if style.venues else None
        # upcoming in-person events serve food: sizing that order is what the demo is for
        food = rng.random() < style.food_rate or (not past and venue is not None)
        slug = series.lower().replace(" ", "-")
        events.append(
            _Event(
                event_id=f"{slug}-{day:%Y-%m-%d}",
                title=f"Demo {series} #{numbers[series]}: {topic}",
                series=series,
                start=pd.Timestamp(dt.datetime.combine(day, clock), tz=MANILA_TZ),
                venue=venue,
                capacity=float(rng.choice(style.capacities)) if style.capacities else np.nan,
                food=bool(food),
                past=past,
            )
        )
    return events


def _nearest_day(
    target: dt.date,
    weekdays: tuple[int, ...],
    *,
    lo: dt.date = dt.date.min,
    hi: dt.date = dt.date.max,
    taken: Collection[dt.date] = (),
) -> dt.date:
    """The closest day to ``target`` on an allowed weekday, within [lo, hi] and not taken."""
    for distance in range(60):
        for day in (target + dt.timedelta(days=distance), target - dt.timedelta(days=distance)):
            if day.weekday() in weekdays and lo <= day <= hi and day not in taken:
                return day
    raise ValueError(f"no free day near {target}; ask for fewer upcoming events")


def _event_row(event: _Event) -> dict:
    venue = event.venue
    return {
        "event_id": event.event_id,
        "title": event.title,
        "series": event.series,
        "start": event.start,
        "venue": venue.name if venue else "",
        "city": venue.city if venue else "",
        "latitude": venue.latitude if venue else np.nan,
        "longitude": venue.longitude if venue else np.nan,
        "is_online": venue is None,
        "capacity": event.capacity,
        "food_provided": event.food,
    }


# --- weather and calendar ---------------------------------------------------------------------


def _make_weather(first: dt.date, last: dt.date, rng: np.random.Generator) -> pd.DataFrame:
    """Daily Metro Manila rain from ``first`` to ``last``, with the odd typhoon spell."""
    days = pd.date_range(first, last, freq="D")
    month = days.month.to_numpy() - 1
    rainy = rng.random(len(days)) < _RAIN_DAY_SHARE[month]
    mm = np.where(rainy, rng.gamma(0.8, _RAIN_MEAN_MM[month] / 0.8), 0.0)
    for start in np.flatnonzero(rng.random(len(days)) < _TYPHOON_START[month]):
        spell = slice(start, start + int(rng.integers(1, 4)))
        mm[spell] = np.maximum(mm[spell], rng.uniform(60, 220, size=len(mm[spell])))
    return pd.DataFrame({"date": days.date, "precipitation_mm": np.round(mm, 1)})


class _Calendar:
    """Philippine holidays, long weekends and paydays, for the ground-truth model."""

    def __init__(self, years: range):
        self.holidays = set(holidays.country_holidays("PH", years=years))

    def day_off(self, day: dt.date) -> bool:
        return day.weekday() >= 5 or day in self.holidays

    def in_long_weekend(self, day: dt.date) -> bool:
        """Part of three or more days off in a row."""
        if not self.day_off(day):
            return False
        run = 1
        for step in (1, -1):
            other = day + dt.timedelta(days=step)
            while self.day_off(other):
                run += 1
                other += dt.timedelta(days=step)
        return run >= 3

    def holidayish(self, day: dt.date) -> bool:
        """A holiday, a long weekend, or the evening before one."""
        tomorrow = day + dt.timedelta(days=1)
        return day in self.holidays or self.in_long_weekend(day) or self.in_long_weekend(tomorrow)

    @staticmethod
    def days_to_payday(day: dt.date) -> int:
        month_end = (pd.Timestamp(day) + pd.offsets.MonthEnd(0)).day
        return 15 - day.day if day.day <= 15 else month_end - day.day


def _event_effect(
    event: _Event, rain_mm: float, calendar: _Calendar, rng: np.random.Generator
) -> float:
    """The event-level part of the show-up logit (see the module docstring)."""
    day = event.start.date()
    noise = rng.normal(0, _EVENT_NOISE)
    if event.venue is None:
        return _ONLINE + (_HOLIDAY_ONLINE if calendar.holidayish(day) else 0.0) + noise
    effect = noise + (_FOOD if event.food else 0.0)
    rush_hour = day.weekday() < 5 and event.start.hour >= 17
    if rush_hour:
        effect += (
            _TRAFFIC_BGC_MAKATI if event.venue.area in ("BGC", "Makati") else _TRAFFIC_ELSEWHERE
        )
    rain = next((value for mm, value in _RAIN_STEPS if rain_mm >= mm), 0.0)
    effect += rain * (_RUSH_HOUR_RAIN if rush_hour else 1.0)
    if 1 <= calendar.days_to_payday(day) <= 5:
        effect += _PETSA_DE_PELIGRO
    if calendar.holidayish(day):
        effect += _HOLIDAY_IN_PERSON
    return effect


# --- people and RSVPs -------------------------------------------------------------------------


def _new_members(rng: np.random.Generator) -> _Members:
    log_affinity = rng.normal(0.0, 1.1, POOL_SIZE)
    reliability = _LOYALTY * log_affinity / 1.1 + rng.normal(0.0, 0.8, POOL_SIZE)
    zeros = np.zeros(POOL_SIZE, dtype=int)
    return _Members(
        affinity=np.exp(log_affinity),
        reliability=reliability,
        rsvps=zeros.copy(),
        shows=zeros.copy(),
        no_shows=zeros.copy(),
        last_rsvp=np.full(POOL_SIZE, np.nan),
    )


def _rsvp_count(event: _Event, rng: np.random.Generator) -> int:
    """RSVPs including the ones later withdrawn: ~30-150 in person, more online."""
    if event.venue is None:
        return int(rng.integers(120, 261))
    if event.series == "PyKids":  # most kids come with a parent, who takes a second seat
        return round(event.capacity * rng.uniform(0.45, 0.8))
    if event.past:
        return int(np.clip(round(event.capacity * rng.uniform(0.6, 1.6)), 30, 150))
    # the demo's upcoming events are popular, so a waitlist is already forming
    return int(np.clip(round(event.capacity * rng.uniform(1.8, 2.2)), 30, 180))


def _pick_attendees(
    members: _Members, n: int, online: bool, day_number: float, rng: np.random.Generator
) -> np.ndarray:
    """First-timers from the unused pool plus returning members weighted by engagement."""
    share_new = rng.uniform(0.35, 0.5) if online else rng.uniform(0.22, 0.38)
    n_new = min(round(n * share_new), POOL_SIZE - members.joined)
    n_back = min(n - n_new, members.joined)
    if n_back < n - n_new:  # early on there aren't enough regulars yet
        n_new = min(n - n_back, POOL_SIZE - members.joined)
    new = np.arange(members.joined, members.joined + n_new)
    members.joined += n_new
    seen = np.arange(members.joined - n_new)
    if n_back == 0:
        return new
    # engagement fades for people who haven't RSVPed in a while (a year's half-life, roughly)
    idle_days = day_number - members.last_rsvp[seen]
    weights = members.affinity[seen] * np.exp(-idle_days / 520.0)
    back = rng.choice(seen, size=n_back, replace=False, p=weights / weights.sum())
    return np.concatenate([new, back])


def _guests(event: _Event, n: int, rng: np.random.Generator) -> np.ndarray:
    if event.venue is None:
        return np.zeros(n, dtype=int)
    p = (0.15, 0.65, 0.20) if event.series == "PyKids" else (0.93, 0.06, 0.01)
    return rng.choice(3, size=n, p=p)


def _lead_days(n: int, rng: np.random.Generator) -> np.ndarray:
    """Days between RSVP and start: a burst at announcement, a trickle, a last-minute rush."""
    announced = rng.uniform(14, 28)
    kind = rng.random(n)
    lead = np.where(
        kind < 0.4,
        announced - rng.exponential(1.5, n),
        np.where(kind < 0.8, rng.uniform(0.5, announced, n), rng.exponential(0.8, n)),
    )
    return np.clip(lead, 0.02, announced)


def _show_up(
    members: _Members,
    people: np.ndarray,
    lead: np.ndarray,
    event_effect: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw who shows up, from the ground-truth model in the module docstring."""
    logit = (
        _BASE
        + event_effect
        + members.reliability[people]
        + _PER_SHOW * np.minimum(members.shows[people], _MAX_SHOWS)
        + _PER_NO_SHOW * np.minimum(members.no_shows[people], _MAX_NO_SHOWS)
        + np.where(members.rsvps[people] == 0, _FIRST_TIMER, 0.0)
        + np.select([lead > 14, lead > 7, lead <= 1], [_VERY_EARLY, _EARLY, _LAST_MINUTE], 0.0)
        + rng.normal(0, _RSVP_NOISE, len(people))
    )
    return rng.random(len(people)) < 1 / (1 + np.exp(-logit))


def _simulate_rsvps(
    events: list[_Event],
    rain: dict[dt.date, float],
    calendar: _Calendar,
    today: pd.Timestamp,
    rng: np.random.Generator,
    salt: str,
) -> pd.DataFrame:
    """RSVPs for every event in date order, updating each member's history as it goes."""
    members = _new_members(rng)
    ids = np.array([hash_identity(f"demo-member-{m}", salt) for m in range(POOL_SIZE)])
    frames = []
    for event in events:
        day_number = float((event.start.normalize() - today).days)
        n_rsvps = _rsvp_count(event, rng)
        people = _pick_attendees(members, n_rsvps, event.venue is None, day_number, rng)
        members.last_rsvp[people] = day_number
        n = len(people)
        lead = _lead_days(n, rng)
        rsvp_at = event.start - pd.to_timedelta(np.round(lead * 86400), unit="s")
        guests = _guests(event, n, rng)
        withdrawn = rng.random(n) < 0.08  # changed their RSVP to "No": not in the export
        collected = np.ones(n, dtype=bool) if event.past else np.asarray(rsvp_at < today)
        order = np.argsort(rsvp_at.asi8, kind="stable")
        keep = order[~withdrawn[order] & collected[order]]
        people, lead, guests, rsvp_at = people[keep], lead[keep], guests[keep], rsvp_at[keep]

        going = np.ones(len(people), dtype=bool)
        if not np.isnan(event.capacity):  # first come, first served; the rest wait
            going = np.cumsum(1 + guests) <= event.capacity
        attended = np.full(len(people), np.nan)
        if event.past:
            event_effect = _event_effect(event, rain.get(event.start.date(), 0.0), calendar, rng)
            showed = _show_up(members, people, lead, event_effect, rng)
            attended[going] = showed[going].astype(float)
            members.shows[people[going]] += showed[going]
            members.no_shows[people[going]] += ~showed[going]
        members.rsvps[people] += 1

        frames.append(
            pd.DataFrame(
                {
                    "event_id": event.event_id,
                    "attendee_id": ids[people],
                    "rsvp_at": rsvp_at,
                    "status": np.where(going, "going", "waitlist"),
                    "guests": guests,
                    "attended": attended,
                    "source": "synthetic",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[RSVP_COLUMNS]


# --- writing ----------------------------------------------------------------------------------

_SAFE_EVENT_ID = r"[A-Za-z0-9][A-Za-z0-9._-]*"

_README = """\
FAKE DEMO DATA, generated by `ilang-pizza demo`.

Every person, RSVP, event and rain reading in this folder is made up. The events are shaped
like Python Philippines meetups (PyMeets, PyWorks, WFH Pythonista, PyKids) so the forecasts
have something realistic to learn from, but none of them happened, and attendee_id values are
random hashes, not people.

  events.csv        {n_events} events, one row each
  rsvps/<id>.csv    {n_rsvps} RSVPs in canonical form, one file per event
  weather.csv       made-up daily rain for Metro Manila
  SYNTHETIC         marker: ilang-pizza labels every number from this folder as demo data

Delete this folder and run `ilang-pizza demo` to make a fresh one.
"""


def write_dataset(ds: Dataset, out_dir: Path) -> Path:
    """Write ``ds`` as a data folder that ``load_dataset`` reads back unchanged; returns it.

    Synthetic datasets also get a SYNTHETIC marker and a README saying the data is fake. A
    folder from an earlier demo is replaced; any other non-empty folder is left alone.
    """
    out = Path(out_dir)
    _clear_old_demo(out)
    rsvp_dir = out / "rsvps"
    rsvp_dir.mkdir(parents=True, exist_ok=True)

    bad = [e for e in ds.events["event_id"] if not re.fullmatch(_SAFE_EVENT_ID, e)]
    if bad:
        raise ValueError(f"event_id values must be file-name safe; got {bad[:3]}")

    events = ds.events[EVENT_COLUMNS].copy()
    events["start"] = _iso(events["start"])
    events["capacity"] = _whole_numbers(events["capacity"])
    events.to_csv(out / "events.csv", index=False)

    for event_id, rows in ds.rsvps.groupby("event_id", sort=False):
        rows = rows[RSVP_COLUMNS].copy()
        rows["rsvp_at"] = _iso(rows["rsvp_at"])
        rows["attended"] = _whole_numbers(rows["attended"])
        rows.to_csv(rsvp_dir / f"{event_id}.csv", index=False)

    if ds.weather is not None:
        weather = ds.weather[["date", "precipitation_mm"]].copy()
        weather["date"] = weather["date"].map(str)
        weather.to_csv(out / "weather.csv", index=False)
    if ds.synthetic:
        (out / SYNTHETIC_MARKER).write_text("Fake demo data. See README.txt.\n", encoding="utf-8")
        readme = _README.format(n_events=len(ds.events), n_rsvps=len(ds.rsvps))
        (out / "README.txt").write_text(readme, encoding="utf-8")
    return out


def _clear_old_demo(out: Path) -> None:
    """Remove files a previous demo wrote, so stale events don't linger; refuse other data."""
    if not out.exists():
        return
    visible = [p for p in out.iterdir() if not p.name.startswith(".")]
    if not visible:
        return
    if not (out / SYNTHETIC_MARKER).exists():
        raise FileExistsError(
            f"{out} already has files and isn't a demo folder (no {SYNTHETIC_MARKER} marker). "
            "Refusing to overwrite it; pick a new or empty folder."
        )
    written = ["events.csv", "weather.csv", "README.txt", SYNTHETIC_MARKER]
    for path in [*(out / name for name in written), *(out / "rsvps").glob("*.csv")]:
        path.unlink(missing_ok=True)


def _iso(stamps: pd.Series) -> pd.Series:
    """ISO 8601 with the +08:00 offset; blank for NaT."""
    return stamps.dt.tz_convert(MANILA_TZ).map(lambda t: "" if pd.isna(t) else t.isoformat())


def _whole_numbers(values: pd.Series) -> pd.Series:
    """Write 60.0 as "60" when every value is whole; leave other numbers alone."""
    finite = values.dropna()
    return values.astype("Int64") if (finite == finite.round()).all() else values
