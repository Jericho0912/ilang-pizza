"""Canonical data shapes shared by every module.

Everything downstream of the loaders speaks these shapes, so a Meetup export, a Luma export
and the synthetic demo data all look the same by the time features are built.

Privacy rule: nothing in this module ever holds a raw name or email. Attendees are identified
by a salted hash (``attendee_id``), and anything that reaches the LLM is aggregate-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MANILA_TZ = "Asia/Manila"

SERIES = ("PyMeets", "PyWorks", "WFH Pythonista", "PyKids", "Other")

# events table: one row per event
EVENT_COLUMNS = [
    "event_id",  # str, unique and file-name safe, e.g. "pymeets-2026-09"
    "title",  # str
    "series",  # str, one of SERIES
    "start",  # pd.Timestamp, tz-aware Asia/Manila
    "venue",  # str, "" when online
    "city",  # str, "" when online
    "latitude",  # float, NaN when unknown or online
    "longitude",  # float, NaN when unknown or online
    "is_online",  # bool
    "capacity",  # float, NaN when unlimited or unknown (float so NaN fits)
    "food_provided",  # bool
]

# rsvps table: one row per RSVP
RSVP_COLUMNS = [
    "event_id",  # str, foreign key to events.event_id
    "attendee_id",  # str, salted hash of the person's email or name; never raw PII
    "rsvp_at",  # pd.Timestamp, tz-aware Asia/Manila
    "status",  # "going" or "waitlist"
    "guests",  # int >= 0, extra heads on the same RSVP
    "attended",  # float: 1.0 came, 0.0 no-show, NaN unknown (upcoming, or not tracked)
    "source",  # "meetup" | "luma" | "tito" | "google_forms" | "canonical" | "synthetic"
]

RSVP_STATUSES = ("going", "waitlist")


@dataclass(frozen=True)
class Dataset:
    """Events plus RSVPs in canonical form.

    ``weather`` is an optional daily rain table (columns ``date`` as ``datetime.date`` and
    ``precipitation_mm`` as float). When present it is used instead of a live weather
    provider: the synthetic demo ships its own rain, and organizers can supply a PAGASA table
    as ``weather.csv``.
    """

    events: pd.DataFrame
    rsvps: pd.DataFrame
    synthetic: bool = False
    weather: pd.DataFrame | None = None

    def event(self, event_id: str) -> pd.Series:
        rows = self.events[self.events["event_id"] == event_id]
        if rows.empty:
            raise KeyError(f"unknown event_id: {event_id}")
        return rows.iloc[0]

    def rsvps_for(self, event_id: str) -> pd.DataFrame:
        return self.rsvps[self.rsvps["event_id"] == event_id]

    def labeled_event_ids(self) -> list[str]:
        """Events where attendance is known for at least one going RSVP, oldest first."""
        going = self.rsvps[self.rsvps["status"] == "going"]
        known = set(going.loc[going["attended"].notna(), "event_id"])
        ordered = self.events.sort_values("start")
        return [e for e in ordered["event_id"] if e in known]

    def upcoming_event_ids(self) -> list[str]:
        """Events with no known attendance yet, oldest first. This includes past events whose
        attendance was never tracked; use ``next_event_id`` for "the next event"."""
        labeled = set(self.labeled_event_ids())
        ordered = self.events.sort_values("start")
        return [e for e in ordered["event_id"] if e not in labeled]

    def next_event_id(self, now: pd.Timestamp | None = None) -> str | None:
        """The soonest unlabeled event starting at or after ``now`` (default: current Manila
        time), else the most recent unlabeled event, else None."""
        candidates = self.upcoming_event_ids()
        if not candidates:
            return None
        now = pd.Timestamp.now(tz=MANILA_TZ) if now is None else now
        starts = self.events.set_index("event_id")["start"]
        future = [e for e in candidates if starts[e] >= now]
        return future[0] if future else candidates[-1]

    def validate(self) -> None:
        """Raise ValueError if the tables do not match the canonical schema."""
        missing_e = [c for c in EVENT_COLUMNS if c not in self.events.columns]
        missing_r = [c for c in RSVP_COLUMNS if c not in self.rsvps.columns]
        if missing_e or missing_r:
            raise ValueError(f"missing columns: events={missing_e} rsvps={missing_r}")
        if self.events["event_id"].duplicated().any():
            raise ValueError("duplicate event_id in events")
        unknown = set(self.rsvps["event_id"]) - set(self.events["event_id"])
        if unknown:
            raise ValueError(f"rsvps reference unknown events: {sorted(unknown)[:5]}")
        bad_status = set(self.rsvps["status"]) - set(RSVP_STATUSES)
        if bad_status:
            raise ValueError(f"unknown RSVP status values: {sorted(bad_status)}")
        for col, frame in (("start", self.events), ("rsvp_at", self.rsvps)):
            if getattr(frame[col].dtype, "tz", None) is None:
                raise ValueError(f"{col} must be tz-aware datetimes")


@dataclass(frozen=True)
class TurnoutForecast:
    """Predicted headcount for one event. Per-RSVP probabilities stay local; only the
    aggregate fields (expected, p10/p50/p90, rates, drivers) may be shown to an LLM."""

    event_id: str
    model_name: str
    going_probs: np.ndarray  # P(show) per going RSVP, in ds.rsvps_for(event_id) going order
    going_heads: np.ndarray  # heads per going RSVP (1 + guests), same order
    waitlist_probs: np.ndarray  # P(show) per waitlisted RSVP if released, waitlist order
    waitlist_heads: np.ndarray  # heads per waitlisted RSVP, same order
    pmf: np.ndarray  # pmf[k] = P(exactly k heads among going RSVPs show up)
    expected: float  # expected heads among going RSVPs
    p10: int
    p50: int
    p90: int
    historical_show_rate: float  # share of going heads that showed up across past events
    drivers: list[str] = field(default_factory=list)  # short human-readable notes
    synthetic: bool = False  # True when the training data is the synthetic demo set
    # Spread of a shared event-day shock on the logit scale (rain, traffic, a rival event): it
    # moves every RSVP's odds together. 0 means independent RSVPs. ``pmf`` already includes it,
    # and the planner applies the same shock to released waitlist RSVPs.
    shock_sd: float = 0.0

    @property
    def rsvp_heads(self) -> int:
        return int(self.going_heads.sum())

    def summary(self) -> dict:
        """Aggregate-only view, safe to hand to an LLM or print."""
        return {
            "event_id": self.event_id,
            "model": self.model_name,
            "rsvp_heads": self.rsvp_heads,
            "waitlist_heads": int(self.waitlist_heads.sum()),
            "expected_heads": round(float(self.expected), 1),
            "p10": self.p10,
            "p50": self.p50,
            "p90": self.p90,
            "predicted_show_rate": round(float(self.expected) / max(self.rsvp_heads, 1), 3),
            "historical_show_rate": round(float(self.historical_show_rate), 3),
            "drivers": list(self.drivers),
            "synthetic_data": self.synthetic,
        }


@dataclass(frozen=True)
class BacktestResult:
    """Rolling-origin evaluation: each past event predicted only from events before it."""

    # one row per evaluated event: event_id, title, series, start, rsvp_heads, actual_heads,
    # model_expected, model_p10, model_p90, rule_pred, history_pred, logistic_expected
    per_event: pd.DataFrame
    # indexed by method ("tabpfn", "logistic", "history", "rule"): mae, mape, bias, n_events,
    # plus coverage_80 for methods that produce intervals
    metrics: pd.DataFrame
    model_name: str
    synthetic: bool = False


@dataclass(frozen=True)
class PlanInputs:
    capacity: int | None = None  # venue limit in heads; None means unlimited
    slices_per_person: float = 3.0
    slices_per_box: int = 8
    price_per_box: float = 0.0  # PHP; 0 hides cost lines
    food_quantile: float = 0.75  # order food for this quantile of predicted heads
    overflow_risk: float = 0.10  # max acceptable P(heads > capacity) when releasing waitlist


@dataclass(frozen=True)
class EventPlan:
    event_id: str
    food_heads: int  # heads the food order is sized for
    pizza_boxes: int
    pizza_boxes_naive: int  # boxes if you fed every RSVP head
    cost_php: float
    cost_naive_php: float
    savings_php: float
    seats: int  # chairs or name tags to prepare (p90 incl. released waitlist)
    certificates: int  # e-certificates to prepare (same basis as seats)
    waitlist_release: int  # how many waitlisted RSVPs to move to going
    overflow_probability: float  # P(heads > capacity) after the release; 0 when unlimited
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Draft:
    kind: str  # one of llm.prompts.EMAIL_KINDS
    subject: str
    body: str
    used_llm: bool  # False when the template fallback was used
    model: str = ""


@dataclass(frozen=True)
class AgentReply:
    answer: str
    steps: list[dict] = field(default_factory=list)  # [{"tool": ..., "args": ..., "result": ...}]
    used_llm: bool = True
