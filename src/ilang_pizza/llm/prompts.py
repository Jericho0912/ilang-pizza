"""Email drafts and forecast explanations: written by Gemma when it runs, by templates otherwise.

Gemma only ever sees aggregate numbers: event details (title, date, venue),
``TurnoutForecast.summary()`` and ``EventPlan`` fields. ``assert_aggregate_only`` checks every
prompt before it is sent. Replies are checked too: a draft that invents numbers, contacts or
links is dropped in favor of the hand-written template, so the tool is just as useful offline.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd

from ilang_pizza.llm.client import ChatModel, LLMError, parse_json
from ilang_pizza.model import turnout
from ilang_pizza.plan import format_php
from ilang_pizza.schema import MANILA_TZ, Draft, EventPlan, TurnoutForecast

log = logging.getLogger(__name__)

EMAIL_KINDS = ("reconfirm", "waitlist_release", "catering_order", "venue_headcount")
TONES = ("taglish", "english")

DEMO_MARKER = "[DEMO DATA — not a real event]"
SIGN_OFF = "— PythonPH Organizers"
DELIVERY_LEAD_MINUTES = 30
MAX_BODY_WORDS = 150  # what we ask Gemma for
_WORD_LIMIT = 230  # longer than this and the template is the better email
_EXPLANATION_WORD_LIMIT = 120

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {"subject": {"type": "string"}, "body": {"type": "string"}},
    "required": ["subject", "body"],
}
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {"explanation": {"type": "string"}},
    "required": ["explanation"],
}


# --- privacy -----------------------------------------------------------------------------


class PrivacyError(ValueError):
    """Text meant for the LLM looks like it contains personal data."""


_PII_PATTERNS = {
    "an email address": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    # attendee_id is a salted hex digest; 16+ hex characters in a row is never aggregate data
    "an attendee hash": re.compile(r"(?<![0-9a-f])[0-9a-f]{16,}(?![0-9a-f])", re.IGNORECASE),
    "a phone number": re.compile(
        r"(?<!\d)\(?(?:\+?63|0)\)?[\s.-]?\(?9\d{2}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)"
    ),
}


def assert_aggregate_only(text: str) -> None:
    """Raise ``PrivacyError`` if ``text`` contains an email, attendee hash or PH phone number.

    The message names what was found but never repeats it, so it is safe to log.
    """
    found = [kind for kind, pattern in _PII_PATTERNS.items() if pattern.search(text)]
    if found:
        raise PrivacyError(
            f"Refusing to send this to the LLM: it looks like it contains {' and '.join(found)}."
        )


def redact_pii(text: str) -> str:
    """Replace anything ``assert_aggregate_only`` would reject with "[redacted]"."""
    for pattern in _PII_PATTERNS.values():
        text = pattern.sub("[redacted]", text)
    return text


def _check_messages(messages: list[dict]) -> None:
    for message in messages:
        assert_aggregate_only(str(message.get("content", "")))


# --- number guard ------------------------------------------------------------------------

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_THOUSANDS = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")


def _numbers_in(text: str) -> list[float]:
    values = []
    for token in _NUMBER.findall(text):
        parts = [token.replace(",", "")] if _THOUSANDS.fullmatch(token) else token.split(",")
        values.extend(float(part) for part in parts if part)
    return values


def invented_numbers(text: str, *sources: str, exact: bool = False) -> list[str]:
    """Numbers in ``text`` that cannot be traced back to ``sources``.

    A rate written as a percent (0.62 -> 62%) or a percent as a rate (10% -> 0.1) counts as
    traceable. So does rounding (24.6 -> 25), unless ``exact``. Anything else, including
    arithmetic, counts as invented.
    """
    allowed: set[float] = set()
    for x in _numbers_in("\n".join(sources)):
        values = [x]
        if 0 < x < 1:
            values.append(x * 100)
        for value in values:
            allowed.add(round(value, 2))
            if not exact:
                allowed.update({float(math.floor(value)), float(math.ceil(value))})
        if 0 < x <= 100:
            allowed.add(round(x / 100, 4))
    invented = [
        n for n in _numbers_in(text) if round(n, 2) not in allowed and round(n, 4) not in allowed
    ]
    return sorted({f"{n:g}" for n in invented})


def _jsonable(value):
    """Round floats so prompts stay short and never contain 17-digit noise."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, 3) if math.isfinite(value) else None
    return value


def to_json(value) -> str:
    """Compact, aggregate-friendly JSON for prompts and tool results."""
    return json.dumps(_jsonable(value), ensure_ascii=False)


# --- event details -----------------------------------------------------------------------


def _manila(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize(MANILA_TZ) if ts.tzinfo is None else ts.tz_convert(MANILA_TZ)


def _clock(ts: pd.Timestamp) -> str:
    return f"{ts.hour % 12 or 12}:{ts:%M} {ts:%p}"


def format_when(start) -> str:
    """Event start in Manila time, e.g. "Thu, Sep 17, 6:30 PM"."""
    ts = _manila(start)
    return f"{ts:%a}, {ts:%b} {ts.day}, {_clock(ts)}"


def _text(value) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _place(event: Mapping) -> str:
    venue, city = _text(event.get("venue")), _text(event.get("city"))
    if venue and city and city.lower() not in venue.lower():
        return f"{venue}, {city}"
    return venue or city


@dataclass(frozen=True)
class _Facts:
    """Everything a draft may mention, already formatted."""

    title: str
    series: str
    when: str  # "Thu, Sep 17, 6:30 PM"
    day: str  # "Thu, Sep 17"
    iso_date: str  # "2026-09-17"
    delivery_time: str  # DELIVERY_LEAD_MINUTES before the start
    venue: str  # "" when online
    online: bool
    rsvp_heads: int
    waitlist_heads: int
    released: int
    low: int  # 10th percentile of total heads, released waitlist included
    high: int  # 90th percentile of total heads
    seats: int
    boxes: int
    boxes_naive: int
    food_heads: int
    cost_php: float
    price_per_box: float

    @property
    def where(self) -> str:
        return "online" if self.online else f"at {self.venue}"

    @property
    def headcount(self) -> str:
        return f"about {self.high}" if self.low == self.high else f"{self.low}–{self.high}"


def _facts(event: Mapping, forecast: TurnoutForecast, plan: EventPlan) -> _Facts:
    if plan.event_id != forecast.event_id:
        raise ValueError(f"plan is for {plan.event_id} but forecast is for {forecast.event_id}")
    start = _manila(event["start"])
    place = _place(event)
    online = bool(event.get("is_online", False)) or not place
    released = min(plan.waitlist_release, len(forecast.waitlist_probs))
    going = forecast.pmf if forecast.pmf.size else np.ones(1)  # no RSVPs: surely 0 heads
    total = turnout.add_rsvps(
        going, forecast.waitlist_probs[:released], forecast.waitlist_heads[:released]
    )
    return _Facts(
        title=_text(event.get("title")) or "our next meetup",
        series=_text(event.get("series")),
        when=format_when(start),
        day=f"{start:%a}, {start:%b} {start.day}",
        iso_date=f"{start:%Y-%m-%d}",
        delivery_time=_clock(start - pd.Timedelta(minutes=DELIVERY_LEAD_MINUTES)),
        venue="" if online else place,
        online=online,
        rsvp_heads=forecast.rsvp_heads,
        waitlist_heads=int(forecast.waitlist_heads.sum()),
        released=released,
        low=turnout.quantile(total, 0.10),
        high=turnout.quantile(total, 0.90),
        seats=plan.seats,
        boxes=plan.pizza_boxes,
        boxes_naive=plan.pizza_boxes_naive,
        food_heads=plan.food_heads,
        cost_php=plan.cost_php,
        price_per_box=plan.cost_php / plan.pizza_boxes if plan.pizza_boxes else 0.0,
    )


# --- templates ---------------------------------------------------------------------------


def _email(*paragraphs: str, closing: str) -> str:
    return "\n\n".join(paragraphs) + f"\n\n{closing}\n{SIGN_OFF}"


def _bullets(*lines: str) -> str:
    return "\n".join(f"- {line}" for line in lines if line)


def _reconfirm(f: _Facts, tone: str) -> tuple[str, str]:
    if tone == "taglish":
        if f.waitlist_heads > 1:
            why = (
                f"para maibigay namin ang slot mo sa isa sa {f.waitlist_heads} Pythonistas "
                "na naka-waitlist"
            )
        elif f.waitlist_heads == 1:
            why = "para maibigay namin ang slot mo sa Pythonista na naka-waitlist"
        elif f.online:
            why = "para alam namin kung ilan ang aasahan"
        else:
            why = "para tama ang bilang namin ng pagkain at upuan"
        return f"Tuloy ka pa ba sa {f.title}?", _email(
            "Hi Pythonistas!",
            f"Excited na kami sa {f.title} on {f.when} {f.where}!",
            "Quick check lang: tuloy ka pa ba? Kung oo, wala ka nang kailangang gawin. Kita-kits!",
            f"Kung hindi ka na makakarating, okay lang! Paki-update lang ang RSVP mo {why}.",
            closing="Salamat!",
        )
    if f.waitlist_heads > 1:
        why = (
            f"so we can give your slot to one of the {f.waitlist_heads} Pythonistas on the waitlist"
        )
    elif f.waitlist_heads == 1:
        why = "so we can give your slot to the Pythonista on the waitlist"
    elif f.online:
        why = "so we know how many to expect"
    else:
        why = "so we can plan the right amount of food and seats"
    return f"Still coming to {f.title}?", _email(
        "Hi Pythonistas!",
        f"We're excited for {f.title} on {f.when} {f.where}!",
        "Quick check: are you still coming? If yes, there's nothing you need to do. See you there!",
        f"If you can't make it anymore, no worries. Please update your RSVP {why}.",
        closing="Thank you!",
    )


def _waitlist_release(f: _Facts, tone: str) -> tuple[str, str]:
    if tone == "taglish":
        practical = (
            "Ito ang join link: [Join link]"
            if f.online
            else "Magdala ng valid ID, kasi karaniwang nagche-check sa lobby ang mga office "
            "building."
        )
        return f"Good news: pasok ka na sa {f.title}!", _email(
            "Hi Pythonista!",
            f"May nagbukas na slot, kaya pasok ka na sa {f.title}! "
            f"Kita-kits on {f.when} {f.where}.",
            'Paki-check lang kung "Going" na ang RSVP mo. Kung hindi ka na pala makakarating, '
            "paki-update agad para maibigay namin ang slot sa susunod sa waitlist.",
            practical,
            closing="See you!",
        )
    practical = (
        "Here's the join link: [Join link]"
        if f.online
        else "Please bring a valid ID. Office buildings usually check at the lobby."
    )
    return f"Good news: you're in for {f.title}!", _email(
        "Hi Pythonista!",
        f"A slot just opened up, so you're in for {f.title}! See you on {f.when} {f.where}.",
        'Please check that your RSVP now says "Going". If you can\'t make it after all, please '
        "update it soon so we can pass the slot to the next person on the waitlist.",
        practical,
        closing="See you!",
    )


def _catering_order(f: _Facts, tone: str) -> tuple[str, str]:
    address = f.venue or "[Delivery address]"
    cost = ""
    if f.cost_php > 0:
        each = format_php(f.price_per_box)
        cost = f"Estimated total: {format_php(f.cost_php)} ({f.boxes} boxes × {each})"
    if tone == "taglish":
        return f"Pizza order: {f.boxes} boxes para sa {f.day}", _email(
            "Hi [Caterer name]!",
            f"Magpapa-order po sana kami ng pizza para sa {f.title}, isang Python Philippines "
            "community meetup.",
            _bullets(
                f"Order: {f.boxes} boxes ([flavors])",
                f"Delivery: {f.delivery_time} on {f.day}, "
                f"{DELIVERY_LEAD_MINUTES} minutes bago mag-start ang event",
                f"Address: {address}",
                "Contact person: [Organizer contact]",
                cost,
            ),
            "Paki-confirm po kung available, pati ang final total at payment details.",
            closing="Maraming salamat po!",
        )
    return f"Pizza order: {f.boxes} boxes for {f.day}", _email(
        "Hi [Caterer name],",
        f"We'd like to order pizza for {f.title}, a Python Philippines community meetup.",
        _bullets(
            f"Order: {f.boxes} boxes ([flavors])",
            f"Delivery: {f.delivery_time} on {f.day}, "
            f"{DELIVERY_LEAD_MINUTES} minutes before the event starts",
            f"Address: {address}",
            "Contact person: [Organizer contact]",
            cost,
        ),
        "Could you please confirm availability, the final total, and payment details?",
        closing="Thank you!",
    )


def _venue_headcount(f: _Facts, tone: str) -> tuple[str, str]:
    numbers = _bullets(f"Expected attendees: {f.headcount} people", f"Seats to prepare: {f.seats}")
    if tone == "taglish":
        return f"Headcount para sa {f.title} ({f.day})", _email(
            "Hi [Venue contact]!",
            f"Para po sa {f.title} ng Python Philippines on {f.when}:",
            numbers,
            "Ipapadala po namin nang hiwalay ang guest list para sa building security bago ang "
            "event. Kung may kailangan pa po kayo mula sa amin, sabihan lang po kami.",
            closing="Maraming salamat po sa pag-host!",
        )
    return f"Expected headcount for {f.title} ({f.day})", _email(
        "Hi [Venue contact],",
        f"Here's our headcount estimate for {f.title} (Python Philippines) on {f.when}:",
        numbers,
        "We'll send the guest list for building security in a separate email before the "
        "event. If you need anything else from us, just let us know.",
        closing="Thank you for hosting us!",
    )


_TEMPLATES = {
    "reconfirm": _reconfirm,
    "waitlist_release": _waitlist_release,
    "catering_order": _catering_order,
    "venue_headcount": _venue_headcount,
}


def _explanation_template(f: _Facts, forecast: TurnoutForecast, plan: EventPlan, tone: str) -> str:
    taglish = tone == "taglish"
    if f.rsvp_heads == 0:
        if taglish:
            return f"Wala pang RSVP para sa {f.title}, kaya wala pang mapo-forecast."
        return f"There are no RSVPs for {f.title} yet, so there's nothing to forecast."
    s = forecast.summary()
    expected, likely = round(forecast.expected), f"{forecast.p10}–{forecast.p90}"
    rate, hist = round(100 * s["predicted_show_rate"]), s["historical_show_rate"]
    # The drivers are full sentences already; two of them keep this at four sentences.
    drivers = [d.strip().rstrip(".") + "." for d in forecast.drivers[:2] if d.strip()]

    if taglish:
        first = (
            f"Mga {expected} sa {f.rsvp_heads} na naka-RSVP ang inaasahang darating sa "
            f"{f.title} (malamang {likely}), o {rate}% show-up rate"
        )
        first += f" kumpara sa {round(100 * hist)}% dati." if math.isfinite(hist) else "."
        act = f"Kaya {f.boxes} boxes ang i-order"
        if f.boxes < f.boxes_naive:
            act = f"Kaya {f.boxes} boxes lang ang i-order imbes na {f.boxes_naive}"
        if f.released:
            act += f", pwedeng i-release ang {f.released} sa waitlist"
        act += f", at maghanda ng {f.seats} na upuan at e-certificates."
    else:
        first = (
            f"We expect about {expected} of the {f.rsvp_heads} people who RSVP'd to "
            f"{f.title} (likely {likely}), a {rate}% show-up rate"
        )
        first += f" vs {round(100 * hist)}% at past events." if math.isfinite(hist) else "."
        act = f"So order {f.boxes} pizza boxes"
        if f.boxes < f.boxes_naive:
            act += f" instead of {f.boxes_naive}"
        if f.released:
            act += f", release {f.released} waitlist spots"
        act += f", and prepare {f.seats} seats and e-certificates."
    return " ".join([first, *drivers, act])


# --- prompts for Gemma -------------------------------------------------------------------

_SYSTEM = """You are the {role} of Python Philippines (PythonPH), a volunteer-run
community of Filipino Python developers. It runs free meetups: PyMeets (weekday-evening talks
at office venues in BGC, Makati and Ortigas), PyWorks (Saturday workshops), WFH Pythonista
(online) and PyKids.

Rules:
- Use only the facts you are given. Never invent numbers, names, links, prices or rules, and
  don't do arithmetic: copy the numbers exactly as given.
- When something is unknown, write a placeholder in square brackets, like [Organizer contact].
- Never include personal data: no attendee names, emails or phone numbers.
- Plain text only, no Markdown headings or bold. No emoji in the subject, at most one in
  the body."""

_TONES = {
    "taglish": (
        "Taglish, the way PythonPH volunteers really write: warm, light and friendly, mixing "
        "English and Filipino inside sentences the way Manila techies chat. For example: "
        '"Excited na kami for PyMeets!", "Kung hindi ka na makakarating, paki-update lang ang '
        'RSVP mo.", "Kita-kits!". Keep it easy to read and don\'t overdo the slang.'
    ),
    "english": "Clear, friendly English: warm, concise and easy to scan.",
}


class _Brief(NamedTuple):
    audience: str
    goal: str
    greeting: str
    business: bool  # caterers and venue hosts get "po"; fellow Pythonistas don't


_BRIEFS = {
    "reconfirm": _Brief(
        audience='everyone who RSVP\'d "going"',
        goal=(
            "Ask whether they are still coming. If they can't make it, ask them to update their "
            "RSVP so a waitlisted Pythonista can take the slot, and mention how many people are "
            "on the waitlist. Don't mention forecasts, predictions or no-show rates."
        ),
        greeting="Hi Pythonistas!",
        business=False,
    ),
    "waitlist_release": _Brief(
        audience="the next people on the waitlist (each one gets this same email)",
        goal=(
            "Tell them a slot opened up and they're in. Ask them to check that their RSVP says "
            '"Going", or to free the slot quickly if they can\'t come after all. For in-person '
            "events, remind them to bring a valid ID for the building lobby. Don't mention "
            "forecasts or numbers of people."
        ),
        greeting="Hi Pythonista!",
        business=False,
    ),
    "catering_order": _Brief(
        audience="the pizza place or caterer",
        goal=(
            "Order the given number of pizza boxes, delivered at the delivery time "
            f"({DELIVERY_LEAD_MINUTES} minutes before the event starts) to the venue. Ask them "
            "to confirm availability, the final "
            "total and payment details. Put the order details in short lines starting with "
            '"- ". Use the placeholders [Caterer name], [flavors] and [Organizer contact]. '
            "Don't mention forecasts or show-up rates."
        ),
        greeting="Hi [Caterer name]!",
        business=True,
    ),
    "venue_headcount": _Brief(
        audience="the venue host or building admin",
        goal=(
            "Share the expected headcount range and the number of seats to prepare, as short "
            'lines starting with "- ". Say the guest list for building security will follow in '
            "a separate email. Use the placeholder [Venue contact] for their name. Don't "
            "mention show-up rates."
        ),
        greeting="Hi [Venue contact]!",
        business=True,
    ),
}


# An empty waitlist is better left unmentioned than announced as "0 people waiting".
_RECONFIRM_WITHOUT_WAITLIST = (
    "Ask whether they are still coming. If they can't make it, ask them to update their RSVP "
    "so we can plan the right amount of food and seats. Don't mention the waitlist, forecasts, "
    "predictions or no-show rates."
)


def _event_json(f: _Facts) -> dict:
    return {
        "title": f.title,
        "series": f.series,
        "when": f.when,
        "date": f.iso_date,
        "venue": f.venue or "online",
    }


def _kind_facts(kind: str, f: _Facts) -> dict:
    if kind == "reconfirm":
        return {"people_on_waitlist": f.waitlist_heads} if f.waitlist_heads else {}
    if kind == "waitlist_release":
        return {"waitlisted_rsvps_released": f.released, "in_person": not f.online}
    if kind == "catering_order":
        facts = {
            "pizza_boxes": f.boxes,
            "delivery_time": f.delivery_time,
            "delivery_day": f.day,
            "minutes_before_start": DELIVERY_LEAD_MINUTES,
            "deliver_to": f.venue or "[Delivery address]",
        }
        if f.cost_php > 0:
            facts["price_per_box"] = format_php(f.price_per_box)
            facts["estimated_total"] = format_php(f.cost_php)
        return facts
    return {
        "expected_attendees_low": f.low,
        "expected_attendees_high": f.high,
        "seats_to_prepare": f.seats,
    }


def draft_messages(
    kind: str, event: Mapping, forecast: TurnoutForecast, plan: EventPlan, *, tone: str = "taglish"
) -> list[dict]:
    """The exact messages Gemma sees for a draft: handy for showing organizers what is shared."""
    _check_choice(kind, tone)
    f = _facts(event, forecast, plan)
    brief = _BRIEFS[kind]
    if kind == "reconfirm" and not f.waitlist_heads:
        brief = brief._replace(goal=_RECONFIRM_WITHOUT_WAITLIST)
    facts = {"event": _event_json(f), "email_facts": _kind_facts(kind, f)}
    tone_text = _TONES[tone]
    if tone == "taglish":
        tone_text += (
            ' This goes to a business, so be polite and use "po".'
            if brief.business
            else ' This goes to fellow Pythonistas, so keep it casual: no "po".'
        )
    user = f"""Write the "{kind}" email.

Audience: {brief.audience}
Goal: {brief.goal}
Tone: {tone_text}
Start with a greeting like "{brief.greeting}". Keep the body under {MAX_BODY_WORDS} words.
End the body with this sign-off on its own line: {SIGN_OFF}

Facts (the only numbers you may use):
{to_json(facts)}

Forecast, as background only (never quote show-up rates to attendees):
{to_json(forecast.summary())}

Reply with a JSON object with two string fields: "subject" and "body"."""
    system = _SYSTEM.format(role="communications helper")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def explain_messages(
    event: Mapping, forecast: TurnoutForecast, plan: EventPlan, *, tone: str = "taglish"
) -> list[dict]:
    """The exact messages Gemma sees when explaining a forecast to organizers."""
    _check_choice(None, tone)
    f = _facts(event, forecast, plan)
    plan_facts = {
        "people_to_feed": plan.food_heads,
        "pizza_boxes": plan.pizza_boxes,
        "pizza_boxes_if_feeding_every_rsvp": plan.pizza_boxes_naive,
        "waitlist_spots_to_release": plan.waitlist_release,
        "seats_and_certificates": plan.seats,
        "chance_of_going_over_capacity": round(plan.overflow_probability, 3),
    }
    if plan.cost_php > 0:
        plan_facts.update(cost_php=plan.cost_php, savings_php=plan.savings_php)
    user = f"""Explain this turnout forecast and plan to the organizers in 2 to 4 short sentences.
Say how many people to expect (and the likely range), name the main drivers, and say what to do
about food, the waitlist and seats.
Tone: {_TONES[tone]}

Event: {to_json(_event_json(f))}
Forecast for the going RSVPs: {to_json(forecast.summary())}
Plan: {to_json(plan_facts)}
Plan notes: {to_json(plan.notes)}

Reply with a JSON object with one string field: "explanation"."""
    system = _SYSTEM.format(role="forecast explainer")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --- checking what Gemma wrote -----------------------------------------------------------

_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_EMOJI_CHARS = "\U0001f000-\U0001faff\u2600-\u27bf\u2b00-\u2bff"
# A flag (two regional letters) or one emoji with its joiners and variants, plus one space.
_EMOJI = re.compile(
    rf"(?:[\U0001f1e6-\U0001f1ff]{{2}}"
    rf"|[{_EMOJI_CHARS}](?:\ufe0f|\u200d[{_EMOJI_CHARS}])*\ufe0f?)[ \t]?"
)
_SIGN_OFF_LINE = re.compile(r"^[\s—–-]*PythonPH Organi[sz]ers[\s.!]*$", re.IGNORECASE)
_SUBJECT_PREFIX = re.compile(r"^subject:\s*", re.IGNORECASE)
# Details an email is useless without; added back when Gemma leaves them out.
_MUST_MENTION = {
    "catering_order": ("[Organizer contact]", "Contact person on the day: [Organizer contact]"),
}
_SUBJECT_LINE = re.compile(r"\A\s*subject:[^\n]*\n", re.IGNORECASE)  # a subject pasted in the body


def _with_sign_off(body: str, *, last_line: str = "") -> str:
    """End ``body`` with exactly one sign-off, after ``last_line`` if one is given."""
    lines = body.rstrip().splitlines()
    while lines and (not lines[-1].strip() or _SIGN_OFF_LINE.match(lines[-1])):
        lines.pop()
    text = "\n".join(lines).rstrip()
    if last_line:
        text += f"\n\n{last_line}"
    return f"{text}\n{SIGN_OFF}"


def _limit_emoji(text: str, keep: int) -> str:
    """Drop every emoji after the first ``keep``, tidying the spaces they leave behind."""
    seen = 0

    def one(match: re.Match) -> str:
        nonlocal seen
        seen += 1
        return match.group(0) if seen <= keep else ""

    lines = _EMOJI.sub(one, text).splitlines()
    return "\n".join(re.sub(r"[ \t]{2,}", " ", line).rstrip() for line in lines).strip()


def _review(text: str, messages: list[dict]) -> str:
    """Return Gemma's text with links replaced by a placeholder, or raise ``LLMError``."""
    try:
        assert_aggregate_only(text)
    except PrivacyError as e:
        raise LLMError("the reply contained contact details we never gave it") from e
    invented = invented_numbers(text, *(str(m["content"]) for m in messages))
    if invented:
        raise LLMError(f"the reply used numbers that are not in the facts: {', '.join(invented)}")
    return _URL.sub("[Meetup link]", text)


def _llm_draft(kind: str, messages: list[dict], llm: ChatModel) -> tuple[str, str]:
    _check_messages(messages)
    reply = parse_json(llm.chat(messages, schema=DRAFT_SCHEMA, temperature=0.4))
    subject = _SUBJECT_PREFIX.sub("", " ".join(str(reply.get("subject") or "").split()))
    body = _SUBJECT_LINE.sub("", str(reply.get("body") or "")).strip()
    if not subject or not body:
        raise LLMError("the reply had no subject or no body")
    if len(body.split()) > _WORD_LIMIT:
        raise LLMError(f"the body ran to {len(body.split())} words")
    subject = _limit_emoji(_review(subject, messages), keep=0)
    body = _limit_emoji(_review(body, messages), keep=1)
    placeholder, line = _MUST_MENTION.get(kind, ("", ""))
    return subject, _with_sign_off(body, last_line="" if placeholder in body else line)


def _mark_demo(subject: str, body: str, synthetic: bool) -> tuple[str, str]:
    if not synthetic:
        return subject, body
    subject = subject.replace(DEMO_MARKER, "").strip()
    body = body.replace(DEMO_MARKER, "").strip()
    return f"{DEMO_MARKER} {subject}", f"{DEMO_MARKER}\n\n{body}"


def _check_choice(kind: str | None, tone: str) -> None:
    if kind is not None and kind not in EMAIL_KINDS:
        raise ValueError(f"unknown email kind {kind!r}; choose one of {', '.join(EMAIL_KINDS)}")
    if tone not in TONES:
        raise ValueError(f"unknown tone {tone!r}; choose one of {', '.join(TONES)}")


def _usable(llm: ChatModel | None) -> bool:
    return llm is not None and llm.available()


# --- public API --------------------------------------------------------------------------


def template_draft(
    kind: str, event: Mapping, forecast: TurnoutForecast, plan: EventPlan, *, tone: str = "taglish"
) -> Draft:
    """The hand-written draft, no LLM involved."""
    _check_choice(kind, tone)
    subject, body = _TEMPLATES[kind](_facts(event, forecast, plan), tone)
    subject, body = _mark_demo(subject, body, forecast.synthetic)
    return Draft(kind=kind, subject=subject, body=body, used_llm=False)


def draft_email(
    kind: str,
    event: Mapping,
    forecast: TurnoutForecast,
    plan: EventPlan,
    *,
    tone: str = "taglish",
    llm: ChatModel | None = None,
) -> Draft:
    """Draft one of ``EMAIL_KINDS`` in ``tone``.

    Gemma writes it when ``llm`` is available. If there is no LLM, it errors, or its reply
    fails a check (bad JSON, invented numbers or contacts, too long), the template is used.
    ``Draft.used_llm`` says which path ran.
    """
    _check_choice(kind, tone)
    if _usable(llm):
        try:
            messages = draft_messages(kind, event, forecast, plan, tone=tone)
            subject, body = _llm_draft(kind, messages, llm)
        except (LLMError, PrivacyError) as e:
            log.warning("Using the %s template instead of %s: %s", kind, llm.model, e)
        else:
            subject, body = _mark_demo(subject, body, forecast.synthetic)
            return Draft(kind=kind, subject=subject, body=body, used_llm=True, model=llm.model)
    return template_draft(kind, event, forecast, plan, tone=tone)


def explain_forecast(
    event: Mapping,
    forecast: TurnoutForecast,
    plan: EventPlan,
    *,
    tone: str = "taglish",
    llm: ChatModel | None = None,
) -> str:
    """Two to four sentences for organizers: expected turnout, why, and what to do about it."""
    _check_choice(None, tone)
    f = _facts(event, forecast, plan)
    text = ""
    if _usable(llm) and f.rsvp_heads > 0:
        try:
            messages = explain_messages(event, forecast, plan, tone=tone)
            _check_messages(messages)
            reply = parse_json(llm.chat(messages, schema=EXPLANATION_SCHEMA, temperature=0.3))
            text = " ".join(str(reply.get("explanation") or "").split())
            if len(text.split()) > _EXPLANATION_WORD_LIMIT:
                raise LLMError(f"the explanation ran to {len(text.split())} words")
            text = _review(text, messages)
        except (LLMError, PrivacyError) as e:
            log.warning("Using the explanation template instead of %s: %s", llm.model, e)
            text = ""
    text = text or _explanation_template(f, forecast, plan, tone)
    return f"{DEMO_MARKER} {text}" if forecast.synthetic else text
