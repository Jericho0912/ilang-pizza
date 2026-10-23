"""Turn a headcount distribution into decisions: waitlist releases, pizza boxes and seats.

Plain arithmetic on the exact distribution from ``model.turnout``. No model and no LLM here,
so every number in a plan can be checked by hand. Released waitlist RSVPs share the event-day
shock of the forecast (``TurnoutForecast.shock_sd``): if rain keeps people home, it keeps the
released ones home too.
"""

from __future__ import annotations

import math

import numpy as np

from ilang_pizza.model.turnout import Headcount, prob_exceeds, quantile
from ilang_pizza.schema import EventPlan, PlanInputs, TurnoutForecast

# Seats and e-certificates should be enough 9 times out of 10.
SEATS_QUANTILE = 0.90

# Float noise in a tail sum must not block a release that is exactly at the limit.
_RISK_TOLERANCE = 1e-9


def make_plan(forecast: TurnoutForecast, inputs: PlanInputs) -> EventPlan:
    """Release the waitlist up to the overflow risk, then size food, seats and certificates."""
    _validate(inputs)
    capacity = _capacity(inputs.capacity)
    going = _going(forecast)

    released, total, notes = _release_waitlist(forecast, going, capacity, inputs.overflow_risk)
    released_heads = int(forecast.waitlist_heads[:released].sum())
    all_heads = forecast.rsvp_heads + released_heads

    pmf = total.pmf()
    food_heads = quantile(pmf, inputs.food_quantile)
    boxes = _boxes(food_heads, inputs)
    boxes_naive = _boxes(all_heads, inputs)
    seats = quantile(pmf, SEATS_QUANTILE)

    price = inputs.price_per_box
    cost, cost_naive = (boxes * price, boxes_naive * price) if price > 0 else (0.0, 0.0)

    if all_heads == 0:
        notes.append("No RSVPs yet, so there's nothing to order.")
    else:
        notes.append(_food_note(boxes, boxes_naive, food_heads, inputs.slices_per_person))
        if price > 0:
            notes.append(_cost_note(cost, cost_naive))
        notes.append(f"Prepare {seats} seats and e-certificates (enough 9 times out of 10).")

    return EventPlan(
        event_id=forecast.event_id,
        food_heads=food_heads,
        pizza_boxes=boxes,
        pizza_boxes_naive=boxes_naive,
        cost_php=float(cost),
        cost_naive_php=float(cost_naive),
        savings_php=float(cost_naive - cost),
        seats=seats,
        certificates=seats,
        waitlist_release=released,
        overflow_probability=prob_exceeds(pmf, capacity) if capacity is not None else 0.0,
        notes=notes,
    )


def total_heads_pmf(forecast: TurnoutForecast, released: int = 0) -> np.ndarray:
    """P(exactly k heads) once the first ``released`` waitlisted RSVPs are let in.

    The released RSVPs share the forecast's event-day shock with the going ones.
    """
    waitlist = forecast.waitlist_probs[:released], forecast.waitlist_heads[:released]
    return _going(forecast).add(*waitlist).pmf()


def _going(forecast: TurnoutForecast) -> Headcount:
    """The going RSVPs' headcount, ready to take waitlisted RSVPs under the same shock."""
    return Headcount(forecast.shock_sd).add(forecast.going_probs, forecast.going_heads)


def _release_waitlist(
    forecast: TurnoutForecast, going: Headcount, capacity: int | None, risk: float
) -> tuple[int, Headcount, list[str]]:
    """Release in waitlist order while P(heads > capacity) stays within ``risk``.

    Returns how many RSVPs to release, the total headcount afterwards, and a note.
    """
    probs, heads = forecast.waitlist_probs, forecast.waitlist_heads
    waiting = len(probs)
    if capacity is None:
        if not waiting:
            return 0, going, []
        note = f"No venue limit set, so release the whole waitlist ({_spots(waiting)})."
        return waiting, going.add(probs, heads), [note]

    over = f"more than {capacity} people"
    base_risk = prob_exceeds(going.pmf(), capacity)
    if base_risk > risk + _RISK_TOLERANCE:
        note = f"the going RSVPs alone have {_chance(base_risk)} of {over} (limit {_pct(risk)})."
        return 0, going, [f"Keep the waitlist closed: {note}" if waiting else f"Heads up: {note}"]
    if not waiting:
        return 0, going, [f"No one is on the waitlist. There's {_chance(base_risk)} of {over}."]

    total, released, next_risk = going, 0, base_risk
    for p, h in zip(probs, heads):
        candidate = total.add([p], [h])
        next_risk = prob_exceeds(candidate.pmf(), capacity)
        if next_risk > risk + _RISK_TOLERANCE:
            break
        total, released = candidate, released + 1

    if released == 0:
        note = f"Keep the waitlist closed for now: releasing even 1 spot means {_chance(next_risk)}"
        return 0, total, [f"{note} of {over}."]
    if released < waiting:
        count = f"{released} of {_spots(waiting)}"
    else:
        count = f"all {_spots(released)}" if released > 1 else "the 1 waitlist spot"
    released_heads = int(heads[:released].sum())
    if released_heads != released:
        count += f" ({released_heads} people with guests)"
    still = _still(prob_exceeds(total.pmf(), capacity))
    return released, total, [f"Release {count}: {still} of {over}."]


def _food_note(boxes: int, boxes_naive: int, food_heads: int, slices_per_person: float) -> str:
    order = (
        f"Order {_plural(boxes, 'box', 'boxes')} "
        f"(feeds {food_heads} at {slices_per_person:g} slices each)"
    )
    if boxes == boxes_naive:
        return f"{order}, the same as ordering for every RSVP."
    return f"{order}. Ordering for every RSVP would be {_plural(boxes_naive, 'box', 'boxes')}."


def _cost_note(cost: float, cost_naive: float) -> str:
    if cost_naive > cost:
        saving = format_php(cost_naive - cost)
        return f"That's {format_php(cost)} instead of {format_php(cost_naive)}, saving {saving}."
    return f"That's {format_php(cost)}."


def _boxes(heads: int, inputs: PlanInputs) -> int:
    # round() first so 24 heads x 3 slices / 8 is 9 boxes, not 10 from float noise
    return math.ceil(round(heads * inputs.slices_per_person / inputs.slices_per_box, 6))


def _capacity(value: float | None) -> int | None:
    """``None`` and NaN (how the events table stores "unlimited") both mean no limit."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def _validate(inputs: PlanInputs) -> None:
    if inputs.capacity is not None and inputs.capacity < 0:
        raise ValueError("capacity cannot be negative")
    if inputs.slices_per_box <= 0:
        raise ValueError("slices_per_box must be positive")
    if inputs.slices_per_person < 0:
        raise ValueError("slices_per_person cannot be negative")
    if inputs.price_per_box < 0:
        raise ValueError("price_per_box cannot be negative")
    if not 0 < inputs.food_quantile <= 1:
        raise ValueError("food_quantile must be in (0, 1]")
    if not 0 <= inputs.overflow_risk <= 1:
        raise ValueError("overflow_risk must be in [0, 1]")


def _chance(p: float) -> str:
    """Phrases like "an 8% chance", "less than a 1% chance" or "no chance"."""
    if p <= 0:
        return "no chance"
    if p < 0.005:
        return "less than a 1% chance"
    if 0.995 <= p < 1:  # rounding to "100%" would claim a certainty the numbers don't have
        return "more than a 99% chance"
    pct = round(p * 100)
    article = "an" if pct in (8, 11, 18) or 80 <= pct <= 89 else "a"
    return f"{article} {pct}% chance"


def _still(p: float) -> str:
    return f"still only {_chance(p)}" if p >= 0.005 else f"still {_chance(p)}"


def _pct(p: float) -> str:
    return "0%" if p <= 0 else ("under 1%" if p < 0.005 else f"{round(p * 100)}%")


def _spots(n: int) -> str:
    return _plural(n, "waitlist spot", "waitlist spots")


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def format_php(amount: float) -> str:
    """Pesos the way organizers write them: "₱4,050", or "₱4,054.50" with centavos."""
    return f"₱{amount:,.0f}" if float(amount).is_integer() else f"₱{amount:,.2f}"
