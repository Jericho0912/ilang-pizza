from __future__ import annotations

import math

import numpy as np
import pytest

from ilang_pizza.model.turnout import add_rsvps, expected, heads_pmf, prob_exceeds, quantile
from ilang_pizza.plan import format_php, make_plan, total_heads_pmf
from ilang_pizza.schema import PlanInputs, TurnoutForecast


def make_forecast(
    going_probs,
    going_heads=None,
    waitlist_probs=(),
    waitlist_heads=None,
    *,
    synthetic=False,
    shock_sd=0.0,
) -> TurnoutForecast:
    gp = np.asarray(going_probs, dtype=float)
    gh = np.ones(gp.size, dtype=int) if going_heads is None else np.asarray(going_heads)
    wp = np.asarray(waitlist_probs, dtype=float)
    wh = np.ones(wp.size, dtype=int) if waitlist_heads is None else np.asarray(waitlist_heads)
    pmf = heads_pmf(gp, gh, shock_sd=shock_sd)
    return TurnoutForecast(
        event_id="pymeets-2026-11",
        model_name="test",
        going_probs=gp,
        going_heads=gh,
        waitlist_probs=wp,
        waitlist_heads=wh,
        pmf=pmf,
        expected=expected(pmf),
        p10=quantile(pmf, 0.10),
        p50=quantile(pmf, 0.50),
        p90=quantile(pmf, 0.90),
        historical_show_rate=0.5,
        synthetic=synthetic,
        shock_sd=shock_sd,
    )


def binomial_tail(n: int, p: float, k: int) -> float:
    """P(Binomial(n, p) > k), computed independently of model.turnout."""
    return sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k + 1, n + 1))


# --- waitlist release --------------------------------------------------------------------


def test_releases_exactly_up_to_capacity_when_turnout_is_certain():
    fc = make_forecast([1.0] * 10, waitlist_probs=[1.0] * 10)
    plan = make_plan(fc, PlanInputs(capacity=15))
    assert plan.waitlist_release == 5
    assert plan.overflow_probability == 0.0
    assert plan.seats == plan.certificates == 15
    assert (
        plan.notes[0] == "Release 5 of 10 waitlist spots: still no chance of more than 15 people."
    )


def test_release_stops_at_the_overflow_risk():
    going, risk, capacity = 20, 0.10, 25
    fc = make_forecast([0.5] * going, waitlist_probs=[0.5] * 40)
    plan = make_plan(fc, PlanInputs(capacity=capacity, overflow_risk=risk))
    allowed = [k for k in range(41) if binomial_tail(going + k, 0.5, capacity) <= risk]
    assert plan.waitlist_release == max(allowed) == 22
    assert plan.overflow_probability == pytest.approx(
        binomial_tail(going + plan.waitlist_release, 0.5, capacity)
    )
    assert plan.overflow_probability <= risk
    assert binomial_tail(going + plan.waitlist_release + 1, 0.5, capacity) > risk


def test_release_note_reads_naturally():
    fc = make_forecast([0.5] * 20, waitlist_probs=[0.5] * 40)
    plan = make_plan(fc, PlanInputs(capacity=25))
    assert plan.notes[0] == (
        "Release 22 of 40 waitlist spots: still only an 8% chance of more than 25 people."
    )


def test_release_follows_waitlist_order_without_skipping():
    # The first person in line brings 2 guests and does not fit; later singles would, but
    # skipping them ahead would be unfair, so nobody is released.
    fc = make_forecast([1.0] * 10, waitlist_probs=[1.0, 1.0, 1.0], waitlist_heads=[3, 1, 1])
    plan = make_plan(fc, PlanInputs(capacity=12))
    assert plan.waitlist_release == 0
    assert plan.overflow_probability == 0.0
    assert plan.notes[0].startswith("Keep the waitlist closed for now: releasing even 1 spot")
    assert "a 100% chance of more than 12 people" in plan.notes[0]


def test_going_rsvps_alone_over_the_risk_release_nobody():
    fc = make_forecast([0.9] * 30, waitlist_probs=[0.5] * 5)
    plan = make_plan(fc, PlanInputs(capacity=20))
    assert plan.waitlist_release == 0
    assert plan.overflow_probability == pytest.approx(binomial_tail(30, 0.9, 20))
    assert plan.notes[0].startswith("Keep the waitlist closed: the going RSVPs alone have")
    assert "of more than 20 people (limit 10%)" in plan.notes[0]


def test_capacity_below_rsvp_count_can_still_overbook_safely():
    # 30 RSVPs for 20 seats, but only ~30% show up: releasing more is still safe.
    fc = make_forecast([0.3] * 30, waitlist_probs=[0.3] * 10)
    plan = make_plan(fc, PlanInputs(capacity=20))
    assert plan.waitlist_release == 10
    assert plan.overflow_probability <= 0.10


@pytest.mark.parametrize("capacity", [None, float("nan")])
def test_no_capacity_releases_the_whole_waitlist(capacity):
    fc = make_forecast([0.5] * 10, waitlist_probs=[0.5] * 4)
    plan = make_plan(fc, PlanInputs(capacity=capacity))
    assert plan.waitlist_release == 4
    assert plan.overflow_probability == 0.0
    assert plan.notes[0] == "No venue limit set, so release the whole waitlist (4 waitlist spots)."


def test_empty_waitlist_with_capacity():
    fc = make_forecast([0.5] * 10)
    plan = make_plan(fc, PlanInputs(capacity=40))
    assert plan.waitlist_release == 0
    assert plan.notes[0] == "No one is on the waitlist. There's no chance of more than 40 people."


def test_empty_waitlist_without_capacity_has_no_waitlist_note():
    plan = make_plan(make_forecast([0.5] * 10), PlanInputs())
    assert not any("waitlist" in note for note in plan.notes)


def test_guests_count_as_heads():
    fc = make_forecast(
        [1.0, 1.0, 1.0], going_heads=[2, 1, 1], waitlist_probs=[1.0], waitlist_heads=[3]
    )
    plan = make_plan(fc, PlanInputs(capacity=7))
    assert plan.waitlist_release == 1
    assert plan.seats == 7
    assert plan.pizza_boxes_naive == math.ceil(7 * 3 / 8)
    assert plan.notes[0] == (
        "Release the 1 waitlist spot (3 people with guests): still no chance of more than 7 people."
    )


def test_overflow_accounts_for_the_shared_shock():
    # Same RSVPs as test_release_stops_at_the_overflow_risk, but whole events swing: released
    # RSVPs swing with the going ones, so fewer spots fit within the same 10% risk.
    going, waiting = [0.5] * 20, [0.5] * 40
    calm = make_plan(make_forecast(going, waitlist_probs=waiting), PlanInputs(capacity=25))
    fc = make_forecast(going, waitlist_probs=waiting, shock_sd=0.5)
    plan = make_plan(fc, PlanInputs(capacity=25))

    assert 0 < plan.waitlist_release < calm.waitlist_release == 22
    total = heads_pmf([0.5] * (20 + plan.waitlist_release), shock_sd=0.5)
    assert plan.overflow_probability == pytest.approx(prob_exceeds(total, 25))
    assert plan.overflow_probability <= 0.10
    one_more = heads_pmf([0.5] * (21 + plan.waitlist_release), shock_sd=0.5)
    assert prob_exceeds(one_more, 25) > 0.10
    # Convolving the released RSVPs in independently would understate the risk.
    independent = add_rsvps(fc.pmf, [0.5] * plan.waitlist_release)
    assert prob_exceeds(independent, 25) < plan.overflow_probability


def test_food_and_seats_use_the_shared_shock_total():
    fc = make_forecast([0.6] * 30, waitlist_probs=[0.5] * 6, shock_sd=0.4)
    plan = make_plan(fc, PlanInputs(capacity=None, food_quantile=0.8))
    total = heads_pmf([0.6] * 30 + [0.5] * 6, shock_sd=0.4)
    assert plan.waitlist_release == 6
    assert plan.food_heads == quantile(total, 0.8)
    assert plan.seats == plan.certificates == quantile(total, 0.9)
    calm = make_plan(make_forecast([0.6] * 30, waitlist_probs=[0.5] * 6), PlanInputs())
    assert plan.seats > calm.seats  # the same expected crowd, a wider range to seat


def test_total_heads_pmf_releases_in_waitlist_order():
    fc = make_forecast([0.7] * 10, waitlist_probs=[0.9, 0.1], waitlist_heads=[2, 1], shock_sd=0.3)
    np.testing.assert_allclose(total_heads_pmf(fc), fc.pmf, atol=1e-15)
    np.testing.assert_allclose(
        total_heads_pmf(fc, released=1),
        heads_pmf([0.7] * 10 + [0.9], [1] * 10 + [2], shock_sd=0.3),
        atol=1e-15,
    )


# --- food, seats, costs ------------------------------------------------------------------


def test_food_matches_the_readme_example():
    # 40 RSVPs, exactly 24 of whom will come: 24 people x 3 slices / 8 = 9 boxes, not 15.
    fc = make_forecast([1.0] * 24 + [0.0] * 16)
    plan = make_plan(fc, PlanInputs(price_per_box=450))
    assert (plan.food_heads, plan.pizza_boxes, plan.pizza_boxes_naive) == (24, 9, 15)
    assert (
        "Order 9 boxes (feeds 24 at 3 slices each). Ordering for every RSVP would be 15 boxes."
        in plan.notes
    )
    assert (plan.cost_php, plan.cost_naive_php, plan.savings_php) == (4050.0, 6750.0, 2700.0)
    assert "That's ₱4,050 instead of ₱6,750, saving ₱2,700." in plan.notes


def test_food_uses_the_quantile_of_total_heads_including_released_waitlist():
    fc = make_forecast([0.6] * 30, waitlist_probs=[0.5] * 6)
    inputs = PlanInputs(capacity=None, food_quantile=0.8, slices_per_person=2.5, slices_per_box=6)
    plan = make_plan(fc, inputs)
    total = add_rsvps(fc.pmf, fc.waitlist_probs, fc.waitlist_heads)
    assert plan.food_heads == quantile(total, 0.8)
    assert plan.pizza_boxes == math.ceil(plan.food_heads * 2.5 / 6)
    assert plan.pizza_boxes_naive == math.ceil(36 * 2.5 / 6)
    assert plan.seats == plan.certificates == quantile(total, 0.9)
    assert f"feeds {plan.food_heads} at 2.5 slices each" in plan.notes[1]


def test_higher_food_quantile_never_orders_less():
    fc = make_forecast(np.linspace(0.2, 0.9, 50))
    boxes = [make_plan(fc, PlanInputs(food_quantile=q)).pizza_boxes for q in (0.5, 0.75, 0.9, 1.0)]
    assert boxes == sorted(boxes)
    assert boxes[-1] <= make_plan(fc, PlanInputs()).pizza_boxes_naive


def test_box_math_is_not_fooled_by_float_noise():
    # 0.1 + 0.2 slices style noise must not round 9 boxes up to 10
    fc = make_forecast([1.0] * 30)
    plan = make_plan(fc, PlanInputs(slices_per_person=0.1 * 3 * 10, slices_per_box=10))
    assert plan.pizza_boxes == 9


def test_no_price_hides_costs():
    plan = make_plan(make_forecast([0.5] * 10), PlanInputs(price_per_box=0))
    assert (plan.cost_php, plan.cost_naive_php, plan.savings_php) == (0.0, 0.0, 0.0)
    assert not any("₱" in note for note in plan.notes)


def test_same_order_as_feeding_everyone_says_so():
    plan = make_plan(make_forecast([1.0] * 8), PlanInputs())
    assert plan.pizza_boxes == plan.pizza_boxes_naive == 3
    assert (
        "Order 3 boxes (feeds 8 at 3 slices each), the same as ordering for every RSVP."
        in plan.notes
    )


def test_seats_note():
    plan = make_plan(make_forecast([1.0] * 12), PlanInputs())
    assert "Prepare 12 seats and e-certificates (enough 9 times out of 10)." in plan.notes


def test_no_rsvps():
    fc = make_forecast([])
    plan = make_plan(fc, PlanInputs(capacity=50, price_per_box=450))
    assert (plan.food_heads, plan.pizza_boxes, plan.pizza_boxes_naive, plan.seats) == (0, 0, 0, 0)
    assert plan.cost_php == plan.savings_php == 0.0
    assert plan.waitlist_release == 0
    assert plan.notes[-1] == "No RSVPs yet, so there's nothing to order."


def test_no_rsvps_tolerates_an_empty_pmf():
    fc = make_forecast([])
    fc = TurnoutForecast(**{**fc.__dict__, "pmf": np.zeros(0)})
    assert make_plan(fc, PlanInputs()).pizza_boxes == 0


def test_plan_keeps_the_event_id():
    assert make_plan(make_forecast([0.5]), PlanInputs()).event_id == "pymeets-2026-11"


@pytest.mark.parametrize(
    "inputs",
    [
        PlanInputs(capacity=-1),
        PlanInputs(slices_per_box=0),
        PlanInputs(slices_per_person=-1),
        PlanInputs(price_per_box=-5),
        PlanInputs(food_quantile=0),
        PlanInputs(food_quantile=1.5),
        PlanInputs(overflow_risk=-0.1),
        PlanInputs(overflow_risk=1.1),
    ],
)
def test_rejects_impossible_inputs(inputs):
    with pytest.raises(ValueError):
        make_plan(make_forecast([0.5] * 5), inputs)


# --- wording -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("going", "capacity", "risk", "phrase"),
    [
        ([0.9] * 30, 25, 0.10, "have an 82% chance of more than 25 people (limit 10%)"),
        ([0.5] * 40, 24, 0.05, "have an 8% chance of more than 24 people (limit 5%)"),
        ([0.5] * 40, 22, 0.10, "have a 21% chance of more than 22 people"),
        ([0.9] * 30, 10, 0.10, "have more than a 99% chance of more than 10 people"),
    ],
)
def test_chance_wording(going, capacity, risk, phrase):
    fc = make_forecast(going, waitlist_probs=[0.5])
    plan = make_plan(fc, PlanInputs(capacity=capacity, overflow_risk=risk))
    assert phrase in plan.notes[0], plan.notes[0]


@pytest.mark.parametrize(
    ("amount", "text"),
    [(4050, "₱4,050"), (4054.5, "₱4,054.50"), (0, "₱0"), (1234567.0, "₱1,234,567")],
)
def test_format_php(amount, text):
    assert format_php(amount) == text
