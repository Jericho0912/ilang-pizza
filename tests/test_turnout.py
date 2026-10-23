from __future__ import annotations

import itertools

import numpy as np
import pytest

from ilang_pizza.model.turnout import (
    Headcount,
    add_rsvps,
    expected,
    fit_shock_sd,
    heads_pmf,
    prob_exceeds,
    quantile,
)


def brute_force_pmf(probs: np.ndarray, heads: np.ndarray) -> np.ndarray:
    """Sum over every show/no-show combination."""
    pmf = np.zeros(int(heads.sum()) + 1)
    for shows in itertools.product((0, 1), repeat=len(probs)):
        mask = np.array(shows, dtype=bool)
        pmf[heads[mask].sum()] += np.prod(np.where(mask, probs, 1 - probs))
    return pmf


@pytest.mark.parametrize("seed", range(5))
def test_matches_brute_force_with_guests(seed):
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0, 1, size=9)
    heads = rng.integers(1, 4, size=9)
    np.testing.assert_allclose(heads_pmf(probs, heads), brute_force_pmf(probs, heads), atol=1e-12)


def test_matches_simulation():
    rng = np.random.default_rng(42)
    probs = rng.uniform(0.1, 0.9, size=150)
    heads = rng.choice([1, 1, 1, 2, 3], size=150)
    draws = (rng.random((100_000, probs.size)) < probs) @ heads
    pmf = heads_pmf(probs, heads)

    simulated = np.bincount(draws, minlength=pmf.size) / draws.size
    assert np.abs(simulated - pmf).max() < 0.004
    for q in (0.1, 0.5, 0.9):
        assert abs(quantile(pmf, q) - np.quantile(draws, q)) <= 1


def test_stable_for_500_rsvps():
    rng = np.random.default_rng(7)
    probs = rng.uniform(0, 1, size=500)
    heads = rng.integers(1, 3, size=500)
    pmf = heads_pmf(probs, heads)

    assert pmf.size == heads.sum() + 1
    assert np.all(pmf >= 0) and np.isfinite(pmf).all()
    assert pmf.sum() == pytest.approx(1.0, abs=1e-12)
    mean = expected(pmf)
    variance = np.arange(pmf.size) ** 2 @ pmf - mean**2
    assert mean == pytest.approx(probs @ heads, rel=1e-10)
    assert variance == pytest.approx((heads**2 * probs * (1 - probs)).sum(), rel=1e-8)


def test_no_rsvps_means_nobody_comes():
    pmf = heads_pmf(np.array([]))
    np.testing.assert_array_equal(pmf, [1.0])
    assert quantile(pmf, 0.9) == 0
    assert expected(pmf) == 0.0
    assert prob_exceeds(pmf, 0) == 0.0


def test_certain_rsvps_and_zero_heads():
    pmf = heads_pmf(np.array([1.0, 0.0, 1.0, 0.5]), np.array([2, 5, 1, 0]))
    assert pmf.size == 2 + 5 + 1 + 0 + 1  # support runs up to every head showing up
    np.testing.assert_allclose(pmf, np.eye(pmf.size)[3])  # 2 + 1 heads for sure


def test_heads_default_to_one_each():
    probs = np.array([0.2, 0.7, 0.4])
    np.testing.assert_allclose(heads_pmf(probs), heads_pmf(probs, np.ones(3, dtype=int)))


def test_add_rsvps_equals_scoring_everyone_together():
    rng = np.random.default_rng(3)
    going, waiting = rng.uniform(size=30), rng.uniform(size=8)
    going_heads, waiting_heads = rng.integers(1, 3, 30), rng.integers(1, 3, 8)
    combined = heads_pmf(np.r_[going, waiting], np.r_[going_heads, waiting_heads])
    stepwise = add_rsvps(heads_pmf(going, going_heads), waiting, waiting_heads)
    np.testing.assert_allclose(stepwise, combined, atol=1e-14)
    np.testing.assert_allclose(add_rsvps(combined, np.array([])), combined)


def test_quantile_is_smallest_k_with_cdf_at_least_q():
    pmf = np.array([0.1, 0.2, 0.3, 0.4])  # cdf 0.1, 0.3, 0.6, 1.0
    assert quantile(pmf, 0.0) == 0
    assert quantile(pmf, 0.1) == 0
    assert quantile(pmf, 0.1000001) == 1
    assert quantile(pmf, 0.3) == 1
    assert quantile(pmf, 0.6) == 2
    assert quantile(pmf, 0.61) == 3
    assert quantile(pmf, 1.0) == 3
    # float noise in the cumulative sum must not push q=1 past the last entry
    assert quantile(heads_pmf(np.full(200, 0.3)), 1.0) <= 200


def test_prob_exceeds_is_the_upper_tail():
    pmf = np.array([0.1, 0.2, 0.3, 0.4])
    assert prob_exceeds(pmf, -1) == 1.0
    assert prob_exceeds(pmf, 0) == pytest.approx(0.9)
    assert prob_exceeds(pmf, 2) == pytest.approx(0.4)
    assert prob_exceeds(pmf, 3) == 0.0
    assert prob_exceeds(pmf, 99) == 0.0


def test_expected():
    assert expected(np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("probs", "heads"),
    [
        ([0.5, 1.2], None),
        ([0.5, np.nan], None),
        ([0.5, 0.5], [1]),
        ([0.5], [-1]),
        ([0.5], [1.5]),
    ],
)
def test_rejects_bad_rsvps(probs, heads):
    with pytest.raises(ValueError):
        heads_pmf(np.array(probs), None if heads is None else np.array(heads))


def test_rejects_bad_pmf_and_q():
    with pytest.raises(ValueError):
        add_rsvps(np.array([]), np.array([0.5]))
    with pytest.raises(ValueError):
        quantile(np.array([0.0, 0.0]), 0.5)
    with pytest.raises(ValueError):
        quantile(np.array([1.0]), 1.5)


# --- the shared event-day shock ----------------------------------------------------------


def independent_dp(probs: np.ndarray, heads: np.ndarray) -> np.ndarray:
    """The plain Poisson-binomial recursion, as turnout computed it before the shock."""
    out = np.zeros(1 + int(heads.sum()))
    out[0], size = 1.0, 1
    for p, h in zip(probs, heads):
        came = p * out[:size]
        out[:size] *= 1.0 - p
        out[h : size + h] += came
        size += h
    return out / out.sum()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mixture_by_integration(probs, heads, shock_sd, points=4001):
    """The shocked pmf by brute force: per-RSVP offsets found by bisection, then a dense
    integral over s of the independent pmf, with no quadrature tricks."""
    s = np.linspace(-8 * shock_sd, 8 * shock_sd, points)
    density = np.exp(-0.5 * (s / shock_sd) ** 2)
    density /= np.trapezoid(density, s)
    lo, hi = np.full(probs.size, -30.0), np.full(probs.size, 30.0)
    for _ in range(100):  # E_s[sigmoid(a + s)] = p
        mid = (lo + hi) / 2
        mean = np.trapezoid(sigmoid(mid[:, None] + s[None, :]) * density, s, axis=1)
        lo, hi = np.where(mean < probs, mid, lo), np.where(mean < probs, hi, mid)
    base = (lo + hi) / 2
    pmfs = np.array([independent_dp(sigmoid(base + x), heads) for x in s])
    return np.trapezoid(pmfs * density[:, None], s, axis=0)


def variance(pmf):
    return np.arange(pmf.size) ** 2 @ pmf - expected(pmf) ** 2


@pytest.mark.parametrize("seed", range(3))
def test_zero_shock_is_the_independent_pmf_bit_for_bit(seed):
    rng = np.random.default_rng(seed)
    probs, heads = rng.uniform(size=200), rng.integers(1, 4, size=200)
    reference = independent_dp(probs, heads)
    np.testing.assert_array_equal(heads_pmf(probs, heads), reference)
    np.testing.assert_array_equal(heads_pmf(probs, heads, shock_sd=0.0), reference)
    np.testing.assert_array_equal(Headcount().add(probs, heads).pmf(), reference)
    np.testing.assert_array_equal(add_rsvps(np.ones(1), probs, heads), reference)


def test_shock_matches_brute_force_integration():
    rng = np.random.default_rng(5)
    probs, heads = rng.uniform(0.05, 0.95, size=8), rng.integers(1, 3, size=8)
    for shock_sd in (0.3, 0.9):
        np.testing.assert_allclose(
            heads_pmf(probs, heads, shock_sd=shock_sd),
            mixture_by_integration(probs, heads, shock_sd),
            atol=2e-6,
        )


def test_shock_widens_the_range_but_keeps_the_mean():
    rng = np.random.default_rng(9)
    probs, heads = rng.uniform(0, 1, size=300), rng.integers(1, 3, size=300)
    probs[:3] = [0.0, 1.0, 1.0]  # certain RSVPs stay certain under any shock
    spreads = []
    for shock_sd in (0.0, 0.2, 0.5, 1.0):
        pmf = heads_pmf(probs, heads, shock_sd=shock_sd)
        assert pmf.sum() == pytest.approx(1.0, abs=1e-12) and np.all(pmf >= 0)
        assert expected(pmf) == pytest.approx(probs @ heads, rel=1e-10)
        assert pmf[: int(heads[1] + heads[2])].sum() == 0.0  # the two sure RSVPs always come
        spreads.append(variance(pmf))
    assert spreads == sorted(spreads) and spreads[-1] > 5 * spreads[0]


def test_adding_rsvps_one_at_a_time_matches_a_batch():
    rng = np.random.default_rng(2)
    going, waiting = rng.uniform(size=40), rng.uniform(size=6)
    going_heads, waiting_heads = rng.integers(1, 3, 40), rng.integers(1, 3, 6)
    start = Headcount(0.6).add(going, going_heads)
    before = start.pmf()

    stepwise = start
    for p, h in zip(waiting, waiting_heads):
        stepwise = stepwise.add([p], [h])
    batch = heads_pmf(np.r_[going, waiting], np.r_[going_heads, waiting_heads], shock_sd=0.6)
    np.testing.assert_allclose(stepwise.pmf(), batch, atol=1e-14)
    np.testing.assert_array_equal(start.pmf(), before)  # add() leaves the original alone
    # Released RSVPs share the shock: wider than convolving them in independently.
    assert variance(stepwise.pmf()) > variance(add_rsvps(before, waiting, waiting_heads))


def simulated_events(shock_sd, n_events=150, seed=0):
    """Events whose RSVPs share a N(0, shock_sd^2) log-odds shock, with calibrated probs."""
    rng = np.random.default_rng(seed)
    s = np.linspace(-6, 6, 1201)
    density = np.exp(-0.5 * s**2) / np.sqrt(2 * np.pi)
    events = []
    for _ in range(n_events):
        base = rng.normal(0, 0.8, size=rng.integers(40, 120))
        probs = np.trapezoid(sigmoid(base[:, None] + shock_sd * s) * density, s, axis=1)
        came = rng.random(base.size) < sigmoid(base + shock_sd * rng.normal())
        events.append((probs, np.ones(base.size, dtype=int), int(came.sum())))
    return events


@pytest.mark.parametrize("true_sd", [0.0, 0.5])
def test_shock_mle_recovers_a_known_spread(true_sd):
    assert fit_shock_sd(simulated_events(true_sd)) == pytest.approx(true_sd, abs=0.15)


def test_rejects_bad_shock():
    for bad in (-0.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            Headcount(bad)
