from __future__ import annotations

import itertools

import numpy as np
import pytest

from ilang_pizza.model.turnout import add_rsvps, expected, heads_pmf, prob_exceeds, quantile


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
