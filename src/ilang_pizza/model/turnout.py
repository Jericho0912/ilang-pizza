"""Headcount distribution from per-RSVP show-up probabilities.

RSVP ``i`` brings ``heads[i]`` people (1 + guests) with probability ``probs[i]`` and nobody
otherwise. If RSVPs were independent, the total would follow a Poisson-binomial distribution
over integer head weights, which we compute exactly by adding one RSVP at a time (dynamic
programming; each step only mixes non-negative numbers, so it stays accurate for hundreds of
RSVPs).

Real events also swing as a whole: a downpour, a transport strike or a rival event moves
everyone's odds together. We model that as one shock ``s ~ N(0, shock_sd²)`` added to every
RSVP's log-odds, and mix the Poisson-binomial over ``s`` with Gauss-Hermite quadrature. Each
RSVP keeps its own chance ``probs[i]`` on average, so the expected headcount is unchanged; the
shock only makes RSVPs move together, which widens the range. ``shock_sd = 0`` is the plain
independent Poisson-binomial.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

import numpy as np

QUADRATURE_NODES = 15
# Candidate shock sizes for fit_shock_sd: 0 (independent RSVPs) to 1.2 log-odds.
SHOCK_GRID = np.round(np.arange(0.0, 1.2001, 0.05), 2)


class Headcount:
    """The headcount distribution of a growing set of RSVPs that share one event-day shock.

    Keeps one Poisson-binomial per quadrature node of the shock, so RSVPs can be added in
    batches or one at a time (the planner releases the waitlist that way) and every RSVP feels
    the same shock. Immutable: ``add`` returns a new ``Headcount``.
    """

    def __init__(self, shock_sd: float = 0.0) -> None:
        if not np.isfinite(shock_sd) or shock_sd < 0:
            raise ValueError(f"shock_sd must be a non-negative number, got {shock_sd}")
        self.shock_sd = float(shock_sd)
        if self.shock_sd == 0:
            self._shifts, self._weights = np.zeros(1), np.ones(1)
        else:
            nodes, weights = np.polynomial.hermite_e.hermegauss(QUADRATURE_NODES)
            self._shifts, self._weights = self.shock_sd * nodes, weights / weights.sum()
        self._rows = np.ones((self._shifts.size, 1))  # one (unnormalized) pmf per node

    def add(self, probs: np.ndarray, heads: np.ndarray | None = None) -> Headcount:
        """A new ``Headcount`` that also counts these RSVPs."""
        probs, heads = _as_rsvps(probs, heads)
        new = copy.copy(self)
        new._rows = _add_rsvps(self._rows, self._at_nodes(probs), heads)
        return new

    def pmf(self) -> np.ndarray:
        """``pmf[k]`` = P(exactly ``k`` heads show up)."""
        mix = self._rows[0] if self._weights.size == 1 else self._weights @ self._rows
        return mix / mix.sum()

    def _at_nodes(self, probs: np.ndarray) -> np.ndarray:
        """P(show) per node and RSVP, keeping each RSVP's average over the nodes at ``probs``."""
        if self._weights.size == 1:
            return probs[None, :]
        at_nodes = np.repeat(probs[None, :], self._shifts.size, axis=0)
        unsure = (probs > 1e-9) & (probs < 1 - 1e-9)  # (near-)certain RSVPs stay as they are
        p = probs[unsure]
        logit = np.log(p) - np.log1p(-p)
        # Newton steps on the per-RSVP offset so the node-weighted mean is exactly p again.
        for _ in range(8):
            q = _sigmoid(logit[None, :] + self._shifts[:, None])
            logit -= (self._weights @ q - p) / (self._weights @ (q * (1 - q)))
        at_nodes[:, unsure] = _sigmoid(logit[None, :] + self._shifts[:, None])
        return at_nodes


def heads_pmf(
    probs: np.ndarray, heads: np.ndarray | None = None, *, shock_sd: float = 0.0
) -> np.ndarray:
    """``pmf[k]`` = P(exactly ``k`` heads show up). No RSVPs gives ``[1.0]``."""
    return Headcount(shock_sd).add(probs, heads).pmf()


def add_rsvps(pmf: np.ndarray, probs: np.ndarray, heads: np.ndarray | None = None) -> np.ndarray:
    """The pmf after adding more RSVPs that are independent of the ones already in ``pmf``.

    To add RSVPs under the same shared shock, build a ``Headcount`` instead.
    """
    base = _as_pmf(pmf)
    probs, heads = _as_rsvps(probs, heads)
    out = _add_rsvps(base[None, :], probs[None, :], heads)[0]
    return out / out.sum()


def fit_shock_sd(
    events: Iterable[tuple[np.ndarray, np.ndarray, int]], *, grid: np.ndarray = SHOCK_GRID
) -> float:
    """Maximum-likelihood ``shock_sd``: the grid value that makes past headcounts most likely.

    ``events`` holds ``(probs, heads, actual_heads)`` per past event, where ``probs`` must be
    out-of-sample predictions (made without that event's attendance).
    """
    loglik = sum((shock_loglik(*event, grid=grid) for event in events), np.zeros(len(grid)))
    return float(grid[int(np.argmax(loglik))])


def shock_loglik(
    probs: np.ndarray, heads: np.ndarray, actual: int, *, grid: np.ndarray = SHOCK_GRID
) -> np.ndarray:
    """log P(``actual`` heads) for each candidate ``shock_sd`` in ``grid``."""
    probs, heads = _as_rsvps(probs, heads)
    candidates = [Headcount(float(shock_sd)) for shock_sd in grid]
    # One pass for every candidate at once: stack all their quadrature nodes as rows.
    nodes = np.vstack([c._at_nodes(probs) for c in candidates])
    rows = _add_rsvps(np.ones((nodes.shape[0], 1)), nodes, heads)
    chance = rows[:, int(actual)] / rows.sum(axis=1)
    sizes = np.cumsum([c._weights.size for c in candidates])[:-1]
    mixed = [c._weights @ part for c, part in zip(candidates, np.split(chance, sizes))]
    return np.log(np.maximum(mixed, 1e-300))


def quantile(pmf: np.ndarray, q: float) -> int:
    """The smallest ``k`` with P(X <= k) >= ``q`` (within float precision)."""
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    cdf = np.cumsum(_as_pmf(pmf))
    cdf /= cdf[-1]  # so the last entry is exactly 1 despite rounding
    return int(min(np.searchsorted(cdf, q, side="left"), cdf.size - 1))


def prob_exceeds(pmf: np.ndarray, k: int) -> float:
    """P(X > ``k``)."""
    pmf = _as_pmf(pmf)
    if k < 0:
        return 1.0
    return float(min(pmf[int(k) + 1 :].sum() / pmf.sum(), 1.0))


def expected(pmf: np.ndarray) -> float:
    """E[X]."""
    pmf = _as_pmf(pmf)
    return float(np.arange(pmf.size) @ pmf / pmf.sum())


def _add_rsvps(rows: np.ndarray, probs: np.ndarray, heads: np.ndarray) -> np.ndarray:
    """Add RSVPs to each row's pmf; row ``r`` uses the probabilities ``probs[r]``."""
    out = np.zeros((rows.shape[0], rows.shape[1] + int(heads.sum())))
    size = rows.shape[1]  # entries that can be non-zero so far
    out[:, :size] = rows
    for j, h in enumerate(heads):
        if h == 0:
            continue
        p = probs[:, j : j + 1]
        came = p * out[:, :size]  # a copy, taken before the no-show update below
        out[:, :size] *= 1.0 - p
        out[:, h : size + h] += came
        size += h
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _as_pmf(pmf: np.ndarray) -> np.ndarray:
    pmf = np.asarray(pmf, dtype=float)
    if pmf.ndim != 1 or pmf.size == 0:
        raise ValueError("pmf must be a non-empty 1-D array")
    if not np.isfinite(pmf).all() or np.any(pmf < 0) or pmf.sum() <= 0:
        raise ValueError("pmf must hold non-negative numbers with a positive sum")
    return pmf


def _as_rsvps(probs: np.ndarray, heads: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=float).ravel()
    if np.isnan(probs).any() or np.any((probs < 0) | (probs > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    if heads is None:
        return probs, np.ones(probs.size, dtype=int)
    heads = np.asarray(heads, dtype=float).ravel()
    if heads.size != probs.size:
        raise ValueError(f"got {probs.size} probabilities but {heads.size} head counts")
    if not np.isfinite(heads).all() or np.any(heads < 0) or np.any(heads != np.round(heads)):
        raise ValueError("heads must be non-negative whole numbers")
    return probs, heads.astype(int)
