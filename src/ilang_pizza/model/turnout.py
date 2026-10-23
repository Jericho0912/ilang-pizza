"""Exact headcount distribution from per-RSVP show-up probabilities.

RSVP ``i`` brings ``heads[i]`` people (1 + guests) with probability ``probs[i]`` and nobody
otherwise, independently of the other RSVPs. The total headcount then follows a
Poisson-binomial distribution over integer head weights, which we compute exactly by adding
one RSVP at a time (dynamic programming). Each step only mixes non-negative numbers, so the
result stays accurate for hundreds of RSVPs.
"""

from __future__ import annotations

import numpy as np


def heads_pmf(probs: np.ndarray, heads: np.ndarray | None = None) -> np.ndarray:
    """``pmf[k]`` = P(exactly ``k`` heads show up). No RSVPs gives ``[1.0]``."""
    return add_rsvps(np.ones(1), probs, heads)


def add_rsvps(pmf: np.ndarray, probs: np.ndarray, heads: np.ndarray | None = None) -> np.ndarray:
    """The pmf of the headcount after adding more independent RSVPs (e.g. released waitlist)."""
    base = _as_pmf(pmf)
    probs, heads = _as_rsvps(probs, heads)

    out = np.zeros(base.size + int(heads.sum()))
    out[: base.size] = base
    size = base.size  # entries that can be non-zero so far
    for p, h in zip(probs, heads):
        if h == 0:
            continue
        came = p * out[:size]  # a copy, taken before the no-show update below
        out[:size] *= 1.0 - p
        out[h : size + h] += came
        size += h
    return out / out.sum()


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
