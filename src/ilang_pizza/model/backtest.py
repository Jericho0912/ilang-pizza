"""Rolling-origin backtest: replay past events as if each were the night before.

Every evaluated event is forecast only from events that started before it, by the chosen
model and three baselines, and each guess is scored against the real check-ins.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ilang_pizza.config import RULE_OF_THUMB_SHOW_RATE
from ilang_pizza.features import build
from ilang_pizza.model import turnout
from ilang_pizza.model.predict import (
    MIN_TRAIN_EVENTS,
    MIN_TRAIN_RSVPS,
    forecast_event,
    rsvp_heads,
    training_rsvps,
)
from ilang_pizza.schema import BacktestResult, Dataset

if TYPE_CHECKING:
    from ilang_pizza.features.weather import WeatherProvider

# per_event column holding each method's point prediction
PREDICTION_COLUMNS = {
    "logistic": "logistic_expected",
    "history": "history_pred",
    "rule": "rule_pred",
}


def backtest(
    ds: Dataset,
    *,
    model_kind: str = "tabpfn",
    min_train_events: int = 6,
    last_n: int | None = None,
    weather: WeatherProvider | None = None,
    rule_rate: float = RULE_OF_THUMB_SHOW_RATE,
    progress: Callable[[int, int], None] | None = None,
) -> BacktestResult:
    """Forecast each labeled event (oldest first) from the events that started before it.

    Events with fewer than ``min_train_events`` earlier labeled events are skipped, and
    ``last_n`` keeps only the most recent evaluated events. ``progress(done, total)`` is called
    before the first event and after each one.
    """
    event_ids = evaluable_events(ds, min_train_events=min_train_events)
    if last_n is not None:
        event_ids = event_ids[-last_n:] if last_n > 0 else []
    if not event_ids:
        raise ValueError(
            f"nothing to backtest: no event has check-in data from at least "
            f"{max(min_train_events, MIN_TRAIN_EVENTS)} earlier events "
            f"({len(ds.labeled_event_ids())} events have check-ins)"
        )

    features = build.build_features(ds, weather=weather)  # leakage-safe, so build once
    rows, model_name = [], ""
    if progress is not None:
        progress(0, len(event_ids))
    for done, event_id in enumerate(event_ids, start=1):
        row, model_name = _replay(ds, event_id, model_kind, features, rule_rate)
        rows.append(row)
        if progress is not None:
            progress(done, len(event_ids))

    per_event = pd.DataFrame(rows)
    main = "tabpfn" if model_kind.startswith("tabpfn") else model_kind
    return BacktestResult(
        per_event=per_event,
        metrics=score(per_event, main),
        model_name=model_name,
        synthetic=ds.synthetic,
    )


def evaluable_events(ds: Dataset, *, min_train_events: int) -> list[str]:
    """Labeled events, oldest first, with enough earlier history to be forecast."""
    needed = max(min_train_events, MIN_TRAIN_EVENTS)
    ready = []
    for event_id in ds.labeled_event_ids():
        past = training_rsvps(ds, before=ds.event(event_id)["start"])
        if past["event_id"].nunique() >= needed and len(past) >= MIN_TRAIN_RSVPS:
            ready.append(event_id)
    return ready


def score(per_event: pd.DataFrame, main: str) -> pd.DataFrame:
    """MAE, MAPE, bias (predicted minus actual) and event count per method; the main model
    (first row) also gets coverage_80, the share of events inside its 80% range."""
    actual = per_event["actual_heads"].astype(float)
    columns = {main: "model_expected"} | {k: v for k, v in PREDICTION_COLUMNS.items() if k != main}
    metrics = {}
    for method, column in columns.items():
        error = per_event[column].astype(float) - actual
        some = actual > 0  # MAPE is undefined for an event nobody came to
        metrics[method] = {
            "mae": error.abs().mean(),
            "mape": (error.abs()[some] / actual[some]).mean() if some.any() else np.nan,
            "bias": error.mean(),
            "n_events": len(error),
            "coverage_80": np.nan,
        }
    inside = (per_event["model_p10"] <= actual) & (actual <= per_event["model_p90"])
    metrics[main]["coverage_80"] = inside.mean()
    return pd.DataFrame.from_dict(metrics, orient="index")


def _replay(
    ds: Dataset, event_id: str, model_kind: str, features: pd.DataFrame, rule_rate: float
) -> tuple[dict, str]:
    """One per_event row, scored on the going RSVPs whose attendance is known."""
    event = ds.event(event_id)
    rsvps = ds.rsvps_for(event_id)
    going = rsvps[rsvps["status"] == "going"]
    known = going["attended"].notna().to_numpy()
    heads = rsvp_heads(going)[known]

    def forecast(kind: str):
        return forecast_event(ds, event_id, model_kind=kind, features=features)

    main = forecast(model_kind)
    pmf = turnout.heads_pmf(main.going_probs[known], heads)
    others = {
        kind: main if kind == model_kind else forecast(kind) for kind in ("logistic", "history")
    }
    row = {
        "event_id": event_id,
        "title": event["title"],
        "series": event["series"],
        "start": event["start"],
        "rsvp_heads": int(heads.sum()),
        "actual_heads": int(going["attended"].to_numpy()[known] @ heads),
        "model_expected": turnout.expected(pmf),
        "model_p10": turnout.quantile(pmf, 0.10),
        "model_p90": turnout.quantile(pmf, 0.90),
        "rule_pred": rule_rate * heads.sum(),
        "history_pred": float(others["history"].going_probs[known] @ heads),
        "logistic_expected": float(others["logistic"].going_probs[known] @ heads),
    }
    return row, main.model_name
