from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ilang_pizza.features.build import build_features
from ilang_pizza.model import predict, turnout
from ilang_pizza.model.backtest import backtest, evaluable_events, score

PER_EVENT_COLUMNS = [
    "event_id",
    "title",
    "series",
    "start",
    "rsvp_heads",
    "actual_heads",
    "model_expected",
    "model_p10",
    "model_p90",
    "rule_pred",
    "history_pred",
    "logistic_expected",
]
METRIC_COLUMNS = ["mae", "mape", "bias", "n_events", "coverage_80"]


@pytest.fixture(scope="module")
def logistic_run(tiny_dataset):
    calls = []
    result = backtest(
        tiny_dataset,
        model_kind="logistic",
        min_train_events=6,
        progress=lambda done, total: calls.append((done, total)),
    )
    return result, calls


def test_per_event_rows(tiny_dataset, logistic_run):
    result, _ = logistic_run
    per = result.per_event
    assert list(per.columns) == PER_EVENT_COLUMNS
    assert list(per["event_id"]) == tiny_dataset.labeled_event_ids()[6:]
    assert per["start"].is_monotonic_increasing

    for row in per.itertuples():
        rsvps = tiny_dataset.rsvps_for(row.event_id).query("status == 'going'")
        heads = 1 + rsvps["guests"]
        assert row.rsvp_heads == heads.sum()
        assert row.actual_heads == (rsvps["attended"] * heads).sum()
    assert ((per["model_p10"] <= per["model_p90"]) & (per["model_p90"] <= per["rsvp_heads"])).all()
    np.testing.assert_allclose(per["rule_pred"], 0.5 * per["rsvp_heads"])
    np.testing.assert_allclose(per["logistic_expected"], per["model_expected"])  # same model
    assert result.model_name == "logistic" and result.synthetic is True


def test_metrics(logistic_run):
    result, _ = logistic_run
    per, metrics = result.per_event, result.metrics
    assert list(metrics.index) == ["logistic", "history", "rule"]
    assert list(metrics.columns) == METRIC_COLUMNS
    for method, column in [("history", "history_pred"), ("rule", "rule_pred")]:
        error = per[column] - per["actual_heads"]
        assert metrics.loc[method, "mae"] == pytest.approx(error.abs().mean())
        assert metrics.loc[method, "bias"] == pytest.approx(error.mean())
        assert metrics.loc[method, "mape"] == pytest.approx(
            (error.abs() / per["actual_heads"]).mean()
        )
        assert metrics.loc[method, "n_events"] == len(per)
        assert np.isnan(metrics.loc[method, "coverage_80"])
    assert 0 <= metrics.loc["logistic", "coverage_80"] <= 1


def test_ranges_use_a_shock_fitted_only_on_earlier_events(tiny_dataset, logistic_run):
    result, _ = logistic_run
    features = build_features(tiny_dataset)
    earlier = []
    for row in result.per_event.itertuples():
        going = tiny_dataset.rsvps_for(row.event_id).query("status == 'going'")
        heads = (1 + going["guests"]).to_numpy()
        fc = predict.forecast_event(
            tiny_dataset, row.event_id, model_kind="logistic", features=features, shock_sd=0.0
        )
        if len(earlier) < predict.MIN_SHOCK_EVENTS:
            shock_sd = predict.PRIOR_SHOCK_SD
        else:
            shock_sd = turnout.fit_shock_sd(earlier)
        pmf = turnout.heads_pmf(fc.going_probs, heads, shock_sd=shock_sd)
        assert (row.model_p10, row.model_p90) == (
            turnout.quantile(pmf, 0.1),
            turnout.quantile(pmf, 0.9),
        ), row.event_id
        assert row.model_expected == pytest.approx(fc.going_probs @ heads)
        earlier.append((fc.going_probs, heads, row.actual_heads))


def test_progress_is_reported(logistic_run):
    result, calls = logistic_run
    n = len(result.per_event)
    assert calls == [(done, n) for done in range(n + 1)]


def test_last_n_and_rule_rate(tiny_dataset, logistic_run):
    full, _ = logistic_run
    result = backtest(tiny_dataset, model_kind="history", last_n=3, rule_rate=0.7)
    assert list(result.per_event["event_id"]) == list(full.per_event["event_id"][-3:])
    np.testing.assert_allclose(result.per_event["rule_pred"], 0.7 * result.per_event["rsvp_heads"])
    assert list(result.metrics.index) == ["history", "logistic", "rule"]


def test_needs_enough_past_events(tiny_dataset):
    with pytest.raises(ValueError, match="nothing to backtest"):
        backtest(tiny_dataset, model_kind="logistic", min_train_events=50)
    # Below the forecast minimum, the first evaluated event still has 3 earlier ones.
    assert (
        evaluable_events(tiny_dataset, min_train_events=1)[0]
        == (tiny_dataset.labeled_event_ids()[predict.MIN_TRAIN_EVENTS])
    )


def test_tabpfn_kinds_report_under_tabpfn(tiny_dataset, monkeypatch):
    real = predict.make_model

    def fake(kind, **kw):
        if not kind.startswith("tabpfn"):
            return real(kind, **kw)
        model = real("logistic", **kw)
        model.name = "tabpfn-v2"
        return model

    monkeypatch.setattr(predict, "make_model", fake)
    result = backtest(tiny_dataset, model_kind="tabpfn", last_n=2)
    assert list(result.metrics.index) == ["tabpfn", "logistic", "history", "rule"]
    assert result.model_name == "tabpfn-v2"


def test_score_by_hand():
    per = pd.DataFrame(
        {
            "actual_heads": [10, 0, 20],
            "model_expected": [12.0, 1.0, 15.0],
            "model_p10": [8, 0, 16],
            "model_p90": [14, 2, 19],
            "rule_pred": [5.0, 5.0, 5.0],
            "history_pred": [10.0, 0.0, 20.0],
            "logistic_expected": [11.0, 0.0, 22.0],
        }
    )
    metrics = score(per, "tabpfn")
    assert list(metrics.index) == ["tabpfn", "logistic", "history", "rule"]
    tabpfn = metrics.loc["tabpfn"]
    assert tabpfn["mae"] == pytest.approx(8 / 3)
    assert tabpfn["mape"] == pytest.approx((2 / 10 + 5 / 20) / 2)  # skips the empty event
    assert tabpfn["bias"] == pytest.approx(-2 / 3)
    assert tabpfn["coverage_80"] == pytest.approx(2 / 3)
    assert metrics.loc["history", "mae"] == 0.0
    assert metrics.loc["rule", "bias"] == pytest.approx((-5 + 5 - 15) / 3)


@pytest.mark.tabpfn
def test_backtest_with_tabpfn_v2(tiny_dataset, monkeypatch):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    result = backtest(tiny_dataset, model_kind="tabpfn-v2", last_n=3)
    assert result.model_name == "tabpfn-v2"
    assert list(result.metrics.index) == ["tabpfn", "logistic", "history", "rule"]
    assert np.isfinite(result.per_event[PER_EVENT_COLUMNS[4:]].to_numpy(float)).all()
    assert result.metrics.loc["tabpfn", "mae"] < result.per_event["rsvp_heads"].mean()
