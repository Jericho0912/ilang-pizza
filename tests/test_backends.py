from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from ilang_pizza import config
from ilang_pizza.features.build import CATEGORICAL_FEATURES, build_features
from ilang_pizza.model import backends
from ilang_pizza.model.backends import (
    HistoryModel,
    LogisticModel,
    RuleOfThumbModel,
    ShowUpModel,
    TabPFNModel,
    TabPFNUnavailableError,
    make_model,
    resolve_tabpfn_version,
    stratified_cap,
    tabpfn_status,
)


@pytest.fixture(scope="module")
def labeled(tiny_dataset):
    """Features and labels of every going RSVP with known attendance."""
    rs = tiny_dataset.rsvps
    known = rs[(rs["status"] == "going") & rs["attended"].notna()]
    X = build_features(tiny_dataset).loc[known.index]
    X = X.loc[:, X.nunique(dropna=False) > 1]
    return X, known["attended"].to_numpy()


@pytest.fixture
def no_token(monkeypatch):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    monkeypatch.setattr(backends, "_UNAVAILABLE", set())


def assert_probabilities(p: np.ndarray, n: int) -> None:
    assert isinstance(p, np.ndarray) and p.shape == (n,)
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))


@pytest.mark.parametrize("kind", ["logistic", "history", "rule"])
def test_baselines_return_probabilities(kind, labeled):
    X, y = labeled
    cats = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    model = make_model(kind, categorical_features=cats)
    assert isinstance(model, ShowUpModel)
    assert model.name == kind
    assert model.fit(X, y) is model
    assert_probabilities(model.predict_proba(X), len(X))


def test_rule_uses_the_configured_rate(labeled):
    X, y = labeled
    np.testing.assert_allclose(RuleOfThumbModel().fit(X, y).predict_proba(X), 0.5)
    np.testing.assert_allclose(RuleOfThumbModel(0.65).fit(X, y).predict_proba(X[:3]), 0.65)
    with pytest.raises(ValueError):
        RuleOfThumbModel(1.5)


def test_history_uses_series_rates_with_an_overall_fallback(labeled):
    X, y = labeled
    model = HistoryModel(min_rsvps=20).fit(X, y)
    series = X["series_code"]
    for code, group in pd.Series(y, index=X.index).groupby(series):
        expected = group.mean() if len(group) >= 20 else y.mean()
        np.testing.assert_allclose(model.predict_proba(X[series == code]), expected)

    unseen = X.head(2).assign(series_code=99.0)
    np.testing.assert_allclose(model.predict_proba(unseen), y.mean())
    no_series = HistoryModel().fit(X.drop(columns="series_code"), y)
    np.testing.assert_allclose(no_series.predict_proba(X.drop(columns="series_code")), y.mean())


def test_logistic_handles_missing_values_and_unseen_categories(labeled):
    X, y = labeled
    X = X.copy()
    X.iloc[::7, 0] = np.nan
    model = LogisticModel(categorical_features=["series_code"]).fit(X, y)
    p = model.predict_proba(X)
    assert_probabilities(p, len(X))
    assert p.mean() == pytest.approx(y.mean(), abs=0.02)  # calibrated in-sample on average

    odd = X.head(3).copy()
    odd["series_code"] = 99.0
    odd.iloc[:, 0] = np.nan
    assert_probabilities(model.predict_proba(odd), 3)


def test_one_class_labels_give_a_smoothed_constant(labeled):
    X, _ = labeled
    ones = np.ones(len(X))
    for model in (LogisticModel(), TabPFNModel("v2")):
        p = model.fit(X, ones).predict_proba(X.head(4))  # TabPFN never loads here
        np.testing.assert_allclose(p, (len(X) + 1) / (len(X) + 2))


def test_labels_must_be_zero_or_one(labeled):
    X, y = labeled
    with pytest.raises(ValueError):
        LogisticModel().fit(X, np.where(y == 1, 2.0, 0.0))
    with pytest.raises(ValueError):
        HistoryModel().fit(X, np.full(len(X), np.nan))


def test_make_model_kinds(no_token, monkeypatch):
    assert make_model("tabpfn-v2").name == "tabpfn-v2"
    assert make_model("tabpfn-3.5").name == "tabpfn-3.5"
    monkeypatch.setattr(config, "TABPFN_VERSION", "auto")
    assert make_model("tabpfn").name == "tabpfn-v2"
    monkeypatch.setattr(config, "TABPFN_VERSION", "3.5")
    assert make_model("tabpfn").name == "tabpfn-3.5"
    assert make_model("rule", rate=0.4).rate == 0.4
    with pytest.raises(ValueError, match="unknown model kind"):
        make_model("xgboost")


def test_auto_uses_tabpfn_35_only_with_a_token(no_token, monkeypatch):
    assert resolve_tabpfn_version("auto") == "v2"
    monkeypatch.setenv("TABPFN_TOKEN", "   ")
    assert resolve_tabpfn_version("auto") == "v2"
    monkeypatch.setenv("TABPFN_TOKEN", "secret")
    assert resolve_tabpfn_version("auto") == "3.5"
    assert resolve_tabpfn_version("v2") == "v2"
    assert resolve_tabpfn_version("tabpfn-3.5") == "3.5"
    backends._UNAVAILABLE.add("3.5")
    assert resolve_tabpfn_version("auto") == "v2"
    with pytest.raises(ValueError, match="unknown TabPFN version"):
        resolve_tabpfn_version("v9")


class FakeClassifier:
    """Stands in for TabPFNClassifier: records what it was fitted on and asked to score."""

    classes_ = np.array([0, 1])

    def __init__(self, version: str, categorical: list[int]):
        self.version, self.categorical = version, categorical

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def predict_proba(self, X):
        self.scored = X
        return np.tile([0.3, 0.7], (len(X), 1))


@pytest.fixture
def fake_tabpfn(monkeypatch):
    """TabPFNModel._load without weights; TabPFN-3.5 always fails its license check."""
    loads = []

    def load(self, version, categorical):
        loads.append(version)
        if version == "3.5":
            raise TabPFNUnavailableError("TabPFN 3.5 could not load (license not accepted).")
        return FakeClassifier(version, categorical)

    monkeypatch.setattr(TabPFNModel, "_load", load)
    return loads


def test_auto_falls_back_to_v2_once(fake_tabpfn, labeled, monkeypatch):
    X, y = labeled
    monkeypatch.setenv("TABPFN_TOKEN", "secret")
    monkeypatch.setattr(backends, "_UNAVAILABLE", set())

    first = TabPFNModel("auto")
    assert first.name == "tabpfn-3.5"
    first.fit(X, y)
    assert first.name == "tabpfn-v2" and fake_tabpfn == ["3.5", "v2"]
    np.testing.assert_allclose(first.predict_proba(X.head(3)), 0.7)

    second = TabPFNModel("auto").fit(X, y)  # no second license attempt in this process
    assert second.name == "tabpfn-v2" and fake_tabpfn == ["3.5", "v2", "v2"]


def test_explicit_35_raises_instead_of_falling_back(fake_tabpfn, labeled, no_token):
    X, y = labeled
    with pytest.raises(TabPFNUnavailableError, match="license"):
        TabPFNModel("3.5").fit(X, y)


def test_tabpfn_caps_context_and_passes_categoricals(fake_tabpfn, labeled, no_token):
    X, y = labeled
    model = TabPFNModel("v2", categorical_features=["series_code", "absent"], max_context_rows=50)
    model.fit(X, y)
    fitted = model.classifier_
    assert fitted.categorical == [list(X.columns).index("series_code")]
    assert len(fitted.X) == 50
    assert fitted.y.mean() == pytest.approx(y.mean(), abs=1 / 50)


def test_tabpfn_gets_explicit_blank_flags(fake_tabpfn, no_token):
    X = pd.DataFrame(
        {
            "days_since": [np.nan, 10.0, 20.0, np.nan, 30.0, 40.0],
            "rain": [1.0, np.nan, 2.0, 3.0, 4.0, 5.0],
            "heavy_rain": [0.0, np.nan, 0.0, 0.0, 0.0, 1.0],  # blank exactly when rain is
            "series_code": [0.0, 1.0, np.nan, 0.0, 1.0, 0.0],
        }
    )
    model = TabPFNModel("v2", categorical_features=["series_code"])
    model.fit(X, np.array([0, 1, 1, 0, 1, 0]))
    seen = model.classifier_.X
    assert list(seen.columns) == [*X.columns, "days_since_blank", "rain_blank"]
    assert seen["days_since"].tolist() == [25.0, 10.0, 20.0, 25.0, 30.0, 40.0]  # median fill
    assert seen["days_since_blank"].tolist() == [1, 0, 0, 1, 0, 0]
    assert np.isnan(seen["series_code"].iloc[2])  # categories keep their blanks
    assert model.classifier_.categorical == [3]

    model.predict_proba(X.iloc[[0, 1]])
    scored = model.classifier_.scored
    assert scored.columns.equals(seen.columns)
    assert scored["rain"].tolist() == [1.0, 3.0] and scored["rain_blank"].tolist() == [0.0, 1.0]


def test_stratified_cap_is_deterministic_and_balanced():
    y = np.r_[np.ones(300), np.zeros(700)].astype(int)
    rows = stratified_cap(y, 101, seed=1)
    assert len(rows) == 101 and len(set(rows)) == 101
    assert np.all(np.diff(rows) > 0)
    assert abs(y[rows].mean() - 0.3) <= 1 / 101
    np.testing.assert_array_equal(rows, stratified_cap(y, 101, seed=1))
    assert not np.array_equal(rows, stratified_cap(y, 101, seed=2))
    np.testing.assert_array_equal(stratified_cap(y[:40], 101), np.arange(40))


def test_license_prompts_are_disabled_only_while_loading(monkeypatch):
    monkeypatch.setenv("TABPFN_NO_BROWSER", "0")
    with backends._license_prompts_disabled():
        assert os.environ["TABPFN_NO_BROWSER"] == "1"
    assert os.environ["TABPFN_NO_BROWSER"] == "0"
    monkeypatch.delenv("TABPFN_NO_BROWSER")
    with backends._license_prompts_disabled():
        assert os.environ["TABPFN_NO_BROWSER"] == "1"
    assert "TABPFN_NO_BROWSER" not in os.environ


def test_tabpfn_status_is_offline(no_token, monkeypatch, tmp_path):
    monkeypatch.setattr(backends, "_checkpoint_path", lambda version: tmp_path / "missing.ckpt")
    monkeypatch.setattr(config, "TABPFN_VERSION", "auto")
    ready, message = tabpfn_status()
    assert ready and "TabPFN v2" in message and "download" in message

    monkeypatch.setattr(config, "TABPFN_VERSION", "3.5")
    ready, message = tabpfn_status()
    assert not ready and "TABPFN_TOKEN" in message

    monkeypatch.setenv("TABPFN_TOKEN", "secret")
    ready, message = tabpfn_status()
    assert ready and "TabPFN-3.5" in message

    monkeypatch.setattr(config, "TABPFN_VERSION", "v7")
    assert tabpfn_status()[0] is False


@pytest.mark.tabpfn
def test_tabpfn_v2_end_to_end(labeled, no_token):
    X, y = labeled
    model = make_model("tabpfn", categorical_features=["series_code"]).fit(X, y)
    assert model.name == "tabpfn-v2"
    p = model.predict_proba(X)
    assert_probabilities(p, len(X))
    assert model.classifier_.categorical_features_indices == [list(X.columns).index("series_code")]
    assert p.mean() == pytest.approx(y.mean(), abs=0.1)
    # Rows are scored independently; half precision on GPUs leaves ~1e-4 of batch-shape noise.
    np.testing.assert_allclose(model.predict_proba(X.head(5)), p[:5], atol=1e-3)
