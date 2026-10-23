"""UI tests: the app builds without launching, and the callbacks work on tiny data.

The callbacks are plain methods on ``AppState``, so they run here without a browser, with the
logistic model, no weather lookups and a ``FakeLLM`` standing in for Gemma.
"""

from __future__ import annotations

import io
import warnings

import gradio as gr
import numpy as np
import pytest
from matplotlib.figure import Figure

from ilang_pizza import ui
from ilang_pizza.data.synth import write_dataset
from ilang_pizza.llm.client import FakeLLM
from ilang_pizza.model.backends import TABPFN_ATTRIBUTION, TabPFNUnavailableError
from ilang_pizza.schema import TurnoutForecast

PLAN = (3.0, 8, 450.0, 0.75, 0.10)  # slices/person, slices/box, price/box, food quantile, risk

# Gradio's Blocks() opens an asyncio event loop it never closes; that warning is not ours.
pytestmark = pytest.mark.filterwarnings(
    "ignore:unclosed event loop:ResourceWarning",
    "ignore::pytest.PytestUnraisableExceptionWarning",
)
TABS = ["🍕 Ilang pizza?", "✉️ Drafts", "🧑‍🌾 Ask Juan", "📈 Backtest", "🗂️ Data"]


@pytest.fixture(autouse=True)
def tabpfn_is_ready(monkeypatch):
    """The real check imports torch; these tests only need its answer."""
    monkeypatch.setattr(ui, "tabpfn_status", lambda: (True, "TabPFN v2 on cpu, weights cached"))


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory, tiny_dataset):
    return write_dataset(tiny_dataset, tmp_path_factory.mktemp("tiny"))


def make_state(data_dir, *responses: str) -> ui.AppState:
    llm = FakeLLM(list(responses), available=bool(responses))
    return ui.AppState(data_dir, offline_weather=True, llm=llm)


@pytest.fixture()
def state(data_dir) -> ui.AppState:
    return make_state(data_dir)


def forecast_args(state: ui.AppState) -> tuple:
    event_id = state.default_event()
    return (event_id, "logistic", state.capacity_of(event_id), *PLAN)


def hand_made_forecast(probs, model_name: str = "logistic") -> TurnoutForecast:
    """A forecast built straight from P(show) values, no model involved."""
    pmf = np.array([1.0])
    for p in probs:
        pmf = np.convolve(pmf, [1 - p, p])
    cdf = np.cumsum(pmf)
    p10, p50, p90 = (int(np.searchsorted(cdf, q)) for q in (0.1, 0.5, 0.9))
    return TurnoutForecast(
        event_id="e",
        model_name=model_name,
        going_probs=np.asarray(probs, dtype=float),
        going_heads=np.ones(len(probs), dtype=int),
        waitlist_probs=np.zeros(0),
        waitlist_heads=np.zeros(0, dtype=int),
        pmf=pmf,
        expected=float(np.sum(probs)),
        p10=p10,
        p50=p50,
        p90=p90,
        historical_show_rate=0.5,
    )


# --- building the app ----------------------------------------------------------------------


def test_build_app_constructs_without_warnings(state):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Gradio 6 deprecations would fail here
        app = ui.build_app(state=state)
    assert isinstance(app, gr.Blocks)
    assert [b.label for b in app.blocks.values() if isinstance(b, gr.Tab)] == TABS


def test_attributions_are_visible(state):
    app = ui.build_app(state=state)
    shown = " ".join(str(b.value) for b in app.blocks.values() if isinstance(b, gr.HTML))
    assert shown.count(TABPFN_ATTRIBUTION) == 2  # the footer and the forecast tab

    forecast = hand_made_forecast([0.5] * 10)
    assert ui._drivers_html(forecast, weather=True).count('href="https://open-meteo.com/"') == 1
    assert "open-meteo" not in ui._drivers_html(forecast, weather=False)
    credits = ui._credits_html("tabpfn-v2", weather=True)
    assert TABPFN_ATTRIBUTION in credits
    assert ">Weather data by Open-Meteo.com</a>" in credits


def test_api_exposes_only_the_main_actions(state):
    app = ui.build_app(state=state)
    expected = {"/forecast", "/explain", "/draft", "/ask", "/backtest", "/reload", "/demo_data"}
    assert set(app.get_api_info()["named_endpoints"]) == expected


def test_sharing_hides_the_data_folder_controls(state):
    app = ui.build_app(state=state, read_only=True)
    buttons = {b.value for b in app.blocks.values() if isinstance(b, gr.Button)}
    assert "🔄 Reload" not in buttons
    assert "🧪 Generate demo data" not in buttons
    assert not {"/reload", "/demo_data"} & set(app.get_api_info()["named_endpoints"])


def test_app_without_data_explains_what_to_do(tmp_path):
    empty = make_state(tmp_path)
    ui.build_app(state=empty)
    assert empty.event_choices() == []
    assert "Generate demo data" in ui._dataset_html(empty.workspace)
    with pytest.raises(gr.Error):
        empty.forecast_view(None, "logistic", None, *PLAN)


# --- 🍕 forecast and plan --------------------------------------------------------------------


def test_forecast_callback_with_logistic(state, tiny_dataset):
    headline, chart, cards, drivers, explanation = state.forecast_view(*forecast_args(state))
    assert "Expect" in headline
    assert "RSVPs" in headline
    assert "synthetic data" in headline
    assert isinstance(chart, Figure)
    chart.savefig(io.BytesIO(), format="webp")  # what gr.Plot does with it
    assert "boxes" in cards
    assert "₱" in cards
    assert "Why this number" in drivers
    assert "Writing the explanation" in explanation
    shown = headline + cards + drivers
    assert not [a for a in tiny_dataset.rsvps["attendee_id"].unique() if a in shown]


def test_plan_changes_never_retrain(state, monkeypatch):
    calls = []
    real = ui.forecast_event

    def counting(*args, **kwargs):
        calls.append(kwargs.get("model_kind"))
        return real(*args, **kwargs)

    monkeypatch.setattr(ui, "forecast_event", counting)
    event_id, model, capacity, *_ = forecast_args(state)
    assert "Press <b>Forecast</b>" in state.replan_view(event_id, model, capacity, *PLAN)[0]

    state.forecast_view(event_id, model, capacity, *PLAN)
    replanned = state.replan_view(event_id, model, 5, 2.0, 10, 0.0, 0.9, 0.05)
    assert "Add a price per box" in replanned[2]  # price 0 hides the pesos
    state.forecast_view(event_id, model, capacity, *PLAN)
    assert calls == ["logistic"]

    state.forecast_view(event_id, "history", capacity, *PLAN)
    assert calls == ["logistic", "history"]


def test_model_trouble_becomes_a_friendly_message(state, monkeypatch):
    def unavailable(*args, **kwargs):
        raise TabPFNUnavailableError("TabPFN weights could not be downloaded.")

    monkeypatch.setattr(ui, "forecast_event", unavailable)
    with pytest.raises(gr.Error, match="Logistic regression"):
        state.forecast_view(*forecast_args(state))

    monkeypatch.undo()
    with pytest.raises(gr.Error, match="unknown event_id"):
        state.forecast_view("no-such-event", "logistic", None, *PLAN)


def test_new_event_resets_capacity(state):
    event_id = state.default_event()
    capacity, headline, *_ = state.event_view(event_id, "logistic", *PLAN)
    assert capacity == state.capacity_of(event_id)
    assert "Forecast" in headline


def test_explain_says_who_wrote_it(data_dir):
    offline = make_state(data_dir)
    args = forecast_args(offline)
    offline.forecast_view(*args)
    assert "Built-in template" in offline.explain_view(*args, "english")

    gemma = make_state(data_dir, '{"explanation": "Mukhang sakto ang order ngayong gabi."}')
    gemma.forecast_view(*args)
    written = gemma.explain_view(*args, "taglish")
    assert "Written by Gemma (fake-gemma)" in written
    assert "Mukhang sakto" in written
    # the explanation is cached for these inputs, so re-planning back to them shows it
    assert "Mukhang sakto" in gemma.replan_view(*args)[4]


# --- other tabs ----------------------------------------------------------------------------


def test_draft_callback_uses_the_main_tab_plan(state):
    subject, body, source = state.draft_view("catering_order", "english", *forecast_args(state))
    assert subject
    assert "PythonPH" in body
    assert "boxes" in source
    assert "Built-in template" in source


def test_ask_callback_returns_chat_messages(state):
    history, cleared = state.ask_view("Ilang pizza ang i-order?", [], "logistic", *PLAN)
    assert cleared == ""
    assert history[0] == {"role": "user", "content": "Ilang pizza ang i-order?"}
    assert history[-1]["role"] == "assistant"
    assert "rule-based" in history[-1]["content"]
    for step in (m for m in history if "metadata" in m):
        assert step["metadata"]["status"] == "done"  # tool steps render collapsed


def test_backtest_callback(state):
    headline, chart, metrics, per_event = state.backtest_view("logistic", 3)
    assert "off by" in headline
    assert isinstance(chart, Figure)
    chart.savefig(io.BytesIO(), format="webp")
    assert {"Method", "MAE (heads)", "80% coverage"} <= set(metrics.columns)
    assert len(per_event) > 0
    assert state.workspace.latest_backtest is not None  # Juan can quote it now


def test_demo_button_writes_and_loads_demo_data(tmp_path, monkeypatch):
    monkeypatch.setattr(ui.config, "DATA_DIR", tmp_path)
    fresh = make_state(tmp_path)
    outputs = fresh.demo_view()
    assert (tmp_path / "demo" / "SYNTHETIC").exists()
    assert fresh.ds is not None and fresh.ds.synthetic
    assert len(outputs) == 11  # one value per component in the page list


# --- charts --------------------------------------------------------------------------------


@pytest.mark.parametrize("capacity", [None, 18, 500])
def test_headcount_chart_renders(capacity):
    fig = ui.headcount_chart(hand_made_forecast([0.5] * 30), capacity)
    fig.savefig(io.BytesIO(), format="webp")


def test_headline_with_no_rsvps(tiny_dataset):
    event = tiny_dataset.event(tiny_dataset.upcoming_event_ids()[0])
    html = ui._headline_html(event, hand_made_forecast([]))
    assert "No RSVPs yet" in html
