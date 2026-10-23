from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ilang_pizza.llm import agent as agent_module
from ilang_pizza.llm.agent import (
    ACTION_SCHEMA,
    FINAL_SCHEMA,
    MAX_STEPS,
    AskJuan,
    detect_tone,
    plan_numbers,
    route,
)
from ilang_pizza.llm.client import LLM, FakeLLM, LLMError
from ilang_pizza.llm.prompts import DEMO_MARKER, SIGN_OFF, assert_aggregate_only
from ilang_pizza.model.turnout import expected, heads_pmf, quantile
from ilang_pizza.plan import make_plan
from ilang_pizza.schema import MANILA_TZ, BacktestResult, Dataset, PlanInputs, TurnoutForecast

SHOW_RATE = 0.6


def make_ds(*, synthetic: bool = True, untracked_old_event: bool = False) -> Dataset:
    """Four past PyMeets with check-ins, a PyMeets next week with a waitlist, then a PyWorks."""
    rng = np.random.default_rng(11)
    today = pd.Timestamp.now(tz=MANILA_TZ).normalize()
    events, rsvps = [], []

    def add(event_id, series, start, capacity, n_going, n_waitlist, known=True):
        venue, city = ("Thinking Machines", "BGC, Taguig")
        events.append(
            {
                "event_id": event_id,
                "title": f"{series} {start:%b %Y}",
                "series": series,
                "start": start,
                "venue": venue,
                "city": city,
                "latitude": 14.55,
                "longitude": 121.05,
                "is_online": False,
                "capacity": float(capacity),
                "food_provided": True,
            }
        )
        for j in range(n_going + n_waitlist):
            going = j < n_going
            rsvps.append(
                {
                    "event_id": event_id,
                    "attendee_id": rng.bytes(8).hex(),  # like a salted hash
                    "rsvp_at": start - pd.Timedelta(days=float(rng.uniform(1, 20))),
                    "status": "going" if going else "waitlist",
                    "guests": int(j % 10 == 0),
                    "attended": float(rng.random() < SHOW_RATE) if known and going else np.nan,
                    "source": "synthetic",
                }
            )

    if untracked_old_event:  # an old event nobody checked in at
        add("pymeets-old", "PyMeets", today - pd.Timedelta(days=400, hours=-18), 30, 20, 0, False)
    for k in range(4, 0, -1):
        add(f"pymeets-past-{k}", "PyMeets", today - pd.Timedelta(days=30 * k, hours=-18), 30, 30, 0)
    add("pymeets-next", "PyMeets", today + pd.Timedelta(days=7, hours=18, minutes=30), 30, 30, 10,
        known=False)  # fmt: skip
    add("pyworks-next", "PyWorks", today + pd.Timedelta(days=14, hours=13), 40, 25, 0, False)
    ev, rs = pd.DataFrame(events), pd.DataFrame(rsvps)
    ev["start"] = pd.to_datetime(ev["start"]).dt.tz_convert(MANILA_TZ)
    rs["rsvp_at"] = pd.to_datetime(rs["rsvp_at"]).dt.tz_convert(MANILA_TZ)
    ds = Dataset(events=ev, rsvps=rs, synthetic=synthetic)
    ds.validate()
    return ds


def fake_forecast(ds: Dataset, event_id: str, **_) -> TurnoutForecast:
    """Everyone shows up with probability SHOW_RATE: easy to check by hand."""
    rs = ds.rsvps_for(event_id)
    going, waiting = rs[rs["status"] == "going"], rs[rs["status"] == "waitlist"]
    gh = (1 + going["guests"]).to_numpy()
    pmf = heads_pmf(np.full(len(going), SHOW_RATE), gh)
    return TurnoutForecast(
        event_id=event_id,
        model_name="fake",
        going_probs=np.full(len(going), SHOW_RATE),
        going_heads=gh,
        waitlist_probs=np.full(len(waiting), SHOW_RATE),
        waitlist_heads=(1 + waiting["guests"]).to_numpy(),
        pmf=pmf,
        expected=expected(pmf),
        p10=quantile(pmf, 0.1),
        p50=quantile(pmf, 0.5),
        p90=quantile(pmf, 0.9),
        historical_show_rate=SHOW_RATE,
        drivers=["Rain expected (12 mm)"],
        synthetic=ds.synthetic,
    )


@pytest.fixture
def forecast_calls(monkeypatch) -> list[str]:
    calls: list[str] = []

    def forecast(ds, event_id, **kw):
        calls.append(event_id)
        return fake_forecast(ds, event_id, **kw)

    monkeypatch.setattr(agent_module, "forecast_event", forecast)
    monkeypatch.setattr(agent_module, "build_features", lambda ds, weather=None: pd.DataFrame())
    return calls


@pytest.fixture
def ds() -> Dataset:
    return make_ds()


def expected_plan(ds: Dataset, event_id: str = "pymeets-next", **inputs):
    capacity = inputs.pop("capacity", int(ds.event(event_id)["capacity"]))
    return make_plan(fake_forecast(ds, event_id), PlanInputs(capacity=capacity, **inputs))


def tool(name: str, **args) -> str:
    return json.dumps({"action": "tool", "tool": name, "args": args, "answer": ""})


def answer(text: str) -> str:
    return json.dumps({"action": "answer", "tool": "none", "args": {}, "answer": text})


def backtest_result(synthetic: bool = True) -> BacktestResult:
    metrics = pd.DataFrame(
        {
            "mae": [3.2, 4.1, 9.5],
            "mape": [0.12, 0.15, 0.31],
            "bias": [0.4, 1.0, 8.9],
            "n_events": [12, 12, 12],
            "coverage_80": [0.83, np.nan, np.nan],
        },
        index=["tabpfn", "history", "rule"],
    )
    per_event = pd.DataFrame(
        {
            "event_id": [f"pymeets-past-{k}" for k in range(3)],
            "title": ["PyMeets A", "PyMeets B", "PyMeets C"],
            "actual_heads": [18, 21, 17],
            "model_expected": [19.2, 20.1, 18.4],
            "rule_pred": [15.0, 15.0, 15.5],
        }
    )
    return BacktestResult(per_event=per_event, metrics=metrics, model_name="TabPFN v2",
                          synthetic=synthetic)  # fmt: skip


# --- pure helpers ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "tool_name"),
    [
        ("Ilang pizza para sa next PyMeets?", "plan"),
        ("Ilan pa pwede i-release sa waitlist kung 80 lang ang venue?", "plan"),
        ("Magkano ang food budget?", "plan"),
        ("Draft the reconfirm email in English", "draft"),
        ("Sulat ka ng email para sa waitlist", "draft"),
        ("How accurate has this been?", "backtest_summary"),
        ("Tama ba ang forecast dati?", "backtest_summary"),
        ("What events are coming up?", "list_events"),
        ("Ano ang schedule natin?", "list_events"),
        ("Kumusta ka, Juan?", None),
    ],
)
def test_route(question, tool_name):
    assert route(question) == tool_name


@pytest.mark.parametrize(
    ("question", "tone"),
    [
        ("Ilang pizza para sa next PyMeets?", "taglish"),
        ("Ilan pa pwede i-release sa waitlist?", "taglish"),
        ("Draft the reconfirm email in English", "english"),
        ("Sulat ka ng email, in English please", "english"),
        ("Draft the venue email in Taglish", "taglish"),
        ("How accurate has this been?", "english"),
        ("What events are coming up?", "english"),
    ],
)
def test_detect_tone(question, tone):
    assert detect_tone(question) == tone


@pytest.mark.parametrize(
    ("question", "numbers"),
    [
        ("Ilan pa pwede i-release sa waitlist kung 80 lang ang venue?", {"capacity": 80}),
        ("How many can we release if the venue fits 60 people?", {"capacity": 60}),
        ("2.5 slices each at ₱450 per box", {"slices_per_person": 2.5, "price_per_box": 450}),
        ("12 slices per box, 3 slices each", {"slices_per_box": 12, "slices_per_person": 3}),
        ("budget 1,200 pesos per box", {"price_per_box": 1200}),
        ("Ilang pizza?", {}),
    ],
)  # fmt: skip
def test_plan_numbers(question, numbers):
    assert plan_numbers(question) == numbers


# --- the keyword router (no LLM) ---------------------------------------------------------


def test_router_plans_pizza_for_the_next_event(ds, forecast_calls):
    reply = AskJuan(ds).ask("Ilang pizza para sa next PyMeets?")
    assert not reply.used_llm
    (step,) = reply.steps
    assert step["tool"] == "plan" and step["args"] == {"event_id": "pymeets-next"}
    plan = expected_plan(ds)
    assert step["result"]["plan"]["pizza_boxes"] == plan.pizza_boxes
    assert reply.answer.startswith(DEMO_MARKER)
    assert f"Pizza: {plan.pizza_boxes} boxes" in reply.answer  # Taglish question, Taglish answer
    assert f"Upuan at e-certificates: {plan.seats}." in reply.answer


def test_router_reads_capacity_from_the_question(ds, forecast_calls):
    reply = AskJuan(ds).ask("Ilan pa pwede i-release sa waitlist kung 28 lang ang venue?")
    (step,) = reply.steps
    assert step["args"] == {"capacity": 28}
    plan = expected_plan(ds, capacity=28)
    assert step["result"]["plan"]["waitlist_release"] == plan.waitlist_release == 6
    assert "pwedeng i-release ang 6 sa 10. 9% lang ang chance na lumampas sa 28" in reply.answer


def test_router_answers_english_questions_with_plan_notes(ds, forecast_calls):
    reply = AskJuan(ds).ask("How many pizzas at ₱450 per box?")
    plan = expected_plan(ds, price_per_box=450)
    for note in plan.notes:
        assert f"- {note}" in reply.answer


def test_router_drafts_with_the_template(ds, forecast_calls):
    reply = AskJuan(ds).ask("Draft the reconfirm email in English")
    (step,) = reply.steps
    assert step["tool"] == "draft"
    assert step["args"] == {"kind": "reconfirm", "tone": "english"}
    assert step["result"]["written_by"] == "template"
    assert "Here's the reconfirm email draft (from the template" in reply.answer
    assert "Subject: " in reply.answer and reply.answer.endswith(SIGN_OFF)


def test_router_drafts_catering_in_taglish(ds, forecast_calls):
    reply = AskJuan(ds).ask("Sulat ka ng email sa caterer para sa pizza")
    assert reply.steps[0]["args"] == {"kind": "catering_order", "tone": "taglish"}
    assert "Magpapa-order po sana kami" in reply.answer


def test_router_backtest_without_results_suggests_the_command(ds, forecast_calls):
    reply = AskJuan(ds).ask("How accurate has this been?")
    assert reply.steps[0]["result"] == {
        "available": False,
        "hint": "Run `ilang-pizza backtest` to measure accuracy on past events "
        "(it takes a few minutes).",
    }
    assert "ilang-pizza backtest" in reply.answer


def test_router_backtest_with_results(ds, forecast_calls):
    reply = AskJuan(ds, backtest=backtest_result()).ask("How accurate has this been?")
    result = reply.steps[0]["result"]
    assert result["available"] and result["main_method"] == "tabpfn"
    assert result["metrics"]["tabpfn"]["coverage_80"] == 0.83
    assert result["metrics"]["history"]["coverage_80"] is None
    assert "off by 3.2 people on average, vs 9.5 for the" in reply.answer
    assert "83% of actual turnouts" in reply.answer


def test_router_lists_only_future_events(forecast_calls):
    ds = make_ds(untracked_old_event=True)
    reply = AskJuan(ds).ask("What events are coming up?")
    events = reply.steps[0]["result"]["events"]
    assert [e["event_id"] for e in events] == ["pymeets-next", "pyworks-next"]
    assert events[0]["going_heads"] == 33 and events[0]["waitlist_heads"] == 11
    assert "- PyMeets" in reply.answer and "(33 going, 11 waitlisted)" in reply.answer


def test_router_lists_recent_events_with_turnout(ds, forecast_calls):
    reply = AskJuan(ds).ask("Show all events, past and upcoming")
    events = reply.steps[0]["result"]["events"]
    assert len(events) == 6
    assert "attended_heads" in events[0] and "attended_heads" not in events[-1]


def test_router_help_for_unknown_questions(ds, forecast_calls):
    reply = AskJuan(ds).ask("Kumusta ka, Juan?")
    assert reply.steps == [] and not reply.used_llm
    assert "Ako si Juan" in reply.answer and "Ilang pizza para sa next PyMeets?" in reply.answer


def test_empty_question_gets_help(ds, forecast_calls):
    assert "I'm Juan" in AskJuan(ds).ask("   ").answer


def test_router_reports_errors_kindly(ds, forecast_calls, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("not enough history before PyMeets: add attendance for more events")

    monkeypatch.setattr(agent_module, "forecast_event", broken)
    reply = AskJuan(ds).ask("Ilang pizza?")
    assert reply.steps[0]["result"] == {
        "error": "not enough history before PyMeets: add attendance for more events"
    }
    assert reply.answer.endswith("Pasensya, hindi ko nagawa: not enough history before PyMeets: "
                                 "add attendance for more events.")  # fmt: skip


def test_router_survives_a_crashing_model(ds, forecast_calls, monkeypatch):
    def crash(*args, **kwargs):
        raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(agent_module, "forecast_event", crash)
    reply = AskJuan(ds).ask("How many pizzas?")
    assert reply.steps[0]["result"] == {"error": "the plan tool failed (RuntimeError)"}


# --- events and caching ------------------------------------------------------------------


def test_default_event_is_the_next_one_not_an_old_untracked_one(forecast_calls):
    ds = make_ds(untracked_old_event=True)
    assert ds.upcoming_event_ids()[0] == "pymeets-old"
    reply = AskJuan(ds).ask("Ilang pizza?")
    assert reply.steps[0]["result"]["event"]["event_id"] == ds.next_event_id() == "pymeets-next"


def test_series_names_pick_the_next_event_of_that_series(ds, forecast_calls):
    reply = AskJuan(ds).ask("Ilang pizza para sa PyWorks?")
    assert reply.steps[0]["args"] == {"event_id": "pyworks-next"}


def test_forecasts_are_cached_per_event(ds, forecast_calls):
    juan = AskJuan(ds)
    juan.ask("Ilang pizza?")
    juan.ask("Draft the reconfirm email")
    juan.ask("Ilang pizza para sa PyWorks?")
    juan.ask("How many pizzas for PyWorks?")
    assert forecast_calls == ["pymeets-next", "pyworks-next"]


def test_event_capacity_is_the_default_venue_limit(ds, forecast_calls):
    reply = AskJuan(ds).ask("Ilang pizza?")
    assert reply.steps[0]["result"]["inputs"]["capacity"] == 30
    reply = AskJuan(ds, plan_inputs=PlanInputs(capacity=25)).ask("Ilang pizza?")
    assert reply.steps[0]["result"]["inputs"]["capacity"] == 25


def test_real_data_is_not_marked_as_demo(forecast_calls):
    reply = AskJuan(make_ds(synthetic=False)).ask("Ilang pizza?")
    assert DEMO_MARKER not in reply.answer


# --- with an LLM -------------------------------------------------------------------------


def test_llm_picks_a_tool_then_answers(ds, forecast_calls):
    plan = expected_plan(ds)
    llm = FakeLLM([tool("plan"), answer(f"Mga {plan.pizza_boxes} boxes ang i-order.")])
    reply = AskJuan(ds, llm).ask("Ilang pizza para sa next PyMeets?")
    assert reply.used_llm
    assert reply.answer == f"{DEMO_MARKER} Mga {plan.pizza_boxes} boxes ang i-order."
    (step,) = reply.steps
    assert step["tool"] == "plan" and step["result"]["plan"]["pizza_boxes"] == plan.pizza_boxes

    first, second = llm.calls
    assert first["schema"] == ACTION_SCHEMA and first["temperature"] == 0.2
    assert first["messages"][-1]["content"].endswith("(Reply in Taglish.)")
    assert "pymeets-next" in first["messages"][0]["content"]  # the default event
    assert second["messages"][-1]["content"].startswith("Result of plan: {")


def test_llm_only_ever_sees_aggregates(ds, forecast_calls):
    llm = FakeLLM([tool("list_events"), tool("forecast"), tool("plan"), answer("Done!")])
    AskJuan(ds, llm).ask("Ilang pizza para sa next PyMeets?")
    sent = llm.sent_text
    assert_aggregate_only(sent)
    assert not any(attendee in sent for attendee in ds.rsvps["attendee_id"])


def test_personal_data_in_the_question_is_redacted(ds, forecast_calls):
    llm = FakeLLM([answer("Okay!")])
    AskJuan(ds, llm).ask("Draft an email to juan.cruz@example.com, 0917 123 4567")
    assert "juan.cruz" not in llm.sent_text and "0917" not in llm.sent_text
    assert "[redacted]" in llm.sent_text


def test_step_limit(ds, forecast_calls):
    llm = FakeLLM([tool("plan")] * 10)
    reply = AskJuan(ds, llm).ask("Ilang pizza?")
    assert len(llm.calls) == MAX_STEPS == 4
    assert [c["schema"] for c in llm.calls] == [ACTION_SCHEMA] * 3 + [FINAL_SCHEMA]
    assert "You have no tool calls left" in llm.calls[-1]["messages"][-1]["content"]
    assert len(reply.steps) == 3
    # Gemma never answered, so the answer comes from the last tool result's template.
    assert not reply.used_llm
    assert f"Pizza: {expected_plan(ds).pizza_boxes} boxes" in reply.answer


def test_invented_numbers_get_one_more_try(ds, forecast_calls):
    plan = expected_plan(ds)
    llm = FakeLLM([tool("plan"), answer("Order 999 boxes."), answer(f"Order {plan.pizza_boxes}.")])
    reply = AskJuan(ds, llm).ask("How many pizzas?")
    assert reply.used_llm and reply.answer.endswith(f"Order {plan.pizza_boxes}.")
    assert "(999)" in llm.calls[2]["messages"][-1]["content"]


def test_invented_numbers_on_the_last_step_fall_back_to_the_template(ds, forecast_calls):
    llm = FakeLLM([tool("plan"), answer("Order 999 boxes.")] + [answer("Still 999.")] * 3)
    reply = AskJuan(ds, llm).ask("How many pizzas?")
    assert not reply.used_llm and "999" not in reply.answer
    assert reply.steps[0]["tool"] == "plan"


def test_malformed_json_gets_a_retry(ds, forecast_calls):
    llm = FakeLLM(["Sure! I'll check the plan.", tool("plan"), answer("Ready na!")])
    reply = AskJuan(ds, llm).ask("Ilang pizza?")
    assert reply.used_llm and reply.answer.endswith("Ready na!")
    assert "not a valid JSON action" in llm.calls[1]["messages"][-1]["content"]


def test_unknown_tools_are_reported_back(ds, forecast_calls):
    llm = FakeLLM([tool("weather"), tool("list_events"), answer("Two events!")])
    reply = AskJuan(ds, llm).ask("What events are coming up?")
    assert "There is no 'weather' tool" in llm.calls[1]["messages"][-1]["content"]
    assert [s["tool"] for s in reply.steps] == ["list_events"]


def test_answers_without_tools_are_sent_back_for_data_questions(ds, forecast_calls):
    plan = expected_plan(ds)
    llm = FakeLLM([answer("Mga sampu."), tool("plan"), answer(f"{plan.pizza_boxes} boxes!")])
    reply = AskJuan(ds, llm).ask("Ilang pizza?")
    assert "Call the plan tool first" in llm.calls[1]["messages"][-1]["content"]
    assert reply.used_llm and reply.answer.endswith(f"{plan.pizza_boxes} boxes!")


def test_small_talk_needs_no_tools(ds, forecast_calls):
    reply = AskJuan(ds, FakeLLM([answer("Ayos lang! Ikaw?")])).ask("Kumusta ka, Juan?")
    assert reply.used_llm and reply.steps == [] and reply.answer.endswith("Ayos lang! Ikaw?")


def test_plan_numbers_must_come_from_the_organizer(ds, forecast_calls):
    junk = tool("plan", capacity=35, slices_per_box=1, food_quantile=0.5, overflow_risk=0)
    llm = FakeLLM([junk, answer("Okay!")])
    reply = AskJuan(ds, llm).ask("Ilan pa pwede i-release kung 35 lang ang venue?")
    assert reply.steps[0]["args"] == {"capacity": 35}
    assert reply.steps[0]["result"]["inputs"]["slices_per_box"] == 8


def test_bad_tool_arguments_become_error_results(ds, forecast_calls):
    llm = FakeLLM([tool("plan", event_id="pyconph-2099"), answer("Hindi ko mahanap.")])
    reply = AskJuan(ds, llm).ask("Ilang pizza para sa pyconph-2099?")
    error = reply.steps[0]["result"]["error"]
    assert "unknown event 'pyconph-2099'" in error and "pymeets-next" in error


def test_llm_draft_is_shown_in_full(ds, forecast_calls):
    plan = expected_plan(ds)
    email = json.dumps(
        {
            "subject": f"Pizza order: {plan.pizza_boxes} boxes",
            "body": f"Hi [Caterer name]!\n\nWe'd like {plan.pizza_boxes} boxes, please.",
        }
    )
    llm = FakeLLM(
        [tool("draft", kind="catering_order", tone="english"), email, answer("Here you go!")]
    )
    reply = AskJuan(ds, llm).ask("Draft the catering email in English")
    assert reply.used_llm
    step = reply.steps[0]
    assert step["result"]["written_by"] == "fake-gemma"
    assert "body" not in step["result"]  # Gemma gets the subject only
    assert f"Here you go!\n\nSubject: {DEMO_MARKER} Pizza order: {plan.pizza_boxes} boxes" in (
        reply.answer
    )
    assert reply.answer.endswith(SIGN_OFF)


def test_llm_errors_fall_back_to_the_router(ds, forecast_calls):
    reply = AskJuan(ds, FakeLLM([LLMError("Can't reach Ollama")])).ask("Ilang pizza?")
    assert not reply.used_llm and reply.steps[0]["tool"] == "plan"


def test_unavailable_llm_is_not_called(ds, forecast_calls):
    llm = FakeLLM([], available=False)
    assert not AskJuan(ds, llm).ask("Ilang pizza?").used_llm
    assert llm.calls == []


# --- the real thing ----------------------------------------------------------------------


@pytest.mark.ollama
def test_real_gemma_answers_the_pizza_question():
    ds = make_ds()
    llm = LLM()
    assert llm.available(), llm.status()[1]
    reply = AskJuan(ds, llm, model_kind="logistic").ask("Ilang pizza para sa next PyMeets?")
    plans = [s for s in reply.steps if s["tool"] == "plan" and "plan" in s["result"]]
    assert plans, reply.steps
    assert reply.used_llm, reply.answer
    assert str(plans[-1]["result"]["plan"]["pizza_boxes"]) in reply.answer
