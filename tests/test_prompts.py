from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ilang_pizza.llm.client import LLM, FakeLLM, LLMError
from ilang_pizza.llm.prompts import (
    DEMO_MARKER,
    DRAFT_SCHEMA,
    EMAIL_KINDS,
    SIGN_OFF,
    TONES,
    PrivacyError,
    assert_aggregate_only,
    draft_email,
    draft_messages,
    explain_forecast,
    explain_messages,
    format_when,
    invented_numbers,
    redact_pii,
    template_draft,
)
from ilang_pizza.model.turnout import expected, heads_pmf, quantile
from ilang_pizza.plan import make_plan
from ilang_pizza.schema import MANILA_TZ, PlanInputs, TurnoutForecast

EVENT = {
    "event_id": "pymeets-2026-09",
    "title": "PyMeets September 2026",
    "series": "PyMeets",
    "start": pd.Timestamp("2026-09-17 18:30", tz=MANILA_TZ),
    "venue": "Thinking Machines, 5th Ave",
    "city": "BGC, Taguig",
    "is_online": False,
    "capacity": 60.0,
    "food_provided": True,
}
ONLINE = EVENT | {
    "event_id": "wfh-pythonista-2026-09",
    "title": "WFH Pythonista September 2026",
    "series": "WFH Pythonista",
    "venue": "",
    "city": "",
    "is_online": True,
}


def make_forecast(n_going=40, n_waitlist=12, *, synthetic=False, event_id="pymeets-2026-09"):
    rng = np.random.default_rng(3)
    gp = rng.uniform(0.35, 0.85, n_going)
    gh = np.where(rng.random(n_going) < 0.1, 2, 1)
    wp = rng.uniform(0.3, 0.7, n_waitlist)
    pmf = heads_pmf(gp, gh)
    return TurnoutForecast(
        event_id=event_id,
        model_name="tabpfn-v2",
        going_probs=gp,
        going_heads=gh,
        waitlist_probs=wp,
        waitlist_heads=np.ones(n_waitlist, dtype=int),
        pmf=pmf,
        expected=expected(pmf),
        p10=quantile(pmf, 0.1),
        p50=quantile(pmf, 0.5),
        p90=quantile(pmf, 0.9),
        historical_show_rate=0.58,
        drivers=["Rain expected (14 mm)", "Payday is 3 days away"],
        synthetic=synthetic,
    )


@pytest.fixture
def forecast():
    return make_forecast()


@pytest.fixture
def plan(forecast):
    return make_plan(forecast, PlanInputs(capacity=50, price_per_box=450))


def reply(subject="Still coming to PyMeets?", body=None, **extra) -> str:
    body = body or (
        "Hi Pythonistas!\n\nExcited na kami for PyMeets September 2026 on Thu, Sep 17, 6:30 PM! "
        "Kung hindi ka na makakarating, paki-update lang ang RSVP mo.\n\n— PythonPH Organizers"
    )
    return json.dumps({"subject": subject, "body": body, **extra})


# --- privacy guard -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "juan.dela.cruz@example.com",
        "RSVP list: a+b@example.org, others",
        "attendee a1b2c3d4e5f60718 came",
        "id_A1B2C3D4E5F60718",
        "sha256 " + "ab" * 32,
        "call 09171234567",
        "call +639171234567",
        "call 0917 123 4567",
        "call +63 917-123-4567",
        "call (0917) 123-4567",
        "call 639171234567",
    ],
)
def test_privacy_guard_rejects_personal_data(text):
    with pytest.raises(PrivacyError):
        assert_aggregate_only(text)


def test_privacy_guard_never_repeats_what_it_found():
    with pytest.raises(PrivacyError) as info:
        assert_aggregate_only("juan@example.com 09171234567 a1b2c3d4e5f60718")
    message = str(info.value)
    assert "juan" not in message and "0917" not in message and "a1b2" not in message
    assert "email" in message and "phone" in message and "hash" in message


@pytest.mark.parametrize(
    "text",
    [
        "pymeets-2026-09 on 2026-09-17 at 6:30 PM",
        "Order 9 boxes for ₱4,050 (saves ₱2,700); 0.613 show rate",
        "p10=35, p90=46, capacity 1234567",
        "deadbeef cafe 0917 is a time",
        "PyMeets #42 @ BGC",
    ],
)
def test_privacy_guard_accepts_aggregate_text(text):
    assert_aggregate_only(text)


def test_forecast_summary_and_prompts_pass_the_guard(forecast, plan):
    assert_aggregate_only(json.dumps(forecast.summary()))
    for kind in EMAIL_KINDS:
        for message in draft_messages(kind, EVENT, forecast, plan):
            assert_aggregate_only(message["content"])


def test_redact_pii():
    text = redact_pii("Draft one for juan@example.com, 0917 123 4567, id a1b2c3d4e5f60718")
    assert text == "Draft one for [redacted], [redacted], id [redacted]"
    assert_aggregate_only(text)


# --- number guard ------------------------------------------------------------------------


def test_invented_numbers_allows_rounding_percent_and_formatting():
    source = '{"expected_heads": 24.6, "predicted_show_rate": 0.615, "cost": 4050.0}'
    assert invented_numbers("About 25 people (62%), ₱4,050.00 total", source) == []
    assert invented_numbers("About 24 people, 61%", source) == []


def test_invented_numbers_flags_new_numbers_and_arithmetic():
    source = '{"rsvp_heads": 40, "expected_heads": 24}'
    assert invented_numbers("16 people won't come; bring 2 IDs", source) == ["16", "2"]


# --- event details -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "start",
    [
        pd.Timestamp("2026-09-17 18:30", tz=MANILA_TZ),
        pd.Timestamp("2026-09-17 10:30", tz="UTC"),
        pd.Timestamp("2026-09-17 18:30"),  # naive means Manila
        "2026-09-17T18:30:00+08:00",
    ],
)
def test_format_when_is_manila_time(start):
    assert format_when(start) == "Thu, Sep 17, 6:30 PM"


def test_format_when_morning_and_noon():
    assert format_when(pd.Timestamp("2026-10-03 09:05", tz=MANILA_TZ)) == "Sat, Oct 3, 9:05 AM"
    assert format_when(pd.Timestamp("2026-10-03 12:00", tz=MANILA_TZ)) == "Sat, Oct 3, 12:00 PM"


# --- templates ---------------------------------------------------------------------------


@pytest.mark.parametrize("tone", TONES)
@pytest.mark.parametrize("kind", EMAIL_KINDS)
def test_templates_for_every_kind_and_tone(kind, tone, forecast, plan):
    draft = draft_email(kind, EVENT, forecast, plan, tone=tone)
    assert draft.kind == kind and not draft.used_llm and draft.model == ""
    assert draft.subject.strip() and draft.body.strip()
    assert draft.body.endswith(SIGN_OFF)
    assert len(draft.body.split()) < 150
    assert DEMO_MARKER not in draft.subject + draft.body
    assert_aggregate_only(draft.subject + "\n" + draft.body)
    assert "PyMeets September 2026" in draft.subject + draft.body


@pytest.mark.parametrize("tone", TONES)
def test_reconfirm_template_mentions_the_waitlist(tone, forecast, plan):
    body = draft_email("reconfirm", EVENT, forecast, plan, tone=tone).body
    assert "Thu, Sep 17, 6:30 PM" in body
    assert "at Thinking Machines, 5th Ave, BGC, Taguig" in body
    assert f"{int(forecast.waitlist_heads.sum())} Pythonistas" in body
    assert body.startswith("Hi Pythonistas!")


@pytest.mark.parametrize("tone", TONES)
def test_catering_template_has_the_order(tone, forecast, plan):
    draft = draft_email("catering_order", EVENT, forecast, plan, tone=tone)
    assert f"{plan.pizza_boxes} boxes" in draft.subject
    assert f"Order: {plan.pizza_boxes} boxes ([flavors])" in draft.body
    assert "Delivery: 6:00 PM on Thu, Sep 17" in draft.body  # 30 minutes before 6:30 PM
    assert "[Caterer name]" in draft.body and "[Organizer contact]" in draft.body
    total = f"Estimated total: ₱{plan.cost_php:,.0f} ({plan.pizza_boxes} boxes × ₱450)"
    assert total in draft.body


def test_catering_template_without_price_has_no_total(forecast):
    plan = make_plan(forecast, PlanInputs(capacity=50))
    assert "Estimated total" not in draft_email("catering_order", EVENT, forecast, plan).body


@pytest.mark.parametrize("tone", TONES)
def test_venue_template_has_range_seats_and_guest_list(tone, forecast, plan):
    body = draft_email("venue_headcount", EVENT, forecast, plan, tone=tone).body
    assert f"Seats to prepare: {plan.seats}" in body
    assert "Expected attendees: " in body and "–" in body
    assert "guest list" in body


def test_venue_range_includes_the_released_waitlist(forecast, plan):
    assert plan.waitlist_release > 0
    body = draft_email("venue_headcount", EVENT, forecast, plan, tone="english").body
    low_high = body.split("Expected attendees: ")[1].split(" people")[0]
    low, high = (int(x) for x in low_high.split("–"))
    assert low > forecast.p10 and high == plan.seats


@pytest.mark.parametrize("tone", TONES)
def test_waitlist_template_reminds_about_id_in_person(tone, forecast, plan):
    body = draft_email("waitlist_release", EVENT, forecast, plan, tone=tone).body
    assert "valid ID" in body and "[Join link]" not in body


@pytest.mark.parametrize("tone", TONES)
def test_online_templates_use_a_join_link_and_no_venue(tone):
    fc = make_forecast(event_id=ONLINE["event_id"])
    plan = make_plan(fc, PlanInputs())
    release = draft_email("waitlist_release", ONLINE, fc, plan, tone=tone).body
    assert "[Join link]" in release and "valid ID" not in release
    reconfirm = draft_email("reconfirm", ONLINE, fc, plan, tone=tone).body
    assert "online" in reconfirm and " at ," not in reconfirm
    assert "[Delivery address]" in draft_email("catering_order", ONLINE, fc, plan, tone=tone).body


@pytest.mark.parametrize("tone", TONES)
def test_templates_without_rsvps(tone):
    fc = make_forecast(n_going=0, n_waitlist=0)
    plan = make_plan(fc, PlanInputs(capacity=40))
    for kind in EMAIL_KINDS:
        assert draft_email(kind, EVENT, fc, plan, tone=tone).body.endswith(SIGN_OFF)
    assert "no RSVPs" in explain_forecast(EVENT, fc, plan, tone="english")


def test_template_tones_differ(forecast, plan):
    taglish = draft_email("reconfirm", EVENT, forecast, plan, tone="taglish").body
    english = draft_email("reconfirm", EVENT, forecast, plan, tone="english").body
    assert "paki-update" in taglish.lower() and "paki-update" not in english.lower()


def test_unknown_kind_or_tone(forecast, plan):
    with pytest.raises(ValueError, match="kind"):
        draft_email("thank_you", EVENT, forecast, plan)
    with pytest.raises(ValueError, match="tone"):
        draft_email("reconfirm", EVENT, forecast, plan, tone="bisaya")


def test_plan_must_match_forecast(forecast, plan):
    other = make_forecast(event_id="pyworks-2026-10")
    with pytest.raises(ValueError, match="pyworks-2026-10"):
        draft_email("reconfirm", EVENT, other, plan)


# --- demo data marker --------------------------------------------------------------------


def test_demo_marker_on_templates():
    fc = make_forecast(synthetic=True)
    plan = make_plan(fc, PlanInputs(capacity=50))
    for kind in EMAIL_KINDS:
        draft = template_draft(kind, EVENT, fc, plan, tone="english")
        assert draft.subject.startswith(DEMO_MARKER) and draft.body.startswith(DEMO_MARKER)
    assert explain_forecast(EVENT, fc, plan).startswith(DEMO_MARKER)


def test_demo_marker_on_llm_drafts_appears_once():
    fc = make_forecast(synthetic=True)
    plan = make_plan(fc, PlanInputs(capacity=50))
    llm = FakeLLM([reply(subject=f"{DEMO_MARKER} Still coming?")])
    draft = draft_email("reconfirm", EVENT, fc, plan, llm=llm)
    assert draft.used_llm
    assert draft.subject == f"{DEMO_MARKER} Still coming?"
    assert draft.body.startswith(DEMO_MARKER) and draft.body.count(DEMO_MARKER) == 1


# --- the Gemma path ----------------------------------------------------------------------


def test_llm_draft_is_used_and_sent_aggregate_only(forecast, plan):
    llm = FakeLLM([reply()])
    draft = draft_email("reconfirm", EVENT, forecast, plan, tone="taglish", llm=llm)
    assert draft.used_llm and draft.model == "fake-gemma"
    assert draft.subject == "Still coming to PyMeets?"
    assert draft.body.endswith(SIGN_OFF)

    (call,) = llm.calls
    assert call["schema"] == DRAFT_SCHEMA and call["temperature"] == 0.4
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    sent = llm.sent_text
    assert_aggregate_only(sent)
    assert "Thu, Sep 17, 6:30 PM" in sent and "PyMeets September 2026" in sent
    summary = forecast.summary()
    for key in ("rsvp_heads", "waitlist_heads", "p10", "p90"):
        assert f'"{key}": {summary[key]}' in sent
    assert "people_on_waitlist" in sent
    assert 'no "po"' in sent  # Taglish to fellow Pythonistas stays casual


def test_llm_prompt_for_business_asks_for_po(forecast, plan):
    (_, user) = draft_messages("catering_order", EVENT, forecast, plan, tone="taglish")
    assert 'use "po"' in user["content"]
    assert f'"pizza_boxes": {plan.pizza_boxes}' in user["content"]
    assert '"delivery_time": "6:00 PM"' in user["content"]
    assert f'"estimated_total": "₱{plan.cost_php:,.0f}"' in user["content"]


def test_english_prompt_has_no_taglish_register_hint(forecast, plan):
    (_, user) = draft_messages("reconfirm", EVENT, forecast, plan, tone="english")
    assert '"po"' not in user["content"]


@pytest.mark.parametrize(
    "bad",
    [
        "Sorry, I can't write that.",  # no JSON at all
        '{"subject": "Hi", "body": "unterminated',
        json.dumps({"subject": "", "body": "Hi"}),
        json.dumps({"subject": "Hi"}),
        reply(body="Hi! We expect 777 no-shows this time.\n— PythonPH Organizers"),
        reply(body="Questions? Email juan@example.com or text 09171234567."),
        reply(body="word " * 300),
        LLMError("Can't reach Ollama"),
    ],
)
def test_bad_llm_replies_fall_back_to_the_template(bad, forecast, plan):
    llm = FakeLLM([bad])
    draft = draft_email("reconfirm", EVENT, forecast, plan, tone="english", llm=llm)
    assert not draft.used_llm
    assert draft == template_draft("reconfirm", EVENT, forecast, plan, tone="english")


def test_unavailable_llm_is_not_called(forecast, plan):
    llm = FakeLLM([reply()], available=False)
    assert not draft_email("reconfirm", EVENT, forecast, plan, llm=llm).used_llm
    assert llm.calls == []


def test_personal_data_in_event_details_never_reaches_the_llm(forecast, plan):
    event = EVENT | {"venue": "Ask juan@example.com for the room"}
    llm = FakeLLM([reply()])
    draft = draft_email("reconfirm", event, forecast, plan, llm=llm)
    assert not draft.used_llm and llm.calls == []


def test_llm_sign_off_is_normalized(forecast, plan):
    body = "Hi Pythonistas!\n\nSee you there!\n\n- PythonPH Organizers\n"
    draft = draft_email("reconfirm", EVENT, forecast, plan, llm=FakeLLM([reply(body=body)]))
    assert draft.body == f"Hi Pythonistas!\n\nSee you there!\n{SIGN_OFF}"
    draft = draft_email("reconfirm", EVENT, forecast, plan, llm=FakeLLM([reply(body="Hi!")]))
    assert draft.body == f"Hi!\n{SIGN_OFF}"


def test_llm_links_become_placeholders_and_emoji_are_limited(forecast, plan):
    body = "Hi Pythonistas! 👋 RSVP at https://meetup.com/pythonph 🐍🎉"
    subject = "🍕 Still coming? 🗓️"
    llm = FakeLLM([reply(subject=subject, body=body)])
    draft = draft_email("reconfirm", EVENT, forecast, plan, llm=llm)
    assert draft.subject == "Still coming?"
    assert draft.body.startswith("Hi Pythonistas! 👋 RSVP at [Meetup link]")
    assert "🐍" not in draft.body and "🎉" not in draft.body


def test_llm_catering_draft_always_has_a_contact(forecast, plan):
    body = f"Hi [Caterer name]!\n\nPlease deliver {plan.pizza_boxes} boxes.\n\nThanks!"
    llm = FakeLLM([reply(subject="Pizza order", body=body)])
    draft = draft_email("catering_order", EVENT, forecast, plan, tone="english", llm=llm)
    assert draft.used_llm
    assert draft.body.endswith(
        f"Thanks!\n\nContact person on the day: [Organizer contact]\n{SIGN_OFF}"
    )


def test_llm_subject_lines_are_not_duplicated(forecast, plan):
    llm = FakeLLM([reply(subject="Subject: Still coming?", body="Subject: Still coming?\nHi!")])
    draft = draft_email("reconfirm", EVENT, forecast, plan, llm=llm)
    assert draft.subject == "Still coming?"
    assert draft.body == f"Hi!\n{SIGN_OFF}"


def test_llm_reply_in_code_fences_is_parsed(forecast, plan):
    llm = FakeLLM([f"```json\n{reply()}\n```"])
    assert draft_email("reconfirm", EVENT, forecast, plan, llm=llm).used_llm


# --- explanations ------------------------------------------------------------------------


@pytest.mark.parametrize("tone", TONES)
def test_explanation_template(tone, forecast, plan):
    text = explain_forecast(EVENT, forecast, plan, tone=tone)
    assert str(round(forecast.expected)) in text
    assert f"{forecast.p10}–{forecast.p90}" in text
    assert f"{plan.pizza_boxes} " in text and str(plan.seats) in text
    assert "Rain expected (14 mm)." in text
    assert 2 <= text.count(". ") + 1 <= 5
    assert_aggregate_only(text)


def test_explanation_by_llm(forecast, plan):
    text = (
        f"Mga {round(forecast.expected)} ang darating, malamang {forecast.p10}–{forecast.p90}. "
        f"Order {plan.pizza_boxes} boxes."
    )
    llm = FakeLLM([json.dumps({"explanation": text})])
    assert explain_forecast(EVENT, forecast, plan, llm=llm) == text
    assert_aggregate_only(llm.sent_text)
    assert "Plan notes" in llm.sent_text


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        json.dumps({"explanation": "Expect 999 people."}),
        json.dumps({"explanation": "Pizza! " * 150}),
    ],
)
def test_explanation_falls_back_to_template(bad, forecast, plan):
    text = explain_forecast(EVENT, forecast, plan, tone="english", llm=FakeLLM([bad]))
    assert text == explain_forecast(EVENT, forecast, plan, tone="english")


def test_explain_messages_are_aggregate_only(forecast, plan):
    for message in explain_messages(EVENT, forecast, plan, tone="taglish"):
        assert_aggregate_only(message["content"])


# --- the real thing ----------------------------------------------------------------------


@pytest.mark.ollama
def test_real_gemma_drafts_a_taglish_reconfirm(forecast, plan):
    llm = LLM()
    assert llm.available(), llm.status()[1]
    draft = draft_email("reconfirm", EVENT, forecast, plan, tone="taglish", llm=llm)
    assert draft.used_llm, "Gemma's draft failed a check; the log says which"
    assert draft.body.endswith(SIGN_OFF)
    assert_aggregate_only(draft.subject + "\n" + draft.body)
