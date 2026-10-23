"""Ask Juan: answers organizer questions by calling the forecast, plan and draft tools.

Juan is named after Juan Tamad, the folk hero who lay under a guava tree waiting for the fruit
to fall into his mouth. Juan does the organizer chores so the volunteers can be lazy too.

With Gemma running, each step Gemma returns one JSON action (call a tool, or answer) and gets
back an aggregate-only JSON result. Without Gemma, a keyword router picks the tool and a
template writes the answer, so Juan still works offline.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field, replace
from string import Template
from typing import TYPE_CHECKING

import pandas as pd

from ilang_pizza.features.build import build_features
from ilang_pizza.llm.client import ChatModel, LLMError, parse_json
from ilang_pizza.llm.prompts import (
    DEMO_MARKER,
    EMAIL_KINDS,
    TONES,
    PrivacyError,
    assert_aggregate_only,
    draft_email,
    format_when,
    invented_numbers,
    redact_pii,
    to_json,
)
from ilang_pizza.model.predict import forecast_event
from ilang_pizza.plan import format_php, make_plan
from ilang_pizza.schema import (
    MANILA_TZ,
    AgentReply,
    BacktestResult,
    Dataset,
    Draft,
    EventPlan,
    PlanInputs,
    TurnoutForecast,
)

if TYPE_CHECKING:
    from ilang_pizza.features.weather import WeatherProvider

log = logging.getLogger(__name__)

MAX_STEPS = 4

# Each tool and the arguments it accepts; anything else Gemma sends is dropped.
TOOL_ARGS = {
    "list_events": ("upcoming_only",),
    "forecast": ("event_id",),
    "plan": (
        "event_id",
        "capacity",
        "slices_per_person",
        "slices_per_box",
        "price_per_box",
        "food_quantile",
        "overflow_risk",
    ),
    "draft": ("kind", "event_id", "tone"),
    "backtest_summary": (),
}
TOOLS = tuple(TOOL_ARGS)
# Plan inputs Gemma may only pass when the organizer typed the number in the question.
ORGANIZER_NUMBERS = (
    "capacity",
    "slices_per_person",
    "slices_per_box",
    "price_per_box",
    "food_quantile",
    "overflow_risk",
)

_ARG_TYPES = {
    "event_id": {"type": "string"},
    "upcoming_only": {"type": "boolean"},
    "capacity": {"type": ["integer", "null"]},
    "slices_per_person": {"type": "number"},
    "slices_per_box": {"type": "integer"},
    "price_per_box": {"type": "number"},
    "food_quantile": {"type": "number"},
    "overflow_risk": {"type": "number"},
    "kind": {"type": "string", "enum": list(EMAIL_KINDS)},
    "tone": {"type": "string", "enum": list(TONES)},
}


def _action_schema(*, final: bool) -> dict:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["answer"] if final else ["tool", "answer"]},
            "tool": {"type": "string", "enum": ["none"] if final else [*TOOLS, "none"]},
            "args": {"type": "object", "properties": _ARG_TYPES},
            "answer": {"type": "string"},
        },
        "required": ["action", "tool", "args", "answer"],
    }


ACTION_SCHEMA = _action_schema(final=False)
FINAL_SCHEMA = _action_schema(final=True)  # the last step has to answer

_SYSTEM = Template("""You are Juan, the helper of the Python Philippines (PythonPH) meetup
organizers. You are named after Juan Tamad, who waited under a guava tree for the fruit to fall:
you do the boring organizer chores so the volunteers don't have to.

Each turn, reply with exactly one JSON object and nothing else:
- To call a tool: {"action": "tool", "tool": "<name>", "args": {...}, "answer": ""}
- To reply: {"action": "answer", "tool": "none", "args": {}, "answer": "<your reply>"}

Tools. Every argument is optional: leave out anything the organizer did not say.
- plan: pizza boxes, waitlist spots to release, seats and e-certificates, with the forecast
  included. Args: event_id, capacity (venue limit in people), slices_per_person,
  slices_per_box, price_per_box, food_quantile, overflow_risk.
- forecast: expected headcount, likely range and what drives it. Args: event_id.
- draft: writes an email. Args: kind, tone (taglish or english), event_id. Kinds: reconfirm
  (ask going RSVPs if they're still coming), waitlist_release (tell the next waitlisted people
  they're in), catering_order (pizza order for the caterer), venue_headcount (headcount for
  the venue host).
- list_events: events with dates, venues and RSVP counts. Args: upcoming_only.
- backtest_summary: how accurate past forecasts were. No args.

Examples:
- "Ilang pizza para sa next PyMeets?" -> {"action": "tool", "tool": "plan", "args": {},
  "answer": ""}
- "How many can we release if the venue fits <N>?" -> {"action": "tool", "tool": "plan",
  "args": {"capacity": <N>}, "answer": ""}
- "Draft the catering email in English" -> {"action": "tool", "tool": "draft",
  "args": {"kind": "catering_order", "tone": "english"}, "answer": ""}

Rules:
- Leave out event_id to use the next upcoming event: $default_event.
- Use only numbers that appear in tool results. Never guess numbers or calculate new ones.
- Reply in the organizer's language: Taglish if they wrote Taglish, English if English.
- Keep replies short and friendly: plain text, a few sentences at most.
- After a draft, don't repeat the email: it is shown to the organizer below your reply.
Today is $today.""")

_LAST_STEP = "You have no tool calls left: answer the organizer now."
_REPLY_IN = {"taglish": "Reply in Taglish.", "english": "Reply in English."}

_EXAMPLES = (
    "Ilang pizza para sa next PyMeets?",
    "Ilan pa pwede i-release sa waitlist kung 80 lang ang venue?",
    "Draft the reconfirm email in English",
    "How accurate has this been?",
    "What events are coming up?",
)

# Keyword router for when Gemma is not available. The first matching route wins.
# fmt: off
_ROUTES = (
    ("draft", ("email", "draft", "sulat", "isulat", "liham", "message", "mensahe")),
    ("backtest_summary", ("accurate", "accuracy", "backtest", "tama", "mintis", "reliable")),
    ("list_events", ("events", "upcoming", "schedule", "sched", "coming up", "calendar")),
    (
        "plan",
        (
            "pizza", "ilang", "ilan", "food", "kain", "pagkain", "waitlist", "boxes", "order",
            "seats", "upuan", "certificates", "headcount", "how many", "magkano", "budget",
        ),
    ),
)
# fmt: on
_KIND_WORDS = (
    ("reconfirm", ("reconfirm", "still coming", "tuloy")),
    ("waitlist_release", ("waitlist", "release", "slot")),
    ("catering_order", ("caterer", "catering", "pizza", "food", "order")),
    ("venue_headcount", ("venue", "building", "admin", "headcount", "guest list", "host")),
)
_SERIES_WORDS = {
    "pymeets": "PyMeets",
    "pyworks": "PyWorks",
    "wfh": "WFH Pythonista",
    "pykids": "PyKids",
}
# fmt: off
_TAGALOG = frozenset({
    "ilang", "ilan", "pwede", "puwede", "para", "sa", "ng", "mga", "ba", "kung", "lang", "naman",
    "po", "pa", "ang", "natin", "tayo", "kami", "sulat", "isulat", "tama", "kain", "pagkain",
    "magkano", "darating", "pupunta", "ano", "anong", "saan", "kailan", "gaano", "paki",
    "salamat", "upuan", "susunod", "yung", "dito", "gawa", "gawin", "kumusta", "kamusta", "ka",
    "ikaw", "ako", "ko", "mo", "nga", "din", "rin", "talaga", "ito", "iyan", "yan", "sino",
    "bakit", "paano", "oo", "opo", "hindi", "wala", "meron",
})
# fmt: on
_NO_LIMIT = ("", "none", "null", "unlimited", "no limit")
_CAPACITY = (
    re.compile(
        r"(\d+)\s*(?:lang\s+)?(?:ang\s+)?(?:venue|capacity|seats|pax|kasya|upuan)", re.IGNORECASE
    ),
    re.compile(r"(?:venue|capacity|room|kasya|fits?|limit|hanggang)\D{0,20}?(\d+)", re.IGNORECASE),
)
_SLICES_PER_BOX = re.compile(r"(\d+)\s*slices?\s*(?:per|a|each|every|bawat)\s*box", re.IGNORECASE)
_SLICES_PER_PERSON = re.compile(r"(\d+(?:\.\d+)?)\s*slices?", re.IGNORECASE)
_PRICE = re.compile(
    r"(?:₱|php\s*)(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s*(?:pesos|php)", re.IGNORECASE
)

_KIND_LABELS = {
    "reconfirm": "reconfirm email",
    "waitlist_release": "waitlist release email",
    "catering_order": "catering order",
    "venue_headcount": "venue headcount email",
}


@dataclass
class _Turn:
    """What one question produces: the tool calls, and drafts to show below the answer."""

    question: str
    tone: str
    steps: list[dict] = field(default_factory=list)
    drafts: list[Draft] = field(default_factory=list)


class AskJuan:
    """Answer organizer questions with tools, using Gemma when it is available.

    ``backtest`` is an optional precomputed ``BacktestResult``: backtests are too slow to run
    mid-conversation. Forecasts are cached per event for the life of the object.
    """

    max_steps = MAX_STEPS

    def __init__(
        self,
        ds: Dataset,
        llm: ChatModel | None = None,
        *,
        plan_inputs: PlanInputs | None = None,
        model_kind: str = "tabpfn",
        weather: WeatherProvider | None = None,
        backtest: BacktestResult | None = None,
    ) -> None:
        self.ds = ds
        self.llm = llm
        self.plan_inputs = plan_inputs or PlanInputs()
        self.model_kind = model_kind
        self.weather = weather
        self.backtest = backtest
        self._forecasts: dict[str, TurnoutForecast] = {}
        self._features: pd.DataFrame | None = None

    def ask(self, question: str) -> AgentReply:
        """Answer one question. ``AgentReply.steps`` lists the tool calls behind the answer."""
        question = " ".join(question.split())
        tone = detect_tone(question)
        if not question:
            return AgentReply(answer=self._finish(_help(tone), _Turn("", tone)), used_llm=False)
        if self.llm is not None and self.llm.available():
            try:
                return self._ask_llm(_Turn(question, tone))
            except (LLMError, PrivacyError) as e:
                log.warning("Juan is answering without %s: %s", self.llm.model, e)
        return self._ask_router(_Turn(question, tone))

    # --- the Gemma loop ------------------------------------------------------------------

    def _ask_llm(self, turn: _Turn) -> AgentReply:
        system = self._system_prompt()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{redact_pii(turn.question)}\n\n({_REPLY_IN[turn.tone]})"},
        ]
        for step in range(1, self.max_steps + 1):
            final = step == self.max_steps
            for message in messages:
                assert_aggregate_only(message["content"])
            schema = FINAL_SCHEMA if final else ACTION_SCHEMA
            raw = self.llm.chat(messages, schema=schema, temperature=0.2)
            reply, feedback = self._handle_action(raw, turn, system, final=final)
            if reply is not None:
                return reply
            if final:
                break
            if step == self.max_steps - 1:
                feedback = f"{feedback}\n{_LAST_STEP}"
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": feedback},
            ]
        if not turn.steps:
            raise LLMError("Gemma did not settle on an answer")
        # Out of steps: answer from the last tool result with a template instead.
        answer = _render(turn.steps[-1], turn.tone)
        return AgentReply(answer=self._finish(answer, turn), steps=turn.steps, used_llm=False)

    def _handle_action(
        self, raw: str, turn: _Turn, system: str, *, final: bool
    ) -> tuple[AgentReply | None, str]:
        """Run one action from Gemma: a reply when it answered well, else feedback for it."""
        try:
            action = parse_json(raw)
        except LLMError:
            return None, "That was not a valid JSON action. Reply with only one JSON object."
        tool = str(action.get("tool") or "none")
        if action.get("action") == "tool" and not final:
            if tool not in TOOLS:
                return None, f"There is no {tool!r} tool. Use one of: {', '.join(TOOLS)}."
            args = action.get("args")
            args = _organizer_args(args if isinstance(args, dict) else {}, turn.question)
            result = self._call(tool, args, turn)
            return None, f"Result of {tool}: {to_json(result)}"
        answer = str(action.get("answer") or "").strip()
        if not answer:
            return None, 'Put your reply to the organizer in the "answer" field.'
        suggested = route(turn.question)
        if not turn.steps and suggested and not final:
            return None, f"Call the {suggested} tool first, so your answer uses real numbers."
        sources = [system, turn.question, *(to_json(s["result"]) for s in turn.steps)]
        invented = invented_numbers(answer, *sources)
        if invented:
            log.info("Juan's answer used numbers not found in any tool result: %s", invented)
            return None, (
                f"Your answer used numbers that are not in any tool result "
                f"({', '.join(invented)}). Call a tool to get them, or answer again using only "
                "numbers from the results."
            )
        return AgentReply(answer=self._finish(answer, turn), steps=turn.steps), ""

    def _system_prompt(self) -> str:
        default = self._next_event_id()
        if default is None:
            default_event = "none yet (there are no events in the data)"
        else:
            event = self.ds.event(default)
            default_event = f"{event['title']} ({default}), {format_when(event['start'])}"
        today = pd.Timestamp.now(tz=MANILA_TZ)
        return _SYSTEM.substitute(default_event=default_event, today=f"{today:%a, %b} {today.day}")

    # --- the keyword router --------------------------------------------------------------

    def _ask_router(self, turn: _Turn) -> AgentReply:
        tool = route(turn.question)
        if tool is None:
            return AgentReply(answer=self._finish(_help(turn.tone), turn), used_llm=False)
        self._call(tool, self._router_args(tool, turn), turn)
        answer = _render(turn.steps[-1], turn.tone)
        return AgentReply(answer=self._finish(answer, turn), steps=turn.steps, used_llm=False)

    def _router_args(self, tool: str, turn: _Turn) -> dict:
        q = turn.question
        args: dict = {}
        if "event_id" in TOOL_ARGS[tool]:
            event_id = self._event_in_text(q)
            if event_id:
                args["event_id"] = event_id
        if tool == "list_events":
            args["upcoming_only"] = not re.search(
                r"\b(past|all|lahat|nakaraan)\b", q, re.IGNORECASE
            )
        elif tool == "plan":
            args.update(plan_numbers(q))
        elif tool == "draft":
            kinds = (kind for kind, words in _KIND_WORDS if _has_any(q, words))
            args["kind"] = next(kinds, "reconfirm")
            args["tone"] = turn.tone
        return args

    # --- tools ---------------------------------------------------------------------------

    def _call(self, tool: str, args: dict, turn: _Turn) -> dict:
        """Run a tool and record the step. A failure becomes an aggregate-only error result."""
        handlers = {
            "list_events": self._list_events,
            "forecast": self._forecast_tool,
            "plan": self._plan_tool,
            "draft": self._draft_tool,
            "backtest_summary": self._backtest_tool,
        }
        clean = {k: v for k, v in args.items() if k in TOOL_ARGS[tool] and v is not None}
        try:
            result = handlers[tool](turn, **clean)
        except (KeyError, ValueError, LLMError) as e:
            result = {"error": redact_pii(str(e).strip("'\""))[:300]}
        except Exception as e:  # noqa: BLE001 - a failing model backend must not end the chat
            log.warning("Juan's %s tool failed: %s", tool, type(e).__name__)
            result = {"error": f"the {tool} tool failed ({type(e).__name__})"}
        turn.steps.append({"tool": tool, "args": clean, "result": result})
        return result

    def _list_events(self, turn: _Turn, upcoming_only: bool | str = True) -> dict:
        upcoming_only = _as_bool(upcoming_only)
        if upcoming_only:
            ids = self._future_ids()
        else:  # recent history plus what is coming
            ids = list(self.ds.events.sort_values("start")["event_id"])[-10:]
        return {"upcoming_only": upcoming_only, "events": [self._card(e) for e in ids]}

    def _forecast_tool(self, turn: _Turn, event_id: str | None = None) -> dict:
        event_id = self._event_id(event_id)
        return {"event": self._card(event_id), "forecast": self._forecast(event_id).summary()}

    def _plan_tool(self, turn: _Turn, event_id: str | None = None, **overrides) -> dict:
        event_id = self._event_id(event_id)
        forecast = self._forecast(event_id)
        inputs = self._plan_inputs(event_id, overrides)
        return {
            "event": self._card(event_id),
            "inputs": {
                "capacity": inputs.capacity,
                "slices_per_person": inputs.slices_per_person,
                "slices_per_box": inputs.slices_per_box,
                "price_per_box": inputs.price_per_box,
                "food_quantile": inputs.food_quantile,
                "overflow_risk": inputs.overflow_risk,
            },
            "forecast": _forecast_brief(forecast),
            "plan": _plan_fields(make_plan(forecast, inputs)),
        }

    def _draft_tool(
        self,
        turn: _Turn,
        kind: str = "reconfirm",
        event_id: str | None = None,
        tone: str | None = None,
    ) -> dict:
        kind, tone = _as_kind(kind), _as_tone(tone) or turn.tone
        event_id = self._event_id(event_id)
        forecast = self._forecast(event_id)
        plan = make_plan(forecast, self._plan_inputs(event_id, {}))
        draft = draft_email(kind, self.ds.event(event_id), forecast, plan, tone=tone, llm=self.llm)
        turn.drafts.append(draft)
        # Gemma gets the subject only: the email itself goes straight to the organizer.
        return {
            "event": self._card(event_id),
            "kind": kind,
            "tone": tone,
            "subject": draft.subject,
            "written_by": draft.model if draft.used_llm else "template",
        }

    def _backtest_tool(self, turn: _Turn) -> dict:
        bt = self.backtest
        if bt is None:
            return {
                "available": False,
                "hint": "Run `ilang-pizza backtest` to measure accuracy on past events "
                "(it takes a few minutes).",
            }
        metrics = {
            str(method): {str(k): _number(v) for k, v in row.items()}
            for method, row in bt.metrics.iterrows()
        }
        recent = [
            {
                "title": str(row["title"]),
                "actual_heads": _number(row["actual_heads"]),
                "model_expected": _number(row["model_expected"]),
                "rule_of_thumb": _number(row["rule_pred"]),
            }
            for _, row in bt.per_event.tail(3).iterrows()
        ]
        return {
            "available": True,
            "model": bt.model_name,
            "main_method": _main_method(metrics),
            "events_evaluated": len(bt.per_event),
            "metrics": metrics,
            "recent_events": recent,
            "synthetic_data": bt.synthetic,
        }

    # --- helpers -------------------------------------------------------------------------

    def _forecast(self, event_id: str) -> TurnoutForecast:
        if event_id not in self._forecasts:
            if self._features is None:
                self._features = build_features(self.ds, weather=self.weather)
            self._forecasts[event_id] = forecast_event(
                self.ds,
                event_id,
                model_kind=self.model_kind,
                weather=self.weather,
                features=self._features,
            )
        return self._forecasts[event_id]

    def _plan_inputs(self, event_id: str, overrides: dict) -> PlanInputs:
        """Organizer overrides first, then the constructor's inputs, then the event capacity."""
        inputs = self.plan_inputs
        if "capacity" not in overrides and inputs.capacity is None:
            inputs = replace(inputs, capacity=_as_capacity(self.ds.event(event_id)["capacity"]))
        changes = {}
        for name, value in overrides.items():
            if name == "capacity":
                changes[name] = _as_capacity(value)
            elif name == "slices_per_box":
                changes[name] = int(_as_float(value, name))
            else:
                changes[name] = _as_float(value, name)
        return replace(inputs, **changes)

    def _future_ids(self) -> list[str]:
        """Events with no attendance yet that start from now on, soonest first."""
        starts = self.ds.events.set_index("event_id")["start"]
        now = pd.Timestamp.now(tz=MANILA_TZ)
        return [e for e in self.ds.upcoming_event_ids() if starts[e] >= now]

    def _next_event_id(self, series: str | None = None) -> str | None:
        """``Dataset.next_event_id``, optionally within one series ("the next PyMeets")."""
        ds = self.ds
        if series is not None:
            events = ds.events[ds.events["series"] == series]
            rsvps = ds.rsvps[ds.rsvps["event_id"].isin(events["event_id"])]
            ds = Dataset(events=events, rsvps=rsvps, synthetic=ds.synthetic)
        return ds.next_event_id()

    def _event_id(self, value: str | None) -> str:
        """Resolve an event reference: an event_id, a title, a series name, or nothing."""
        text = str(value or "").strip()
        if text.lower() in ("", "next", "default", "none", "null", "upcoming"):
            default = self._next_event_id()
            if default is None:
                raise ValueError("there are no events in the data yet")
            return default
        for event_id, title in zip(self.ds.events["event_id"], self.ds.events["title"]):
            if text.lower() in (event_id.lower(), str(title).lower()):
                return event_id
        found = self._event_in_text(text)
        if found:
            return found
        upcoming = ", ".join(self._future_ids()) or "none"
        raise ValueError(f"unknown event {text!r}; upcoming events: {upcoming}")

    def _event_in_text(self, text: str) -> str | None:
        """An event_id mentioned in ``text``, or the next event of a series it names."""
        lowered = text.lower()
        for event_id in self.ds.events["event_id"]:
            if event_id.lower() in lowered:
                return event_id
        for word, series in _SERIES_WORDS.items():
            if word in lowered:
                found = self._next_event_id(series)
                if found is None:
                    raise ValueError(f"there is no upcoming {series} event in the data")
                return found
        return None

    def _card(self, event_id: str) -> dict:
        """Aggregate facts about one event, safe for the LLM."""
        event = self.ds.event(event_id)
        rsvps = self.ds.rsvps_for(event_id)
        heads = 1 + rsvps["guests"].fillna(0)
        going = rsvps["status"] == "going"
        place = ", ".join(p for p in (event["venue"], event["city"]) if isinstance(p, str) and p)
        card = {
            "event_id": event_id,
            "title": str(event["title"]),
            "series": str(event["series"]),
            "when": format_when(event["start"]),
            "date": f"{event['start'].tz_convert(MANILA_TZ):%Y-%m-%d}",
            "venue": "online" if bool(event["is_online"]) or not place else place,
            "capacity": _as_capacity(event["capacity"]),
            "going_heads": int(heads[going].sum()),
            "waitlist_heads": int(heads[~going].sum()),
        }
        attended = rsvps.loc[going, "attended"]
        if attended.notna().any():
            card["attended_heads"] = int((heads[going] * attended.fillna(0)).sum())
        return card

    def _finish(self, answer: str, turn: _Turn) -> str:
        """The answer, then any drafts in full, labeled when the data is the demo set."""
        parts = [answer.strip()]
        parts += [f"Subject: {d.subject}\n\n{d.body}" for d in turn.drafts]
        text = "\n\n".join(parts)
        return f"{DEMO_MARKER} {text}" if self.ds.synthetic else text


# --- routing and parsing -----------------------------------------------------------------


def detect_tone(text: str) -> str:
    """The tone to answer in: an explicit "English"/"Taglish" wins, else the question's own."""
    lowered = text.lower()
    if re.search(r"\b(english|ingles)\b", lowered):
        return "english"
    if re.search(r"\b(taglish|tagalog|filipino)\b", lowered):
        return "taglish"
    return "taglish" if set(re.findall(r"[a-zñ]+", lowered)) & _TAGALOG else "english"


def route(question: str) -> str | None:
    """The tool the keyword router picks for ``question``, or None."""
    return next((tool for tool, words in _ROUTES if _has_any(question, words)), None)


def plan_numbers(question: str) -> dict:
    """Plan inputs typed into a question: "kung 80 lang ang venue", "3 slices each", "₱450"."""
    found: dict = {}
    for pattern in _CAPACITY:
        match = pattern.search(question)
        if match:
            found["capacity"] = int(match.group(1))
            break
    per_box = _SLICES_PER_BOX.search(question)
    if per_box:
        found["slices_per_box"] = int(per_box.group(1))
    per_person = _SLICES_PER_PERSON.search(_SLICES_PER_BOX.sub("", question))
    if per_person:
        found["slices_per_person"] = float(per_person.group(1))
    price = _PRICE.search(question)
    if price:
        found["price_per_box"] = float((price.group(1) or price.group(2)).replace(",", ""))
    return found


def _organizer_args(args: dict, question: str) -> dict:
    """Drop plan numbers the organizer never typed: small models like to fill in every field."""

    def typed(value) -> bool:
        text = str(value)
        return bool(re.search(r"\d", text)) and not invented_numbers(text, question, exact=True)

    return {k: v for k, v in args.items() if k not in ORGANIZER_NUMBERS or typed(v)}


def _has_any(text: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) for w in words)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "no", "0", "hindi")
    return bool(value)


def _as_float(value, name: str) -> float:
    try:
        number = float(str(value).replace(",", "").replace("₱", "").strip())
    except ValueError:
        raise ValueError(f"{name} must be a number, not {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _as_capacity(value) -> int | None:
    """None, NaN (how the events table says "unlimited") and words like "unlimited": no limit."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str) and value.strip().lower() in _NO_LIMIT:
        return None
    number = _as_float(value, "capacity")
    if number < 0:
        raise ValueError("capacity cannot be negative")
    return int(number)


def _as_kind(value) -> str:
    text = str(value or "").strip().lower()
    if text in EMAIL_KINDS:
        return text
    for kind, words in _KIND_WORDS:
        if any(w in text for w in words):
            return kind
    raise ValueError(f"unknown email kind {value!r}; choose one of {', '.join(EMAIL_KINDS)}")


def _as_tone(value) -> str | None:
    text = str(value or "").strip().lower()
    if text in ("taglish", "tagalog", "filipino"):
        return "taglish"
    if text in ("english", "en"):
        return "english"
    return None


def _number(value):
    """JSON-friendly number: whole numbers become ints, NaN becomes None."""
    if value is None or isinstance(value, str):
        return value
    number = float(value)
    if math.isnan(number):
        return None
    return int(number) if number.is_integer() else round(number, 3)


def _main_method(metrics: dict) -> str | None:
    """The backtested model, as opposed to the baselines it is compared with."""
    if "tabpfn" in metrics:
        return "tabpfn"
    models = [m for m in metrics if m not in ("history", "rule")]
    return models[0] if models else next(iter(metrics), None)


def _forecast_brief(forecast: TurnoutForecast) -> dict:
    s = forecast.summary()
    keys = ("rsvp_heads", "waitlist_heads", "expected_heads", "p10", "p50", "p90", "drivers")
    return {k: s[k] for k in keys} | {"waitlist_rsvps": len(forecast.waitlist_probs)}


def _plan_fields(plan: EventPlan) -> dict:
    fields = {
        "food_heads": plan.food_heads,
        "pizza_boxes": plan.pizza_boxes,
        "pizza_boxes_naive": plan.pizza_boxes_naive,
        "waitlist_release": plan.waitlist_release,
        "overflow_probability": round(plan.overflow_probability, 3),
        "seats": plan.seats,
        "certificates": plan.certificates,
        "notes": list(plan.notes),
    }
    if plan.cost_php > 0:
        fields |= {
            "cost_php": plan.cost_php,
            "cost_naive_php": plan.cost_naive_php,
            "savings_php": plan.savings_php,
        }
    return fields


# --- templated answers -------------------------------------------------------------------


def _render(step: dict, tone: str) -> str:
    """A templated answer for one tool result."""
    result, taglish = step["result"], tone == "taglish"
    if "error" in result:
        prefix = "Pasensya, hindi ko nagawa" if taglish else "Sorry, I couldn't do that"
        return f"{prefix}: {result['error']}."
    renderers = {
        "list_events": _render_events,
        "forecast": _render_forecast,
        "plan": _render_plan,
        "draft": _render_draft,
        "backtest_summary": _render_backtest,
    }
    return renderers[step["tool"]](result, taglish)


def _render_events(r: dict, taglish: bool) -> str:
    if not r["events"]:
        if taglish:
            return "Wala pang upcoming event sa data. Idagdag muna ito sa events.csv."
        return "There are no upcoming events in the data yet. Add the next one to events.csv."
    if taglish:
        header = "Mga susunod na event:" if r["upcoming_only"] else "Mga event:"
        waiting, came = "sa waitlist", "dumating"
    else:
        header = "Coming up:" if r["upcoming_only"] else "Recent and upcoming events:"
        waiting, came = "waitlisted", "came"
    lines = [header]
    for e in r["events"]:
        counts = f"{e['going_heads']} going, {e['waitlist_heads']} {waiting}"
        if "attended_heads" in e:
            counts += f", {e['attended_heads']} {came}"
        lines.append(f"- {e['title']}: {e['when']}, {e['venue']} ({counts})")
    return "\n".join(lines)


def _headline(event: dict, forecast: dict, taglish: bool) -> str:
    expected = round(forecast["expected_heads"])
    low, high = forecast["p10"], forecast["p90"]
    likely = str(low) if low == high else f"{low}–{high}"
    if taglish:
        return (
            f"Para sa {event['title']} ({event['when']}): mga {expected} sa "
            f"{forecast['rsvp_heads']} na naka-RSVP ang inaasahang darating (malamang {likely})."
        )
    return (
        f"{event['title']} ({event['when']}): expect about {expected} of the "
        f"{forecast['rsvp_heads']} RSVP'd people (likely {likely})."
    )


def _render_forecast(r: dict, taglish: bool) -> str:
    text = _headline(r["event"], r["forecast"], taglish)
    drivers = "; ".join(d.strip().rstrip(".") for d in r["forecast"]["drivers"][:3])
    if drivers:
        text += f" {'Mga dahilan' if taglish else 'Main factors'}: {drivers}."
    return text


def _render_plan(r: dict, taglish: bool) -> str:
    plan = r["plan"]
    lines = [_headline(r["event"], r["forecast"], taglish)]
    if not taglish:
        return "\n".join(lines + [f"- {note}" for note in plan["notes"]])
    lines += [
        (
            f"- Pizza: {plan['pizza_boxes']} boxes, pang-{plan['food_heads']} na tao. Kung "
            f"lahat ng RSVP ang papakainin, {plan['pizza_boxes_naive']} boxes sana."
        ),
        f"- Waitlist: {_waitlist_taglish(r)}",
        f"- Upuan at e-certificates: {plan['seats']}.",
    ]
    if "cost_php" in plan:
        lines.append(
            f"- Gastos: {format_php(plan['cost_php'])} imbes na "
            f"{format_php(plan['cost_naive_php'])} (tipid na {format_php(plan['savings_php'])})."
        )
    return "\n".join(lines)


def _waitlist_taglish(r: dict) -> str:
    plan, inputs = r["plan"], r["inputs"]
    capacity, risk = inputs["capacity"], plan["overflow_probability"]
    waiting, released = r["forecast"]["waitlist_rsvps"], plan["waitlist_release"]
    if not waiting:
        return "walang naka-waitlist."
    if capacity is None:
        return f"walang venue limit, kaya i-release na lahat ({released})."
    over = f"lumampas sa {capacity} na tao"
    if released == 0 and risk > inputs["overflow_risk"]:
        pct = "halos 100%" if 0.995 <= risk < 1 else f"{round(risk * 100)}%"
        return (
            f"huwag munang mag-release. Kahit wala pang i-release, {pct} na ang chance na {over}."
        )
    if released == 0:
        return "huwag munang mag-release: kahit isa lang, sobra na ang risk na lumampas sa venue."
    count = f"lahat ng {released}" if released == waiting else f"{released} sa {waiting}"
    if risk < 0.005:
        return f"pwedeng i-release ang {count}. Halos walang chance na {over}."
    return f"pwedeng i-release ang {count}. {round(risk * 100)}% lang ang chance na {over}."


def _render_draft(r: dict, taglish: bool) -> str:
    label = _KIND_LABELS[r["kind"]]
    template = r["written_by"] == "template"
    if taglish:
        note = " (template ito; i-start ang Ollama para si Gemma ang sumulat)" if template else ""
        return f"Heto ang draft ng {label}{note}:"
    note = " (from the template; start Ollama for a Gemma-written one)" if template else ""
    return f"Here's the {label} draft{note}:"


def _render_backtest(r: dict, taglish: bool) -> str:
    if not r["available"]:
        if taglish:
            return (
                "Wala pang backtest results. Patakbuhin muna ang `ilang-pizza backtest` "
                "(ilang minuto lang), tapos magtanong ulit."
            )
        return (
            "No backtest results are loaded yet. Run `ilang-pizza backtest` (it takes a few "
            "minutes), then ask again."
        )
    model = r["metrics"].get(r["main_method"], {})
    rule_mae = r["metrics"].get("rule", {}).get("mae")
    mae, coverage, n = model.get("mae"), model.get("coverage_80"), r["events_evaluated"]
    if mae is None:
        return (
            "Wala pang accuracy numbers ang backtest."
            if taglish
            else ("The backtest has no accuracy numbers yet.")
        )
    if taglish:
        text = f"Sa {n} na nakaraang event, mga {mae:.1f} tao ang average na mintis ng forecast"
        if rule_mae is not None:
            text += f', kumpara sa {rule_mae:.1f} ng "kalahati ang darating" na rule of thumb'
        text += "."
        if coverage is not None:
            text += f" {round(100 * coverage)}% ng actual turnout ang pasok sa 80% range."
        return text
    text = f"Over {n} past events, the forecast was off by {mae:.1f} people on average"
    if rule_mae is not None:
        text += f', vs {rule_mae:.1f} for the "half will show up" rule of thumb'
    text += "."
    if coverage is not None:
        text += f" {round(100 * coverage)}% of actual turnouts landed inside the 80% range."
    return text


def _help(tone: str) -> str:
    examples = "\n".join(f"- {q}" for q in _EXAMPLES)
    if tone == "taglish":
        return (
            "Ako si Juan, ako na ang bahala sa mga gawaing organizer. Subukan mong itanong:\n"
            f"{examples}\n(Para sa kahit anong tanong, i-start ang Ollama with Gemma 4.)"
        )
    return (
        "I'm Juan, and I do the organizer chores so you don't have to. Try asking:\n"
        f"{examples}\n(For free-form questions, start Ollama with Gemma 4.)"
    )
