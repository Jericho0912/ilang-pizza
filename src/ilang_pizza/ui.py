"""Ilang pizza? web app (Gradio 6).

``build_app()`` lays out five tabs over an ``AppState``, which keeps the loaded dataset, its
feature matrix and every forecast in memory: moving a plan slider re-plans instantly, and
TabPFN only retrains when the event or the model changes. ``main()`` serves the app on
127.0.0.1 with sharing and Gradio telemetry off, so nothing about attendees leaves the machine.
"""

from __future__ import annotations

import html
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import gradio as gr
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter

from ilang_pizza import cli, config
from ilang_pizza.data.loaders import load_dataset
from ilang_pizza.data.synth import make_synthetic_dataset, write_dataset
from ilang_pizza.features.build import build_features
from ilang_pizza.features.weather import default_provider
from ilang_pizza.llm.agent import AskJuan
from ilang_pizza.llm.client import LLM, ChatModel
from ilang_pizza.llm.prompts import draft_email, explain_forecast
from ilang_pizza.model.backends import TABPFN_ATTRIBUTION, TabPFNUnavailableError, tabpfn_status
from ilang_pizza.model.backtest import backtest
from ilang_pizza.model.predict import forecast_event
from ilang_pizza.model.turnout import quantile
from ilang_pizza.plan import make_plan
from ilang_pizza.schema import (
    MANILA_TZ,
    BacktestResult,
    Dataset,
    EventPlan,
    PlanInputs,
    TurnoutForecast,
)

PY_BLUE = "#3776AB"
PY_YELLOW = "#FFD43B"
BLUE_SOFT = "#A9C4DD"  # PY_BLUE at ~40% on white: bars outside the 80% range
RULE_ORANGE = "#C96F2B"  # passes contrast and CVD checks next to PY_BLUE on light and dark
INK = "#7A8699"  # chart text: ~3.7:1 on white and ~4:1 on Gradio's dark blocks

# Gradio gives a live progress bar to any callback parameter whose default is a gr.Progress.
PROGRESS = gr.Progress()

MODEL_CHOICES = [("TabPFN", "tabpfn"), ("Logistic regression", "logistic")]
TONE_CHOICES = [("Taglish", "taglish"), ("English", "english")]
DATA_GUIDE_URL = "https://github.com/Jericho0912/ilang-pizza/blob/main/docs/data-guide.md"
# Open-Meteo's rain data is CC BY 4.0: credit it with a link wherever rain is shown or used.
_WEATHER_LINK = (
    f'<a href="{cli.WEATHER_CREDIT_URL}" target="_blank" rel="noopener">{cli.WEATHER_CREDIT}</a>'
)
ASK_EXAMPLES = [
    "Ilang pizza ang dapat i-order para sa next PyMeets?",
    "Ilan ang darating sa susunod na event?",
    "Pwede na ba i-release ang waitlist? Ilan?",
    "How many people should I expect at the next event?",
    "How much do we save versus feeding every RSVP?",
]


# --- data and caches -----------------------------------------------------------------------


class Explanation(NamedTuple):
    text: str
    by_llm: bool  # False when the template wrote it
    tone: str


@contextmanager
def _friendly_errors() -> Iterator[None]:
    """Show the errors an organizer can act on as a message instead of a stack trace."""
    try:
        yield
    except TabPFNUnavailableError as e:
        raise gr.Error(f"{e} Switch the model to Logistic regression meanwhile.") from e
    except (KeyError, ValueError) as e:
        raise gr.Error(str(e).strip("'\"")) from e


class Workspace:
    """One data folder: the dataset plus cached features, forecasts, backtests, explanations."""

    def __init__(self, data_dir: Path, *, offline_weather: bool = False) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.ds: Dataset | None = None
        if (self.data_dir / "events.csv").exists():
            self.ds = load_dataset(self.data_dir)
        self.weather = default_provider(offline=offline_weather)
        self.explanations: dict[tuple[str, str, PlanInputs], Explanation] = {}
        self._features: pd.DataFrame | None = None
        self._forecasts: dict[tuple[str, str], TurnoutForecast] = {}
        self._backtests: dict[tuple[str, int], BacktestResult] = {}
        self.latest_backtest: BacktestResult | None = None
        self._lock = threading.Lock()  # one model fit at a time; the caches stay consistent

    def require(self) -> Dataset:
        if self.ds is None:
            raise gr.Error(
                f"No events.csv in {self.data_dir}. Open the 🗂️ Data tab and press "
                "Generate demo data, or point it at your exports."
            )
        return self.ds

    def cached_forecast(self, event_id: str | None, model_kind: str) -> TurnoutForecast | None:
        return self._forecasts.get((event_id or "", model_kind))

    def forecast(self, event_id: str, model_kind: str) -> TurnoutForecast:
        """Train and forecast once per (event, model); later calls are free."""
        ds = self.require()
        with self._lock, _friendly_errors():
            key = (event_id, model_kind)
            if key not in self._forecasts:
                if self._features is None:
                    self._features = build_features(ds, weather=self.weather)
                self._forecasts[key] = forecast_event(
                    ds,
                    event_id,
                    model_kind=model_kind,
                    weather=self.weather,
                    features=self._features,
                )
            return self._forecasts[key]

    def backtest(
        self, model_kind: str, last_n: int, progress: Callable[[int, int], None] | None = None
    ) -> BacktestResult:
        ds = self.require()
        with self._lock, _friendly_errors():
            key = (model_kind, last_n)
            if key not in self._backtests:
                self._backtests[key] = backtest(
                    ds,
                    model_kind=model_kind,
                    last_n=last_n,
                    weather=self.weather,
                    progress=progress,
                )
            self.latest_backtest = self._backtests[key]
            return self.latest_backtest


class AppState:
    """What every callback shares: the current workspace and the local LLM.

    The methods ending in ``_view`` are the Gradio callbacks; they take and return plain
    values, so tests can call them without a browser.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        *,
        offline_weather: bool = False,
        llm: ChatModel | None = None,
    ) -> None:
        self.offline_weather = offline_weather
        self.llm = llm if llm is not None else LLM()
        self.tabpfn_ready, self.tabpfn_message = tabpfn_status()
        self.default_model = "tabpfn" if self.tabpfn_ready else "logistic"
        self.workspace = Workspace(cli.resolve_data_dir(data_dir), offline_weather=offline_weather)
        self._agents: dict[str, AskJuan] = {}

    @property
    def ds(self) -> Dataset | None:
        return self.workspace.ds

    def _ready_llm(self) -> ChatModel | None:
        return self.llm if self.llm.available() else None

    # --- events and plan inputs

    def event_choices(self) -> list[tuple[str, str]]:
        """Events without check-ins: upcoming ones soonest first, then past ones, latest first."""
        if self.ds is None:
            return []
        now = pd.Timestamp.now(tz=MANILA_TZ)
        events = [self.ds.event(e) for e in self.ds.upcoming_event_ids()]
        upcoming = [e for e in events if e["start"] >= now]
        past = [e for e in reversed(events) if e["start"] < now]
        choices = []
        for event in upcoming + past:
            label = f"{event['title']} · {_short_date(event['start'])}"
            if event["start"] < now:
                label += " (past)"
            choices.append((label, event["event_id"]))
        return choices

    def default_event(self) -> str | None:
        return self.ds.next_event_id() if self.ds is not None else None

    def capacity_of(self, event_id: str | None) -> int | None:
        if self.ds is None or not event_id:
            return None
        return cli.event_capacity(self.ds.event(event_id))

    def _inputs(self, capacity, slices, box, price, food_q, risk) -> PlanInputs:
        return PlanInputs(
            capacity=int(capacity) if capacity else None,
            slices_per_person=float(slices),
            slices_per_box=int(box),
            price_per_box=float(price or 0),
            food_quantile=float(food_q),
            overflow_risk=float(risk),
        )

    # --- 🍕 forecast and plan

    def forecast_view(
        self,
        event_id,
        model_kind,
        capacity,
        slices,
        box,
        price,
        food_q,
        risk,
        progress=PROGRESS,
    ):
        """Forecast button: train if needed, then headline, chart, plan cards and drivers."""
        if not event_id:
            raise gr.Error("Pick an event first.")
        if self.workspace.cached_forecast(event_id, model_kind) is None:
            progress(0.15, desc=f"Training {cli.model_label(model_kind)} on past RSVPs…")
        fc = self.workspace.forecast(event_id, model_kind)
        inputs = self._inputs(capacity, slices, box, price, food_q, risk)
        return self._results(
            event_id, fc, inputs, _explain_html_pending("Writing the explanation…")
        )

    def replan_view(self, event_id, model_kind, capacity, slices, box, price, food_q, risk):
        """Plan inputs, event or model changed: re-plan from the cache, never retrain."""
        fc = self.workspace.cached_forecast(event_id, model_kind)
        if fc is None:
            label = cli.model_label(model_kind)
            return _idle_results(f"Press <b>Forecast</b> to train {html.escape(label)}.")
        inputs = self._inputs(capacity, slices, box, price, food_q, risk)
        pending = _explain_html_pending("Settings changed. Press ✨ Explain to update this.")
        return self._results(event_id, fc, inputs, pending)

    def event_view(self, event_id, model_kind, slices, box, price, food_q, risk):
        """A new event: reset capacity to its venue limit, then re-plan."""
        capacity = self.capacity_of(event_id)
        return capacity, *self.replan_view(
            event_id, model_kind, capacity, slices, box, price, food_q, risk
        )

    def _results(self, event_id: str, fc: TurnoutForecast, inputs: PlanInputs, pending: str):
        event = self.workspace.require().event(event_id)
        plan = make_plan(fc, inputs)
        cached = self.workspace.explanations.get((event_id, fc.model_name, inputs))
        explanation = _explain_html(cached, self.llm.model) if cached else pending
        return (
            _headline_html(event, fc),
            headcount_chart(fc, inputs.capacity),
            _cards_html(plan, fc, inputs),
            _drivers_html(fc, weather=not self.offline_weather),
            explanation,
        )

    def explain_view(self, event_id, model_kind, capacity, slices, box, price, food_q, risk, tone):
        """Gemma explains the forecast and plan; the template does when Gemma isn't running."""
        fc = self.workspace.cached_forecast(event_id, model_kind)
        if fc is None:
            return _explain_html_pending("Press Forecast first.")
        event = self.workspace.require().event(event_id)
        inputs = self._inputs(capacity, slices, box, price, food_q, risk)
        plan = make_plan(fc, inputs)
        key = (event_id, fc.model_name, inputs)
        cached = self.workspace.explanations.get(key)
        if cached is None or cached.tone != tone:
            llm = self._ready_llm()
            text = explain_forecast(event, fc, plan, tone=tone, llm=llm)
            # explain_forecast falls back to the template quietly; comparing tells us which ran.
            template = explain_forecast(event, fc, plan, tone=tone) if llm else text
            cached = Explanation(text, by_llm=llm is not None and text != template, tone=tone)
            self.workspace.explanations[key] = cached
        return _explain_html(cached, self.llm.model)

    # --- ✉️ drafts

    def draft_view(
        self,
        kind,
        tone,
        event_id,
        model_kind,
        capacity,
        slices,
        box,
        price,
        food_q,
        risk,
        progress=PROGRESS,
    ):
        """Draft an email from the same forecast and plan as the main tab."""
        if not event_id:
            raise gr.Error("Pick an event in the 🍕 tab first.")
        if self.workspace.cached_forecast(event_id, model_kind) is None:
            progress(0.1, desc=f"Training {cli.model_label(model_kind)} on past RSVPs…")
        fc = self.workspace.forecast(event_id, model_kind)
        event = self.workspace.require().event(event_id)
        plan = make_plan(fc, self._inputs(capacity, slices, box, price, food_q, risk))
        llm = self._ready_llm()
        progress(0.5, desc="Gemma is writing…" if llm else "Filling in the template…")
        result = draft_email(kind, event, fc, plan, tone=tone, llm=llm)
        if result.used_llm:
            source = f"✍️ Written by Gemma ({html.escape(result.model)}) on this machine."
        elif llm is not None:
            source = "📝 Gemma didn't produce a usable draft this time, so this is the template."
        else:
            source = f"📝 Built-in template. {html.escape(_gemma_hint(self.llm))}"
        context = (
            f"For <b>{html.escape(str(event['title']))}</b> · {plan.pizza_boxes} boxes · "
            f"{plan.seats} seats · release {plan.waitlist_release} from the waitlist"
        )
        return result.subject, result.body, f'<div class="ip-source">{context}<br>{source}</div>'

    # --- 🧑‍🌾 Ask Juan

    def ask_view(self, message, history, model_kind, slices, box, price, food_q, risk):
        """One question to Juan; his tool calls show as a collapsible step list."""
        message = (message or "").strip()
        if not message:
            return history, ""
        # Juan plans with the main tab's knobs; capacity stays per event (None = the event's own).
        inputs = replace(self._inputs(None, slices, box, price, food_q, risk), capacity=None)
        reply = self._agent(model_kind, inputs).ask(message)
        history = [*history, {"role": "user", "content": message}]
        if reply.steps:
            n = len(reply.steps)
            history.append(
                {
                    "role": "assistant",
                    "content": _steps_markdown(reply.steps),
                    "metadata": {
                        "title": f"🔧 Juan used {n} tool{'s' * (n != 1)}",
                        "status": "done",
                    },
                }
            )
        source = f"Gemma ({self.llm.model}), local" if reply.used_llm else "rule-based, no LLM"
        history.append({"role": "assistant", "content": f"{reply.answer}\n\n*— {source}*"})
        return history, ""

    def _agent(self, model_kind: str, inputs: PlanInputs) -> AskJuan:
        """One Juan per model, so his forecast cache survives between questions."""
        agent = self._agents.get(model_kind)
        if agent is None:
            agent = AskJuan(
                self.workspace.require(),
                self.llm,  # Juan checks llm.available() on every question
                model_kind=model_kind,
                weather=self.workspace.weather,
            )
            self._agents[model_kind] = agent
        agent.plan_inputs = inputs
        agent.backtest = self.workspace.latest_backtest
        return agent

    # --- 📈 backtest

    def backtest_view(self, model_kind, last_n, progress=PROGRESS):
        """Replay past events and compare the model with the rule of thumb."""

        def report(done: int, total: int) -> None:
            progress((done, total), desc="Replaying past events", unit="events")

        result = self.workspace.backtest(model_kind, int(last_n), report)
        return (
            _backtest_headline_html(result, weather=not self.offline_weather),
            backtest_chart(result),
            _metrics_frame(result),
            _per_event_frame(result),
        )

    # --- 🗂️ data

    def reload_view(self, data_dir: str):
        """Load another data folder; caches start fresh."""
        folder = Path((data_dir or "").strip() or config.DATA_DIR).expanduser()
        try:
            self.workspace = Workspace(folder, offline_weather=self.offline_weather)
        except (ValueError, KeyError, FileNotFoundError) as e:
            raise gr.Error(f"Could not load {folder}: {e}") from e
        self._agents.clear()
        if self.ds is None:
            gr.Warning(f"No events.csv in {folder}.")
        return self.page_view()

    def demo_view(self):
        """Write fresh synthetic data to data/demo and load it."""
        out = config.DATA_DIR / "demo"
        try:
            write_dataset(make_synthetic_dataset(), out)  # replaces an earlier demo only
        except FileExistsError as e:
            raise gr.Error(str(e)) from e
        gr.Info(f"Wrote synthetic demo data to {out}.")
        return self.reload_view(str(out))

    def page_view(self):
        """Everything that depends on the loaded dataset, for page loads and reloads."""
        choices = self.event_choices()
        event_id = self.default_event()
        synthetic = bool(self.ds is not None and self.ds.synthetic)
        return (
            gr.HTML(visible=synthetic),
            _header_html(self),
            gr.Dropdown(choices=choices, value=event_id),
            self.capacity_of(event_id),
            str(self.workspace.data_dir),
            _dataset_html(self.workspace),
            *_idle_results("Pick an event and press <b>Forecast</b>."),
        )


# --- charts --------------------------------------------------------------------------------


def _new_figure(height: float) -> tuple[Figure, Axes]:
    """Transparent figure with recessive axes, readable on light and dark themes."""
    fig = Figure(figsize=(8.0, height), dpi=120, layout="constrained")
    fig.patch.set_alpha(0)
    ax = fig.add_subplot()
    ax.set_facecolor("none")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=9, length=0, pad=6)
    ax.grid(axis="y", color=INK, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def headcount_chart(fc: TurnoutForecast, capacity: int | None) -> Figure:
    """Bars of P(exactly k people show up): the 80% range in blue, the most likely in yellow."""
    fig, ax = _new_figure(3.3)
    pmf = fc.pmf
    lo = max(0, quantile(pmf, 0.002) - 2)
    hi = quantile(pmf, 0.998) + 2
    # Widen the view to the capacity line when it is close; a far-away line would squash the
    # bars, so the legend points to it instead.
    if capacity is not None and capacity <= hi + max(10, 0.3 * (hi - lo)):
        hi = max(hi, capacity + 2)
    heads = np.arange(lo, hi + 1)
    probs = np.pad(pmf, (0, max(0, hi + 1 - len(pmf))))[lo : hi + 1]
    colors = np.where((heads >= fc.p10) & (heads <= fc.p90), PY_BLUE, BLUE_SOFT).astype(object)
    colors[heads == fc.p50] = PY_YELLOW
    ax.bar(heads, probs, width=0.8, color=list(colors), linewidth=0)

    ticks = sorted({fc.p10, fc.p50, fc.p90})
    legend = [
        Patch(color=PY_BLUE, label=f"80% likely: {fc.p10}–{fc.p90}"),
        Patch(color=PY_YELLOW, label=f"most likely: {fc.p50}"),
    ]
    if capacity is not None and capacity <= hi:
        ax.axvline(capacity, color=INK, linewidth=1.3, linestyle=(0, (4, 3)))
        ticks = sorted({*ticks, capacity})
        legend.append(
            Line2D(
                [],
                [],
                color=INK,
                linewidth=1.3,
                linestyle=(0, (4, 3)),
                label=f"capacity: {capacity}",
            )
        )
    elif capacity is not None:
        legend.append(Line2D([], [], linestyle="none", label=f"capacity: {capacity} →"))
    ax.set_xticks(ticks)
    ax.set_xlim(lo - 0.8, hi + 0.8)
    ax.set_ylim(0, probs.max() * 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.set_xlabel("people who show up", color=INK, fontsize=9)
    ax.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0, 1),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        ncols=len(legend),
        handlelength=1.2,
        borderaxespad=0.2,
    )
    return fig


def backtest_chart(result: BacktestResult) -> Figure:
    """Per past event: actual (ring), model guess (dot) with 80% interval, rule of thumb (dash)."""
    fig, ax = _new_figure(3.6)
    df = result.per_event.sort_values("start").reset_index(drop=True)
    x = np.arange(len(df))
    label = cli.model_label(cli.main_method(result))
    ax.vlines(
        x,
        df["model_p10"],
        df["model_p90"],
        color=PY_BLUE,
        alpha=0.28,
        linewidth=7,
        capstyle="round",
        label=f"{label} 80% interval",
    )
    ax.scatter(x, df["model_expected"], color=PY_BLUE, s=26, zorder=3, label=f"{label} guess")
    ax.scatter(
        x,
        df["rule_pred"],
        color=RULE_ORANGE,
        marker="_",
        s=150,
        linewidths=2.4,
        zorder=3,
        label=cli.model_label("rule"),
    )
    ax.scatter(
        x,
        df["actual_heads"],
        facecolors="none",
        edgecolors=INK,
        s=72,
        linewidths=1.6,
        zorder=4,
        label="actual",
    )
    step = max(1, int(np.ceil(len(df) / 12)))
    ax.set_xticks(x[::step])
    ax.set_xticklabels([f"{ts:%b} '{ts:%y}" for ts in df["start"].iloc[::step]])
    ax.set_xlim(-0.7, len(df) - 0.3)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.set_ylabel("people", color=INK, fontsize=9)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1),
        frameon=False,
        fontsize=9,
        labelcolor=INK,
        ncols=4,
        handlelength=1.4,
        borderaxespad=0.2,
    )
    return fig


# --- HTML fragments ------------------------------------------------------------------------


def _e(value) -> str:
    return html.escape(str(value))


def _short_date(ts: pd.Timestamp) -> str:
    return f"{ts:%a} {ts.day} {ts:%b %Y}"


def _headline_html(event: pd.Series, fc: TurnoutForecast) -> str:
    s = fc.summary()
    if fc.rsvp_heads == 0:
        big, sub = "No RSVPs yet", "Nothing to forecast for this event."
    elif fc.p10 == fc.p90:
        big = f'Expect <span class="ip-num">about {fc.p50}</span> people'
        sub = f"out of <b>{fc.rsvp_heads}</b> RSVPs"
    else:
        big = f'Expect <span class="ip-num">{fc.p10}–{fc.p90}</span> people'
        sub = f"(most likely <b>{fc.p50}</b>) out of <b>{fc.rsvp_heads}</b> RSVPs"
    rule = round(config.RULE_OF_THUMB_SHOW_RATE * fc.rsvp_heads)
    chips = [
        cli.model_label(fc.model_name),
        f"predicted show-up {s['predicted_show_rate']:.0%}",
    ]
    if np.isfinite(s["historical_show_rate"]):
        chips.append(f"past events {s['historical_show_rate']:.0%}")
    chips.append(f"rule of thumb says {rule}")
    if len(fc.waitlist_probs):
        chips.append(f"{len(fc.waitlist_probs)} on the waitlist")
    chip_html = "".join(f'<span class="ip-chip">{_e(c)}</span>' for c in chips)
    if fc.synthetic:
        chip_html += '<span class="ip-chip ip-chip-demo">synthetic data</span>'
    return (
        f'<div class="ip-hero"><div class="ip-kicker">{_e(event["title"])} · '
        f"{_e(cli.when_where(event))}</div>"
        f'<div class="ip-hero-line">{big}</div><div class="ip-hero-sub">{sub}</div>'
        f'<div class="ip-chips">{chip_html}</div></div>'
    )


def _card(label: str, value: str, note: str, accent: str = "") -> str:
    return (
        f'<div class="ip-card {accent}"><div class="ip-card-label">{label}</div>'
        f'<div class="ip-card-value">{value}</div><div class="ip-card-note">{note}</div></div>'
    )


def _cards_html(plan: EventPlan, fc: TurnoutForecast, inputs: PlanInputs) -> str:
    fewer = plan.pizza_boxes_naive - plan.pizza_boxes
    cards = [
        _card(
            "🍕 Pizza to order",
            f"{plan.pizza_boxes} <small>boxes</small>",
            f"vs {plan.pizza_boxes_naive} if you fed every RSVP · enough for "
            f"{plan.food_heads} people",
        )
    ]
    if inputs.price_per_box > 0:
        cards.append(
            _card(
                "💸 Saved",
                f"₱{plan.savings_php:,.0f}",
                f"₱{plan.cost_php:,.0f} instead of ₱{plan.cost_naive_php:,.0f}",
                "ip-yellow",
            )
        )
    else:
        cards.append(
            _card(
                "📦 Boxes saved",
                f"{fewer} <small>boxes</small>",
                "Add a price per box to see the pesos.",
                "ip-yellow",
            )
        )
    waiting = len(fc.waitlist_probs)
    cards.append(
        _card(
            "🎟️ Waitlist release",
            f"{plan.waitlist_release} <small>of {waiting}</small>",
            _e(cli.waitlist_advice(plan, fc, inputs.capacity)),
        )
    )
    certs = (
        "chairs, name tags and e-certificates"
        if plan.seats == plan.certificates
        else f"chairs and name tags · {plan.certificates} e-certificates"
    )
    cards.append(_card("🪑 Seats", f"{plan.seats}", certs))
    return f'<div class="ip-cards">{"".join(cards)}</div>'


def _drivers_html(fc: TurnoutForecast, *, weather: bool) -> str:
    items = "".join(f"<li>{_e(d)}</li>" for d in fc.drivers) or "<li>No standout drivers.</li>"
    credit = f'<div class="ip-source">{_WEATHER_LINK}</div>' if weather else ""
    return (
        '<div class="ip-panel"><div class="ip-panel-title">Why this number</div>'
        f"<ul>{items}</ul>{credit}</div>"
    )


def _credits_html(model_name: str, *, weather: bool) -> str:
    """The attributions owed for a number: TabPFN's license line and Open-Meteo's link."""
    credits = cli.credits(model_name, weather=weather)
    parts = [_WEATHER_LINK if c == cli.WEATHER_CREDIT else _e(c) for c in credits]
    return f'<div class="ip-credits">{" · ".join(parts)}</div>' if parts else ""


def _explain_html(explanation: Explanation, model: str) -> str:
    if explanation.by_llm:
        source = f"✍️ Written by Gemma ({_e(model)}) on this machine"
    else:
        source = "📝 Built-in template · start Ollama with Gemma for a written explanation"
    return (
        f'<div class="ip-panel"><div class="ip-panel-title">In plain words</div>'
        f'<p>{_e(explanation.text)}</p><div class="ip-source">{source}</div></div>'
    )


def _explain_html_pending(message: str) -> str:
    return (
        f'<div class="ip-panel"><div class="ip-panel-title">In plain words</div>'
        f'<p class="ip-muted">{_e(message)}</p></div>'
    )


def _idle_results(message: str) -> tuple:
    return (f'<div class="ip-idle">{message}</div>', None, "", "", "")


def _gemma_hint(llm: ChatModel) -> str:
    status = getattr(llm, "status", None)
    return status()[1] if callable(status) else "Gemma is not available."


def _header_html(state: AppState) -> str:
    """Title, tagline and what is running right now (refreshed on every page load)."""
    tabpfn = "TabPFN ready" if state.tabpfn_ready else "TabPFN unavailable, using logistic"
    gemma = (
        f"Gemma ready ({state.llm.model})" if state.llm.available() else "Gemma offline: templates"
    )
    status = " · ".join(_e(item) for item in ("🔒 Runs on this machine", tabpfn, gemma))
    return (
        '<div id="ip-header"><div class="ip-logo">🍕</div><div>'
        "<h1>Ilang <span>pizza?</span></h1>"
        "<p>How many RSVPs will actually show up? TabPFN predicts it, plain Python sizes the "
        "food and waitlist, and a local Gemma drafts the emails.</p>"
        f'<div class="ip-status">{status}</div></div></div><div class="ip-stripe"></div>'
    )


def _dataset_html(ws: Workspace) -> str:
    if ws.ds is None:
        return (
            f'<div class="ip-idle">No events.csv in <code>{_e(ws.data_dir)}</code>. Press '
            "<b>Generate demo data</b> to try the app, or point the folder at your exports.</div>"
        )
    stats = cli.dataset_stats(ws.ds)
    rate = stats["show_rate"]
    tiles = [
        ("Events", f"{stats['events']}", f"{stats['upcoming_events']} without check-ins yet"),
        ("RSVPs", f"{stats['rsvps']:,}", "going and waitlisted"),
        ("Labeled events", f"{stats['labeled_events']}", "with check-in data to learn from"),
        (
            "Show-up rate",
            f"{rate:.0%}" if np.isfinite(rate) else "—",
            "of going RSVPs, past events",
        ),
    ]
    series = ws.ds.events["series"].value_counts()
    mix = " · ".join(f"{_e(name)} {count}" for name, count in series.items())
    span = f"{_short_date(stats['first_event'])} to {_short_date(stats['last_event'])}"
    kind = "synthetic demo data" if ws.ds.synthetic else "your exports"
    return (
        '<div class="ip-cards">'
        + "".join(_card(label, value, note) for label, value, note in tiles)
        + f'</div><div class="ip-source">{_e(ws.data_dir)} · {kind} · {span} · {mix}</div>'
    )


def _backtest_headline_html(result: BacktestResult, *, weather: bool) -> str:
    demo = '<span class="ip-chip ip-chip-demo">synthetic data</span>' if result.synthetic else ""
    return (
        f'<div class="ip-hero"><div class="ip-kicker">Backtest · {len(result.per_event)} past '
        f"events, each predicted only from earlier ones</div>"
        f'<div class="ip-hero-sub ip-strong">{_e(cli.backtest_headline(result))}</div>'
        f'<div class="ip-chips">{demo}</div>'
        f"{_credits_html(result.model_name, weather=weather)}</div>"
    )


def _metrics_frame(result: BacktestResult) -> pd.DataFrame:
    return pd.DataFrame(cli.metrics_rows(result))


def _per_event_frame(result: BacktestResult) -> pd.DataFrame:
    df = result.per_event.sort_values("start", ascending=False)
    return pd.DataFrame(
        {
            "Event": df["title"],
            "Date": [f"{ts:%b %d, %Y}" for ts in df["start"]],
            "RSVPs": df["rsvp_heads"].round().astype(int),
            "Actual": df["actual_heads"].round().astype(int),
            "Predicted": df["model_expected"].round().astype(int),
            "80% range": [f"{lo:.0f}–{hi:.0f}" for lo, hi in zip(df["model_p10"], df["model_p90"])],
            "Rule of thumb": df["rule_pred"].round().astype(int),
        }
    )


def _example_text(evt: gr.SelectData) -> str:
    return evt.value["text"]


def _steps_markdown(steps: list[dict]) -> str:
    lines = []
    for i, step in enumerate(steps, 1):
        args = ", ".join(f"{k}={v!r}" for k, v in (step.get("args") or {}).items())
        result = cli.step_summary(step.get("result"), limit=400)
        lines.append(f"{i}. `{step.get('tool', '')}({args})`\n\n   {result}")
    return "\n".join(lines)


# --- layout --------------------------------------------------------------------------------

_SYSTEM_FONT = ("ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif")
_MONO_FONT = ("ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace")

# Shades 50..950 around Python blue (#3776ab) and Python yellow (#ffd43b).
_BLUE = gr.themes.Color(
    name="python-blue",
    c50="#eef4fa",
    c100="#d9e7f4",
    c200="#b3cfe8",
    c300="#86b1d8",
    c400="#5a92c5",
    c500="#3776ab",
    c600="#2f6595",
    c700="#28557c",
    c800="#214564",
    c900="#1a3750",
    c950="#102338",
)
_YELLOW = gr.themes.Color(
    name="python-yellow",
    c50="#fffbea",
    c100="#fff4c5",
    c200="#ffeb95",
    c300="#ffe066",
    c400="#ffd43b",
    c500="#f2c21b",
    c600="#d1a10c",
    c700="#a77d09",
    c800="#7d5d08",
    c900="#5b4407",
    c950="#3a2b04",
)

THEME = gr.themes.Soft(
    primary_hue=_BLUE,
    secondary_hue=_YELLOW,
    neutral_hue="slate",
    font=_SYSTEM_FONT,
    font_mono=_MONO_FONT,
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
    slider_color="*primary_500",
)

CSS = """
:root { --ip-blue: #3776ab; --ip-yellow: #ffd43b; --ip-ink: #2f6595;
        --ip-tint: rgba(55, 118, 171, 0.07); }
.dark { --ip-ink: #8fbde6; --ip-tint: rgba(55, 118, 171, 0.18); }
#ip-header { display: flex; align-items: center; gap: 14px; padding: 4px 2px 0; }
#ip-header .ip-logo { font-size: 2.7rem; line-height: 1; }
#ip-header h1 { margin: 0; font-size: clamp(1.7rem, 3.6vw, 2.3rem); font-weight: 800;
                letter-spacing: -0.02em; }
#ip-header h1 span { color: var(--ip-ink); }
#ip-header p { margin: 2px 0 0; color: var(--body-text-color-subdued); }
.ip-stripe { height: 4px; border-radius: 4px; margin: 10px 0 2px;
             background: linear-gradient(90deg, var(--ip-blue) 0 62%, var(--ip-yellow) 62% 100%); }
.ip-status { font-size: .85rem; margin-top: 4px; color: var(--body-text-color-subdued); }
.ip-banner { background: var(--ip-yellow); color: #1f2328; border-radius: 12px;
             padding: 10px 14px; font-size: .95rem; }
.ip-banner b { letter-spacing: .05em; color: inherit !important; }  /* dark themes recolor b */
.ip-hero { padding: 4px 2px 2px; }
.ip-kicker { text-transform: uppercase; letter-spacing: .06em; font-size: .76rem;
             font-weight: 600; color: var(--body-text-color-subdued); }
.ip-hero-line { font-size: clamp(2rem, 5vw, 3.1rem); font-weight: 800; line-height: 1.08;
                letter-spacing: -0.02em; margin: 6px 0 2px; }
.ip-num { color: var(--ip-ink); }
.ip-hero-sub { font-size: 1.15rem; color: var(--body-text-color-subdued); }
.ip-hero-sub b, .ip-strong { color: var(--body-text-color); font-weight: 700; }
.ip-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.ip-chip { font-size: .8rem; padding: 3px 10px; border-radius: 999px; background: var(--ip-tint);
           border: 1px solid var(--border-color-primary); }
.ip-chip-demo { background: var(--ip-yellow); color: #1f2328; border-color: transparent; }
.ip-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.ip-card { position: relative; overflow: hidden; border: 1px solid var(--border-color-primary);
           border-radius: 14px; padding: 16px 16px 12px; background: var(--block-background-fill); }
.ip-card::before { content: ""; position: absolute; inset: 0 0 auto 0; height: 4px;
                   background: var(--ip-blue); }
.ip-card.ip-yellow::before { background: var(--ip-yellow); }
.ip-card-label { font-size: .85rem; font-weight: 600; color: var(--body-text-color-subdued); }
.ip-card-value { font-size: 2rem; font-weight: 800; line-height: 1.15; margin: 4px 0 2px; }
.ip-card-value small { font-size: 1rem; font-weight: 600; color: var(--body-text-color-subdued); }
.ip-card-note { font-size: .85rem; color: var(--body-text-color-subdued); }
.ip-panel { border: 1px solid var(--border-color-primary); border-radius: 14px; height: 100%;
            padding: 12px 16px; background: var(--block-background-fill); }
.ip-panel-title { font-weight: 700; margin-bottom: 6px; }
.ip-panel ul { margin: 0; padding-left: 18px; }
.ip-panel p { margin: 0 0 8px; line-height: 1.5; }
.ip-muted, .ip-idle, .ip-source { color: var(--body-text-color-subdued); }
.ip-idle { padding: 18px 4px; font-size: 1.05rem; }
.ip-source { font-size: .85rem; margin-top: 6px; }
.ip-footer { text-align: center; font-size: .85rem; color: var(--body-text-color-subdued); }
.ip-credits { font-size: .8rem; color: var(--body-text-color-subdued); margin-top: 8px; }
.ip-credits a, .ip-source a, .ip-footer a { color: inherit; text-decoration: underline; }
@media (max-width: 640px) {
  .ip-card-value { font-size: 1.6rem; }
  #ip-header .ip-logo { font-size: 2.1rem; }
}
"""

# An emoji favicon, so the browser tab reads as the app and not as "Gradio".
HEAD = (
    '<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 '
    'viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🍕</text></svg>">'
)

_BANNER = (
    '<div class="ip-banner">🧪 <b>SYNTHETIC DEMO DATA</b> · these events and RSVPs are '
    "simulated, so every number here is made up. Load your own exports in the 🗂️ Data tab.</div>"
)
_PRIVACY = """
### 🔒 Your attendees stay on this laptop
- **Everything runs locally.** TabPFN runs on this machine and Gemma runs through Ollama on
  127.0.0.1. Gradio telemetry is off, and the app is not shared unless you start it with `--share`.
- **The LLM sees aggregates only:** headcounts, rates, the plan, and the event's title, date and
  venue. Never a name, email or attendee id; every prompt is checked before it is sent.
- **Names and emails are hashed at load time** with a salt that stays in your data folder, and
  the raw columns are dropped right away.
- Weather lookups send only the venue's coordinates and the date to Open-Meteo.
"""
_DATA_GUIDE = f"""
### 📄 Bring your own data
Export the RSVP list from Meetup, Luma, ti.to or Google Forms, add one row per event to
`events.csv`, and see [docs/data-guide.md]({DATA_GUIDE_URL}) for the layout. Check-in lists
are optional per event; the more past events with check-ins, the better the forecasts.
"""


def _footer_html(*, weather: bool) -> str:
    weather_credit = f" · {_WEATHER_LINK}" if weather else ""
    return (
        '<div class="ip-footer">Made for Python Philippines · open source (MIT) · '
        f"{_e(TABPFN_ATTRIBUTION)} · Gemma 4 by Google DeepMind, served by Ollama"
        f"{weather_credit}</div>"
    )


def build_app(
    data_dir: Path | None = None,
    *,
    offline_weather: bool = False,
    read_only: bool = False,
    state: AppState | None = None,
) -> gr.Blocks:
    """Lay out the app. ``read_only`` hides the data-folder controls (used when sharing)."""
    state = state or AppState(data_dir, offline_weather=offline_weather)
    model_events = {"concurrency_id": "model", "concurrency_limit": 1}
    choices, event_id = state.event_choices(), state.default_event()
    synthetic = bool(state.ds is not None and state.ds.synthetic)

    with gr.Blocks(title="Ilang pizza?", analytics_enabled=False) as app:
        header = gr.HTML(_header_html(state))
        banner = gr.HTML(_BANNER, visible=synthetic)

        with gr.Tabs():
            with gr.Tab("🍕 Ilang pizza?"):
                with gr.Row(equal_height=True):
                    event = gr.Dropdown(
                        choices, value=event_id, label="Event", scale=5, min_width=280
                    )
                    model = gr.Radio(
                        MODEL_CHOICES, value=state.default_model, label="Model", scale=2
                    )
                    forecast_btn = gr.Button(
                        "🍕 Forecast", variant="primary", size="lg", scale=1, min_width=150
                    )
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2, min_width=280):
                        gr.Markdown("**Plan settings** · change them any time; nothing retrains.")
                        capacity = gr.Number(
                            state.capacity_of(event_id),
                            label="Capacity (seats)",
                            precision=0,
                            minimum=1,
                            info="Blank means no limit.",
                        )
                        slices = gr.Slider(1, 6, value=3, step=0.5, label="Slices per person")
                        box = gr.Slider(4, 12, value=8, step=1, label="Slices per box")
                        price = gr.Number(
                            0, label="Price per box (₱)", minimum=0, info="0 hides the pesos."
                        )
                        food_q = gr.Slider(
                            0.5,
                            0.95,
                            value=0.75,
                            step=0.05,
                            label="Food quantile",
                            info="Order enough food for this share of likely turnouts.",
                        )
                        risk = gr.Slider(
                            0.0,
                            0.3,
                            value=0.10,
                            step=0.01,
                            label="Overflow risk",
                            info="Max chance of going over capacity after releasing the waitlist.",
                        )
                        gr.HTML(_credits_html("tabpfn", weather=not state.offline_weather))
                    with gr.Column(scale=5, min_width=340):
                        headline = gr.HTML(
                            '<div class="ip-idle">Pick an event and press <b>Forecast</b>.</div>'
                        )
                        chart = gr.Plot(label="Headcount distribution", show_label=False)
                        cards = gr.HTML()
                        with gr.Row(equal_height=True):
                            drivers = gr.HTML()
                            with gr.Column():
                                with gr.Row(equal_height=True):
                                    explain_tone = gr.Radio(
                                        TONE_CHOICES,
                                        value="english",
                                        show_label=False,
                                        container=False,
                                        scale=3,
                                    )
                                    explain_btn = gr.Button("✨ Explain", size="sm", scale=1)
                                explanation = gr.HTML()

            with gr.Tab("✉️ Drafts"):
                gr.Markdown(
                    "Emails for the event picked in **🍕 Ilang pizza?**, using the same plan. "
                    "Edit them here, then copy."
                )
                with gr.Row():
                    kind = gr.Dropdown(
                        [(label, k) for k, label in cli.EMAIL_LABELS.items()],
                        value="reconfirm",
                        label="Email",
                        scale=3,
                    )
                    tone = gr.Radio(TONE_CHOICES, value="taglish", label="Tone", scale=2)
                    draft_btn = gr.Button("✍️ Generate", variant="primary", scale=1)
                draft_source = gr.HTML()
                subject = gr.Textbox(label="Subject", buttons=["copy"], interactive=True)
                body = gr.Textbox(
                    label="Body", lines=14, max_lines=40, buttons=["copy"], interactive=True
                )

            with gr.Tab("🧑‍🌾 Ask Juan"):
                gr.Markdown(
                    "Juan answers with the same forecasting and planning tools. Gemma picks the "
                    "tools; it only ever sees their aggregate results."
                )
                chat = gr.Chatbot(
                    label="Juan",
                    height=460,
                    buttons=["copy"],
                    placeholder="**Kumusta!** Ask about turnout, pizza or the waitlist.",
                    examples=[{"text": q} for q in ASK_EXAMPLES],
                )
                question = gr.Textbox(
                    placeholder="Ilang pizza ang i-order para sa next PyMeets?",
                    show_label=False,
                    submit_btn="Ask",
                )

            with gr.Tab("📈 Backtest"):
                gr.Markdown(
                    "Replays past events: each one is predicted only from the events before it, "
                    "the way you would have the night before."
                )
                with gr.Row():
                    bt_model = gr.Radio(MODEL_CHOICES, value=state.default_model, label="Model")
                    bt_last_n = gr.Slider(4, 40, value=12, step=1, label="Most recent events")
                    bt_btn = gr.Button("▶ Run backtest", variant="primary")
                bt_headline = gr.HTML()
                bt_chart = gr.Plot(label="Predicted vs actual", show_label=False)
                bt_metrics = gr.Dataframe(label="How far off, on average", interactive=False)
                with gr.Accordion("Per-event results", open=False):
                    bt_events = gr.Dataframe(interactive=False, wrap=True)

            with gr.Tab("🗂️ Data"):
                with gr.Row(equal_height=False):
                    data_box = gr.Textbox(
                        str(state.workspace.data_dir),
                        label="Data folder",
                        scale=4,
                        interactive=not read_only,
                    )
                    if not read_only:
                        reload_btn = gr.Button("🔄 Reload", scale=1)
                        demo_btn = gr.Button("🧪 Generate demo data", scale=1)
                summary = gr.HTML(_dataset_html(state.workspace))
                with gr.Row():
                    gr.Markdown(_PRIVACY)
                    gr.Markdown(_DATA_GUIDE)

        gr.HTML(_footer_html(weather=not state.offline_weather))

        # --- wiring: the main actions get readable API names; UI plumbing stays private
        private = {"api_visibility": "private"}
        plan_controls = [capacity, slices, box, price, food_q, risk]
        results = [headline, chart, cards, drivers, explanation]
        explain_inputs = [event, model, *plan_controls, explain_tone]
        page = [banner, header, event, capacity, data_box, summary, *results]

        forecast_btn.click(
            state.forecast_view,
            [event, model, *plan_controls],
            results,
            api_name="forecast",
            **model_events,
        ).success(state.explain_view, explain_inputs, explanation, **model_events, **private)
        explain_btn.click(
            state.explain_view, explain_inputs, explanation, api_name="explain", **model_events
        )
        event.input(
            state.event_view, [event, model, *plan_controls[1:]], [capacity, *results], **private
        )
        for control in (model, *plan_controls):
            control.input(
                state.replan_view,
                [event, model, *plan_controls],
                results,
                trigger_mode="always_last",
                show_progress="hidden",
                **private,
            )

        draft_btn.click(
            state.draft_view,
            [kind, tone, event, model, *plan_controls],
            [subject, body, draft_source],
            api_name="draft",
            **model_events,
        )

        ask_args = {
            "fn": state.ask_view,
            "inputs": [question, chat, model, *plan_controls[1:]],
            "outputs": [chat, question],
            **model_events,
        }
        question.submit(**ask_args, api_name="ask")
        chat.example_select(_example_text, None, question, **private).then(**ask_args, **private)

        bt_btn.click(
            state.backtest_view,
            [bt_model, bt_last_n],
            [bt_headline, bt_chart, bt_metrics, bt_events],
            api_name="backtest",
            **model_events,
        )

        if not read_only:
            reload_btn.click(state.reload_view, data_box, page, api_name="reload")
            data_box.submit(state.reload_view, data_box, page, **private)
            demo_btn.click(state.demo_view, None, page, api_name="demo_data")
        app.load(state.page_view, None, page, **private)
    return app


def main(
    data_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    *,
    offline_weather: bool = False,
) -> None:
    """Serve the app until Ctrl+C. Sharing switches the data-folder controls off."""
    app = build_app(data_dir, offline_weather=offline_weather, read_only=share)
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=THEME,
        css=CSS,
        head=HEAD,
        footer_links=["gradio", "settings"],
        show_error=True,
    )


if __name__ == "__main__":
    main()
