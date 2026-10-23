"""The ``ilang-pizza`` command line.

Heavy imports (pandas, torch, TabPFN, Gradio) happen inside the commands, so ``--help`` stays
instant. Everything printed is aggregate: counts, rates and ranges, never attendee ids.

The helpers in the first section are shared with the web UI (``ui.py``) so both speak the
same numbers in the same words.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NoReturn

import typer
from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ilang_pizza import config

if TYPE_CHECKING:
    import pandas as pd

    from ilang_pizza.features.weather import WeatherProvider
    from ilang_pizza.llm.client import LLM
    from ilang_pizza.schema import BacktestResult, Dataset, EventPlan, PlanInputs, TurnoutForecast

app = typer.Typer(
    name="ilang-pizza",
    help="🍕 [bold]Ilang pizza?[/] Forecast how many RSVPs will actually show up, then size the "
    "food, seats and waitlist. Runs on this machine; attendee data never leaves it.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,  # a crash must never dump attendee rows
)
console = Console()
err_console = Console(stderr=True)

ModelKind = Literal["tabpfn", "tabpfn-v2", "tabpfn-3.5", "logistic", "history", "rule"]
EmailKind = Literal["reconfirm", "waitlist_release", "catering_order", "venue_headcount"]
Tone = Literal["taglish", "english"]

MIN_PYTHON = (3, 11)
TABPFN_LICENSE_URL = "https://ux.priorlabs.ai"
DATA_GUIDE = "docs/data-guide.md"
WEATHER_CREDIT = "Weather data by Open-Meteo.com"  # Open-Meteo data is CC BY 4.0
WEATHER_CREDIT_URL = "https://open-meteo.com/"

MODEL_LABELS = {
    "tabpfn": "TabPFN",
    "tabpfn-v2": "TabPFN v2",
    "tabpfn-3.5": "TabPFN-3.5",
    "logistic": "Logistic regression",
    "history": "Attendee history",
}

EMAIL_LABELS = {
    "reconfirm": "Reconfirm attendance (to everyone going)",
    "waitlist_release": "You're in! (to released waitlist)",
    "catering_order": "Pizza order (to the caterer)",
    "venue_headcount": "Headcount (to the venue host)",
}


# --- shared helpers (also used by ui.py) ---------------------------------------------------


def resolve_data_dir(data: Path | None = None) -> Path:
    """``--data`` when given, else ``config.DATA_DIR``, or the only dataset folder inside it."""
    if data is not None:
        return data
    base = config.DATA_DIR
    if (base / "events.csv").exists():
        return base
    found = _datasets_under(base)
    return found[0] if len(found) == 1 else base


def _datasets_under(base: Path) -> list[Path]:
    return sorted(p.parent for p in base.glob("*/events.csv"))


def event_capacity(event: pd.Series) -> int | None:
    """The venue limit in heads, or None when unlimited or unknown."""
    capacity = event.get("capacity")
    if capacity is None or (isinstance(capacity, float) and math.isnan(capacity)):
        return None
    return int(capacity)


def when_where(event: pd.Series) -> str:
    """'Thu, Sep 17, 6:30 PM · BGC office' (or '· online')."""
    from ilang_pizza.llm.prompts import format_when

    place = _text(event.get("venue")) or _text(event.get("city")) or "venue TBA"
    if event.get("is_online"):
        place = "online"
    return f"{format_when(event['start'])} · {place}"


def _text(value) -> str:
    """A table cell as text: "" for missing values (CSV blanks load as NaN)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _finite(value) -> bool:
    return isinstance(value, int | float) and math.isfinite(value)


def model_label(name: str) -> str:
    """Human name for a model kind or backtest method."""
    if name == "rule":
        return f"Rule of thumb ({config.RULE_OF_THUMB_SHOW_RATE:.0%})"
    return MODEL_LABELS.get(name, name)


def forecast_headline(fc: TurnoutForecast) -> str:
    """'Expect 46–61 people (most likely 53) out of 98 RSVPs'."""
    if fc.rsvp_heads == 0:
        return "No RSVPs yet, so nothing to forecast"
    if fc.p10 == fc.p90:
        return f"Expect about {fc.p50} people out of {fc.rsvp_heads} RSVPs"
    return f"Expect {fc.p10}–{fc.p90} people (most likely {fc.p50}) out of {fc.rsvp_heads} RSVPs"


def waitlist_advice(plan: EventPlan, fc: TurnoutForecast, capacity: int | None) -> str:
    """One sentence on why the plan releases this many waitlisted RSVPs."""
    if len(fc.waitlist_probs) == 0:
        return "Nobody is on the waitlist."
    if capacity is None:
        return "No capacity limit, so everyone can come."
    risk = f"{chance(plan.overflow_probability)} chance of going over {capacity}"
    if plan.waitlist_release == 0:
        return f"Hold for now: already a {risk}."
    return f"{risk} after the release."


def credits(model_name: str, *, weather: bool) -> list[str]:
    """Attributions owed wherever a number is shown: TabPFN's weights license asks for its
    line on every interface built on it, and Open-Meteo's rain data is CC BY 4.0."""
    from ilang_pizza.model.backends import TABPFN_ATTRIBUTION

    owed = [TABPFN_ATTRIBUTION] if model_name.startswith("tabpfn") else []
    return owed + ([WEATHER_CREDIT] if weather else [])


def chance(p: float) -> str:
    """A probability for people: '7%', and '<1%' rather than a misleading '0%'."""
    if 0 < p < 0.005:
        return "<1%"
    if 0.995 < p < 1:
        return ">99%"
    return f"{p:.0%}"


def plan_inputs_for(
    event: pd.Series,
    *,
    capacity: int | None = None,
    slices_per_person: float = 3.0,
    slices_per_box: int = 8,
    price_per_box: float = 0.0,
    food_quantile: float = 0.75,
    overflow_risk: float = 0.10,
) -> PlanInputs:
    """``PlanInputs`` with the capacity defaulting to the event's own."""
    from ilang_pizza.schema import PlanInputs

    return PlanInputs(
        capacity=event_capacity(event) if capacity is None else capacity,
        slices_per_person=slices_per_person,
        slices_per_box=slices_per_box,
        price_per_box=price_per_box,
        food_quantile=food_quantile,
        overflow_risk=overflow_risk,
    )


def dataset_stats(ds: Dataset) -> dict:
    """Aggregate counts for summaries: events, RSVPs and the past show-up rate."""
    rs = ds.rsvps
    known = rs[(rs["status"] == "going") & rs["attended"].notna()]
    heads = 1 + known["guests"].fillna(0)
    shown = float((known["attended"] * heads).sum())
    return {
        "events": len(ds.events),
        "labeled_events": len(ds.labeled_event_ids()),
        "upcoming_events": len(ds.upcoming_event_ids()),
        "rsvps": len(rs),
        "show_rate": shown / float(heads.sum()) if len(known) else float("nan"),
        "first_event": ds.events["start"].min(),
        "last_event": ds.events["start"].max(),
    }


def main_method(result: BacktestResult) -> str:
    """The metrics row of the model that was backtested (the rest are baselines)."""
    index = list(result.metrics.index)
    for candidate in (result.model_name, result.model_name.split("-")[0]):
        if candidate in index:
            return candidate
    baselines = {"logistic", "history", "rule"}
    return next((m for m in index if m not in baselines), index[0])


def step_summary(result, limit: int = 200) -> str:
    """What an agent tool returned, as 'forecast.p50=40, plan.pizza_boxes=20, ...'.

    The event card and the plan inputs are skipped (they repeat the question), and so are
    lists such as drivers; nested values are flattened one level.
    """
    if not isinstance(result, dict):
        text = str(result)
    else:
        parts = []
        for key, value in _plain(result).items():
            if key in ("event", "inputs"):
                continue
            nested = value if isinstance(value, dict) else {"": value}
            for name, item in nested.items():
                if isinstance(item, bool | int | float | str) and item != "":
                    parts.append(f"{key}.{name}={item}" if name else f"{key}={item}")
        text = ", ".join(parts)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def metrics_rows(result: BacktestResult) -> list[dict[str, str]]:
    """The backtest metrics as display strings, one row per method."""
    rows = []
    for method, row in result.metrics.iterrows():
        coverage = row.get("coverage_80")
        rows.append(
            {
                "Method": model_label(method),
                "MAE (heads)": f"{row['mae']:.1f}",
                "MAPE": f"{row['mape']:.0%}",
                "Bias": f"{row['bias']:+.1f}",
                "80% coverage": f"{coverage:.0%}" if _finite(coverage) else "—",
                "Events": f"{row['n_events']:.0f}",
            }
        )
    return rows


def backtest_headline(result: BacktestResult) -> str:
    """'Rule of thumb (50%) was off by 23 heads per event on average; TabPFN by 7.'"""
    metrics = result.metrics
    method = main_method(result)
    model_mae = metrics.loc[method, "mae"]
    label = model_label(method)
    if "rule" not in metrics.index or method == "rule":
        return f"{label} was off by {model_mae:.0f} heads per event on average."
    return (
        f"{model_label('rule')} was off by {metrics.loc['rule', 'mae']:.0f} heads per event "
        f"on average; {label} by {model_mae:.0f}."
    )


# --- terminal rendering --------------------------------------------------------------------


def _fail(message: str, hint: str = "") -> NoReturn:
    """Print a friendly error (no traceback) and exit with status 1."""
    body = f"[bold]{escape(message)}[/]" + (f"\n{hint}" if hint else "")
    err_console.print(Panel(body, title="ilang-pizza", border_style="red", expand=False))
    raise typer.Exit(1)


def _banner(ds: Dataset, out: Console = console) -> None:
    if ds.synthetic:
        out.print(
            Panel(
                "[bold]SYNTHETIC DEMO DATA[/] · these events and RSVPs are simulated, so every "
                "number below is made up. Point --data at your own exports for real forecasts.",
                style="black on yellow",
                expand=True,
            )
        )


def _no_data_hint(data_dir: Path) -> str:
    found = _datasets_under(data_dir)
    if found:
        names = ", ".join(p.name for p in found)
        return f"Found several datasets in {escape(str(data_dir))}/: {names}. Pick one with --data."
    return (
        "Run [bold]ilang-pizza demo[/] to create synthetic data to try, or see "
        f"{DATA_GUIDE} to load your Meetup, Luma, ti.to or Google Forms exports."
    )


def _load(data: Path | None) -> Dataset:
    data_dir = resolve_data_dir(data)
    if not (data_dir / "events.csv").exists():
        _fail(f"No events.csv in {data_dir}/", _no_data_hint(data_dir))
    from ilang_pizza.data.loaders import load_dataset

    try:
        return load_dataset(data_dir)
    except (ValueError, KeyError, FileNotFoundError) as e:
        _fail(f"Could not load {data_dir}: {e}", f"Check the file layout in {DATA_GUIDE}.")


def _pick_event(ds: Dataset, event_id: str | None) -> pd.Series:
    upcoming = ", ".join(ds.upcoming_event_ids()[:5]) or "none"
    if event_id is None:
        event_id = ds.next_event_id()
        if event_id is None:
            _fail(
                "Every event already has check-ins, so there is nothing to forecast.",
                "Add the next event to events.csv with its RSVP export, or pass --event.",
            )
    try:
        return ds.event(event_id)
    except KeyError:
        _fail(f"Unknown event {event_id!r}.", f"Upcoming events: {escape(upcoming)}")


def _weather(no_weather: bool) -> WeatherProvider:
    from ilang_pizza.features.weather import default_provider

    return default_provider(offline=no_weather)


def _local_llm(no_llm: bool, out: Console = console) -> LLM | None:
    """The local Gemma when it is ready; None (templates) otherwise, with a hint why."""
    if no_llm:
        return None
    from ilang_pizza.llm.client import LLM

    llm = LLM()
    ready, message = llm.status()
    if not ready:
        out.print(f"[yellow]Using the built-in template.[/] {escape(message)}")
        return None
    return llm


def _forecast(
    data: Path | None, event_id: str | None, model: str, no_weather: bool, out: Console
) -> tuple[Dataset, pd.Series, TurnoutForecast, float]:
    """Load, pick the event and forecast it; returns the seconds the forecast took too."""
    ds = _load(data)
    _banner(ds, out)
    event = _pick_event(ds, event_id)
    # imported after the checks above, so a typo in --data or --event fails instantly
    from ilang_pizza.model.backends import TabPFNUnavailableError
    from ilang_pizza.model.predict import forecast_event

    started = time.perf_counter()
    with out.status(f"Forecasting {escape(str(event['title']))} with {model_label(model)}..."):
        try:
            fc = forecast_event(
                ds, event["event_id"], model_kind=model, weather=_weather(no_weather)
            )
        except TabPFNUnavailableError as e:
            _fail(str(e), "Try [bold]--model logistic[/] meanwhile.")
        except (KeyError, ValueError) as e:
            _fail(f"Could not forecast {event['event_id']}: {e}")
    return ds, event, fc, time.perf_counter() - started


def _footer(out: Console, model_name: str, *, weather: bool, timing: str = "") -> None:
    """A dim last line: what ran, how long it took, and the credits we owe."""
    parts = [f"{escape(model_label(model_name))} · {escape(timing)}"] if timing else []
    for credit in credits(model_name, weather=weather):
        link = WEATHER_CREDIT_URL if credit == WEATHER_CREDIT else ""
        parts.append(f"[link={link}]{escape(credit)}[/link]" if link else escape(credit))
    if parts:
        out.print(f"[dim]{' · '.join(parts)}[/]")


def _forecast_panel(event: pd.Series, fc: TurnoutForecast) -> Panel:
    s = fc.summary()
    facts = Table.grid(padding=(0, 2))
    facts.add_column(style="dim")
    facts.add_column()
    facts.add_row("Most likely", f"[bold]{fc.p50}[/] people")
    facts.add_row("80% range", f"{fc.p10}–{fc.p90}")
    facts.add_row("Expected", f"{fc.expected:.1f}")
    waiting = len(fc.waitlist_probs)  # RSVPs, as the waitlist decision counts them
    waitlist = f" · {waiting} waitlisted" if waiting else ""
    facts.add_row("RSVPs", f"{fc.rsvp_heads} going{waitlist}")
    history = s["historical_show_rate"]
    past = f" · {history:.0%} at past events" if math.isfinite(history) else ""
    facts.add_row("Show-up rate", f"{s['predicted_show_rate']:.0%} predicted{past}")
    facts.add_row("Model", model_label(fc.model_name))
    parts: list = [Text(forecast_headline(fc), style="bold"), Text(""), facts]
    if fc.drivers:
        parts += [Text(""), Text("Why", style="bold")]
        parts += [Text(f"• {driver}") for driver in fc.drivers]
    title = f"{escape(str(event['title']))} · {escape(when_where(event))}"
    return Panel(Group(*parts), title=title, title_align="left", border_style="blue")


def _plan_panel(plan: EventPlan, fc: TurnoutForecast, inputs: PlanInputs) -> Panel:
    rows = Table.grid(padding=(0, 2))
    rows.add_column()
    rows.add_column(style="bold")
    rows.add_column(style="dim")
    rows.add_row(
        "🍕 Pizza",
        f"{plan.pizza_boxes} boxes",
        f"vs {plan.pizza_boxes_naive} if you fed every RSVP · sized for {plan.food_heads} people",
    )
    if inputs.price_per_box > 0:
        rows.add_row(
            "💸 Saved",
            f"₱{plan.savings_php:,.0f}",
            f"₱{plan.cost_php:,.0f} instead of ₱{plan.cost_naive_php:,.0f}",
        )
    waiting = len(fc.waitlist_probs)
    rows.add_row(
        "🎟️ Waitlist",
        f"release {plan.waitlist_release} of {waiting}",
        waitlist_advice(plan, fc, inputs.capacity),
    )
    rows.add_row("🪑 Seats", f"{plan.seats}", "chairs and name tags")
    rows.add_row("📜 E-certificates", f"{plan.certificates}", "to prepare")
    return Panel(rows, title="Plan", title_align="left", border_style="yellow")


def _event_info(event: pd.Series) -> dict:
    return {
        "event_id": event["event_id"],
        "title": event["title"],
        "series": event["series"],
        "start": event["start"].isoformat(),
        "venue": event["venue"],
        "city": event["city"],
        "is_online": bool(event["is_online"]),
        "capacity": event_capacity(event),
    }


def _plain(value):
    """JSON-safe copy: numpy values to Python, NaN to null, timestamps to ISO strings."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "tolist"):  # numpy scalar or array
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _print_json(payload: dict) -> None:
    typer.echo(json.dumps(_plain(payload), indent=2, ensure_ascii=False))


# --- options shared by several commands ----------------------------------------------------

DataOpt = Annotated[
    Path | None,
    typer.Option(
        "--data",
        "-d",
        help="Dataset folder with events.csv and rsvps/. "
        "[dim]Default: $ILANG_PIZZA_DATA or ./data, or the one dataset inside it.[/]",
        show_default=False,
    ),
]
EventOpt = Annotated[
    str | None,
    typer.Option("--event", "-e", help="Event id. [dim]Default: the next upcoming event.[/]"),
]
ModelOpt = Annotated[
    ModelKind,
    typer.Option(
        "--model",
        "-m",
        help="Show-up model. tabpfn picks TabPFN-3.5 when TABPFN_TOKEN is set, else v2.",
    ),
]
NoWeatherOpt = Annotated[
    bool, typer.Option("--no-weather", help="Skip the Open-Meteo rain lookup (works offline).")
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON instead of tables.")]
NoLlmOpt = Annotated[bool, typer.Option("--no-llm", help="Use the built-in templates, not Gemma.")]
ToneOpt = Annotated[Tone, typer.Option("--tone", "-t", help="Language of the text.")]
CapacityOpt = Annotated[
    int | None,
    typer.Option("--capacity", min=1, help="Venue limit in heads. [dim]Default: the event's.[/]"),
]
SlicesOpt = Annotated[float, typer.Option("--slices", min=0.5, help="Pizza slices per person.")]
BoxOpt = Annotated[int, typer.Option("--box", min=1, help="Slices per pizza box.")]
PriceOpt = Annotated[
    float, typer.Option("--price", min=0, help="Price per box in PHP; 0 hides the cost lines.")
]
FoodQuantileOpt = Annotated[
    float,
    typer.Option(
        "--food-quantile", min=0.5, max=0.99, help="Order food for this quantile of heads."
    ),
]
RiskOpt = Annotated[
    float,
    typer.Option(
        "--risk", min=0.0, max=0.5, help="Max chance of going over capacity after a release."
    ),
]


# --- commands ------------------------------------------------------------------------------


@app.command()
def demo(
    out: Annotated[Path, typer.Option("--out", "-o", help="Folder to write it to.")] = Path(
        "data/demo"
    ),
    events: Annotated[
        int, typer.Option("--events", min=8, help="Past events to simulate (plus 2 upcoming).")
    ] = 60,
    seed: Annotated[int, typer.Option("--seed", help="Random seed, for repeatable data.")] = 2026,
) -> None:
    """Write a synthetic PythonPH-style dataset to try every command on."""
    from ilang_pizza.data.synth import make_synthetic_dataset, write_dataset

    with console.status("Simulating PythonPH events and RSVPs..."):
        ds = make_synthetic_dataset(n_past=events, seed=seed)
        try:
            write_dataset(ds, out)  # replaces an earlier demo, never other files
        except FileExistsError as e:
            _fail(str(e), "Pick another folder with --out, or delete that one first.")
    _banner(ds)
    stats = dataset_stats(ds)
    console.print(
        f"Wrote [bold]{stats['events']}[/] events ({stats['labeled_events']} with check-ins, "
        f"{stats['upcoming_events']} upcoming) and [bold]{stats['rsvps']:,}[/] RSVPs to "
        f"[bold]{escape(str(out))}[/]. Past show-up rate: {stats['show_rate']:.0%}."
    )
    flag = "" if resolve_data_dir(None).resolve() == out.resolve() else f" --data {out}"
    steps = [
        ("ilang-pizza doctor", "check TabPFN, Gemma and your data"),
        (f"ilang-pizza backtest{flag} --last-n 10", "how well it predicted past events"),
        (f"ilang-pizza forecast{flag}", "headcount for the next event"),
        (f"ilang-pizza plan{flag} --price 450", "pizza boxes, waitlist spots, seats"),
        (f"ilang-pizza draft{flag} --kind catering_order", "an email to the pizza place"),
        (f'ilang-pizza ask "Ilang pizza ang i-order?"{flag}', "ask Juan, the agent"),
        (f"ilang-pizza ui{flag}", "the web app"),
    ]
    table = Table.grid(padding=(0, 2))
    for command, what in steps:
        table.add_row(f"[bold cyan]{escape(command)}[/]", f"[dim]{what}[/]")
    console.print(Panel(table, title="Next steps", title_align="left", border_style="blue"))


@dataclass(frozen=True)
class _Check:
    level: Literal["ok", "warn", "fail"]
    name: str
    status: str
    fix: str = ""


_ICONS = {"ok": "[green]✔[/]", "warn": "[yellow]![/]", "fail": "[red]✘[/]"}


def _check_python() -> _Check:
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info >= MIN_PYTHON:
        return _Check("ok", "Python", version)
    return _Check("fail", "Python", version, "Install Python 3.11 or newer.")


def _check_tabpfn() -> _Check:
    upgrade = (
        f"TabPFN v2 works without a token. To upgrade to TabPFN-3.5, accept the license "
        f"yourself at {TABPFN_LICENSE_URL}, then `export TABPFN_TOKEN=...`."
    )
    try:
        from ilang_pizza.model.backends import tabpfn_status

        ready, message = tabpfn_status()
    except ImportError as e:
        return _Check(
            "fail", "TabPFN", f"not importable: {e}", "Reinstall with `pip install -e .`."
        )
    if not ready:
        return _Check("fail", "TabPFN", message, f"Use --model logistic meanwhile. {upgrade}")
    if os.getenv("TABPFN_TOKEN"):
        return _Check("ok", "TabPFN", message)
    return _Check("ok", "TabPFN", message, upgrade)


def _check_ollama() -> list[_Check]:
    import httpx
    import ollama

    from ilang_pizza.llm.client import PROBE_TIMEOUT_SECONDS, match_model

    model = config.LLM_MODEL
    try:
        client = ollama.Client(host=config.OLLAMA_HOST, timeout=PROBE_TIMEOUT_SECONDS)
        installed = [m.model or "" for m in client.list().models]
    except (ollama.ResponseError, ConnectionError, httpx.HTTPError):
        return [
            _Check(
                "warn",
                "Ollama",
                f"not reachable at {config.OLLAMA_HOST}",
                "Install it from https://ollama.com, then `ollama serve`. "
                "Templates write the emails meanwhile.",
            ),
            _Check("warn", "Gemma 4", f"{model}: unknown", f"`ollama pull {model}`"),
        ]
    ollama_ok = _Check("ok", "Ollama", f"running at {config.OLLAMA_HOST}")
    resolved = match_model(model, installed)
    if resolved:
        return [ollama_ok, _Check("ok", "Gemma 4", f"{resolved} pulled")]
    return [
        ollama_ok,
        _Check("warn", "Gemma 4", f"{model} not pulled", f"`ollama pull {model}` (~10 GB)"),
    ]


def _check_data(data: Path | None) -> _Check:
    data_dir = resolve_data_dir(data)
    if not (data_dir / "events.csv").exists():
        hint = Text.from_markup(_no_data_hint(data_dir)).plain
        return _Check("fail", "Data", f"no events.csv in {data_dir}/", hint)
    from ilang_pizza.data.loaders import load_dataset

    try:
        stats = dataset_stats(load_dataset(data_dir))
    except (ValueError, KeyError, FileNotFoundError) as e:
        return _Check("fail", "Data", f"{data_dir}: {e}", f"See {DATA_GUIDE}.")
    synthetic = " (synthetic demo)" if (data_dir / "SYNTHETIC").exists() else ""
    status = (
        f"{data_dir}{synthetic}: {stats['events']} events, {stats['labeled_events']} labeled, "
        f"{stats['upcoming_events']} upcoming, {stats['rsvps']:,} RSVPs"
    )
    if stats["labeled_events"] < 6:
        return _Check(
            "warn", "Data", status, "Forecasts get better with 6+ past events with check-ins."
        )
    return _Check("ok", "Data", status)


def _check_cache() -> _Check:
    cache = config.CACHE_DIR
    move = "Set ILANG_PIZZA_CACHE to move it; safe to delete any time."
    if cache.exists():
        files = [f for f in cache.rglob("*") if f.is_file()]
        size_mb = sum(f.stat().st_size for f in files) / 1e6
        if not os.access(cache, os.W_OK):
            return _Check("warn", "Cache", f"{cache} is read-only", move)
        return _Check("ok", "Cache", f"{cache} ({len(files)} files, {size_mb:.1f} MB)", move)
    return _Check("ok", "Cache", f"{cache} (created on the first weather lookup)", move)


@app.command()
def doctor(data: DataOpt = None) -> None:
    """Check Python, TabPFN, Ollama + Gemma, your data and the cache, with a fix for each."""
    with console.status("Checking..."):
        checks = [_check_python(), _check_tabpfn(), *_check_ollama(), _check_data(data)]
        checks.append(_check_cache())
    table = Table(show_header=True, header_style="bold", show_lines=False, expand=True)
    table.add_column("", width=1)
    table.add_column("Check", no_wrap=True)
    table.add_column("Status", ratio=3)
    table.add_column("Fix", ratio=3, style="dim")
    for check in checks:
        table.add_row(_ICONS[check.level], check.name, escape(check.status), escape(check.fix))
    console.print(Panel(table, title="ilang-pizza doctor", title_align="left", border_style="blue"))
    if any(check.level == "fail" for check in checks):
        raise typer.Exit(1)


@app.command()
def backtest(
    data: DataOpt = None,
    model: ModelOpt = "tabpfn",
    last_n: Annotated[
        int, typer.Option("--last-n", min=1, help="Score only the most recent N events.")
    ] = 20,
    min_train: Annotated[
        int, typer.Option("--min-train", min=2, help="Past events needed before the first guess.")
    ] = 6,
    no_weather: NoWeatherOpt = False,
    csv: Annotated[
        Path | None, typer.Option("--csv", help="Also write the per-event results to a CSV.")
    ] = None,
) -> None:
    """Replay past events, each predicted only from the events before it, and score the guesses."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeElapsedColumn

    from ilang_pizza.model.backends import TabPFNUnavailableError
    from ilang_pizza.model.backtest import backtest as run_backtest

    ds = _load(data)
    _banner(ds)
    started = time.perf_counter()
    columns = (
        "[progress.description]{task.description}",
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task(f"Backtesting {model_label(model)}", total=None)
        try:
            result = run_backtest(
                ds,
                model_kind=model,
                min_train_events=min_train,
                last_n=last_n,
                weather=_weather(no_weather),
                progress=lambda done, total: progress.update(task, completed=done, total=total),
            )
        except TabPFNUnavailableError as e:
            _fail(str(e), "Try [bold]--model logistic[/] meanwhile.")
        except ValueError as e:
            _fail(f"Could not backtest: {e}", "Try a smaller --min-train or more past events.")
    elapsed = time.perf_counter() - started
    console.print(_per_event_table(result, last_n))
    console.print(_metrics_table(result))
    console.print(Panel(Text(backtest_headline(result), style="bold"), border_style="green"))
    timing = f"{len(result.per_event)} events replayed in {elapsed:.1f}s"
    _footer(console, result.model_name, weather=not no_weather, timing=timing)
    if csv is not None:
        result.per_event.to_csv(csv, index=False)
        console.print(f"Wrote {escape(str(csv))}")


def _per_event_table(result: BacktestResult, last_n: int) -> Table:
    label = model_label(main_method(result))
    table = Table(title=f"Last {min(last_n, len(result.per_event))} events", title_justify="left")
    table.add_column("Event")
    table.add_column("Date", no_wrap=True)
    for name in ("RSVPs", "Actual", f"{label} (80% range)", "Rule", "Error"):
        table.add_column(name, justify="right", no_wrap=True)
    for row in result.per_event.sort_values("start").tail(last_n).itertuples():
        error = row.model_expected - row.actual_heads
        better = abs(error) <= abs(row.rule_pred - row.actual_heads)
        table.add_row(
            escape(str(row.title)),
            f"{row.start:%b %d, %Y}",
            f"{row.rsvp_heads:.0f}",
            f"{row.actual_heads:.0f}",
            f"{row.model_expected:.0f} ({row.model_p10:.0f}–{row.model_p90:.0f})",
            f"{row.rule_pred:.0f}",
            f"[{'green' if better else 'red'}]{error:+.0f}[/]",
        )
    return table


def _metrics_table(result: BacktestResult) -> Table:
    rows = metrics_rows(result)
    table = Table(title="How far off, on average", title_justify="left")
    for name in rows[0]:
        table.add_column(name, justify="left" if name == "Method" else "right")
    main = model_label(main_method(result))
    for row in rows:
        style = "bold" if row["Method"] == main else ""
        table.add_row(*(escape(cell) for cell in row.values()), style=style)
    return table


@app.command()
def forecast(
    data: DataOpt = None,
    event: EventOpt = None,
    model: ModelOpt = "tabpfn",
    no_weather: NoWeatherOpt = False,
    as_json: JsonOpt = False,
) -> None:
    """Predict the headcount for an event: a range, the most likely number, and why."""
    out = err_console if as_json else console
    _, ev, fc, seconds = _forecast(data, event, model, no_weather, out)
    if as_json:
        _print_json({"event": _event_info(ev), "forecast": fc.summary()})
    else:
        console.print(_forecast_panel(ev, fc))
    _footer(out, fc.model_name, weather=not no_weather, timing=f"forecast in {seconds:.1f}s")


@app.command()
def plan(
    data: DataOpt = None,
    event: EventOpt = None,
    capacity: CapacityOpt = None,
    slices: SlicesOpt = 3.0,
    box: BoxOpt = 8,
    price: PriceOpt = 0.0,
    food_quantile: FoodQuantileOpt = 0.75,
    risk: RiskOpt = 0.10,
    model: ModelOpt = "tabpfn",
    no_weather: NoWeatherOpt = False,
    as_json: JsonOpt = False,
) -> None:
    """Turn the forecast into decisions: pizza boxes, waitlist spots to release, seats."""
    from ilang_pizza.plan import make_plan

    out = err_console if as_json else console
    ds, ev, fc, seconds = _forecast(data, event, model, no_weather, out)
    inputs = plan_inputs_for(
        ev,
        capacity=capacity,
        slices_per_person=slices,
        slices_per_box=box,
        price_per_box=price,
        food_quantile=food_quantile,
        overflow_risk=risk,
    )
    result = make_plan(fc, inputs)
    if as_json:
        _print_json(
            {
                "event": _event_info(ev),
                "forecast": fc.summary(),
                "inputs": asdict(inputs),
                "plan": asdict(result),
                "synthetic_data": ds.synthetic,
            }
        )
    else:
        console.print(_forecast_panel(ev, fc))
        console.print(_plan_panel(result, fc, inputs))
    _footer(out, fc.model_name, weather=not no_weather, timing=f"forecast in {seconds:.1f}s")


@app.command()
def draft(
    kind: Annotated[EmailKind, typer.Option("--kind", "-k", help="Which email to write.")],
    data: DataOpt = None,
    event: EventOpt = None,
    tone: ToneOpt = "taglish",
    no_llm: NoLlmOpt = False,
    capacity: CapacityOpt = None,
    slices: SlicesOpt = 3.0,
    box: BoxOpt = 8,
    price: PriceOpt = 0.0,
    food_quantile: FoodQuantileOpt = 0.75,
    risk: RiskOpt = 0.10,
    model: ModelOpt = "tabpfn",
    no_weather: NoWeatherOpt = False,
) -> None:
    """Draft an organizer email from the forecast and plan. Gemma writes it when it runs."""
    from ilang_pizza.llm.prompts import draft_email
    from ilang_pizza.plan import make_plan

    _, ev, fc, seconds = _forecast(data, event, model, no_weather, console)
    inputs = plan_inputs_for(
        ev,
        capacity=capacity,
        slices_per_person=slices,
        slices_per_box=box,
        price_per_box=price,
        food_quantile=food_quantile,
        overflow_risk=risk,
    )
    llm = _local_llm(no_llm)
    writer = f"Gemma ({llm.model}) is writing..." if llm else "Filling in the template..."
    started = time.perf_counter()
    with console.status(writer):
        result = draft_email(kind, ev, fc, make_plan(fc, inputs), tone=tone, llm=llm)
    elapsed = time.perf_counter() - started
    if result.used_llm:
        source = f"written by Gemma ({result.model}) on this machine in {elapsed:.1f}s"
    elif llm is not None:
        source = "Gemma didn't produce a usable draft (see above), so this is the template"
    else:
        source = "built-in template"
    console.rule(f"✉️  {EMAIL_LABELS[kind]}")
    console.print(Text("Subject: ", style="bold") + Text(result.subject))
    console.print()
    console.print(Text(result.body), soft_wrap=True)
    console.rule(f"[dim]{source}[/]")
    _footer(console, fc.model_name, weather=not no_weather, timing=f"forecast in {seconds:.1f}s")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Your question, in Taglish or English.")],
    data: DataOpt = None,
    no_llm: NoLlmOpt = False,
    model: ModelOpt = "tabpfn",
    no_weather: NoWeatherOpt = False,
) -> None:
    """Ask Juan, the organizer's helper, about turnout, food or the waitlist."""
    from rich.markdown import Markdown

    from ilang_pizza.llm.agent import AskJuan

    ds = _load(data)
    _banner(ds)
    llm = _local_llm(no_llm)
    agent = AskJuan(ds, llm, model_kind=model, weather=_weather(no_weather))
    started = time.perf_counter()
    with console.status("Juan is thinking..."):
        reply = agent.ask(question)
    elapsed = time.perf_counter() - started
    source = f"Gemma ({llm.model})" if reply.used_llm and llm else "rule-based answer, no LLM"
    console.print(
        Panel(
            Markdown(reply.answer),
            title="🧑‍🌾 Juan",
            title_align="left",
            subtitle=f"{source} · {elapsed:.1f}s",
            subtitle_align="right",
            border_style="green",
        )
    )
    if reply.steps:
        console.print(_steps_table(reply.steps))
    _footer(console, model, weather=not no_weather)


def _steps_table(steps: list[dict]) -> Table:
    table = Table(title="Tools Juan used", title_justify="left", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Tool", no_wrap=True)
    table.add_column("Arguments")
    table.add_column("Result", style="dim", overflow="ellipsis")
    for i, step in enumerate(steps, 1):
        args = ", ".join(f"{k}={v}" for k, v in (step.get("args") or {}).items())
        result = step_summary(step.get("result"))
        table.add_row(str(i), escape(str(step.get("tool", ""))), escape(args), escape(result))
    return table


@app.command()
def ui(
    data: DataOpt = None,
    port: Annotated[int, typer.Option("--port", "-p", help="Port on 127.0.0.1.")] = 7860,
    share: Annotated[
        bool, typer.Option("--share", help="Also open a temporary public gradio.live link.")
    ] = False,
    no_weather: NoWeatherOpt = False,
) -> None:
    """Open the web app at http://127.0.0.1:7860 (only this machine can reach it)."""
    if share:
        err_console.print(
            Panel(
                "--share makes a public link: anyone who has it sees this dataset's aggregate "
                "numbers and can use Ask Juan. Attendee rows stay on this machine, and the data "
                "folder controls are switched off while sharing.",
                border_style="yellow",
            )
        )
    from ilang_pizza.ui import main as launch

    launch(data_dir=resolve_data_dir(data), port=port, share=share, offline_weather=no_weather)
