"""CLI tests: every command runs in-process on a tiny canonical dataset.

The forecasting commands use ``--model logistic --no-weather`` and the LLM commands
``--no-llm``, so nothing here needs network, model weights or Ollama.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import get_args

import pytest
from typer.testing import CliRunner

from ilang_pizza import cli, config
from ilang_pizza.model.backends import TABPFN_ATTRIBUTION
from ilang_pizza.schema import Dataset, PlanInputs

runner = CliRunner()
FAST = ["--model", "logistic", "--no-weather"]


def write_canonical(ds: Dataset, out: Path) -> Path:
    """The documented layout: events.csv plus one rsvps/<event_id>.csv per event."""
    (out / "rsvps").mkdir(parents=True)
    ds.events.to_csv(out / "events.csv", index=False)
    for event_id, rows in ds.rsvps.groupby("event_id"):
        rows.to_csv(out / "rsvps" / f"{event_id}.csv", index=False)
    if ds.synthetic:
        (out / "SYNTHETIC").touch()
    return out


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory, tiny_dataset) -> Path:
    return write_canonical(tiny_dataset, tmp_path_factory.mktemp("tiny"))


def run(*args: str):
    result = runner.invoke(cli.app, [str(a) for a in args])
    assert result.exit_code == 0, result.output
    return result


def assert_no_attendee_ids(text: str, ds: Dataset) -> None:
    leaked = [a for a in ds.rsvps["attendee_id"].unique() if a in text]
    assert not leaked, f"attendee ids printed: {leaked[:3]}"


# --- entry points --------------------------------------------------------------------------


def test_help_lists_every_command():
    output = run("--help").output
    for command in ("demo", "doctor", "backtest", "forecast", "plan", "draft", "ask", "ui"):
        assert command in output


def test_help_stays_light():
    code = (
        "import sys, ilang_pizza.cli; "
        "print(sorted(m for m in ('pandas', 'torch', 'tabpfn', 'gradio') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_python_dash_m():
    command = [sys.executable, "-m", "ilang_pizza", "--help"]
    out = subprocess.run(command, capture_output=True, text=True, check=True)
    assert "forecast" in out.stdout


# --- demo ----------------------------------------------------------------------------------


def test_demo_writes_a_dataset(tmp_path):
    out = tmp_path / "demo"
    result = run("demo", "--out", out, "--events", 10, "--seed", 1)
    assert (out / "events.csv").exists()
    assert (out / "SYNTHETIC").exists()
    assert "SYNTHETIC DEMO DATA" in result.output
    assert "Next steps" in result.output


def test_demo_never_overwrites_real_exports(tmp_path):
    (tmp_path / "events.csv").write_text("event_id\n")
    result = runner.invoke(cli.app, ["demo", "--out", str(tmp_path)])
    assert result.exit_code == 1
    assert "--out" in result.output
    assert (tmp_path / "events.csv").read_text() == "event_id\n"


# --- forecasting commands ------------------------------------------------------------------


def test_forecast_prints_the_headline(data_dir, tiny_dataset):
    result = run("forecast", "--data", data_dir, *FAST)
    assert "Expect" in result.output
    assert "out of" in result.output
    assert "SYNTHETIC DEMO DATA" in result.output
    assert TABPFN_ATTRIBUTION not in result.output  # logistic did the work
    assert_no_attendee_ids(result.output, tiny_dataset)


def test_tabpfn_forecasts_carry_the_attribution(data_dir, monkeypatch):
    from ilang_pizza.model import predict

    real = predict.forecast_event

    def tabpfn_named(ds, event_id, **kwargs):  # no weights in tests: only the name matters
        fc = real(ds, event_id, **{**kwargs, "model_kind": "logistic"})
        return replace(fc, model_name="tabpfn-v2")

    monkeypatch.setattr(predict, "forecast_event", tabpfn_named)
    result = run("forecast", "--data", data_dir, "--no-weather")
    assert result.output.count(TABPFN_ATTRIBUTION) == 1
    assert "Open-Meteo" not in result.output  # --no-weather: no rain data was used

    as_json = run("forecast", "--data", data_dir, "--no-weather", "--json")
    assert TABPFN_ATTRIBUTION in as_json.stderr
    json.loads(as_json.stdout)


def test_credits_follow_the_model_and_the_weather():
    assert cli.credits("tabpfn-v2", weather=True) == [TABPFN_ATTRIBUTION, cli.WEATHER_CREDIT]
    assert cli.credits("logistic", weather=True) == [cli.WEATHER_CREDIT]
    assert cli.credits("logistic", weather=False) == []


def test_forecast_json_is_clean_on_stdout(data_dir):
    result = run("forecast", "--data", data_dir, "--json", *FAST)
    payload = json.loads(result.stdout)
    assert payload["event"]["event_id"] == "pymeets-2026-03"
    forecast = payload["forecast"]
    assert forecast["p10"] <= forecast["p50"] <= forecast["p90"] <= forecast["rsvp_heads"]
    assert forecast["synthetic_data"] is True
    assert "SYNTHETIC DEMO DATA" in result.stderr


def test_forecast_unknown_event(data_dir):
    result = runner.invoke(cli.app, ["forecast", "--data", str(data_dir), "--event", "nope", *FAST])
    assert result.exit_code == 1
    assert "Unknown event" in result.output
    assert "pymeets-2026-03" in result.output


def test_tabpfn_trouble_suggests_logistic(data_dir, monkeypatch):
    from ilang_pizza.model import predict
    from ilang_pizza.model.backends import TabPFNUnavailableError

    def unavailable(*args, **kwargs):
        raise TabPFNUnavailableError("TabPFN weights could not be downloaded.")

    monkeypatch.setattr(predict, "forecast_event", unavailable)
    result = runner.invoke(cli.app, ["forecast", "--data", str(data_dir), "--no-weather"])
    assert result.exit_code == 1
    assert "could not be downloaded" in result.output
    assert "--model logistic" in result.output


def test_missing_data_suggests_the_demo(tmp_path):
    result = runner.invoke(cli.app, ["forecast", "--data", str(tmp_path), *FAST])
    assert result.exit_code == 1
    assert "ilang-pizza demo" in result.output


def test_plan_defaults_capacity_to_the_event(data_dir, tiny_dataset):
    payload = json.loads(run("plan", "--data", data_dir, "--json", *FAST).stdout)
    expected = int(tiny_dataset.event(payload["event"]["event_id"])["capacity"])
    assert payload["inputs"]["capacity"] == expected
    assert payload["plan"]["pizza_boxes"] <= payload["plan"]["pizza_boxes_naive"]
    assert payload["synthetic_data"] is True

    custom = json.loads(run("plan", "--data", data_dir, "--json", "--capacity", 12, *FAST).stdout)
    assert custom["inputs"]["capacity"] == 12


def test_plan_shows_pesos_when_priced(data_dir, tiny_dataset):
    result = run("plan", "--data", data_dir, "--price", 450, "--slices", 2.5, *FAST)
    assert "boxes" in result.output
    assert "₱" in result.output
    assert_no_attendee_ids(result.output, tiny_dataset)


def test_backtest_scores_against_the_rule_of_thumb(data_dir, tmp_path):
    csv = tmp_path / "per_event.csv"
    result = run("backtest", "--data", data_dir, "--last-n", 3, "--csv", csv, *FAST)
    assert "Rule of thumb (50%) was off by" in result.output
    assert "80% coverage" in result.output
    assert csv.exists()


# --- LLM commands (templates only) ---------------------------------------------------------


def test_draft_with_the_template(data_dir, tiny_dataset):
    result = run("draft", "--data", data_dir, "--kind", "catering_order", "--no-llm", *FAST)
    assert "Subject:" in result.output
    assert "built-in template" in result.output
    assert "DEMO DATA" in result.output  # synthetic drafts are labeled
    assert_no_attendee_ids(result.output, tiny_dataset)


def test_draft_needs_a_kind(data_dir):
    result = runner.invoke(cli.app, ["draft", "--data", str(data_dir), "--no-llm"])
    assert result.exit_code == 2


def test_ask_without_the_llm(data_dir, tiny_dataset):
    result = run("ask", "Ilang pizza ang i-order?", "--data", data_dir, "--no-llm", *FAST)
    assert "Juan" in result.output
    assert "no LLM" in result.output
    assert_no_attendee_ids(result.output, tiny_dataset)


# --- doctor --------------------------------------------------------------------------------


@pytest.fixture()
def no_tabpfn_import(monkeypatch):
    """doctor's TabPFN check imports torch; the answer is all these tests need."""
    from ilang_pizza.model import backends

    monkeypatch.setattr(backends, "tabpfn_status", lambda: (True, "TabPFN v2 (open weights)"))


def test_doctor_checks_everything(data_dir, monkeypatch, no_tabpfn_import):
    monkeypatch.setattr(
        cli,
        "_check_ollama",
        lambda: [cli._Check("ok", "Ollama", "running"), cli._Check("ok", "Gemma 4", "pulled")],
    )
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    result = run("doctor", "--data", data_dir)
    for text in ("Python", "TabPFN v2", "TABPFN_TOKEN", "ux.priorlabs.ai", "15 events", "Cache"):
        assert text in result.output


def test_doctor_fails_without_data(tmp_path, monkeypatch, no_tabpfn_import):
    monkeypatch.setattr(cli, "_check_ollama", list)
    result = runner.invoke(cli.app, ["doctor", "--data", str(tmp_path)])
    assert result.exit_code == 1
    assert "ilang-pizza demo" in result.output


# --- shared helpers ------------------------------------------------------------------------


def test_resolve_data_dir_finds_the_only_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    assert cli.resolve_data_dir() == tmp_path
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "events.csv").touch()
    assert cli.resolve_data_dir() == tmp_path / "demo"
    (tmp_path / "pythonph").mkdir()
    (tmp_path / "pythonph" / "events.csv").touch()
    assert cli.resolve_data_dir() == tmp_path  # ambiguous: the caller asks for --data
    assert cli.resolve_data_dir(Path("elsewhere")) == Path("elsewhere")


def test_step_summary_keeps_the_numbers_not_the_event_card():
    result = {
        "event": {"event_id": "pymeets-2026-03", "title": "Tiny PyMeets"},
        "inputs": {"capacity": 30},
        "forecast": {"p10": 12, "p50": 15, "p90": 18, "drivers": ["Regulars"]},
        "plan": {"pizza_boxes": 6},
    }
    summary = cli.step_summary(result)
    assert summary == "forecast.p10=12, forecast.p50=15, forecast.p90=18, plan.pizza_boxes=6"
    assert cli.step_summary({"error": "unknown event"}) == "error=unknown event"
    assert len(cli.step_summary({"x": "y" * 500}, limit=50)) == 50


def test_plan_option_defaults_match_plan_inputs():
    defaults = PlanInputs()
    params = inspect.signature(cli.plan).parameters
    assert params["slices"].default == defaults.slices_per_person
    assert params["box"].default == defaults.slices_per_box
    assert params["price"].default == defaults.price_per_box
    assert params["food_quantile"].default == defaults.food_quantile
    assert params["risk"].default == defaults.overflow_risk


def test_choices_match_the_prompts_module():
    from ilang_pizza.llm import prompts

    assert get_args(cli.EmailKind) == prompts.EMAIL_KINDS
    assert tuple(cli.EMAIL_LABELS) == prompts.EMAIL_KINDS
    assert get_args(cli.Tone) == prompts.TONES
