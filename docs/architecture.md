# Architecture

```
exports (Meetup / Luma / ti.to / Google Forms)          synthetic demo data
                    │                                            │
                    ▼                                            ▼
        data.loaders.load_dataset ──────────► Dataset (schema.py) ◄── data.synth
                                                     │
                    features.build.build_features ◄──┤  (calendar_ph, weather)
                                                     ▼
              model.backends (TabPFN, logistic, history, rule)
                                                     │  P(show) per RSVP
                                                     ▼
              model.turnout (Poisson-binomial mixed over a shared event-day shock)
                    │                                │
        model.backtest (rolling origin)     model.predict.forecast_event → TurnoutForecast
                                                     │
                                          plan.make_plan → EventPlan (pizza, seats, waitlist)
                                                     │  aggregate numbers only
                                                     ▼
                         llm.prompts / llm.agent (Gemma 4 via Ollama) → Draft, AgentReply
                                                     │
                                          cli.py (Typer)   ui.py (Gradio)
```

## Invariants

1. **No leakage.** Features for an RSVP use only information available before its event starts.
   Attendee history counts only events that started strictly earlier. An event's own
   attendance is never a feature. Backtests train only on events that started before the
   event being predicted.
2. **Aggregate-only LLM.** Gemma never sees names, emails, or `attendee_id`s. Prompts are built
   from `TurnoutForecast.summary()`, `EventPlan`, and event details (title, date, venue).
   `llm.prompts.assert_aggregate_only` enforces this.
3. **PII is hashed at the door.** Loaders hash email (or name when there is no email) with a
   per-dataset salt and drop raw PII columns immediately.
4. **Synthetic is labeled.** `Dataset.synthetic` propagates into forecasts, backtests, CLI
   output, UI banners, and drafts, so demo numbers are never mistaken for real ones.
5. **Asia/Manila time** everywhere. All timestamps are tz-aware.
6. **Tests need no network, model weights, or Ollama** by default. Tests that do are marked
   `tabpfn`, `network`, or `ollama` and run only when `ILANG_PIZZA_TEST_<MARKER>=1`.

## Data layout

```
data/<name>/
  events.csv              canonical event columns (schema.EVENT_COLUMNS); start in local time
  rsvps/<event_id>.csv    one export per event, any supported format (auto-detected)
  rsvps/<event_id>.xlsx   Meetup organizer exports can be xlsx
  attendance/<event_id>.csv  optional check-in list (emails or names) for exports without one
  weather.csv             optional daily rain: date, precipitation_mm (loaded into Dataset.weather)
  .ilang-pizza-salt       created on first load, never commit
  SYNTHETIC               marker file written by the demo generator
```

Rain lookup order for an event day: `Dataset.weather` table → the `WeatherProvider` passed to
`build_features` → NaN. Online events always get NaN rain.

## Module interfaces

### `ilang_pizza.schema`
`Dataset`, `TurnoutForecast`, `BacktestResult`, `PlanInputs`, `EventPlan`, `Draft`,
`AgentReply`, `EVENT_COLUMNS`, `RSVP_COLUMNS`, `SERIES`, `MANILA_TZ`.

`Dataset.next_event_id(now=None)` is the default event for forecasts, plans, drafts and Ask
Juan: the soonest unlabeled event from now on, else the most recent unlabeled one.
`Dataset.upcoming_event_ids()` lists every unlabeled event, including past untracked ones.

### `ilang_pizza.data.loaders`
- `DataError(ValueError)`: every friendly loading error. CLI and UI print its message, no traceback.
- `load_dataset(data_dir: Path | str) -> Dataset`
- `load_rsvp_export(path: Path, event_id: str, *, salt: str, attendance_path: Path | None = None) -> pd.DataFrame`
- `detect_format(columns: Iterable[str]) -> str`: `"meetup" | "luma" | "tito" | "google_forms" | "canonical"`
- `hash_identity(value: str, salt: str) -> str`
- `get_or_create_salt(data_dir: Path) -> str`

### `ilang_pizza.data.synth`
- `make_synthetic_dataset(n_past: int = 60, n_upcoming: int = 2, seed: int = 2026, today: pd.Timestamp | None = None) -> Dataset`
- `write_dataset(ds: Dataset, out_dir: Path) -> Path`: returns `out_dir`; refuses to overwrite a
  non-empty folder that lacks the `SYNTHETIC` marker (`FileExistsError`).

### `ilang_pizza.features.calendar_ph`
- `days_to_next_payday(d: date) -> int`, `days_since_payday(d: date) -> int`
  (paydays on the 15th and the last day of the month: *kinsenas at katapusan*)
- `is_petsa_de_peligro(d: date, window: int = 5) -> bool`
- `is_ph_holiday(d: date) -> bool`, `holiday_name(d: date) -> str | None`, `is_long_weekend(d: date) -> bool`
- `is_wet_season(d: date) -> bool` (June to November)

### `ilang_pizza.features.weather`
- `WeatherProvider` protocol: `precipitation_mm(lat: float, lon: float, day: date) -> float | None`
- `OpenMeteoWeather(cache_dir: Path = CACHE_DIR, timeout: float = 10.0)`: archive API for past
  days, forecast API up to 16 days ahead, JSON disk cache, returns `None` on any failure.
- `NoWeather`: always `None`.
- `default_provider(offline: bool = False) -> WeatherProvider`: `NoWeather()` when offline,
  otherwise a cached `OpenMeteoWeather()`. CLI and UI use this.
- Optional `prefetch(points: Iterable[tuple[float, float, date]])` on a provider: when present,
  `build_features` calls it first so a backtest makes a handful of requests instead of one per
  event. `OpenMeteoWeather` implements it.

### `ilang_pizza.features.build`
- `FEATURE_COLUMNS: list[str]`, `CATEGORICAL_FEATURES: list[str]` (integer-coded),
  `FEATURE_DESCRIPTIONS: dict[str, str]` (plain-English name per column, for explanations),
  `FEATURE_GROUPS: dict[str, list[str]]` (rsvp, history, event, calendar, weather)
- `build_features(ds: Dataset, *, weather: WeatherProvider | None = None) -> pd.DataFrame`:
  index aligned with `ds.rsvps.index`, float64 columns, NaN allowed.

### `ilang_pizza.model.backends`
- `ShowUpModel` protocol: `name: str`, `fit(X: pd.DataFrame, y: np.ndarray) -> ShowUpModel`,
  `predict_proba(X: pd.DataFrame) -> np.ndarray` (P(show), 1-D)
- `make_model(kind: str = "tabpfn", *, categorical_features: list[str] | None = None, **kw) -> ShowUpModel`;
  kinds: `"tabpfn"` (auto version), `"tabpfn-v2"`, `"tabpfn-3.5"`, `"logistic"`, `"history"`, `"rule"`
- `tabpfn_status() -> tuple[bool, str]`: whether TabPFN can run and which checkpoint "auto" would use.
  TabPFN-3.5 is used only when `TABPFN_TOKEN` is set or explicitly requested. Never triggers an
  interactive license or browser flow.

### `ilang_pizza.model.turnout`
- `heads_pmf(probs: np.ndarray, heads: np.ndarray | None = None, *, shock_sd: float = 0.0) -> np.ndarray`:
  exact Poisson-binomial over integer head weights, mixed over a shared logit shock
  s ~ N(0, shock_sd²) that moves every RSVP's odds together (rain, traffic). `shock_sd = 0`
  is the plain independent Poisson-binomial. `TurnoutForecast.shock_sd` carries the estimate
  (maximum likelihood on earlier events' out-of-sample predictions).
- `add_rsvps(pmf: np.ndarray, probs: np.ndarray, heads: np.ndarray | None = None) -> np.ndarray`
- `quantile(pmf: np.ndarray, q: float) -> int`, `prob_exceeds(pmf: np.ndarray, k: int) -> float` (P(X > k)),
  `expected(pmf: np.ndarray) -> float`

### `ilang_pizza.model.predict`
- `forecast_event(ds: Dataset, event_id: str, *, model_kind: str = "tabpfn", weather: WeatherProvider | None = None, features: pd.DataFrame | None = None) -> TurnoutForecast`

### `ilang_pizza.model.backtest`
- `backtest(ds: Dataset, *, model_kind: str = "tabpfn", min_train_events: int = 6, last_n: int | None = None, weather: WeatherProvider | None = None, rule_rate: float = RULE_OF_THUMB_SHOW_RATE, progress: Callable[[int, int], None] | None = None) -> BacktestResult`

### `ilang_pizza.plan`
- `make_plan(forecast: TurnoutForecast, inputs: PlanInputs) -> EventPlan`

### `ilang_pizza.llm.client`
- `LLM(model: str = LLM_MODEL, host: str = OLLAMA_HOST, timeout: float = 120.0)`:
  `available() -> bool`, `chat(messages: list[dict], *, schema: dict | None = None, temperature: float = 0.4) -> str`
- `FakeLLM(responses: list[str])`: same interface, for tests.

### `ilang_pizza.llm.prompts`
- `EMAIL_KINDS = ("reconfirm", "waitlist_release", "catering_order", "venue_headcount")`, `TONES = ("taglish", "english")`
- `draft_email(kind: str, event: pd.Series | dict, forecast: TurnoutForecast, plan: EventPlan, *, tone: str = "taglish", llm: LLM | None = None) -> Draft`
- `explain_forecast(event, forecast, plan, *, tone: str = "taglish", llm: LLM | None = None) -> str`
- `assert_aggregate_only(text: str) -> None`

### `ilang_pizza.llm.agent`
- `AskJuan(ds: Dataset, llm: LLM | None = None, *, plan_inputs: PlanInputs | None = None, model_kind: str = "tabpfn", weather: WeatherProvider | None = None)`
  with `ask(question: str) -> AgentReply`.

### `ilang_pizza.cli`
Typer `app`: `demo`, `doctor`, `backtest`, `forecast`, `plan`, `draft`, `ask`, `ui`.

### `ilang_pizza.ui`
- `build_app(data_dir: Path | None = None) -> gradio.Blocks`
- `main(data_dir: Path | None = None, host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None`
