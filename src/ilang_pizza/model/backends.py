"""Show-up models: P(show) for each RSVP, from the feature matrix.

Every backend has the same small interface (``fit``, then ``predict_proba``) and returns
probabilities in [0, 1], so the rest of the pipeline never cares which one ran:

- ``tabpfn``: Prior Labs' tabular foundation model. The past RSVPs are its context; there is
  no training loop and nothing to tune.
- ``logistic``: a regularized logistic regression, the classic baseline.
- ``history``: each series' past show-up rate.
- ``rule``: the organizers' rule of thumb ("about half will show").

TabPFN v2's weights are openly licensed (Prior Labs License: Apache 2.0 plus attribution)
and download without an account. TabPFN-3.5's weights are non-commercial and need a license
that the user accepts at https://ux.priorlabs.ai. "auto" picks 3.5 only when ``TABPFN_TOKEN``
is set, and nothing here ever opens a browser or waits for a license prompt.
"""

from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from ilang_pizza import config

logger = logging.getLogger(__name__)

KINDS = ("tabpfn", "tabpfn-v2", "tabpfn-3.5", "logistic", "history", "rule")

TABPFN_LICENSE_URL = "https://ux.priorlabs.ai"
# The v2 weights' license asks products built on them to show this line.
TABPFN_ATTRIBUTION = "Built with PriorLabs-TabPFN"

# TabPFN defaults, benchmarked on an Apple M5 (16 GB) with ~25 features: one fit + predict
# takes ~4 s on MPS, so a forecast stays under 10 s and a 20-event backtest under 3 min.
# In a 20-event backtest, 4 estimators scored as well as 8 at half the time; 3,000 context
# rows cut the error ~7% more but took twice as long.
DEFAULT_N_ESTIMATORS = 4
DEFAULT_MAX_CONTEXT_ROWS = 2_000

# The history baseline groups RSVPs by this integer-coded feature from features.build.
SERIES_FEATURE = "series_code"

# TabPFN versions that failed to load in this process ("auto" then stays on v2).
_UNAVAILABLE: set[str] = set()

_VERSION_ALIASES = {
    "auto": "auto",
    "v2": "v2",
    "2": "v2",
    "tabpfn-v2": "v2",
    "3.5": "3.5",
    "v3.5": "3.5",
    "tabpfn-3.5": "3.5",
}


class TabPFNUnavailableError(RuntimeError):
    """TabPFN's weights could not be loaded: no download, no license, or a broken install."""


@runtime_checkable
class ShowUpModel(Protocol):
    """Anything that learns P(show) from labeled RSVPs."""

    name: str

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> ShowUpModel: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


def make_model(
    kind: str = "tabpfn", *, categorical_features: list[str] | None = None, **kw
) -> ShowUpModel:
    """Build a show-up model by kind; ``kw`` goes to the backend's constructor.

    ``"tabpfn"`` picks its version from ``ILANG_PIZZA_TABPFN`` (default ``auto``).
    """
    cats = list(categorical_features or [])
    if kind == "tabpfn":
        return TabPFNModel(config.TABPFN_VERSION, categorical_features=cats, **kw)
    if kind == "tabpfn-v2":
        return TabPFNModel("v2", categorical_features=cats, **kw)
    if kind == "tabpfn-3.5":
        return TabPFNModel("3.5", categorical_features=cats, **kw)
    if kind == "logistic":
        return LogisticModel(categorical_features=cats, **kw)
    if kind == "history":
        return HistoryModel(**kw)
    if kind == "rule":
        return RuleOfThumbModel(**kw)
    raise ValueError(f"unknown model kind {kind!r}; choose one of: {', '.join(KINDS)}")


class RuleOfThumbModel:
    """Everyone shows up with the same probability (default ``RULE_OF_THUMB_SHOW_RATE``)."""

    name = "rule"

    def __init__(self, rate: float | None = None) -> None:
        self.rate = config.RULE_OF_THUMB_SHOW_RATE if rate is None else float(rate)
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {self.rate}")

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> RuleOfThumbModel:
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.rate)


class HistoryModel:
    """Each series' past show-up rate; the overall rate for series with little history."""

    name = "history"

    def __init__(self, group_feature: str = SERIES_FEATURE, min_rsvps: int = 20) -> None:
        self.group_feature = group_feature
        self.min_rsvps = min_rsvps

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> HistoryModel:
        y = _labels(y)
        self.overall_ = float(y.mean())
        self.rates_: dict[float, float] = {}
        if self.group_feature in X.columns:
            stats = pd.Series(y, index=X.index).groupby(X[self.group_feature]).agg(["mean", "size"])
            self.rates_ = stats.loc[stats["size"] >= self.min_rsvps, "mean"].to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.rates_ or self.group_feature not in X.columns:
            return np.full(len(X), self.overall_)
        return X[self.group_feature].map(self.rates_).fillna(self.overall_).to_numpy(float)


class LogisticModel:
    """Logistic regression with median imputation, missing flags and one-hot categories.

    ``C=0.1`` regularizes firmly: event-level features take only a few dozen distinct values,
    and a looser fit extrapolated badly to new events.
    """

    name = "logistic"

    def __init__(self, categorical_features: Sequence[str] = (), C: float = 0.1) -> None:
        self.categorical_features = list(categorical_features)
        self.C = C

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> LogisticModel:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        y = _labels(y)
        self.columns_ = list(X.columns)
        self.constant_ = _one_class_rate(y)
        if self.constant_ is not None:
            return self

        cats = [c for c in self.columns_ if c in self.categorical_features]
        nums = [c for c in self.columns_ if c not in cats]
        steps = []
        if nums:
            numeric = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True), StandardScaler()
            )
            steps.append(("numeric", numeric, nums))
        if cats:  # NaN becomes its own category; unseen codes get all-zero columns
            steps.append(("categorical", OneHotEncoder(handle_unknown="ignore"), cats))
        self.pipeline_ = make_pipeline(
            ColumnTransformer(steps), LogisticRegression(C=self.C, max_iter=1000)
        )
        self.pipeline_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.constant_ is not None:
            return np.full(len(X), self.constant_)
        return self.pipeline_.predict_proba(X[self.columns_])[:, 1]


class TabPFNModel:
    """TabPFN as a show-up classifier.

    Args:
        version: ``"auto"``, ``"v2"`` or ``"3.5"``. ``"auto"`` uses 3.5 only when
            ``TABPFN_TOKEN`` is set, and falls back to v2 if 3.5 cannot load.
        categorical_features: integer-coded columns that TabPFN should treat as categories.
        device: ``"auto"`` (CUDA, then Apple's MPS, then CPU) or a torch device string. On an
            M5, MPS was 7-10x faster than the CPU and ~2x faster than routing attention
            through MLX, with equivalent predictions.
        n_estimators: ensemble members; each sees a differently preprocessed view of the data.
        max_context_rows: past RSVPs TabPFN reads at most. Larger histories are cut down by a
            deterministic, class-stratified sample, which keeps runtime bounded (cost grows
            roughly with the square of the context size).
        random_state: seed for that sample and for TabPFN's ensemble.

    Blank (NaN) values get an explicit 0/1 ``<column>_blank`` column and a median fill.
    TabPFN reads NaN by itself, but as "missing at random", while a blank in RSVP data
    usually means something: never attended, no capacity limit, online so no rain. Left to
    TabPFN, never-attended people were scored like regulars (0.50 predicted vs 0.30 actual
    for online first-timers in a backtest); with the flag they came out at 0.33.
    """

    def __init__(
        self,
        version: str = "auto",
        *,
        categorical_features: Sequence[str] = (),
        device: str = "auto",
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_context_rows: int = DEFAULT_MAX_CONTEXT_ROWS,
        random_state: int = 0,
    ) -> None:
        self.requested_version = _normalize_version(version)
        self.version = resolve_tabpfn_version(self.requested_version)
        self.name = f"tabpfn-{self.version}"
        self.categorical_features = list(categorical_features)
        self.device = device
        self.n_estimators = n_estimators
        self.max_context_rows = max_context_rows
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> TabPFNModel:
        y = _labels(y)
        self.columns_ = list(X.columns)
        self.constant_ = _one_class_rate(y)
        if self.constant_ is not None:
            return self

        categorical = [i for i, c in enumerate(self.columns_) if c in self.categorical_features]
        try:
            self.classifier_ = self._load(self.version, categorical)
        except TabPFNUnavailableError as error:
            if self.requested_version != "auto" or self.version == "v2":
                raise
            logger.warning("%s Using the open TabPFN v2 weights instead.", error)
            _UNAVAILABLE.add(self.version)  # don't retry for every event of a backtest
            self.version, self.name = "v2", "tabpfn-v2"
            self.classifier_ = self._load("v2", categorical)

        numeric = [c for c in self.columns_ if c not in self.categorical_features]
        self.fill_ = X[numeric].median()
        patterns: dict[bytes, str] = {}  # one flag per distinct blank pattern
        for column in numeric:
            blank = X[column].isna().to_numpy()
            if blank.any():
                patterns.setdefault(blank.tobytes(), column)
        self.flagged_ = list(patterns.values())

        rows = stratified_cap(y, self.max_context_rows, seed=self.random_state)
        with _quiet_tabpfn():
            self.classifier_.fit(self._with_blank_flags(X).iloc[rows], y[rows])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.constant_ is not None:
            return np.full(len(X), self.constant_)
        with _quiet_tabpfn():
            proba = self.classifier_.predict_proba(self._with_blank_flags(X))
        shows = list(self.classifier_.classes_).index(1)
        return np.clip(proba[:, shows], 0.0, 1.0)

    def _with_blank_flags(self, X: pd.DataFrame) -> pd.DataFrame:
        """The fitted columns, blanks filled with the median, plus a flag per blank pattern."""
        X = X[self.columns_]
        flags = X[self.flagged_].isna().astype(float).add_suffix("_blank")
        return pd.concat([X.fillna(self.fill_), flags], axis=1)

    def _load(self, version: str, categorical: list[int]):
        """A TabPFN classifier with its weights loaded (downloaded on first use)."""
        from tabpfn import TabPFNClassifier

        classifier = TabPFNClassifier.create_default_for_version(
            _model_version(version),
            n_estimators=self.n_estimators,
            categorical_features_indices=categorical or None,
            device=self.device,
            # The context is capped in fit(). This also lifts v2's 1,000-row limit on CPU.
            ignore_pretraining_limits=True,
            random_state=self.random_state,
        )
        try:
            with _quiet_tabpfn(), _license_prompts_disabled():
                classifier.get_inference_config()  # loads the checkpoint
        except Exception as error:
            raise TabPFNUnavailableError(_load_failure(version, error)) from error
        return classifier


def stratified_cap(y: np.ndarray, max_rows: int, *, seed: int = 0) -> np.ndarray:
    """Sorted indices of at most ``max_rows`` rows with the class mix of ``y``.

    Deterministic: the same labels and seed always pick the same rows.
    """
    y = np.asarray(y)
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    if y.size <= max_rows:
        return np.arange(y.size)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    # Largest-remainder rounding, so the per-class quotas add up to exactly max_rows.
    quotas = counts * max_rows / y.size
    take = np.floor(quotas).astype(int)
    take[np.argsort(take - quotas)[: max_rows - take.sum()]] += 1
    picked = [
        rng.choice(np.flatnonzero(y == label), size=n, replace=False)
        for label, n in zip(classes, take)
    ]
    return np.sort(np.concatenate(picked))


def resolve_tabpfn_version(requested: str = "auto") -> str:
    """``"v2"`` or ``"3.5"``. ``"auto"`` means 3.5 when ``TABPFN_TOKEN`` is set, else v2.

    Once 3.5 has failed to load in this process, ``"auto"`` stays on v2.
    """
    version = _normalize_version(requested)
    if version != "auto":
        return version
    has_token = bool(os.getenv("TABPFN_TOKEN", "").strip())
    return "3.5" if has_token and "3.5" not in _UNAVAILABLE else "v2"


def tabpfn_status() -> tuple[bool, str]:
    """Whether TabPFN can run here, and which checkpoint ``make_model("tabpfn")`` would use.

    Instant and offline: it reads the environment and the local weight cache only, so it never
    downloads, checks a license or opens a browser.
    """
    try:
        requested = _normalize_version(config.TABPFN_VERSION)
    except ValueError as e:
        return False, f"ILANG_PIZZA_TABPFN: {e}"
    try:
        from tabpfn.utils import infer_devices
    except ImportError as e:
        return False, f"the tabpfn package is not installed ({e})"

    version = resolve_tabpfn_version(requested)
    device = infer_devices("auto")[0].type
    cached = _checkpoint_path(version).exists()
    if version == "v2":
        weights = "weights cached" if cached else "~30 MB of weights download on first use"
        return True, f"TabPFN v2 on {device}, {weights}"
    if cached:
        return True, f"TabPFN-3.5 on {device}, weights cached"
    if os.getenv("TABPFN_TOKEN", "").strip():
        fallback = ", falls back to v2 if the license check fails" if requested == "auto" else ""
        return True, f"TabPFN-3.5 on {device}, weights download on first use{fallback}"
    return False, (
        "TabPFN-3.5 was requested but TABPFN_TOKEN is not set. Accept its license at "
        f"{TABPFN_LICENSE_URL} and export TABPFN_TOKEN, or set ILANG_PIZZA_TABPFN=v2"
    )


def _normalize_version(requested: str) -> str:
    version = _VERSION_ALIASES.get(str(requested).strip().lower())
    if version is None:
        raise ValueError(f"unknown TabPFN version {requested!r}; use auto, v2 or 3.5")
    return version


def _model_version(version: str):
    from tabpfn.constants import ModelVersion

    return {"v2": ModelVersion.V2, "3.5": ModelVersion.V3_5}[version]


def _checkpoint_path(version: str) -> Path:
    from tabpfn.model_loading import ModelSource, get_cache_dir

    source = ModelSource.get_classifier_v2() if version == "v2" else ModelSource.get_v3_5()
    return get_cache_dir() / source.default_filename


def _load_failure(version: str, error: Exception) -> str:
    if version == "v2":
        hint = "Its weights download from Hugging Face on first use, so check your connection."
    else:
        hint = (
            f"TabPFN-3.5 needs its non-commercial license accepted at {TABPFN_LICENSE_URL} "
            "and TABPFN_TOKEN set to your API key. TabPFN v2 (tabpfn-v2) needs neither."
        )
    return f"TabPFN {version} could not load ({type(error).__name__}: {error}). {hint}"


@contextmanager
def _license_prompts_disabled() -> Iterator[None]:
    """Make TabPFN raise instead of opening a browser for a gated download without a token."""
    previous = os.environ.get("TABPFN_NO_BROWSER")
    os.environ["TABPFN_NO_BROWSER"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TABPFN_NO_BROWSER", None)
        else:
            os.environ["TABPFN_NO_BROWSER"] = previous


@contextmanager
def _quiet_tabpfn() -> Iterator[None]:
    """Hide TabPFN's advisory warnings (e.g. "CPU may be slow"); errors still raise."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module=r"tabpfn(\.|$)")
        yield


def _labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    if y.size == 0 or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y must be a non-empty array of 0/1 attendance labels")
    return y.astype(int)


def _one_class_rate(y: np.ndarray) -> float | None:
    """Laplace-smoothed show rate when every label is the same (nothing to learn), else None."""
    if np.unique(y).size > 1:
        return None
    return (y.sum() + 1.0) / (y.size + 2.0)
