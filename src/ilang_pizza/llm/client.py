"""Thin wrapper around a local Ollama server running Gemma 4.

The server runs on the organizer's own machine, and callers only ever send it aggregate
numbers (see ``llm.prompts.assert_aggregate_only``). Every failure surfaces as ``LLMError``
with a message that says how to fix it, so callers can fall back to templates.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Protocol

import httpx
import ollama

from ilang_pizza.config import LLM_MODEL, OLLAMA_HOST

# How long an available() answer is trusted before asking the server again.
STATUS_TTL_SECONDS = 20.0
# Probing should fail fast so a CLI or UI never hangs when Ollama is not running.
PROBE_TIMEOUT_SECONDS = 3.0
# Loading Gemma 4 E4B from disk can take minutes on a busy 16 GB laptop.
WARM_UP_TIMEOUT_SECONDS = 600.0
# Replies here are short emails and JSON actions; a cap stops a runaway reply hogging the
# laptop (a truncated reply fails to parse, and callers fall back to their templates).
MAX_REPLY_TOKENS = 800

_REGISTRY_PREFIXES = ("registry.ollama.ai/", "library/")

# Ollama's Gemma 4 parser moves reasoning into the separate `thinking` field. These catch it
# when it leaks into the text anyway (other models, older servers): <think> tags and Gemma 4's
# "<|channel>thought ... <channel|>" tokens.
_THINKING_BLOCKS = [
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|channel>\s*thought\b.*?<channel\|>", re.DOTALL),
]
_THINKING_OPENERS = re.compile(r"<think>|<thinking>|<\|channel>\s*thought\b", re.IGNORECASE)
_THINKING_CLOSERS = re.compile(r"</think>|</thinking>|<channel\|>", re.IGNORECASE)
_CODE_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


class LLMError(RuntimeError):
    """The local LLM could not answer. The message tells the organizer how to fix it."""


class ChatModel(Protocol):
    """What prompts and the agent need from an LLM: ``LLM`` and ``FakeLLM`` both fit."""

    model: str

    def available(self) -> bool: ...

    def chat(
        self, messages: list[dict], *, schema: dict | None = None, temperature: float = 0.4
    ) -> str: ...


def normalize_model_name(name: str) -> str:
    """Canonical form of an Ollama model name: lowercase, no registry prefix, explicit tag."""
    name = name.strip().lower()
    for prefix in _REGISTRY_PREFIXES:
        name = name.removeprefix(prefix)
    repo, _, tag = name.rpartition(":") if ":" in name.split("/")[-1] else (name, "", "")
    return f"{repo}:{tag or 'latest'}"


def match_model(wanted: str, installed: Iterable[str]) -> str | None:
    """Return the installed name to use for ``wanted``, or None when it is not pulled.

    An exact match wins ("gemma4" means "gemma4:latest"). Otherwise a quantization variant of
    the same tag counts ("gemma4:e4b" accepts "gemma4:e4b-it-q8_0"), but a different size never
    does ("gemma4:e4b" never accepts "gemma4:e2b" or "gemma4:latest").
    """
    installed = [name for name in installed if name]
    target = normalize_model_name(wanted)
    for name in installed:
        if normalize_model_name(name) == target:
            return name
    repo, _, tag = target.rpartition(":")
    if tag == "latest":
        return None
    variants = sorted(
        name for name in installed if normalize_model_name(name).startswith(f"{repo}:{tag}-")
    )
    return variants[0] if variants else None


def strip_thinking(text: str) -> str:
    """Remove reasoning segments so only the model's final answer remains."""
    for pattern in _THINKING_BLOCKS:
        text = pattern.sub("", text)
    opener = _THINKING_OPENERS.search(text)
    if opener:  # unterminated: the model ran out of tokens while thinking
        text = text[: opener.start()]
    closers = list(_THINKING_CLOSERS.finditer(text))
    if closers:  # the template opened the thought for the model; keep what follows it
        text = text[closers[-1].end() :]
    return text.strip()


def parse_json(text: str) -> dict:
    """Parse the JSON object in a model reply, tolerating code fences and chatter around it."""
    fenced = _CODE_FENCE.match(text)
    candidate = (fenced.group(1) if fenced else text).strip()
    decoder = json.JSONDecoder()
    start = candidate.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(candidate, start)
        except json.JSONDecodeError:
            start = candidate.find("{", start + 1)
            continue
        if isinstance(obj, dict):
            return obj
        start = candidate.find("{", start + 1)
    raise LLMError("The model's reply was not the JSON object we asked for.")


def _with_schema_hint(messages: list[dict], schema: dict) -> list[dict]:
    """Describe ``schema`` in the last message, for servers that cannot enforce it."""
    hint = (
        "Reply with only a JSON object that matches this JSON schema, and nothing else:\n"
        + json.dumps(schema, separators=(",", ":"))
    )
    messages = [dict(m) for m in messages]
    messages[-1]["content"] = f"{messages[-1].get('content', '')}\n\n{hint}"
    return messages


class LLM:
    """Chat with a model served by a local Ollama.

    ``think=False`` turns off Gemma 4's reasoning mode, which is on by default: drafts come
    back about 3x faster and are just as good. Pass ``think=None`` to keep the model default.

    ``schema`` is enforced with Ollama structured outputs when the server supports them for
    the model. Ollama's MLX runner (Gemma 4 on Apple Silicon) does not, so the schema is then
    described in the prompt instead and the reply is parsed leniently by ``parse_json``.
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        host: str = OLLAMA_HOST,
        timeout: float = 120.0,
        *,
        think: bool | None = False,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout
        self.think = think
        self.structured_outputs: bool | None = None  # None until the server tells us
        self._client = ollama.Client(host=host, timeout=timeout)
        self._probe = ollama.Client(host=host, timeout=min(timeout, PROBE_TIMEOUT_SECONDS))
        self._checked_at: float | None = None
        self._resolved: str | None = None
        self._problem = ""

    def available(self) -> bool:
        """True when the Ollama server answers and the model (or a variant of it) is pulled."""
        return self._resolve() is not None

    def status(self) -> tuple[bool, str]:
        """``(ready, message)`` for diagnostics such as ``ilang-pizza doctor``."""
        resolved = self._resolve()
        if resolved is None:
            return False, self._problem
        return True, f"{resolved} is ready on Ollama at {self.host}"

    def chat(
        self, messages: list[dict], *, schema: dict | None = None, temperature: float = 0.4
    ) -> str:
        """Send a conversation and return the reply text, with any reasoning removed.

        With ``schema`` (a JSON schema) Ollama constrains the reply to match it.
        """
        name = self._resolve() or self._resolve(refresh=True)
        if name is None:
            raise LLMError(self._problem)
        options = {"temperature": temperature, "num_predict": MAX_REPLY_TOKENS}
        request = {"model": name, "messages": messages, "options": options}
        if schema is not None:
            if self.structured_outputs is False:
                request["messages"] = _with_schema_hint(messages, schema)
            else:
                request["format"] = schema
        if self.think is not None:
            request["think"] = self.think
        with self._friendly_errors(name):
            response = self._send(request, schema)
        text = strip_thinking(response.message.content or "")
        if not text:
            raise LLMError(f"{name} returned an empty answer.")
        return text

    def warm_up(self) -> None:
        """Load the model into memory now, so the first real request is not the slow one.

        Uses a long timeout of its own: a cold load can take minutes on a busy laptop.
        """
        name = self._resolve() or self._resolve(refresh=True)
        if name is None:
            raise LLMError(self._problem)
        loader = ollama.Client(host=self.host, timeout=WARM_UP_TIMEOUT_SECONDS)
        with self._friendly_errors(name, timeout=WARM_UP_TIMEOUT_SECONDS):
            loader.chat(model=name, messages=[])

    def _send(self, request: dict, schema: dict | None) -> ollama.ChatResponse:
        while True:
            try:
                response = self._client.chat(**request)
            except ollama.ResponseError as e:
                error = str(e.error).lower()
                if "format" in request and (e.status_code == 501 or "structured" in error):
                    self.structured_outputs = False
                    request = {k: v for k, v in request.items() if k != "format"}
                    request["messages"] = _with_schema_hint(request["messages"], schema)
                    continue
                if "think" in request and "think" in error:  # model has no reasoning mode
                    self.think = None
                    request = {k: v for k, v in request.items() if k != "think"}
                    continue
                raise
            if "format" in request:
                self.structured_outputs = True
            return response

    @contextmanager
    def _friendly_errors(self, name: str, *, timeout: float | None = None) -> Iterator[None]:
        try:
            yield
        except ollama.ResponseError as e:
            if e.status_code == 404:
                self._forget()
                raise LLMError(self._pull_hint()) from e
            raise LLMError(f"Ollama could not answer with {name}: {e.error}") from e
        except httpx.TimeoutException as e:
            raise LLMError(
                f"{name} took longer than {timeout or self.timeout:.0f}s to answer. The first "
                "request after a while also loads the model; try again, or close other heavy "
                "apps so it has enough memory."
            ) from e
        except (ConnectionError, httpx.HTTPError) as e:
            self._forget()
            raise LLMError(self._unreachable_hint()) from e

    def _resolve(self, *, refresh: bool = False) -> str | None:
        now = time.monotonic()
        fresh = self._checked_at is not None and now - self._checked_at < STATUS_TTL_SECONDS
        if fresh and not refresh:
            return self._resolved
        try:
            installed = [m.model or "" for m in self._probe.list().models]
        except (ollama.ResponseError, ConnectionError, httpx.HTTPError):
            self._resolved, self._problem = None, self._unreachable_hint()
        else:
            self._resolved = match_model(self.model, installed)
            self._problem = "" if self._resolved else self._pull_hint()
        self._checked_at = now
        return self._resolved

    def _forget(self) -> None:
        self._checked_at = None

    def _pull_hint(self) -> str:
        return (
            f"Ollama is running but {self.model} is not pulled yet. "
            f"Run `ollama pull {self.model}` (a one-time download of several GB)."
        )

    def _unreachable_hint(self) -> str:
        return (
            f"Can't reach Ollama at {self.host}. Start it with `ollama serve` (or open the "
            f"Ollama app), then run `ollama pull {self.model}`."
        )


class FakeLLM:
    """Scripted stand-in for ``LLM`` in tests.

    Replays ``responses`` in order (an Exception in the list is raised instead) and records
    every call in ``calls`` so tests can assert exactly what would have been sent.
    """

    def __init__(
        self,
        responses: Sequence[str | Exception],
        *,
        model: str = "fake-gemma",
        available: bool = True,
    ) -> None:
        self.model = model
        self._responses = list(responses)
        self._available = available
        self.calls: list[dict] = []

    def available(self) -> bool:
        return self._available

    def chat(
        self, messages: list[dict], *, schema: dict | None = None, temperature: float = 0.4
    ) -> str:
        self.calls.append(
            {"messages": [dict(m) for m in messages], "schema": schema, "temperature": temperature}
        )
        if not self._responses:
            raise LLMError("FakeLLM has no scripted responses left.")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return strip_thinking(response)

    @property
    def sent_text(self) -> str:
        """Everything sent so far, as one string, for privacy assertions."""
        return "\n".join(str(m.get("content", "")) for c in self.calls for m in c["messages"])
