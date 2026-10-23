from __future__ import annotations

import httpx
import ollama
import pytest

from ilang_pizza.llm import client
from ilang_pizza.llm.client import (
    LLM,
    FakeLLM,
    LLMError,
    match_model,
    normalize_model_name,
    parse_json,
    strip_thinking,
)

SCHEMA = {"type": "object", "properties": {"subject": {"type": "string"}}, "required": ["subject"]}


class FakeServer:
    """Stands in for a local Ollama: which models are pulled, and scripted chat replies."""

    def __init__(self) -> None:
        self.up = True
        self.installed = ["gemma4:e4b"]
        self.replies: list[str | Exception] = []
        self.list_calls = 0
        self.chat_calls: list[dict] = []
        self.chat_timeouts: list[float | None] = []


class FakeOllamaClient:
    def __init__(self, server: FakeServer, host=None, **kwargs) -> None:
        self.server = server
        self.host = host
        self.timeout = kwargs.get("timeout")

    def list(self) -> ollama.ListResponse:
        self.server.list_calls += 1
        if not self.server.up:
            raise ConnectionError("Failed to connect to Ollama.")
        models = [ollama.ListResponse.Model(model=name) for name in self.server.installed]
        return ollama.ListResponse(models=models)

    def chat(self, **request) -> ollama.ChatResponse:
        self.server.chat_calls.append(request)
        self.server.chat_timeouts.append(self.timeout)
        reply = self.server.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        message = ollama.Message(role="assistant", content=reply)
        return ollama.ChatResponse(model=request.get("model"), message=message)


@pytest.fixture
def server(monkeypatch) -> FakeServer:
    srv = FakeServer()
    monkeypatch.setattr(
        client.ollama, "Client", lambda host=None, **kw: FakeOllamaClient(srv, host, **kw)
    )
    return srv


# --- model names -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "normal"),
    [
        ("gemma4", "gemma4:latest"),
        ("Gemma4:E4B", "gemma4:e4b"),
        ("registry.ollama.ai/library/gemma4:e4b", "gemma4:e4b"),
        ("hf.co/unsloth/gemma-4-E4B-it-GGUF:Q4_K_M", "hf.co/unsloth/gemma-4-e4b-it-gguf:q4_k_m"),
        ("localhost:5000/gemma4", "localhost:5000/gemma4:latest"),
    ],
)
def test_normalize_model_name(name, normal):
    assert normalize_model_name(name) == normal


@pytest.mark.parametrize(
    ("wanted", "installed", "expected"),
    [
        ("gemma4:e4b", ["gemma4:e4b"], "gemma4:e4b"),
        ("gemma4:e4b", ["llama3.2:latest", "gemma4:e4b"], "gemma4:e4b"),
        ("gemma4", ["gemma4:latest"], "gemma4:latest"),
        ("gemma4:latest", ["gemma4"], "gemma4"),
        ("gemma4:e4b", ["library/gemma4:e4b"], "library/gemma4:e4b"),
        # a quantization variant of the same size is fine...
        ("gemma4:e4b", ["gemma4:e4b-it-q8_0"], "gemma4:e4b-it-q8_0"),
        ("gemma4:e4b", ["gemma4:e4b-it-qat", "gemma4:e4b-it-bf16"], "gemma4:e4b-it-bf16"),
        # ...but the exact tag wins over variants
        ("gemma4:e4b", ["gemma4:e4b-it-q8_0", "gemma4:e4b"], "gemma4:e4b"),
        # a different size or family never counts
        ("gemma4:e4b", ["gemma4:e2b"], None),
        ("gemma4:e4b", ["gemma4:latest"], None),
        ("gemma4:e4b", ["gemma4:e4"], None),
        ("gemma4:e4b", ["gemma3:e4b", "gemma4-custom:e4b"], None),
        ("gemma4", ["gemma4:e4b"], None),
        ("gemma4:e4b", [], None),
    ],
)
def test_match_model(wanted, installed, expected):
    assert match_model(wanted, installed) == expected


# --- reply cleanup -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("Hello!", "Hello!"),
        ("<think>count the boxes</think>\n9 boxes", "9 boxes"),
        ("<THINKING>hmm</THINKING>Answer", "Answer"),
        ("<|channel>thought\nlet me see<channel|>Final answer", "Final answer"),
        ("Answer first <think>then rambling that never ends", "Answer first"),
        ("reasoning the template opened</think>\nThe answer", "The answer"),
    ],
)
def test_strip_thinking(raw, clean):
    assert strip_thinking(raw) == clean


@pytest.mark.parametrize(
    "raw",
    [
        '{"subject": "Hi", "body": "Kita-kits!"}',
        '```json\n{"subject": "Hi", "body": "Kita-kits!"}\n```',
        'Sure! Here it is:\n{"subject": "Hi", "body": "Kita-kits!"}\nHope that helps.',
        'Using {braces} first, then {"subject": "Hi", "body": "Kita-kits!"}',
    ],
)
def test_parse_json_tolerates_fences_and_chatter(raw):
    assert parse_json(raw) == {"subject": "Hi", "body": "Kita-kits!"}


@pytest.mark.parametrize("raw", ["", "no json here", "[1, 2, 3]", '{"subject": "unterminated'])
def test_parse_json_rejects_non_objects(raw):
    with pytest.raises(LLMError):
        parse_json(raw)


# --- availability ------------------------------------------------------------------------


def test_available_when_model_is_pulled(server):
    llm = LLM()
    assert llm.available()
    ready, message = llm.status()
    assert ready and "gemma4:e4b" in message


def test_available_is_cached_briefly(server, monkeypatch):
    llm = LLM()
    assert llm.available() and llm.available() and llm.available()
    assert server.list_calls == 1
    monkeypatch.setattr(client, "STATUS_TTL_SECONDS", 0.0)
    assert llm.available()
    assert server.list_calls == 2


def test_unavailable_when_server_is_down(server):
    server.up = False
    llm = LLM()
    assert not llm.available()
    ready, message = llm.status()
    assert not ready
    assert "ollama serve" in message and "ollama pull gemma4:e4b" in message


def test_unavailable_when_model_is_not_pulled(server):
    server.installed = ["llama3.2:latest", "gemma4:e2b"]
    llm = LLM()
    assert not llm.available()
    assert "ollama pull gemma4:e4b" in llm.status()[1]


def test_probe_uses_a_short_timeout(server):
    llm = LLM(timeout=120.0)
    assert llm._probe.timeout == client.PROBE_TIMEOUT_SECONDS
    assert llm._client.timeout == 120.0


# --- chat --------------------------------------------------------------------------------


def test_chat_sends_schema_temperature_and_think(server):
    server.replies = ['{"subject": "Hi"}']
    llm = LLM()
    messages = [{"role": "user", "content": "Draft it"}]
    assert llm.chat(messages, schema=SCHEMA, temperature=0.2) == '{"subject": "Hi"}'
    request = server.chat_calls[0]
    assert request["model"] == "gemma4:e4b"
    assert request["format"] == SCHEMA
    assert request["options"] == {"temperature": 0.2, "num_predict": client.MAX_REPLY_TOKENS}
    assert request["think"] is False
    assert request["messages"] == messages
    assert llm.structured_outputs is True


def test_chat_uses_the_installed_variant(server):
    server.installed = ["gemma4:e4b-it-q8_0"]
    server.replies = ["ok"]
    LLM().chat([{"role": "user", "content": "hi"}])
    assert server.chat_calls[0]["model"] == "gemma4:e4b-it-q8_0"


def test_chat_strips_thinking(server):
    server.replies = ["<think>3 slices each...</think>Order 9 boxes."]
    assert LLM().chat([{"role": "user", "content": "hi"}]) == "Order 9 boxes."


def test_chat_falls_back_when_structured_outputs_are_unavailable(server):
    # Ollama's MLX runner (Gemma 4 on Apple Silicon) answers 501 to any `format`.
    server.replies = [
        ollama.ResponseError("structured output is unavailable", 501),
        '```json\n{"subject": "Hi"}\n```',
        '{"subject": "Again"}',
    ]
    llm = LLM()
    messages = [{"role": "user", "content": "Draft it"}]
    assert parse_json(llm.chat(messages, schema=SCHEMA)) == {"subject": "Hi"}
    retry = server.chat_calls[1]
    assert "format" not in retry
    assert retry["messages"][-1]["content"].startswith("Draft it")
    assert '"required":["subject"]' in retry["messages"][-1]["content"]
    assert messages == [{"role": "user", "content": "Draft it"}]  # caller's list untouched
    assert llm.structured_outputs is False

    llm.chat(messages, schema=SCHEMA)  # remembered: no second 501 round trip
    assert len(server.chat_calls) == 3
    assert "format" not in server.chat_calls[2]


def test_chat_retries_without_think_for_models_without_reasoning(server):
    server.replies = [ollama.ResponseError('"llama3.2" does not support thinking', 400), "ok"]
    llm = LLM(model="llama3.2")
    server.installed = ["llama3.2:latest"]
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert "think" not in server.chat_calls[1]
    assert llm.think is None


@pytest.mark.parametrize(
    ("error", "hint"),
    [
        (ollama.ResponseError("model 'gemma4:e4b' not found", 404), "ollama pull gemma4:e4b"),
        (ConnectionError("Failed to connect to Ollama."), "ollama serve"),
        (httpx.ReadTimeout("timed out"), "took longer than 120s"),
        (ollama.ResponseError("out of memory", 500), "out of memory"),
    ],
)
def test_chat_errors_are_friendly(server, error, hint):
    server.replies = [error]
    with pytest.raises(LLMError, match=hint):
        LLM().chat([{"role": "user", "content": "hi"}])


def test_chat_refuses_early_when_unavailable(server):
    server.up = False
    with pytest.raises(LLMError, match="ollama serve"):
        LLM().chat([{"role": "user", "content": "hi"}])
    assert server.chat_calls == []


def test_chat_rejects_empty_replies(server):
    server.replies = ["<think>only thoughts</think>"]
    with pytest.raises(LLMError, match="empty"):
        LLM().chat([{"role": "user", "content": "hi"}])


def test_warm_up_loads_the_model_with_a_long_timeout(server):
    server.replies = [""]
    LLM(timeout=120.0).warm_up()
    assert server.chat_calls == [{"model": "gemma4:e4b", "messages": []}]
    assert server.chat_timeouts == [client.WARM_UP_TIMEOUT_SECONDS]


# --- FakeLLM -----------------------------------------------------------------------------


def test_fake_llm_replays_and_records():
    fake = FakeLLM(["first", "<think>hmm</think>second", LLMError("boom")])
    assert fake.available()
    assert fake.chat([{"role": "user", "content": "one"}], schema=SCHEMA) == "first"
    assert fake.chat([{"role": "user", "content": "two"}], temperature=0.1) == "second"
    with pytest.raises(LLMError, match="boom"):
        fake.chat([{"role": "user", "content": "three"}])
    with pytest.raises(LLMError, match="no scripted responses"):
        fake.chat([{"role": "user", "content": "four"}])
    assert [c["messages"][0]["content"] for c in fake.calls] == ["one", "two", "three", "four"]
    assert fake.calls[0]["schema"] == SCHEMA
    assert fake.calls[1]["temperature"] == 0.1
    assert "two" in fake.sent_text


def test_fake_llm_copies_messages():
    fake = FakeLLM(["ok"])
    messages = [{"role": "user", "content": "before"}]
    fake.chat(messages)
    messages[0]["content"] = "after"
    assert fake.calls[0]["messages"][0]["content"] == "before"


def test_fake_llm_can_be_unavailable():
    assert not FakeLLM([], available=False).available()


# --- the real thing ----------------------------------------------------------------------


@pytest.mark.ollama
def test_real_gemma_answers_json():
    llm = LLM()
    assert llm.available(), llm.status()[1]
    messages = [
        {
            "role": "user",
            "content": 'Reply with a JSON object {"subject": ...} for a PyMeets reminder email.',
        }
    ]
    reply = parse_json(llm.chat(messages, schema=SCHEMA, temperature=0.2))
    assert isinstance(reply.get("subject"), str) and reply["subject"].strip()
