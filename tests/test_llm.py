import pytest
from pydantic import BaseModel

from factcheck.llm import (
    LLMError,
    LLMFormatError,
    LLMTransportError,
    OpenAICompatClient,
    complete_json,
    extract_json,
)
from tests.fakes import FakeClient


class Answer(BaseModel):
    value: int


def test_extract_json_ignores_surrounding_prose_and_fences():
    assert extract_json('Sure!\n```json\n{"value": 3}\n```') == {"value": 3}


def test_extract_json_drops_think_block_with_braces():
    text = '<think>maybe {"value": 1} or not</think>\n{"value": 2}'
    assert extract_json(text) == {"value": 2}


def _clear_settings(monkeypatch):
    for name in ("BASE_URL", "API_KEY", "MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_openai_client_reads_settings_from_env_file(tmp_path, monkeypatch):
    _clear_settings(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("BASE_URL=http://127.0.0.1:9/v1\nMODEL=local-m\nAPI_KEY=k\n", encoding="utf-8")
    assert OpenAICompatClient(env_file=env).model == "local-m"
    assert OpenAICompatClient("override", env_file=env).model == "override"


def test_openai_client_env_var_beats_env_file(tmp_path, monkeypatch):
    _clear_settings(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("BASE_URL=http://127.0.0.1:9/v1\nMODEL=from-file\nAPI_KEY=k\n", encoding="utf-8")
    monkeypatch.setenv("MODEL", "from-env")
    assert OpenAICompatClient(env_file=env).model == "from-env"


def test_openai_client_missing_settings_names_them(tmp_path, monkeypatch):
    _clear_settings(monkeypatch)
    env = tmp_path / ".env"
    env.write_text("MODEL=m\n", encoding="utf-8")
    with pytest.raises(LLMError, match=r"BASE_URL.*API_KEY"):
        OpenAICompatClient(env_file=env)


def test_extract_json_without_object_raises():
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_complete_json_retries_until_valid():
    client = FakeClient(["garbage", '{"value": "not an int"}', '{"value": 7}'])
    assert complete_json(client, "p", Answer, retries=2).value == 7
    assert len(client.prompts) == 3


def test_complete_json_gives_up_after_retries():
    client = FakeClient(["garbage"] * 3)
    with pytest.raises(LLMError):
        complete_json(client, "p", Answer, retries=2)
    assert len(client.prompts) == 3


def test_retry_prompt_includes_the_rejection_reason():
    client = FakeClient(["garbage", '{"value": 1}'])
    complete_json(client, "原始問題", Answer, retries=2)
    assert client.prompts[0] == "原始問題"
    assert "原始問題" in client.prompts[1]
    assert "previous answer was rejected" in client.prompts[1]


def test_transport_errors_are_not_retried():
    class Broken:
        model = "m"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str, system: str | None = None) -> str:
            self.calls += 1
            raise LLMTransportError("401")

    client = Broken()
    with pytest.raises(LLMTransportError):
        complete_json(client, "p", Answer, retries=5)
    assert client.calls == 1


def test_format_failure_after_retries_is_a_format_error():
    with pytest.raises(LLMFormatError):
        complete_json(FakeClient(["garbage"] * 3), "p", Answer, retries=2)
    assert issubclass(LLMFormatError, LLMError)
    assert issubclass(LLMTransportError, LLMError)
