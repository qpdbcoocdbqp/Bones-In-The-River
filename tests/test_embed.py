import pytest

from factcheck.module.utils.embed import OpenAICompatEmbedder, make_reranker
from factcheck.module.utils.llm import LLMError

_NAMES = (
    "BASE_URL",
    "API_KEY",
    "EMBED_MODEL",
    "EMBED_BASE_URL",
    "EMBED_API_KEY",
    "RERANK_MODEL",
    "RERANK_BASE_URL",
    "RERANK_API_KEY",
)


@pytest.fixture
def env(tmp_path, monkeypatch):
    for name in _NAMES:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"

    def write(text: str):
        path.write_text(text, encoding="utf-8")
        return path

    return write


def test_embedder_falls_back_to_chat_settings(env):
    path = env("BASE_URL=http://chat:9/v1\nAPI_KEY=chat-key\nEMBED_MODEL=e\n")
    embedder = OpenAICompatEmbedder(env_file=path)
    assert str(embedder._client.base_url).startswith("http://chat:9/v1")
    assert embedder._client.api_key == "chat-key"


def test_embedder_prefixed_settings_beat_chat_settings(env):
    path = env(
        "BASE_URL=http://chat:9/v1\nAPI_KEY=chat-key\n"
        "EMBED_MODEL=e\nEMBED_BASE_URL=http://embed:9/v1\nEMBED_API_KEY=embed-key\n"
    )
    embedder = OpenAICompatEmbedder(env_file=path)
    assert str(embedder._client.base_url).startswith("http://embed:9/v1")
    assert embedder._client.api_key == "embed-key"


def test_embedder_explicit_arguments_beat_prefixed_settings(env):
    path = env("EMBED_MODEL=e\nEMBED_BASE_URL=http://embed:9/v1\nEMBED_API_KEY=embed-key\n")
    embedder = OpenAICompatEmbedder(env_file=path, base_url="http://arg:9/v1", api_key="arg-key")
    assert str(embedder._client.base_url).startswith("http://arg:9/v1")
    assert embedder._client.api_key == "arg-key"


def test_embedder_missing_settings_names_them(env):
    with pytest.raises(LLMError, match=r"EMBED_MODEL.*BASE_URL.*API_KEY"):
        OpenAICompatEmbedder(env_file=env(""))


def test_reranker_none_without_model(env):
    assert make_reranker(env_file=env("BASE_URL=http://chat:9/v1\nAPI_KEY=k\n")) is None


def test_reranker_explicit_arguments_beat_settings(env):
    path = env(
        "BASE_URL=http://chat:9/v1\nAPI_KEY=chat-key\n"
        "RERANK_MODEL=r\nRERANK_BASE_URL=http://rr:9/v1\nRERANK_API_KEY=rr-key\n"
    )
    from_env = make_reranker(env_file=path)
    assert from_env is not None
    assert from_env._url == "http://rr:9/v1/rerank"
    assert from_env._headers == {"Authorization": "Bearer rr-key"}

    explicit = make_reranker(env_file=path, base_url="http://arg:9/v1", api_key="arg-key")
    assert explicit is not None
    assert explicit._url == "http://arg:9/v1/rerank"
    assert explicit._headers == {"Authorization": "Bearer arg-key"}


def test_reranker_model_without_endpoint_raises(env):
    with pytest.raises(LLMError, match="BASE_URL or API_KEY"):
        make_reranker(env_file=env("RERANK_MODEL=r\n"))
