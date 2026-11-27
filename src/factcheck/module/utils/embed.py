from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import httpx

from .llm import LLMError, LLMTransportError


class Embedder(Protocol):
    model: str

    def embed(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        """Vectors for `texts`; query=True for search queries (asymmetric models use different
        prefixes for queries and documents)."""
        ...


class Reranker(Protocol):
    model: str

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        """One relevance score per document, in the order given (higher = more relevant)."""
        ...


def _setting(name: str, env_file: Path, default: str) -> str:
    """One setting, where an empty value in the environment / .env is a real value (no prefix)."""
    from dotenv import dotenv_values

    if name in os.environ:
        return os.environ[name]
    values = dotenv_values(env_file) if env_file.exists() else {}
    value = values.get(name)
    return default if value is None else value


def _resolve(
    prefix: str,
    model: str | None,
    env_file: Path,
    base_url: str | None = None,
    api_key: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """(model, base_url, api_key): explicit arguments first, then `<prefix>_MODEL/_BASE_URL/
    _API_KEY`, falling back to the chat settings BASE_URL / API_KEY. Environment variables take
    precedence over the .env file."""
    from dotenv import dotenv_values  # lazy, like OpenAICompatClient

    file_values = dotenv_values(env_file) if env_file.exists() else {}

    def setting(name: str) -> str | None:
        return os.environ.get(name) or file_values.get(name)

    return (
        model or setting(f"{prefix}_MODEL"),
        base_url or setting(f"{prefix}_BASE_URL") or setting("BASE_URL"),
        api_key or setting(f"{prefix}_API_KEY") or setting("API_KEY"),
    )


# Task prefixes the model was trained with. Without them embeddinggemma-2 recalled 3/9 quotes in
# the top 3 passages, with them 8/9 (examples/shopify).
_PREFIXES = {
    "embeddinggemma": ("task: search result | query: ", "title: none | text: "),
}


def _default_prefixes(model: str) -> tuple[str, str]:
    name = model.lower()
    return next((p for key, p in _PREFIXES.items() if key in name), ("", ""))


class OpenAICompatEmbedder:
    """`/v1/embeddings` of an OpenAI-compatible server (llama.cpp, vLLM, LM Studio, ...).

    Settings: EMBED_MODEL (required), EMBED_BASE_URL / EMBED_API_KEY (default to BASE_URL /
    API_KEY), EMBED_QUERY_PREFIX / EMBED_DOC_PREFIX (default: the model's own task prefixes, known
    for embeddinggemma). The source is English and the statements Chinese, so use a multilingual
    model (e.g. embeddinggemma, bge-m3).
    """

    def __init__(
        self,
        model: str | None = None,
        env_file: Path = Path(".env"),
        batch_size: int = 32,
        timeout: float = 300.0,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        from openai import OpenAI

        name, base_url, api_key = _resolve("EMBED", model, env_file, base_url, api_key)
        missing = [
            n
            for n, v in (("EMBED_MODEL", name), ("BASE_URL", base_url), ("API_KEY", api_key))
            if not v
        ]
        if missing or not name:
            raise LLMError(
                f"Missing settings: {', '.join(missing)} "
                f"(put them in {env_file} or the environment)"
            )
        self.model = name
        default_query, default_doc = _default_prefixes(name)
        self._query_prefix = _setting("EMBED_QUERY_PREFIX", env_file, default_query)
        self._doc_prefix = _setting("EMBED_DOC_PREFIX", env_file, default_doc)
        self._batch_size = batch_size
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def embed(self, texts: Sequence[str], *, query: bool = False) -> list[list[float]]:
        prefix = self._query_prefix if query else self._doc_prefix
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [prefix + t for t in texts[start : start + self._batch_size]]
            try:
                response = self._client.embeddings.create(model=self.model, input=batch)
            except Exception as exc:  # same reasoning as OpenAICompatClient.complete
                raise LLMTransportError(
                    f"Embedding call failed: {type(exc).__name__}: {exc}"
                ) from exc
            data = sorted(response.data, key=lambda d: d.index)
            if len(data) != len(batch):
                raise LLMTransportError(
                    f"Embedding response has {len(data)} vectors for {len(batch)} texts"
                )
            vectors.extend(list(d.embedding) for d in data)
        return vectors


class HTTPReranker:
    """`POST {base_url}/rerank` with `{model, query, documents}` -> `{results: [{index,
    relevance_score}]}`, the Jina / Cohere shape that vLLM and llama.cpp servers also speak."""

    def __init__(self, model: str, base_url: str, api_key: str, timeout: float = 120.0) -> None:
        self.model = model
        self._url = base_url.rstrip("/") + "/rerank"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout = timeout

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        payload = {"model": self.model, "query": query, "documents": list(documents)}
        try:
            response = httpx.post(
                self._url, json=payload, headers=self._headers, timeout=self._timeout
            )
            response.raise_for_status()
            scores = [float("-inf")] * len(documents)
            for item in response.json()["results"]:
                scores[int(item["index"])] = float(item["relevance_score"])
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMTransportError(f"Rerank call failed: {type(exc).__name__}: {exc}") from exc
        return scores


def make_reranker(
    model: str | None = None,
    env_file: Path = Path(".env"),
    base_url: str | None = None,
    api_key: str | None = None,
) -> HTTPReranker | None:
    """A reranker from RERANK_MODEL / RERANK_BASE_URL / RERANK_API_KEY (the last two default to
    BASE_URL / API_KEY). None if no rerank model is configured."""
    name, base_url, api_key = _resolve("RERANK", model, env_file, base_url, api_key)
    if not name:
        return None
    if not base_url or not api_key:
        raise LLMError(f"Missing BASE_URL or API_KEY for the reranker (put them in {env_file})")
    return HTTPReranker(name, base_url, api_key)
